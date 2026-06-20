// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_intersection.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshIntersectionLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_intersection.h"
#include "app_cli_options.h"
#include "../utilities/PairHitTracking.h"
#include "../utilities/ContainmentTracking.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
};

enum class QueryDirection {
    Both,
    Mesh1ToMesh2,
    Mesh2ToMesh1
};

QueryDirection parseQueryDirection(const std::string& direction) {
    if (direction == "both") {
        return QueryDirection::Both;
    }
    if (direction == "mesh1_to_mesh2") {
        return QueryDirection::Mesh1ToMesh2;
    }
    if (direction == "mesh2_to_mesh1") {
        return QueryDirection::Mesh2ToMesh1;
    }
    throw std::invalid_argument("Invalid query direction: " + direction);
}

void writeIntersectionPairsCsv(
    const std::string& outputPath,
    const std::vector<MeshQueryResult>& pairs
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open intersection pairs output: " + outputPath);
    }

    out << "a_object_id,b_object_id\n";
    for (const auto& p : pairs) {
        out << p.object_id_mesh1 << ',' << p.object_id_mesh2 << "\n";
    }
}

// Execute the intersection query using hash table deduplication
QueryResults executeHashQuery(
    MeshIntersectionLauncher& intersectionLauncher,
    MeshIntersectionLaunchParams& params1,
    MeshIntersectionLaunchParams& params2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    int mesh1NumObjects,
    int mesh2NumObjects,
    unsigned long long* d_hash_table,
    int hash_table_size,
    QueryDirection queryDirection,
    PerformanceTimer* timer = nullptr,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    params1.use_hash_table = true;
    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    
    params2.use_hash_table = true;
    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;

    const bool runMesh1ToMesh2 = (queryDirection == QueryDirection::Both || queryDirection == QueryDirection::Mesh1ToMesh2);
    const bool runMesh2ToMesh1 = (queryDirection == QueryDirection::Both || queryDirection == QueryDirection::Mesh2ToMesh1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    if (runMesh1ToMesh2) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchOverlapMesh1ToMesh2(params1, mesh1NumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh2ToMesh1) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchOverlapMesh2ToMesh1(params2, mesh2NumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh1ToMesh2) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh1ToMesh2(params1, mesh1NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (runMesh2ToMesh1) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh2ToMesh1(params2, mesh2NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    int max_output = hash_table_size; 

    MeshQueryResult* d_merged_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshQueryResult)));

    auto t_dedup_start = std::chrono::high_resolution_clock::now();
    int numUnique = compact_hash_table_pairs(d_hash_table, hash_table_size, d_merged_results, max_output);
    auto t_dedup_end = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "compact_hash_table_pairs",
            std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_end - t_dedup_start).count()
        );
    }
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
    }
    
    return {d_merged_results, numUnique};
}


// Helper to calculate global average size of objects from grid statistics
float calculateGlobalAvgSize(const std::vector<SparseGridEntry>& sparseCells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            // Un-average to get sum of sizes in this cell
            totalSize += (double)entry.stats.AvgSizeMean * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 0.0f;
    return (float)(totalSize / totalCount);
}

// Helper to calculate global average VolRatio from grid statistics
float calculateGlobalAvgVolRatio(const std::vector<SparseGridEntry>& sparseCells) {
    double totalRatio = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            totalRatio += (double)entry.stats.VolRatio * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 1.0f;  // Default to 1.0 (no correction)
    return (float)(totalRatio / totalCount);
}

static long long estimateIntersectionPairs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    float epsilon,
    float gamma,
    bool verbose
) {
    long long estimatedPairs = 0;

    if (mesh1.grid.hasGrid && mesh2.grid.hasGrid) {
        if (std::abs(mesh1.grid.cellSize - mesh2.grid.cellSize) > 1e-5f) {
            if (verbose) {
                std::cerr << "Warning: Grid cell sizes mismatch (Mesh1: " << mesh1.grid.cellSize
                          << ", Mesh2: " << mesh2.grid.cellSize << "). Estimation may be invalid." << std::endl;
            }
        }

        float cellVolume = mesh1.grid.cellSize * mesh1.grid.cellSize * mesh1.grid.cellSize;

        struct Int3Hash {
            size_t operator()(const int3& k) const {
                return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
            }
        };
        struct Int3Equal {
            bool operator()(const int3& a, const int3& b) const {
                return a.x == b.x && a.y == b.y && a.z == b.z;
            }
        };

        std::unordered_map<int3, GridCell, Int3Hash, Int3Equal> mapA;
        for (const auto& entry : mesh1.grid.sparseCells) {
            mapA[entry.index] = entry.stats;
        }

        std::vector<GridCell> matchedA;
        std::vector<GridCell> matchedB;

        for (const auto& entry : mesh2.grid.sparseCells) {
            auto it = mapA.find(entry.index);
            if (it != mapA.end()) {
                matchedA.push_back(it->second);
                matchedB.push_back(entry.stats);
            }
        }

        int numMatchedCells = matchedA.size();

        float estimatedPairsFloat = 0.0f;
        if (numMatchedCells > 0) {
            estimatedPairsFloat = estimateIntersectionSelectivity(
                matchedA.data(),
                matchedB.data(),
                numMatchedCells,
                cellVolume,
                epsilon,
                gamma
            );
        }

        float avgSize1 = calculateGlobalAvgSize(mesh1.grid.sparseCells);
        float avgSize2 = calculateGlobalAvgSize(mesh2.grid.sparseCells);
        float avgVolRatio1 = calculateGlobalAvgVolRatio(mesh1.grid.sparseCells);
        float avgVolRatio2 = calculateGlobalAvgVolRatio(mesh2.grid.sparseCells);

        float effectiveSize1 = avgSize1 * std::cbrt(avgVolRatio1);
        float effectiveSize2 = avgSize2 * std::cbrt(avgVolRatio2);

        float combinedSize = effectiveSize1 + effectiveSize2;
        float minkowskiVol = combinedSize * combinedSize * combinedSize;

        if (cellVolume < 1e-9f) cellVolume = 1e-9f;

        float alpha = minkowskiVol / cellVolume;
        if (alpha < 1.0f) alpha = 1.0f;

        estimatedPairs = (long long)(estimatedPairsFloat / alpha);

        if (verbose) {
            std::cout << "\n=== Selectivity Estimation ===" << std::endl;
            std::cout << "Matched Sparse Cells:      " << numMatchedCells << std::endl;
            std::cout << "Raw Potential Pairs:       " << (long long)estimatedPairsFloat << std::endl;
            std::cout << "Avg Object Size (Mesh1):   " << avgSize1 << std::endl;
            std::cout << "Avg Object Size (Mesh2):   " << avgSize2 << std::endl;
            std::cout << "Avg VolRatio (Mesh1):      " << avgVolRatio1 << std::endl;
            std::cout << "Avg VolRatio (Mesh2):      " << avgVolRatio2 << std::endl;
            std::cout << "Effective Size (Mesh1):    " << effectiveSize1 << std::endl;
            std::cout << "Effective Size (Mesh2):    " << effectiveSize2 << std::endl;
            std::cout << "Replication Factor (alpha):" << alpha << std::endl;
            std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
            std::cout << "==============================\n" << std::endl;
        }
    } else if (verbose) {
        std::cout << "Skipping estimation: Grid data not found in one or both datasets." << std::endl;
        std::cout << "Run pierce_preprocess with --generate-grid to enable estimation." << std::endl;
    }

    return estimatedPairs;
}

static int chooseIntersectionHashTableSize(long long estimatedPairs, float hashLoadFactor) {
    int hashTableSize = 16777216;
    if (estimatedPairs > 0) {
        unsigned long long target = (unsigned long long)(estimatedPairs / hashLoadFactor);
        if (target < 1024) target = 1024;

        // Cap to reasonable int size for hash table param
        if (target > 1073741824ULL) target = 1073741824ULL;
        hashTableSize = (int)target;
        if (hashTableSize % 2 == 0) {
            hashTableSize += 1;
        }
    }
    return hashTableSize;
}

class IntersectionEstimatedCliOptions : public BenchmarkMeshPairCliOptions {
public:
    IntersectionEstimatedCliOptions() : BenchmarkMeshPairCliOptions("estimated_intersection_timing.json") {}

    std::string queryDirectionArg = "both";
    bool estimateOnly = false;
    bool enableTracking = false;
    bool enableProfilingStats = false;
    std::string pairsOutputPath;
    std::string pairHitsOutputPath;
    std::string containmentTrackingOutputPath;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    float hashLoadFactor = 0.5f;
    int overlapMaxIterations = 100;

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(options);
        appendBenchmarkRunHelp(options);
        options.emplace_back("--gamma <float>", "Estimation gamma (default: 0.8)");
        options.emplace_back("--epsilon <float>", "Estimation epsilon (default: 0.001)");
        options.emplace_back("--estimate-only", "Run only selectivity estimation");
        options.emplace_back("--query-direction <both|mesh1_to_mesh2|mesh2_to_mesh1>", "Control query direction (default: both)");
        options.emplace_back("--overlap-max-iterations <int>", "Overlap ray iteration cap (default: 100)");
        options.emplace_back("--hash-load-factor <float>", "Hash load factor in (0,1] (default: 0.5)");
        options.emplace_back("--enable-tracking", "Enable optional containment candidate tracking and CSV exports");
        options.emplace_back("--enable-profiling-stats", "Enable device-side profiling counters");
        options.emplace_back("--pairs-output <path>", "Intersection pairs CSV path (default: intersection_pairs.csv)");
        options.emplace_back("--pair-hits-output <path>", "Pair hit tracking CSV path (default: intersection_pair_hits.csv)");
        options.emplace_back("--containment-tracking-output <path>", "Containment tracking CSV path (default: intersection_containment_tracking.csv)");
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "--mesh1 <path> --mesh2 <path> [options]",
            "Intersection estimated query: overlap plus containment passes with hash-based deduplication.",
            options
        );
    }

protected:
    bool parseApplicationOption(const std::string& arg, int& i, int argc, char* argv[]) override {
        if (arg == "--gamma" && i + 1 < argc) {
            gamma = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--epsilon" && i + 1 < argc) {
            epsilon = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--estimate-only") {
            estimateOnly = true;
            return true;
        }
        if (arg == "--query-direction" && i + 1 < argc) {
            queryDirectionArg = argv[++i];
            return true;
        }
        if (arg == "--overlap-max-iterations" && i + 1 < argc) {
            overlapMaxIterations = std::stoi(argv[++i]);
            return true;
        }
        if (arg == "--hash-load-factor" && i + 1 < argc) {
            hashLoadFactor = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--enable-tracking") {
            enableTracking = true;
            return true;
        }
        if (arg == "--enable-profiling-stats") {
            enableProfilingStats = true;
            return true;
        }
        if (arg == "--pairs-output" && i + 1 < argc) {
            pairsOutputPath = argv[++i];
            return true;
        }
        if (arg == "--pair-hits-output" && i + 1 < argc) {
            pairHitsOutputPath = argv[++i];
            return true;
        }
        if (arg == "--containment-tracking-output" && i + 1 < argc) {
            containmentTrackingOutputPath = argv[++i];
            return true;
        }
        return false;
    }
};

static unsigned long long packIntersectionPairKey(int mesh1ObjectId, int mesh2ObjectId) {
    return (static_cast<unsigned long long>(static_cast<unsigned int>(mesh1ObjectId)) << 32) |
        static_cast<unsigned long long>(static_cast<unsigned int>(mesh2ObjectId));
}

static void writeIntersectionTrackingCsv(
    const std::string& outputPath,
    int maxTargetsPerSource,
    const std::vector<int>& mesh1TargetIds,
    const std::vector<unsigned int>& mesh1TargetHits,
    const std::vector<int>& mesh2TargetIds,
    const std::vector<unsigned int>& mesh2TargetHits,
    const std::unordered_set<unsigned long long>& finalPairs
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open intersection tracking output: " + outputPath);
    }

    out << "direction,source_object_id,target_object_id,target_ray_hits,final_pair\n";

    const int mesh1Sources = static_cast<int>(mesh1TargetIds.size()) / maxTargetsPerSource;
    for (int src = 0; src < mesh1Sources; ++src) {
        const int base = src * maxTargetsPerSource;
        for (int i = 0; i < maxTargetsPerSource; ++i) {
            const int tgt = mesh1TargetIds[base + i];
            if (tgt < 0) {
                continue;
            }
            const bool isFinalPair = finalPairs.count(packIntersectionPairKey(src, tgt)) > 0;
            out << "mesh1_to_mesh2," << src << ',' << tgt << ',' << mesh1TargetHits[base + i] << ','
                << (isFinalPair ? 1 : 0) << "\n";
        }
    }

    const int mesh2Sources = static_cast<int>(mesh2TargetIds.size()) / maxTargetsPerSource;
    for (int src = 0; src < mesh2Sources; ++src) {
        const int base = src * maxTargetsPerSource;
        for (int i = 0; i < maxTargetsPerSource; ++i) {
            const int tgt = mesh2TargetIds[base + i];
            if (tgt < 0) {
                continue;
            }
            const bool isFinalPair = finalPairs.count(packIntersectionPairKey(tgt, src)) > 0;
            out << "mesh2_to_mesh1," << src << ',' << tgt << ',' << mesh2TargetHits[base + i] << ','
                << (isFinalPair ? 1 : 0) << "\n";
        }
    }
}

static void writeIntersectionTrackingSummaryCsv(
    const std::string& outputPath,
    int maxTargetsPerSource,
    const std::vector<int>& mesh1TargetIds,
    const std::vector<unsigned int>& mesh1TargetHits,
    const std::vector<unsigned int>& mesh1Iterations,
    const std::vector<unsigned int>& mesh1CandidateCounts,
    const std::vector<unsigned int>& mesh1CandidateOverflows,
    const std::vector<int>& mesh2TargetIds,
    const std::vector<unsigned int>& mesh2TargetHits,
    const std::vector<unsigned int>& mesh2Iterations,
    const std::vector<unsigned int>& mesh2CandidateCounts,
    const std::vector<unsigned int>& mesh2CandidateOverflows,
    const std::unordered_set<unsigned long long>& finalPairs
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open containment tracking output: " + outputPath);
    }

    out << "direction,source_object_id,iterations,candidate_count,tracked_target_count,tracked_hit_total,candidate_overflow_events,final_pair_count,tracked_final_pair_count\n";

    const int mesh1Sources = static_cast<int>(mesh1Iterations.size());
    for (int src = 0; src < mesh1Sources; ++src) {
        const int base = src * maxTargetsPerSource;
        int trackedTargetCount = 0;
        unsigned long long trackedHitTotal = 0;
        int trackedFinalPairCount = 0;
        for (int i = 0; i < maxTargetsPerSource; ++i) {
            const int tgt = mesh1TargetIds[base + i];
            if (tgt < 0) {
                continue;
            }
            trackedTargetCount++;
            trackedHitTotal += mesh1TargetHits[base + i];
            if (finalPairs.count(packIntersectionPairKey(src, tgt)) > 0) {
                trackedFinalPairCount++;
            }
        }

        const int finalPairCount = trackedFinalPairCount;
        out << "mesh1_to_mesh2," << src << ',' << mesh1Iterations[src] << ','
            << mesh1CandidateCounts[src] << ',' << trackedTargetCount << ',' << trackedHitTotal << ','
            << mesh1CandidateOverflows[src] << ',' << finalPairCount << ',' << trackedFinalPairCount << "\n";
    }

    const int mesh2Sources = static_cast<int>(mesh2Iterations.size());
    for (int src = 0; src < mesh2Sources; ++src) {
        const int base = src * maxTargetsPerSource;
        int trackedTargetCount = 0;
        unsigned long long trackedHitTotal = 0;
        int trackedFinalPairCount = 0;
        for (int i = 0; i < maxTargetsPerSource; ++i) {
            const int tgt = mesh2TargetIds[base + i];
            if (tgt < 0) {
                continue;
            }
            trackedTargetCount++;
            trackedHitTotal += mesh2TargetHits[base + i];
            if (finalPairs.count(packIntersectionPairKey(tgt, src)) > 0) {
                trackedFinalPairCount++;
            }
        }

        const int finalPairCount = trackedFinalPairCount;
        out << "mesh2_to_mesh1," << src << ',' << mesh2Iterations[src] << ','
            << mesh2CandidateCounts[src] << ',' << trackedTargetCount << ',' << trackedHitTotal << ','
            << mesh2CandidateOverflows[src] << ',' << finalPairCount << ',' << trackedFinalPairCount << "\n";
    }
}

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    IntersectionEstimatedCliOptions options;
    options.ptxPath = detectPTXPath("mesh_intersection.ptx");
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }

    const std::string& mesh1Path = options.mesh1Path;
    const std::string& mesh2Path = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const std::string& queryDirectionArg = options.queryDirectionArg;
    const bool estimateOnly = options.estimateOnly;
    const bool enableTracking = options.enableTracking;
    const bool enableProfilingStats = options.enableProfilingStats;
    const std::string pairsOutputPath = options.pairsOutputPath.empty()
        ? "intersection_pairs.csv"
        : options.pairsOutputPath;
    const std::string pairHitsOutputPath = options.pairHitsOutputPath.empty()
        ? "intersection_pair_hits.csv"
        : options.pairHitsOutputPath;
    const std::string containmentTrackingOutputPath = options.containmentTrackingOutputPath.empty()
        ? "intersection_containment_tracking.csv"
        : options.containmentTrackingOutputPath;
    const float gamma = options.gamma;
    const float epsilon = options.epsilon;
    const float hashLoadFactor = options.hashLoadFactor;
    const int overlapMaxIterations = options.overlapMaxIterations;
    const int warmupRuns = options.warmupRuns;

    QueryDirection queryDirection = QueryDirection::Both;
    try {
        queryDirection = parseQueryDirection(queryDirectionArg);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << "Valid values for --query-direction are: both, mesh1_to_mesh2, mesh2_to_mesh1" << std::endl;
        return 1;
    }

    if (hashLoadFactor <= 0.0f || hashLoadFactor > 1.0f) {
        std::cerr << "Invalid --hash-load-factor. Expected value in (0, 1]." << std::endl;
        return 1;
    }
    if (overlapMaxIterations <= 0) {
        std::cerr << "--overlap-max-iterations must be > 0." << std::endl;
        return 1;
    }
    
    if (!options.hasRequiredMeshInputs()) {
        std::cerr << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
        return 1;
    }
    
    timer.start("Load Mesh1");
    GeometryData mesh1 = loadGeometryFromFile(mesh1Path);
    if (mesh1.vertices.empty()) {
        std::cerr << "Error loading mesh1." << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh1, mesh1Path, "Mesh1")) {
        return 1;
    }
    
    timer.next("Load Mesh2");
    GeometryData mesh2 = loadGeometryFromFile(mesh2Path);
    
    if (mesh2.vertices.empty()) {
        std::cerr << "Error loading mesh2." << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh2, mesh2Path, "Mesh2")) {
        return 1;
    }

    if (warmupRuns > 0) {
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            (void)estimateIntersectionPairs(mesh1, mesh2, epsilon, gamma, false);
        }
    }

    // --- ESTIMATION PHASE ---
    timer.next("Selectivity Estimation");
    long long estimatedPairs = estimateIntersectionPairs(mesh1, mesh2, epsilon, gamma, false);

    int hash_table_size = chooseIntersectionHashTableSize(estimatedPairs, hashLoadFactor);

    if (estimateOnly) {
        std::cout << "\n=== Query Configuration ===" << std::endl;
        std::cout << "Estimated Pairs:    " << estimatedPairs << std::endl;
        std::cout << "Hash Table Size:    " << hash_table_size << " (Load Factor ~" << hashLoadFactor << ")" << std::endl;
        std::cout << "Query Direction:    " << queryDirectionArg << std::endl;
        std::cout << "Overlap Max Iter:   " << overlapMaxIterations << std::endl;
        std::cout << "Tracking:           " << (enableTracking ? "enabled" : "disabled") << std::endl;
        std::cout << "Profiling Stats:    " << (enableProfilingStats ? "enabled" : "disabled") << std::endl;
        std::cout << "===========================\n" << std::endl;
        timer.finish(outputJsonPath);
        return 0;
    }

    timer.next("Init OptiX");

    std::cout << "\n=== Query Configuration ===" << std::endl;
    std::cout << "Estimated Pairs:    " << estimatedPairs << std::endl;
    std::cout << "Hash Table Size:    " << hash_table_size << " (Load Factor ~" << hashLoadFactor << ")" << std::endl;
    std::cout << "Query Direction:    " << queryDirectionArg << std::endl;
    std::cout << "Overlap Max Iter:   " << overlapMaxIterations << std::endl;
    std::cout << "Tracking:           " << (enableTracking ? "enabled" : "disabled") << std::endl;
    std::cout << "Profiling Stats:    " << (enableProfilingStats ? "enabled" : "disabled") << std::endl;
    std::cout << "===========================\n" << std::endl;

    // --- EXECUTION PHASE ---
    // Create OptiX context and pipeline (reuse existing project patterns)
    OptixContext context;
    MeshIntersectionLauncher intersectionLauncher(context, options.ptxPath);

    timer.next("Upload Mesh1");

    GeometryUploader mesh1Uploader;
    mesh1Uploader.upload(mesh1);

    timer.next("Upload Mesh2");
    GeometryUploader mesh2Uploader;
    mesh2Uploader.upload(mesh2);

    timer.next("Build Mesh1 Index");
    OptixAccelerationStructure mesh1AS(context, mesh1Uploader);
    mesh1AS.build();

    timer.next("Build Mesh2 Index");
    OptixAccelerationStructure mesh2AS(context, mesh2Uploader);
    mesh2AS.build();

    timer.next("Prepare Kernel Parameters");

    int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());
    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh1);
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh2);
    int mesh1NumEdges = mesh1EdgeData.num_edges;
    int mesh2NumEdges = mesh2EdgeData.num_edges;

    // Allocate Hash Table
    unsigned long long* d_hash_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

    MeshIntersectionProfilingStats* d_profiling_stats = nullptr;
    MeshIntersectionProfilingStats h_profiling_stats = {};
    if (enableProfilingStats) {
        CUDA_CHECK(cudaMalloc(&d_profiling_stats, sizeof(MeshIntersectionProfilingStats)));
        CUDA_CHECK(cudaMemset(d_profiling_stats, 0, sizeof(MeshIntersectionProfilingStats)));
    }

    // Allocate object tracking buffers
    std::set<int> mesh1Objects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    std::set<int> mesh2Objects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh1NumObjects = mesh1Objects.size();
    int mesh2NumObjects = mesh2Objects.size();

    std::vector<int> firstTriangleMesh1(mesh1NumObjects, -1);
    std::vector<int> firstTriangleMesh2(mesh2NumObjects, -1);
    for (int tri = 0; tri < mesh1NumTriangles; ++tri) {
        int obj = mesh1.triangleToObject[tri];
        if (firstTriangleMesh1[obj] == -1) {
            firstTriangleMesh1[obj] = tri;
        }
    }
    for (int tri = 0; tri < mesh2NumTriangles; ++tri) {
        int obj = mesh2.triangleToObject[tri];
        if (firstTriangleMesh2[obj] == -1) {
            firstTriangleMesh2[obj] = tri;
        }
    }

    int* d_first_triangle_mesh1 = nullptr;
    int* d_first_triangle_mesh2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh1, mesh1NumObjects * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh2, mesh2NumObjects * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_first_triangle_mesh1, firstTriangleMesh1.data(), mesh1NumObjects * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_first_triangle_mesh2, firstTriangleMesh2.data(), mesh2NumObjects * sizeof(int), cudaMemcpyHostToDevice));

    constexpr int kAnyhitMaxTargetsPerSource = 256;
    int* d_anyhit_candidate_object_ids_mesh1 = nullptr;
    unsigned int* d_anyhit_candidate_parity_mesh1 = nullptr;
    unsigned int* d_anyhit_candidate_hit_counts_mesh1 = nullptr;
    unsigned int* d_anyhit_candidate_count_mesh1 = nullptr;
    unsigned int* d_anyhit_candidate_overflow_mesh1 = nullptr;
    int* d_anyhit_candidate_object_ids_mesh2 = nullptr;
    unsigned int* d_anyhit_candidate_parity_mesh2 = nullptr;
    unsigned int* d_anyhit_candidate_hit_counts_mesh2 = nullptr;
    unsigned int* d_anyhit_candidate_count_mesh2 = nullptr;
    unsigned int* d_anyhit_candidate_overflow_mesh2 = nullptr;

    const size_t anyhitSlotsMesh1 = static_cast<size_t>(mesh1NumObjects) * static_cast<size_t>(kAnyhitMaxTargetsPerSource);
    const size_t anyhitSlotsMesh2 = static_cast<size_t>(mesh2NumObjects) * static_cast<size_t>(kAnyhitMaxTargetsPerSource);
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_object_ids_mesh1, anyhitSlotsMesh1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_parity_mesh1, anyhitSlotsMesh1 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_hit_counts_mesh1, anyhitSlotsMesh1 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_count_mesh1, mesh1NumObjects * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_overflow_mesh1, mesh1NumObjects * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_object_ids_mesh2, anyhitSlotsMesh2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_parity_mesh2, anyhitSlotsMesh2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_hit_counts_mesh2, anyhitSlotsMesh2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_count_mesh2, mesh2NumObjects * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_anyhit_candidate_overflow_mesh2, mesh2NumObjects * sizeof(unsigned int)));

    MeshIntersectionLaunchParams params1{};
    params1.mesh1_vertices = mesh1Uploader.getVertices();
    params1.mesh1_indices = mesh1Uploader.getIndices();
    params1.mesh1_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh1_num_objects = mesh1NumObjects;
    params1.edge_starts = mesh1EdgeData.d_edge_starts;
    params1.edge_ends = mesh1EdgeData.d_edge_ends;
    params1.edge_source_object_ids = mesh1EdgeData.d_source_object_ids;
    params1.num_edges = mesh1NumEdges;
    params1.mesh2_handle = mesh2AS.getHandle();
    params1.mesh2_vertices = mesh2Uploader.getVertices();
    params1.mesh2_indices = mesh2Uploader.getIndices();
    params1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params1.mesh2_num_objects = mesh2NumObjects;

    MeshIntersectionLaunchParams params2{};
    params2.mesh1_vertices = mesh2Uploader.getVertices();
    params2.mesh1_indices = mesh2Uploader.getIndices();
    params2.mesh1_triangle_to_object = mesh2Uploader.getTriangleToObject();
    params2.mesh1_num_triangles = mesh2NumTriangles;
    params2.mesh1_num_objects = mesh2NumObjects;
    params2.edge_starts = mesh2EdgeData.d_edge_starts;
    params2.edge_ends = mesh2EdgeData.d_edge_ends;
    params2.edge_source_object_ids = mesh2EdgeData.d_source_object_ids;
    params2.num_edges = mesh2NumEdges;
    params2.mesh2_handle = mesh1AS.getHandle();
    params2.mesh2_vertices = mesh1Uploader.getVertices();
    params2.mesh2_indices = mesh1Uploader.getIndices();
    params2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    params2.mesh2_num_objects = mesh1NumObjects;

    params1.hash_table = d_hash_table;
    params1.hash_table_size = hash_table_size;
    params1.use_hash_table = true;

    params2.hash_table = d_hash_table;
    params2.hash_table_size = hash_table_size;
    params2.use_hash_table = true;

    params1.first_triangle_index_per_object = d_first_triangle_mesh1;
    params2.first_triangle_index_per_object = d_first_triangle_mesh2;

    params1.overlap_max_iterations = overlapMaxIterations;
    params1.profiling_enabled = enableProfilingStats ? 1 : 0;
    params1.profiling_stats = d_profiling_stats;
    params1.anyhit_max_pair_targets_per_source = kAnyhitMaxTargetsPerSource;
    params1.anyhit_candidate_object_ids = d_anyhit_candidate_object_ids_mesh1;
    params1.anyhit_candidate_parity = d_anyhit_candidate_parity_mesh1;
    params1.anyhit_candidate_hit_counts = d_anyhit_candidate_hit_counts_mesh1;
    params1.anyhit_candidate_count_per_source = d_anyhit_candidate_count_mesh1;
    params1.anyhit_candidate_overflow_per_source = d_anyhit_candidate_overflow_mesh1;

    params2.overlap_max_iterations = overlapMaxIterations;
    params2.profiling_enabled = enableProfilingStats ? 1 : 0;
    params2.profiling_stats = d_profiling_stats;
    params2.anyhit_max_pair_targets_per_source = kAnyhitMaxTargetsPerSource;
    params2.anyhit_candidate_object_ids = d_anyhit_candidate_object_ids_mesh2;
    params2.anyhit_candidate_parity = d_anyhit_candidate_parity_mesh2;
    params2.anyhit_candidate_hit_counts = d_anyhit_candidate_hit_counts_mesh2;
    params2.anyhit_candidate_count_per_source = d_anyhit_candidate_count_mesh2;
    params2.anyhit_candidate_overflow_per_source = d_anyhit_candidate_overflow_mesh2;

    PairHitTrackingBuffers pairHitBuffers;
    ContainmentTrackingBuffers containmentTrackingBuffers;
    if (enableTracking) {
        pairHitBuffers.allocate(mesh1NumObjects, mesh2NumObjects);
        pairHitBuffers.setupLaunchParams(params1, params2);
        containmentTrackingBuffers.allocate(mesh1NumObjects, mesh2NumObjects);
        containmentTrackingBuffers.setupLaunchParams(params1, params2);
    } else {
        params1.enable_pair_hit_tracking = 0;
        params1.max_pair_targets_per_source = 0;
        params1.pair_target_object_ids = nullptr;
        params1.pair_target_hit_counts = nullptr;
        params2.enable_pair_hit_tracking = 0;
        params2.max_pair_targets_per_source = 0;
        params2.pair_target_object_ids = nullptr;
        params2.pair_target_hit_counts = nullptr;

        params1.enable_containment_tracking = 0;
        params1.containment_iterations_per_source = nullptr;
        params1.containment_candidate_count_per_source = nullptr;
        params1.containment_candidate_overflow_per_source = nullptr;
        params2.enable_containment_tracking = 0;
        params2.containment_iterations_per_source = nullptr;
        params2.containment_candidate_count_per_source = nullptr;
        params2.containment_candidate_overflow_per_source = nullptr;
    }

    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (hash query)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            QueryResults warmupResults = executeHashQuery(
                intersectionLauncher,
                params1, params2,
                mesh1NumEdges, mesh2NumEdges,
                mesh1NumObjects, mesh2NumObjects,
                d_hash_table, hash_table_size, queryDirection,
                nullptr,
                false
            );
            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }

    timer.next("Query");
    std::cout << "Running Intersection Query..." << std::endl;

    QueryResults results = executeHashQuery(
        intersectionLauncher, params1, params2,
        mesh1NumEdges, mesh2NumEdges,
        mesh1NumObjects, mesh2NumObjects,
        d_hash_table, hash_table_size, queryDirection,
        &timer
    );

    std::cout << "Actual Intersection Pairs: " << results.numUnique << std::endl;
    timer.addCounter("Profile_Actual_Intersection_Pairs", static_cast<unsigned long long>(results.numUnique));


    std::vector<MeshQueryResult> h_pairs(results.numUnique);
    if (results.numUnique > 0) {
        CUDA_CHECK(cudaMemcpy(
            h_pairs.data(),
            results.d_merged_results,
            static_cast<size_t>(results.numUnique) * sizeof(MeshQueryResult),
            cudaMemcpyDeviceToHost
        ));
    }

    writeIntersectionPairsCsv(pairsOutputPath, h_pairs);

    if (enableTracking) {
        pairHitBuffers.copyFromDevice(mesh1NumObjects, mesh2NumObjects);
        containmentTrackingBuffers.copyFromDevice(mesh1NumObjects, mesh2NumObjects);

        std::unordered_set<unsigned long long> finalPairs;
        finalPairs.reserve(h_pairs.size() * 2 + 1);
        for (const auto& pair : h_pairs) {
            finalPairs.insert(packIntersectionPairKey(pair.object_id_mesh1, pair.object_id_mesh2));
        }

        writeIntersectionTrackingCsv(
            pairHitsOutputPath,
            PairHitTrackingBuffers::kMaxPairTargetsPerSource,
            pairHitBuffers.h_mesh1_pair_target_ids,
            pairHitBuffers.h_mesh1_pair_target_hits,
            pairHitBuffers.h_mesh2_pair_target_ids,
            pairHitBuffers.h_mesh2_pair_target_hits,
            finalPairs
        );
        writeIntersectionTrackingSummaryCsv(
            containmentTrackingOutputPath,
            PairHitTrackingBuffers::kMaxPairTargetsPerSource,
            pairHitBuffers.h_mesh1_pair_target_ids,
            pairHitBuffers.h_mesh1_pair_target_hits,
            containmentTrackingBuffers.h_mesh1_iterations,
            containmentTrackingBuffers.h_mesh1_candidate_counts,
            containmentTrackingBuffers.h_mesh1_candidate_overflows,
            pairHitBuffers.h_mesh2_pair_target_ids,
            pairHitBuffers.h_mesh2_pair_target_hits,
            containmentTrackingBuffers.h_mesh2_iterations,
            containmentTrackingBuffers.h_mesh2_candidate_counts,
            containmentTrackingBuffers.h_mesh2_candidate_overflows,
            finalPairs
        );
    }
    std::cout << "Intersection pairs CSV: " << pairsOutputPath << std::endl;
    if (enableTracking) {
        std::cout << "Pair hit tracking CSV: " << pairHitsOutputPath << std::endl;
        std::cout << "Containment tracking CSV: " << containmentTrackingOutputPath << std::endl;
    }

    if (enableProfilingStats) {
        CUDA_CHECK(cudaMemcpy(&h_profiling_stats, d_profiling_stats, sizeof(MeshIntersectionProfilingStats), cudaMemcpyDeviceToHost));

        timer.addCounter("Profile_Overlap_Trace_Calls", h_profiling_stats.overlap_trace_calls);
        timer.addCounter("Profile_Overlap_Iterations_Total", h_profiling_stats.overlap_iterations_total);
        timer.addCounter("Profile_Overlap_Hits_Total", h_profiling_stats.overlap_hits_total);
        timer.addCounter("Profile_Overlap_Max_Iterations_Per_Trace", h_profiling_stats.overlap_max_iterations_per_trace);

        timer.addCounter("Profile_Containment_Rays_Total", h_profiling_stats.containment_rays_total);
        timer.addCounter("Profile_Containment_Iterations_Total", h_profiling_stats.containment_iterations_total);
        timer.addCounter("Profile_Containment_Hits_Total", h_profiling_stats.containment_hits_total);
        timer.addCounter("Profile_Containment_Max_Iterations_Per_Ray", h_profiling_stats.containment_max_iterations_per_ray);
        timer.addCounter("Profile_Containment_Same_Hit_Suppressed", h_profiling_stats.containment_same_hit_suppressed);
        timer.addCounter("Profile_Containment_Candidate_Additions", h_profiling_stats.containment_candidate_additions);
        timer.addCounter("Profile_Containment_Candidate_Toggles", h_profiling_stats.containment_candidate_toggles);
        timer.addCounter("Profile_Containment_Candidate_Overflow", h_profiling_stats.containment_candidate_overflow);
        timer.addCounter("Profile_Containment_Targets_Total", h_profiling_stats.containment_targets_total);

        const unsigned long long overlapCalls = h_profiling_stats.overlap_trace_calls;
        const unsigned long long containmentRays = h_profiling_stats.containment_rays_total;
        const unsigned long long avgOverlapIterScaled = overlapCalls ? (h_profiling_stats.overlap_iterations_total * 1000ULL / overlapCalls) : 0ULL;
        const unsigned long long avgContainmentIterScaled = containmentRays ? (h_profiling_stats.containment_iterations_total * 1000ULL / containmentRays) : 0ULL;
        timer.addCounter("Profile_Overlap_Avg_Iterations_x1000", avgOverlapIterScaled);
        timer.addCounter("Profile_Containment_Avg_Iterations_x1000", avgContainmentIterScaled);
    }

    timer.next("Cleanup");
    if (d_profiling_stats) {
        CUDA_CHECK(cudaFree(d_profiling_stats));
    }
    pairHitBuffers.free();
    containmentTrackingBuffers.free();
    CUDA_CHECK(cudaFree(d_anyhit_candidate_object_ids_mesh1));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_parity_mesh1));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_hit_counts_mesh1));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_count_mesh1));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_overflow_mesh1));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_object_ids_mesh2));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_parity_mesh2));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_hit_counts_mesh2));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_count_mesh2));
    CUDA_CHECK(cudaFree(d_anyhit_candidate_overflow_mesh2));
    CUDA_CHECK(cudaFree(d_hash_table));
    CUDA_CHECK(cudaFree(results.d_merged_results));
    CUDA_CHECK(cudaFree(d_first_triangle_mesh1));
    CUDA_CHECK(cudaFree(d_first_triangle_mesh2));
    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);

    mesh1Uploader.free();
    mesh2Uploader.free();

    timer.finish(outputJsonPath);
    return 0;
}
