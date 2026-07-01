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
#include <cmath>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <iomanip>
#include <cuda_runtime.h>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshOverlapEdgesLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_overlap.h"
#include "app_cli_options.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
    unsigned long long hashAccesses;
    unsigned long long hashContentions;
    unsigned long long resultBufferCapacity;
};

enum class QueryDirection {
    Both,
    Mesh1ToMesh2,
    Mesh2ToMesh1
};

static const char* directionToString(QueryDirection direction) {
    switch (direction) {
        case QueryDirection::Mesh1ToMesh2: return "mesh1_to_mesh2";
        case QueryDirection::Mesh2ToMesh1: return "mesh2_to_mesh1";
        case QueryDirection::Both:
        default: return "both";
    }
}

static bool parseDirection(const std::string& raw, QueryDirection& outDirection) {
    if (raw == "both") {
        outDirection = QueryDirection::Both;
        return true;
    }
    if (raw == "mesh1_to_mesh2") {
        outDirection = QueryDirection::Mesh1ToMesh2;
        return true;
    }
    if (raw == "mesh2_to_mesh1") {
        outDirection = QueryDirection::Mesh2ToMesh1;
        return true;
    }
    return false;
}

// Execute the overlap query using hash table deduplication
QueryResults executeHashQuery(
    MeshOverlapEdgesLauncher& edgesLauncher,
    MeshOverlapEdgesLaunchParams& edgesParams1,
    MeshOverlapEdgesLaunchParams& edgesParams2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    unsigned long long* d_hash_table,
    unsigned long long hash_table_size,
    long long estimated_pairs,
    QueryDirection direction,
    PerformanceTimer* timer = nullptr,
    bool verbose = true,
    bool trackHashContention = false
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    
    // Ensure params use hash table (no bitwise opt - table size is not power-of-two)
    edgesParams1.use_hash_table = 1;
    edgesParams1.use_bitwise_hash = 0;
    edgesParams1.hash_table = d_hash_table;
    edgesParams1.hash_table_size = hash_table_size;

    edgesParams2.use_hash_table = 1;
    edgesParams2.use_bitwise_hash = 0;
    edgesParams2.hash_table = d_hash_table;
    edgesParams2.hash_table_size = hash_table_size;

    unsigned long long* d_hash_access_counter = nullptr;
    unsigned long long* d_hash_contention_counter = nullptr;
    if (trackHashContention) {
        CUDA_CHECK(cudaMalloc(&d_hash_access_counter, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_hash_contention_counter, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_hash_access_counter, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_hash_contention_counter, 0, sizeof(unsigned long long)));
    }

    edgesParams1.hash_access_counter = d_hash_access_counter;
    edgesParams1.hash_contention_counter = d_hash_contention_counter;
    edgesParams1.track_hash_contention = trackHashContention ? 1 : 0;

    edgesParams2.hash_access_counter = d_hash_access_counter;
    edgesParams2.hash_contention_counter = d_hash_contention_counter;
    edgesParams2.track_hash_contention = trackHashContention ? 1 : 0;

    if (direction == QueryDirection::Both || direction == QueryDirection::Mesh1ToMesh2) {
        auto t0 = std::chrono::high_resolution_clock::now();
        edgesLauncher.launchMesh1ToMesh2(edgesParams1, mesh1NumEdges);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    if (direction == QueryDirection::Both || direction == QueryDirection::Mesh2ToMesh1) {
        auto t0 = std::chrono::high_resolution_clock::now();
        edgesLauncher.launchMesh2ToMesh1(edgesParams2, mesh2NumEdges);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
            );
        }
    }

    auto t_count_start = std::chrono::high_resolution_clock::now();
    unsigned long long occupied_slots = count_hash_table_pairs(d_hash_table, hash_table_size);
    auto t_count_end = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "count_hash_table_pairs",
            std::chrono::duration_cast<std::chrono::microseconds>(t_count_end - t_count_start).count()
        );
    }

    if (occupied_slots > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        occupied_slots = static_cast<unsigned long long>(std::numeric_limits<int>::max());
    }

    int max_output = static_cast<int>(occupied_slots);

    MeshQueryResult* d_merged_results = nullptr;
    if (max_output > 0) {
        CUDA_CHECK(cudaMalloc(&d_merged_results, static_cast<size_t>(max_output) * sizeof(MeshQueryResult)));
    }

    auto t_dedup_start = std::chrono::high_resolution_clock::now();
    int numUnique = 0;
    bool outputOverflowed = false;
    if (max_output > 0) {
        numUnique = compact_hash_table_pairs(
            d_hash_table,
            hash_table_size,
            d_merged_results,
            max_output,
            &outputOverflowed
        );
    }
    auto t_dedup_end = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "compact_hash_table_pairs",
            std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_end - t_dedup_start).count()
        );
    }

    unsigned long long hashAccesses = 0;
    unsigned long long hashContentions = 0;
    if (trackHashContention) {
        CUDA_CHECK(cudaMemcpy(&hashAccesses, d_hash_access_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hashContentions, d_hash_contention_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_hash_access_counter));
        CUDA_CHECK(cudaFree(d_hash_contention_counter));
    }
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
         if (outputOverflowed) {
             std::cerr << "WARNING: Output buffer full! Results may be truncated. Max output: " << max_output << std::endl;
         }
    }
    
    return {d_merged_results, numUnique, hashAccesses, hashContentions, static_cast<unsigned long long>(max_output)};
}

// Helper to calculate global average size of objects from grid statistics
float calculateGlobalAvgSize(const std::vector<SparseGridEntry>& sparseCells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
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

class OverlapDirectCliOptions : public BenchmarkMeshPairCliOptions {
public:
    OverlapDirectCliOptions() : BenchmarkMeshPairCliOptions("estimated_overlap_timing.json") {}

    bool estimateOnly = false;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    QueryDirection queryDirection = QueryDirection::Both;
    std::string pairsOutputPath;
    bool trackHashContention = false;
    bool useAlphaCorrection = true;
    unsigned long long manualHashTableSize = 0;
    float hashTableFreeMemFraction = 0.0f;
    int overlapMaxIterations = 100;

    bool valid = true;

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(options);
        appendBenchmarkRunHelp(options);
        options.emplace_back("--gamma <float>", "Gamma parameter for estimation (default: 0.8)");
        options.emplace_back("--epsilon <float>", "Epsilon parameter for estimation (default: 0.001)");
        options.emplace_back("--query-direction <d>", "Query direction: both|mesh1_to_mesh2|mesh2_to_mesh1 (default: both)");
        options.emplace_back("--pairs-output <path>", "Optional CSV export path for unique result pairs");
        options.emplace_back("--track-hash-contention", "Track hash accesses and contention rate");
        options.emplace_back("--no-alpha-correction", "Disable replication-factor alpha correction for estimated pairs");
        options.emplace_back("--hash-table-size <ull>", "Override hash table size (slots); 0 = auto-compute");
        options.emplace_back("--hash-table-free-mem-fraction <f>", "Use f in (0,1] of free GPU memory for hash table sizing");
        options.emplace_back("--overlap-max-iterations <int>", "Overlap ray iteration cap (default: 100)");
        options.emplace_back("--estimate-only", "Only run selectivity estimation, skip actual query");
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "--mesh1 <path> --mesh2 <path> [options]",
            "Overlap direct-estimation query with optional pair export and hash contention diagnostics.",
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
        if (arg == "--query-direction" && i + 1 < argc) {
            const std::string directionRaw = argv[++i];
            if (!parseDirection(directionRaw, queryDirection)) {
                std::cerr << "Error: Invalid --query-direction value: " << directionRaw
                          << ". Expected one of: both, mesh1_to_mesh2, mesh2_to_mesh1" << std::endl;
                valid = false;
            }
            return true;
        }
        if (arg == "--pairs-output" && i + 1 < argc) {
            pairsOutputPath = argv[++i];
            return true;
        }
        if (arg == "--track-hash-contention") {
            trackHashContention = true;
            return true;
        }
        if (arg == "--no-alpha-correction") {
            useAlphaCorrection = false;
            return true;
        }
        if (arg == "--hash-table-size" && i + 1 < argc) {
            manualHashTableSize = std::stoull(argv[++i]);
            return true;
        }
        if (arg == "--hash-table-free-mem-fraction" && i + 1 < argc) {
            hashTableFreeMemFraction = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--estimate-only") {
            estimateOnly = true;
            return true;
        }
        if (arg == "--overlap-max-iterations" && i + 1 < argc) {
            overlapMaxIterations = std::stoi(argv[++i]);
            return true;
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    OverlapDirectCliOptions options;
    options.ptxPath = detectPTXPath("mesh_overlap_edges.ptx");
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }
    if (!options.valid) {
        return 1;
    }

    options.sanitizeRunCounts();

    const std::string& mesh1Path = options.mesh1Path;
    const std::string& mesh2Path = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const int numberOfRuns = options.numberOfRuns;
    const int warmupRuns = options.warmupRuns;
    const bool estimateOnly = options.estimateOnly;
    const float gamma = options.gamma;
    const float epsilon = options.epsilon;
    const QueryDirection queryDirection = options.queryDirection;
    const std::string& pairsOutputPath = options.pairsOutputPath;
    const bool trackHashContention = options.trackHashContention;
    const bool useAlphaCorrection = options.useAlphaCorrection;
    const unsigned long long manualHashTableSize = options.manualHashTableSize;
    const float hashTableFreeMemFraction = options.hashTableFreeMemFraction;
    const int overlapMaxIterations = options.overlapMaxIterations;

    if (!options.hasRequiredMeshInputs()) {
        std::cerr << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
        return 1;
    }

    if (hashTableFreeMemFraction < 0.0f || hashTableFreeMemFraction > 1.0f) {
        std::cerr << "Error: --hash-table-free-mem-fraction must be in [0, 1]." << std::endl;
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

    auto estimatePairs = [&](bool verbose) -> long long {
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
                estimatedPairsFloat = estimateOverlapSelectivity(
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

            float alpha = 1.0f;
            if (useAlphaCorrection) {
                alpha = minkowskiVol / cellVolume;
                if (alpha < 1.0f) alpha = 1.0f;
            }

            estimatedPairs = (long long)(estimatedPairsFloat / alpha);

            if (verbose) {
                std::cout << "\n=== Selectivity Estimation (Overlap - Direct) ===" << std::endl;
                std::cout << "Matched Sparse Cells:      " << numMatchedCells << std::endl;
                std::cout << "Raw Potential Pairs:       " << (long long)estimatedPairsFloat << std::endl;
                std::cout << "Avg Object Size (Mesh1):   " << avgSize1 << std::endl;
                std::cout << "Avg Object Size (Mesh2):   " << avgSize2 << std::endl;
                std::cout << "Avg VolRatio (Mesh1):      " << avgVolRatio1 << std::endl;
                std::cout << "Avg VolRatio (Mesh2):      " << avgVolRatio2 << std::endl;
                std::cout << "Effective Size (Mesh1):    " << effectiveSize1 << std::endl;
                std::cout << "Effective Size (Mesh2):    " << effectiveSize2 << std::endl;
                std::cout << "Alpha Correction Enabled:  " << (useAlphaCorrection ? "yes" : "no") << std::endl;
                std::cout << "Replication Factor (alpha):" << alpha << std::endl;
                std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
                std::cout << "==============================\n" << std::endl;
            }
        } else if (verbose) {
            std::cout << "Skipping estimation: Grid data not found." << std::endl;
        }

        return estimatedPairs;
    };

    auto computeHashTableSize = [&](long long estimatedPairs) -> unsigned long long {
        if (hashTableFreeMemFraction > 0.0f) {
            size_t freeMemBytes = 0;
            size_t totalMemBytes = 0;
            cudaError_t memInfoStatus = cudaMemGetInfo(&freeMemBytes, &totalMemBytes);
            if (memInfoStatus == cudaSuccess && freeMemBytes > 0) {
                double fraction = static_cast<double>(hashTableFreeMemFraction);
                unsigned long long targetBytes = static_cast<unsigned long long>(fraction * static_cast<double>(freeMemBytes));
                unsigned long long targetSlots = targetBytes / sizeof(unsigned long long);
                if (targetSlots < 1024ULL) targetSlots = 1024ULL;
                if ((targetSlots % 2ULL) == 0ULL) {
                    targetSlots += 1ULL;
                }
                return targetSlots;
            }
            std::cerr << "Warning: cudaMemGetInfo failed for free-memory hash sizing. Falling back to estimate-based sizing." << std::endl;
        }

        unsigned long long hash_table_size = 16777216;
        if (estimatedPairs > 0) {
            unsigned long long target = (unsigned long long)(estimatedPairs / 0.5);
            if (target < 1024) target = 1024;
            if (target > 2147483648ULL) target = 2147483648ULL;

            hash_table_size = target;
            if (hash_table_size % 2 == 0) hash_table_size++;
        }
        return hash_table_size;
    };

    if (estimateOnly) {
        timer.next("Selectivity Estimation");
        const long long estimatedPairs = estimatePairs(false);
        const unsigned long long hash_table_size = (manualHashTableSize > 0)
            ? manualHashTableSize
            : computeHashTableSize(estimatedPairs);
        std::cout << "\n=== Selectivity Estimation (Overlap - Direct) ===" << std::endl;
        std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
        if (manualHashTableSize > 0) {
            std::cout << "Using Manual Hash Table Size: " << hash_table_size << std::endl;
        } else if (hashTableFreeMemFraction > 0.0f) {
            std::cout << "Using Free GPU Memory Hash Table Size: " << hash_table_size << std::endl;
        } else {
            std::cout << "Using Direct Estimated Hash Table Size: " << hash_table_size << std::endl;
        }
        std::cout << "Hash Table Allocated Bytes: "
                  << (hash_table_size * sizeof(unsigned long long)) << std::endl;
        timer.finish(outputJsonPath);
        return 0;
    }

    // --- EXECUTION PHASE ---
    timer.next("Init OptiX");
    OptixContext context;

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

    timer.next("Upload Mesh1 Edges");
    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh1);
    int mesh1NumEdges = mesh1EdgeData.num_edges;
    std::cout << "Mesh1 edges uploaded: " << mesh1NumEdges << " unique edges" << std::endl;

    timer.next("Upload Mesh2 Edges");
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh2);
    int mesh2NumEdges = mesh2EdgeData.num_edges;
    std::cout << "Mesh2 edges uploaded: " << mesh2NumEdges << " unique edges" << std::endl;

    timer.next("Create Edge Launcher");
    MeshOverlapEdgesLauncher edgesLauncher(context, options.ptxPath);

    timer.next("Prepare Kernel Parameters");
    int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());

    MeshOverlapEdgesLaunchParams edgesParams1 = {};
    edgesParams1.edge_starts = mesh1EdgeData.d_edge_starts;
    edgesParams1.edge_ends = mesh1EdgeData.d_edge_ends;
    edgesParams1.edge_source_object_ids = mesh1EdgeData.d_source_object_ids;
    edgesParams1.num_edges = mesh1NumEdges;
    edgesParams1.mesh2_handle = mesh2AS.getHandle();
    edgesParams1.mesh2_vertices = mesh2Uploader.getVertices();
    edgesParams1.mesh2_indices = (uint3*)mesh2Uploader.getIndices();
    edgesParams1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    edgesParams1.swap_pair_order = 0;
    edgesParams1.overlap_max_iterations = overlapMaxIterations;

    MeshOverlapEdgesLaunchParams edgesParams2 = {};
    edgesParams2.edge_starts = mesh2EdgeData.d_edge_starts;
    edgesParams2.edge_ends = mesh2EdgeData.d_edge_ends;
    edgesParams2.edge_source_object_ids = mesh2EdgeData.d_source_object_ids;
    edgesParams2.num_edges = mesh2NumEdges;
    edgesParams2.mesh2_handle = mesh1AS.getHandle();
    edgesParams2.mesh2_vertices = mesh1Uploader.getVertices();
    edgesParams2.mesh2_indices = (uint3*)mesh1Uploader.getIndices();
    edgesParams2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    edgesParams2.swap_pair_order = 1;
    edgesParams2.overlap_max_iterations = overlapMaxIterations;

    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (estimation + hash query)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            long long warmupEstimatedPairs = estimatePairs(false);
            unsigned long long warmupHashSize = computeHashTableSize(warmupEstimatedPairs);
            unsigned long long* d_warmup_hash_table = nullptr;
            CUDA_CHECK(cudaMalloc(&d_warmup_hash_table, warmupHashSize * sizeof(unsigned long long)));

            QueryResults warmupResults = executeHashQuery(
                edgesLauncher,
                edgesParams1,
                edgesParams2,
                mesh1NumEdges,
                mesh2NumEdges,
                d_warmup_hash_table,
                warmupHashSize,
                warmupEstimatedPairs,
                queryDirection,
                nullptr,
                false,
                false
            );

            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
            CUDA_CHECK(cudaFree(d_warmup_hash_table));
        }
    }

    int finalNumUnique = 0;
    unsigned long long finalHashAccesses = 0;
    unsigned long long finalHashContentions = 0;
    unsigned long long finalHashTableSize = 0;
    unsigned long long finalResultBufferCapacity = 0;
    std::vector<MeshQueryResult> hostResults;
    for (int run = 0; run < numberOfRuns; ++run) {
        bool verboseRun = (run == 0);

        timer.next("Selectivity Estimation");
        long long estimatedPairs = estimatePairs(false);
        unsigned long long hash_table_size = (manualHashTableSize > 0)
            ? manualHashTableSize
            : computeHashTableSize(estimatedPairs);

        timer.next("Execute Hash Query");

        if (verboseRun) {
            std::cout << "\n=== Selectivity Estimation (Overlap - Direct) ===" << std::endl;
            std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
            if (manualHashTableSize > 0) {
                std::cout << "Using Manual Hash Table Size: " << hash_table_size << std::endl;
            } else if (hashTableFreeMemFraction > 0.0f) {
                std::cout << "Using Free GPU Memory Hash Table Size: " << hash_table_size << std::endl;
            } else {
                std::cout << "Using Direct Estimated Hash Table Size: " << hash_table_size << std::endl;
            }
            std::cout << "Hash Table Allocated Bytes: "
                      << (hash_table_size * sizeof(unsigned long long)) << std::endl;
        }

        unsigned long long* d_hash_table = nullptr;
        CUDA_CHECK(cudaMalloc(&d_hash_table, hash_table_size * sizeof(unsigned long long)));

        QueryResults queryResults = executeHashQuery(
            edgesLauncher,
            edgesParams1,
            edgesParams2,
            mesh1NumEdges,
            mesh2NumEdges,
            d_hash_table,
            hash_table_size,
            estimatedPairs,
            queryDirection,
            &timer,
            verboseRun,
            trackHashContention
        );

        timer.next("Download Results");
        hostResults.clear();
        if (queryResults.numUnique > 0) {
            hostResults.resize(queryResults.numUnique);
            CUDA_CHECK(cudaMemcpy(hostResults.data(), queryResults.d_merged_results,
                                  (size_t)queryResults.numUnique * sizeof(MeshQueryResult),
                                  cudaMemcpyDeviceToHost));
        }
        finalNumUnique = queryResults.numUnique;
        finalHashAccesses = queryResults.hashAccesses;
        finalHashContentions = queryResults.hashContentions;
        finalHashTableSize = hash_table_size;
        finalResultBufferCapacity = queryResults.resultBufferCapacity;

        if (trackHashContention) {
            double contentionPct = (queryResults.hashAccesses > 0)
                ? (100.0 * static_cast<double>(queryResults.hashContentions) / static_cast<double>(queryResults.hashAccesses))
                : 0.0;
            std::cout << std::fixed << std::setprecision(2)
                      << "Hash contention (run " << (run + 1) << "): "
                      << queryResults.hashContentions << "/" << queryResults.hashAccesses
                      << " accesses (" << contentionPct << "%)" << std::endl;
            std::cout.unsetf(std::ios::floatfield);
        }

        if (queryResults.d_merged_results) CUDA_CHECK(cudaFree(queryResults.d_merged_results));
        CUDA_CHECK(cudaFree(d_hash_table));
    }

    timer.next("Cleanup");

    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);

    std::set<int> mesh1UniqueObjects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    int mesh1NumObjects = mesh1UniqueObjects.size();
    std::set<int> mesh2UniqueObjects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh2NumObjects = mesh2UniqueObjects.size();

    std::cout << "\n=== Mesh Overlap Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh1 objects: " << mesh1NumObjects << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Mesh2 objects: " << mesh2NumObjects << std::endl;

    std::cout << "Query direction: " << directionToString(queryDirection) << std::endl;
    std::cout << "Hash contention tracking: " << (trackHashContention ? "enabled" : "disabled") << std::endl;
    if (trackHashContention) {
        double contentionPct = (finalHashAccesses > 0)
            ? (100.0 * static_cast<double>(finalHashContentions) / static_cast<double>(finalHashAccesses))
            : 0.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "Hash contention (last run): " << finalHashContentions << "/" << finalHashAccesses
                  << " accesses (" << contentionPct << "%)" << std::endl;
        std::cout.unsetf(std::ios::floatfield);
    }
    std::cout << "Unique object pairs: " << finalNumUnique << std::endl;
    std::cout << "Result Buffer Capacity: " << finalResultBufferCapacity << std::endl;
    std::cout << "Result Buffer Allocated Bytes: "
              << (finalResultBufferCapacity * sizeof(MeshQueryResult)) << std::endl;
    std::cout << "Result Buffer Used Bytes: "
              << (static_cast<unsigned long long>(finalNumUnique) * sizeof(MeshQueryResult)) << std::endl;
    std::cout << "Hash Table Slots: " << finalHashTableSize << std::endl;
    std::cout << "Hash Table Allocated Bytes: "
              << (finalHashTableSize * sizeof(unsigned long long)) << std::endl;

    if (!pairsOutputPath.empty()) {
        std::ofstream csvFile(pairsOutputPath);
        if (!csvFile.is_open()) {
            std::cerr << "Warning: Failed to open pairs output file: " << pairsOutputPath << std::endl;
        } else {
            csvFile << "object_id_mesh1,object_id_mesh2\n";
            for (const auto& result : hostResults) {
                csvFile << result.object_id_mesh1 << "," << result.object_id_mesh2 << "\n";
            }
            csvFile.close();
            std::cout << "Pair results written to: " << pairsOutputPath << std::endl;
        }
    }

    timer.finish(outputJsonPath);
    
    std::cout << "\nQuery completed in " << (double)timer.getTotalDuration() / 1000.0 << " ms." << std::endl;
    std::cout << "Results saved to: " << outputJsonPath << std::endl;

    return 0;
}
