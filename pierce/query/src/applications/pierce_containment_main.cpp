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
#include <unordered_map>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_containment.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshContainmentLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_intersection.h"
#include "app_cli_options.h"

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
    
    if (totalCount == 0) return 1.0f;
    return (float)(totalRatio / totalCount);
}

static long long estimateContainmentPairs(
    const GeometryData& meshAData,
    const GeometryData& meshBData,
    float epsilon,
    float gamma,
    bool verbose
) {
    if (!meshAData.grid.hasGrid || !meshBData.grid.hasGrid) {
        return 0;
    }

    float cellVolume = meshAData.grid.cellSize * meshAData.grid.cellSize * meshAData.grid.cellSize;

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
    for (const auto& entry : meshAData.grid.sparseCells) {
        mapA[entry.index] = entry.stats;
    }

    std::vector<GridCell> matchedA;
    std::vector<GridCell> matchedB;

    for (const auto& entry : meshBData.grid.sparseCells) {
        auto it = mapA.find(entry.index);
        if (it != mapA.end()) {
            matchedA.push_back(it->second);
            matchedB.push_back(entry.stats);
        }
    }

    int numMatchedCells = (int)matchedA.size();
    float estimatedPairsFloat = 0.0f;
    if (numMatchedCells > 0) {
        estimatedPairsFloat = estimateIntersectionSelectivity(
            matchedA.data(), matchedB.data(), numMatchedCells,
            cellVolume, epsilon, gamma);
    }

    float avgSize1 = calculateGlobalAvgSize(meshAData.grid.sparseCells);
    float avgSize2 = calculateGlobalAvgSize(meshBData.grid.sparseCells);
    float avgVolRatio1 = calculateGlobalAvgVolRatio(meshAData.grid.sparseCells);
    float avgVolRatio2 = calculateGlobalAvgVolRatio(meshBData.grid.sparseCells);
    float effectiveSize1 = avgSize1 * std::cbrt(avgVolRatio1);
    float effectiveSize2 = avgSize2 * std::cbrt(avgVolRatio2);
    float combinedSize = effectiveSize1 + effectiveSize2;
    float minkowskiVol = combinedSize * combinedSize * combinedSize;
    if (cellVolume < 1e-9f) cellVolume = 1e-9f;
    float alpha = minkowskiVol / cellVolume;
    if (alpha < 1.0f) alpha = 1.0f;

    const long long estimatedPairs = (long long)(estimatedPairsFloat / alpha);

    if (verbose) {
        std::cout << "\n=== Containment Selectivity Estimation ===" << std::endl;
        std::cout << "Matched Sparse Cells:      " << numMatchedCells << std::endl;
        std::cout << "Raw Potential Pairs:       " << (long long)estimatedPairsFloat << std::endl;
        std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }

    return estimatedPairs;
}

static int chooseContainmentHashTableSize(long long estimatedPairs, float hashLoadFactor) {
    if (estimatedPairs <= 0) {
        return 16777216;
    }

    unsigned long long target = (unsigned long long)(estimatedPairs / hashLoadFactor);
    if (target < 1024) target = 1024;
    if (target > 536870912ULL) target = 536870912ULL; // Cap at 512M slots (~4GB each)

    int hashTableSize = (int)target;
    if (hashTableSize % 2 == 0) hashTableSize++;
    return hashTableSize;
}

static constexpr int kContainmentAnyhitLegacyDefaultMaxUniqueAObjects = 512;
static constexpr int kContainmentAnyhitHardMaxUniqueAObjects = 512;

struct ContainmentAnyhitUsageSummary {
    unsigned int maxUniqueTargets = 0;
    unsigned long long overflowEvents = 0;
    unsigned int overflowSources = 0;
};

static ContainmentAnyhitUsageSummary summarizeContainmentAnyhitUsage(
    const std::vector<unsigned int>& uniqueCounts,
    const std::vector<unsigned int>& overflowEvents
) {
    ContainmentAnyhitUsageSummary summary;
    for (size_t i = 0; i < uniqueCounts.size(); ++i) {
        summary.maxUniqueTargets = std::max(summary.maxUniqueTargets, uniqueCounts[i]);
        summary.overflowEvents += overflowEvents[i];
        if (overflowEvents[i] > 0) {
            summary.overflowSources++;
        }
    }
    return summary;
}

// ---------------------------------------------------------------------
// Containment query
//
// Determines which B-objects are fully contained inside which A-objects.
//
// Algorithm (per pair (A_obj, B_obj)):
//   1. Phase 1 – Edge intersection:
//      Cast all B-triangle edges against A's acceleration structure AND
//      all A-triangle edges against B's acceleration structure.
//      Any (A_obj, B_obj) pair found to intersect is NOT a containment
//      pair – these are recorded in the intersection hash table.
//
//   2. Phase 2 – Point-in-mesh:
//      For every B-object, pick one vertex and cast a +Z ray against
//      A's acceleration structure.  Count surface crossings per A-object
//      (odd/even parity rule).  If the vertex is inside A_obj AND the
//      pair was NOT flagged in Phase 1 → containment.
// ---------------------------------------------------------------------

class ContainmentCliOptions : public BenchmarkMeshPairCliOptions {
public:
    ContainmentCliOptions() : BenchmarkMeshPairCliOptions("containment_timing.json") {
        allowNoExportFlag = true;
    }

    bool includeOverlapPairs = false;
    bool enableTracking = false;
    int overlapMaxIterations = 256;
    int anyhitMaxUniqueAObjects = -1;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    float hashLoadFactor = 0.5f;

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(
            options,
            "Dataset A (container meshes)",
            "Dataset B (objects tested for containment)"
        );
        appendBenchmarkRunHelp(options);
        options.emplace_back("--include-overlap-pairs", "Include overlap/touch pairs in output (union of overlap + strict containment)");
        options.emplace_back("--overlap-max-iterations <int>", "Max iterations for edge-scan overlap check (default: 256)");
        options.emplace_back("--containment-anyhit-max-targets <int>", "Point-in-mesh any-hit scratch cap per B object (default: 512)");
        options.emplace_back("--track-overflow", "Enable containment any-hit usage/overflow tracking");
        options.emplace_back("--gamma <float>", "Estimation gamma (default: 0.8)");
        options.emplace_back("--epsilon <float>", "Estimation epsilon (default: 0.001)");
        options.emplace_back("--hash-load-factor <float>", "Hash load factor for table sizing (default: 0.5)");
        appendNoExportHelp(options);
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "--mesh1 <A_dataset> --mesh2 <B_dataset> [options]",
            "Containment query: finds pairs where B objects are fully contained in A objects.",
            options
        );
    }

protected:
    bool parseApplicationOption(const std::string& arg, int& i, int argc, char* argv[]) override {
        (void)i;
        (void)argc;
        (void)argv;
        if (arg == "--include-overlap-pairs") {
            includeOverlapPairs = true;
            return true;
        }
        if (arg == "--overlap-max-iterations" && i + 1 < argc) {
            overlapMaxIterations = std::stoi(argv[++i]);
            return true;
        }
        if (arg == "--containment-anyhit-max-targets" && i + 1 < argc) {
            anyhitMaxUniqueAObjects = std::stoi(argv[++i]);
            return true;
        }
        if (arg == "--track-overflow") {
            enableTracking = true;
            return true;
        }
        if (arg == "--gamma" && i + 1 < argc) {
            gamma = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--epsilon" && i + 1 < argc) {
            epsilon = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--hash-load-factor" && i + 1 < argc) {
            hashLoadFactor = std::stof(argv[++i]);
            return true;
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");

    ContainmentCliOptions options;
    options.ptxPath = detectPTXPath("mesh_containment.ptx");
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }

    options.sanitizeRunCounts();

    const std::string& meshAPath = options.mesh1Path;
    const std::string& meshBPath = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const std::string& ptxPath = options.ptxPath;
    const int numberOfRuns = options.numberOfRuns;
    const int warmupRuns = options.warmupRuns;
    const bool exportResults = options.exportResults;
    const bool includeOverlapPairs = options.includeOverlapPairs;
    const bool enableTracking = options.enableTracking;
    const float gamma = options.gamma;
    const float epsilon = options.epsilon;
    const float hashLoadFactor = options.hashLoadFactor;
    const bool anyhitCapExplicitlyConfigured = options.anyhitMaxUniqueAObjects > 0;
    int anyhitMaxUniqueAObjects = anyhitCapExplicitlyConfigured
        ? options.anyhitMaxUniqueAObjects
        : kContainmentAnyhitLegacyDefaultMaxUniqueAObjects;

    if (anyhitMaxUniqueAObjects <= 0 ||
        anyhitMaxUniqueAObjects > kContainmentAnyhitHardMaxUniqueAObjects) {
        std::cerr << "--containment-anyhit-max-targets must be in [1, "
                  << kContainmentAnyhitHardMaxUniqueAObjects << "]." << std::endl;
        return 1;
    }

    std::cout << "=== Mesh Containment Query ===" << std::endl;
    std::cout << "(checks which B-objects are fully contained inside A-objects)" << std::endl;
    std::cout << "Point-in-mesh mode: anyhit" << std::endl;
    std::cout << "AnyHit scratch cap: " << anyhitMaxUniqueAObjects
              << " (" << (anyhitCapExplicitlyConfigured ? "explicit" : "default") << ")" << std::endl;
    std::cout << "Overflow Tracking: " << (enableTracking ? "enabled" : "disabled") << std::endl;
    std::cout << "Include overlap pairs: " << (includeOverlapPairs ? "yes" : "no") << std::endl;

    if (!options.hasRequiredMeshInputs()) {
        if (meshAPath.empty()) { std::cerr << "Error: --mesh1 (dataset A) required\n"; }
        if (meshBPath.empty()) { std::cerr << "Error: --mesh2 (dataset B) required\n"; }
        return 1;
    }

    // ---------------------------------------------------------------
    // Load geometry
    // ---------------------------------------------------------------
    timer.next("Application Creation");

    OptixContext context;
    MeshContainmentLauncher launcher(context, ptxPath);

    timer.next("Load Mesh A");
    std::cout << "Loading dataset A from: " << meshAPath << std::endl;
    GeometryData meshAData = loadGeometryFromFile(meshAPath);
    if (meshAData.vertices.empty()) { std::cerr << "Failed to load A\n"; return 1; }
    if (!requirePrecomputedEdges(meshAData, meshAPath, "MeshA")) { return 1; }
    std::cout << "  A: " << meshAData.vertices.size() << " vertices, "
              << meshAData.indices.size() << " triangles" << std::endl;

    timer.next("Load Mesh B");
    std::cout << "Loading dataset B from: " << meshBPath << std::endl;
    GeometryData meshBData = loadGeometryFromFile(meshBPath);
    if (meshBData.vertices.empty()) { std::cerr << "Failed to load B\n"; return 1; }
    if (!requirePrecomputedEdges(meshBData, meshBPath, "MeshB")) { return 1; }
    std::cout << "  B: " << meshBData.vertices.size() << " vertices, "
              << meshBData.indices.size() << " triangles" << std::endl;

    // Count unique objects
    std::set<int> aObjSet(meshAData.triangleToObject.begin(), meshAData.triangleToObject.end());
    std::set<int> bObjSet(meshBData.triangleToObject.begin(), meshBData.triangleToObject.end());
    int numAObjects = static_cast<int>(aObjSet.size());
    int numBObjects = static_cast<int>(bObjSet.size());
    std::cout << "  A objects: " << numAObjects << "   B objects: " << numBObjects << std::endl;

    // ---------------------------------------------------------------
    // Upload & build acceleration structures
    // ---------------------------------------------------------------
    timer.next("Upload Mesh A");
    GeometryUploader aUploader;
    aUploader.upload(meshAData);

    timer.next("Upload Mesh B");
    GeometryUploader bUploader;
    bUploader.upload(meshBData);

    timer.next("Build A Index");
    OptixAccelerationStructure aAS(context, aUploader);
    aAS.build();

    timer.next("Build B Index");
    OptixAccelerationStructure bAS(context, bUploader);
    bAS.build();

    // ---------------------------------------------------------------
    // Pre-compute first vertex per B-object (host side)
    // ---------------------------------------------------------------
    timer.next("Prepare Kernel Parameters");

    int aNumTriangles = static_cast<int>(aUploader.getNumIndices());
    int bNumTriangles = static_cast<int>(bUploader.getNumIndices());
    EdgeMeshData aEdgeData = PrecomputedEdgeData::uploadFromGeometry(meshAData);
    EdgeMeshData bEdgeData = PrecomputedEdgeData::uploadFromGeometry(meshBData);
    int aNumEdges = aEdgeData.num_edges;
    int bNumEdges = bEdgeData.num_edges;

    std::vector<float3> bFirstVertices(numBObjects);
    {
        std::vector<bool> seen(numBObjects, false);
        for (int tri = 0; tri < bNumTriangles; ++tri) {
            int obj = meshBData.triangleToObject[tri];
            if (!seen[obj]) {
                uint3  idx = meshBData.indices[tri];
                bFirstVertices[obj] = meshBData.vertices[idx.x];
                seen[obj] = true;
            }
        }
    }

    float3* d_bFirstVertices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bFirstVertices, numBObjects * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_bFirstVertices, bFirstVertices.data(),
                          numBObjects * sizeof(float3), cudaMemcpyHostToDevice));

    // The point-in-mesh kernel uses one scratch row per B object ray.
    // Each row tracks up to anyhitMaxUniqueAObjects distinct A objects.
    const int trackedBObjects = numBObjects;
    int* d_anyhit_a_ids = nullptr;
    unsigned int* d_anyhit_a_parity = nullptr;
    unsigned int* d_anyhit_num_unique = nullptr;
    unsigned int* d_anyhit_overflow_events = nullptr;
    auto freeAnyhitBuffers = [&]() {
        if (d_anyhit_overflow_events) CUDA_CHECK(cudaFree(d_anyhit_overflow_events));
        if (d_anyhit_num_unique) CUDA_CHECK(cudaFree(d_anyhit_num_unique));
        if (d_anyhit_a_parity) CUDA_CHECK(cudaFree(d_anyhit_a_parity));
        if (d_anyhit_a_ids) CUDA_CHECK(cudaFree(d_anyhit_a_ids));
        d_anyhit_overflow_events = nullptr;
        d_anyhit_num_unique = nullptr;
        d_anyhit_a_parity = nullptr;
        d_anyhit_a_ids = nullptr;
    };
    auto allocateAnyhitBuffers = [&](int cap) {
        freeAnyhitBuffers();
        const size_t slots = static_cast<size_t>(trackedBObjects) * static_cast<size_t>(cap);
        CUDA_CHECK(cudaMalloc(&d_anyhit_a_ids, slots * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_anyhit_a_parity, slots * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_anyhit_num_unique, trackedBObjects * sizeof(unsigned int)));
        if (enableTracking) {
            CUDA_CHECK(cudaMalloc(&d_anyhit_overflow_events, trackedBObjects * sizeof(unsigned int)));
        }
    };
    allocateAnyhitBuffers(anyhitMaxUniqueAObjects);

    auto copyContainmentAnyhitDiagnostics = [&]() {
        std::vector<unsigned int> uniqueCounts(trackedBObjects, 0);
        std::vector<unsigned int> overflowEvents(trackedBObjects, 0);
        CUDA_CHECK(cudaMemcpy(
            uniqueCounts.data(),
            d_anyhit_num_unique,
            trackedBObjects * sizeof(unsigned int),
            cudaMemcpyDeviceToHost
        ));
        if (enableTracking && d_anyhit_overflow_events) {
            CUDA_CHECK(cudaMemcpy(
                overflowEvents.data(),
                d_anyhit_overflow_events,
                trackedBObjects * sizeof(unsigned int),
                cudaMemcpyDeviceToHost
            ));
        }
        return summarizeContainmentAnyhitUsage(uniqueCounts, overflowEvents);
    };

    // ------------------------------------------------------------------
    // Warmup
    // ------------------------------------------------------------------
    timer.next("Warmup");

    struct ContainmentRunResult {
        std::vector<MeshQueryResult> reportedPairs;
        int containmentPairCount = 0;
        int overlapPairCount = 0;
    };

    int intersectionHTSize = 16777216;
    int containmentHTSize = 16777216;

    unsigned long long* d_intersectionHT = nullptr;
    unsigned long long* d_containmentHT  = nullptr;
    auto run_alloc_tables = [&]() {
        CUDA_CHECK(cudaMalloc(&d_intersectionHT, (size_t)intersectionHTSize * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_containmentHT,  (size_t)containmentHTSize  * sizeof(unsigned long long)));
    };

    auto runOnce = [&](bool verbose, bool recordBreakdownPhases) {
        if (recordBreakdownPhases) {
            auto t_est_0 = std::chrono::high_resolution_clock::now();
            const long long estimatedPairs = estimateContainmentPairs(
                meshAData,
                meshBData,
                epsilon,
                gamma,
                false
            );
            intersectionHTSize = chooseContainmentHashTableSize(estimatedPairs, hashLoadFactor);
            containmentHTSize = intersectionHTSize;
            auto t_est_1 = std::chrono::high_resolution_clock::now();
            timer.addMeasurement(
                "Selectivity Estimation",
                std::chrono::duration_cast<std::chrono::microseconds>(t_est_1 - t_est_0).count());

            if (verbose) {
                std::cout << "\n=== Containment Selectivity Estimation ===" << std::endl;
                std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
                std::cout << "Hash Table Size:           " << containmentHTSize
                          << " (Load Factor ~" << hashLoadFactor << ")" << std::endl;
                std::cout << "==========================================\n" << std::endl;
            }
            
            run_alloc_tables();
        }

        CUDA_CHECK(cudaMemset(d_intersectionHT, 0xFF,
                              (size_t)intersectionHTSize * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_containmentHT,  0xFF,
                              (size_t)containmentHTSize  * sizeof(unsigned long long)));
        MeshContainmentLaunchParams params{};

        // Phase 1a: B edges → A
        params.src_edge_starts             = bEdgeData.d_edge_starts;
        params.src_edge_ends               = bEdgeData.d_edge_ends;
        params.src_edge_source_object_ids    = bEdgeData.d_source_object_ids;
        params.src_num_edges               = bNumEdges;
        params.target_handle           = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();
        params.intersection_hash_table      = d_intersectionHT;
        params.intersection_hash_table_size = intersectionHTSize;
        params.swap_ids                = 0;
        params.b_first_vertices        = d_bFirstVertices;
        params.b_num_objects           = numBObjects;
        params.containment_hash_table       = d_containmentHT;
        params.containment_hash_table_size  = containmentHTSize;
        params.trace_phase = 0;
        params.anyhit_max_unique_a_objects = anyhitMaxUniqueAObjects;
        params.anyhit_a_ids = d_anyhit_a_ids;
        params.anyhit_a_parity = d_anyhit_a_parity;
        params.anyhit_num_unique = d_anyhit_num_unique;
        params.anyhit_overflow_events = d_anyhit_overflow_events;
        params.overlap_max_iterations = options.overlapMaxIterations;

        auto t0 = std::chrono::high_resolution_clock::now();
        launcher.launchEdgeCheck(params, bNumEdges);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (recordBreakdownPhases) {
            timer.addMeasurement(
                "Raytrace_Overlap_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }

        // Phase 1b: A edges → B
        params.src_edge_starts             = aEdgeData.d_edge_starts;
        params.src_edge_ends               = aEdgeData.d_edge_ends;
        params.src_edge_source_object_ids    = aEdgeData.d_source_object_ids;
        params.src_num_edges               = aNumEdges;
        params.target_handle           = bAS.getHandle();
        params.target_triangle_to_object = bUploader.getTriangleToObject();
        params.swap_ids                = 1;
        params.trace_phase = 0;

        t0 = std::chrono::high_resolution_clock::now();
        launcher.launchEdgeCheck(params, aNumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        if (recordBreakdownPhases) {
            timer.addMeasurement(
                "Raytrace_Overlap_Hash_Mesh1ToMesh2",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }

        // Phase 2: Point-in-mesh
        params.target_handle             = aAS.getHandle();
        params.target_triangle_to_object = aUploader.getTriangleToObject();
        params.trace_phase = 1;

        t0 = std::chrono::high_resolution_clock::now();
        launcher.launchPointInMesh(params, numBObjects);
        t1 = std::chrono::high_resolution_clock::now();
        if (recordBreakdownPhases) {
            timer.addMeasurement(
                "Raytrace_Containment_Hash_Mesh2ToMesh1",
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }

        // Compact results
        int maxContainmentOutput = 2000000;
        MeshQueryResult* d_containment_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_containment_results, maxContainmentOutput * sizeof(MeshQueryResult)));

        auto t_dedup_0 = std::chrono::high_resolution_clock::now();
        int numContained = compact_hash_table_pairs(
            d_containmentHT, containmentHTSize, d_containment_results, maxContainmentOutput);
        auto t_dedup_1 = std::chrono::high_resolution_clock::now();
        if (recordBreakdownPhases) {
            timer.addMeasurement(
                "compact_hash_table_pairs (containment)",
                std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_1 - t_dedup_0).count());
        }

        if (verbose) {
            std::cout << "Containment pairs found: " << numContained << std::endl;
        }

        ContainmentRunResult runResult;
        runResult.containmentPairCount = numContained;

        // Copy strict containment results to host first.
        std::vector<MeshQueryResult> containmentResults(numContained);
        if (numContained > 0) {
            CUDA_CHECK(cudaMemcpy(containmentResults.data(), d_containment_results,
                                  numContained * sizeof(MeshQueryResult),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(d_containment_results));

        runResult.reportedPairs = std::move(containmentResults);

        if (includeOverlapPairs) {
            int maxOverlapOutput = 2000000;
            MeshQueryResult* d_overlap_results = nullptr;
            CUDA_CHECK(cudaMalloc(&d_overlap_results, maxOverlapOutput * sizeof(MeshQueryResult)));

            auto t_dedup_overlap_0 = std::chrono::high_resolution_clock::now();
            int numOverlap = compact_hash_table_pairs(
                d_intersectionHT, intersectionHTSize, d_overlap_results, maxOverlapOutput);
            auto t_dedup_overlap_1 = std::chrono::high_resolution_clock::now();
            if (recordBreakdownPhases) {
                timer.addMeasurement(
                    "compact_hash_table_pairs (overlap)",
                    std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_overlap_1 - t_dedup_overlap_0).count());
            }
            runResult.overlapPairCount = numOverlap;

            if (verbose) {
                std::cout << "Overlap/touch pairs found: " << numOverlap << std::endl;
            }

            if (numOverlap > 0) {
                std::vector<MeshQueryResult> overlapResults(numOverlap);
                CUDA_CHECK(cudaMemcpy(overlapResults.data(), d_overlap_results,
                                      numOverlap * sizeof(MeshQueryResult),
                                      cudaMemcpyDeviceToHost));
                runResult.reportedPairs.insert(
                    runResult.reportedPairs.end(),
                    overlapResults.begin(),
                    overlapResults.end());
            }

            CUDA_CHECK(cudaFree(d_overlap_results));
        }

        return runResult;
    };

    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations..." << std::endl;
        const long long estimatedPairs = estimateContainmentPairs(
            meshAData,
            meshBData,
            epsilon,
            gamma,
            false
        );
        intersectionHTSize = chooseContainmentHashTableSize(estimatedPairs, hashLoadFactor);
        containmentHTSize = intersectionHTSize;
        run_alloc_tables();
        for (int w = 0; w < warmupRuns; ++w) runOnce(false, false);
    }

    // ---------------------------------------------------------------
    // Timed run(s)
    // ---------------------------------------------------------------
    timer.next("Query");

    std::cout << "\n=== Executing containment query ===" << std::endl;

    std::vector<MeshQueryResult> finalResults;
    int finalContainmentCount = 0;
    int finalOverlapCount = 0;
    for (int run = 0; run < numberOfRuns; ++run) {
        // Free tables from previous run if any
        if (d_intersectionHT) CUDA_CHECK(cudaFree(d_intersectionHT));
        if (d_containmentHT) CUDA_CHECK(cudaFree(d_containmentHT));
        d_intersectionHT = d_containmentHT = nullptr;

        ContainmentRunResult runResult = runOnce(run == numberOfRuns - 1, true);
        finalContainmentCount = runResult.containmentPairCount;
        finalOverlapCount = runResult.overlapPairCount;
        finalResults = std::move(runResult.reportedPairs);
    }
    int numReported = static_cast<int>(finalResults.size());

    timer.next("Download Results");

    // ---------------------------------------------------------------
    // Output
    // ---------------------------------------------------------------
    timer.next("Output");

    std::cout << "\n=== Containment Query Summary ===" << std::endl;
    std::cout << "A triangles: " << aNumTriangles << "  objects: " << numAObjects << std::endl;
    std::cout << "B triangles: " << bNumTriangles << "  objects: " << numBObjects << std::endl;
    std::cout << "Containment pairs (B in A): " << finalContainmentCount << std::endl;
    if (includeOverlapPairs) {
        std::cout << "Overlap/Touch pairs (A vs B): " << finalOverlapCount << std::endl;
        std::cout << "Reported pairs: " << numReported << std::endl;
    }
    if (enableTracking) {
        const ContainmentAnyhitUsageSummary anyhitUsageSummary = copyContainmentAnyhitDiagnostics();
        std::cout << "Containment any-hit usage: max_unique_targets=" << anyhitUsageSummary.maxUniqueTargets
                  << ", overflow_events=" << anyhitUsageSummary.overflowEvents
                  << ", overflow_sources=" << anyhitUsageSummary.overflowSources << std::endl;
        timer.addCounter("Profile_Containment_Anyhit_Max_Unique_Targets", anyhitUsageSummary.maxUniqueTargets);
        timer.addCounter("Profile_Containment_Anyhit_Overflow_Events", anyhitUsageSummary.overflowEvents);
        timer.addCounter("Profile_Containment_Anyhit_Overflow_Sources", anyhitUsageSummary.overflowSources);
    }

    if (exportResults) {
        std::string csvFile = "mesh_containment_results.csv";
        std::cout << "Exporting results to " << csvFile << std::endl;
        std::ofstream csv(csvFile);
        if (includeOverlapPairs) {
            csv << "a_object_id,b_object_id,pair_type\n";
            const size_t containmentCount = static_cast<size_t>(std::max(0, finalContainmentCount));
            for (size_t i = 0; i < finalResults.size(); ++i) {
                const auto& r = finalResults[i];
                const char* pairType = (i < containmentCount) ? "containment" : "overlap";
                csv << r.object_id_mesh1 << "," << r.object_id_mesh2 << "," << pairType << "\n";
            }
        } else {
            csv << "a_object_id,b_object_id\n";
            for (const auto& r : finalResults) {
                csv << r.object_id_mesh1 << "," << r.object_id_mesh2 << "\n";
            }
        }
    }

    // ---------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------
    timer.next("Cleanup");

    CUDA_CHECK(cudaFree(d_intersectionHT));
    CUDA_CHECK(cudaFree(d_containmentHT));
    CUDA_CHECK(cudaFree(d_bFirstVertices));
    freeAnyhitBuffers();
    PrecomputedEdgeData::freeEdgeData(aEdgeData);
    PrecomputedEdgeData::freeEdgeData(bEdgeData);

    timer.finish(outputJsonPath);

    std::cout << "\nContainment query completed successfully." << std::endl;
    return 0;
}
