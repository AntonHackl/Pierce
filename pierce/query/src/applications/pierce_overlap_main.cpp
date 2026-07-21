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
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <cuda_runtime.h>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "MultiGpuPartitioning.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshOverlapEdgesLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "../cuda/estimated_overlap.h"
#include "../utilities/GpuMemoryTracker.h"
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

struct OverlapExecutionConfig {
    QueryDirection direction = QueryDirection::Both;
    bool trackHashContention = false;
    int overlapMaxIterations = 100;
    bool trackGpuMemory = false;
    int warmupRuns = 0;
    std::string ptxPath;
};

struct OverlapHashQueryTimingStats {
    long long raytraceMesh1ToMesh2Us = 0;
    long long raytraceMesh2ToMesh1Us = 0;
    long long countHashTablePairsUs = 0;
    long long compactHashTablePairsUs = 0;
};

struct OverlapGpuWorkerResult {
    MeshQueryResult* dPairs = nullptr;
    long long numPairs = 0;
    int deviceId = 0;
    unsigned long long hashAccesses = 0;
    unsigned long long hashContentions = 0;
    unsigned long long hashTableSize = 0;
    unsigned long long resultBufferCapacity = 0;
    unsigned long long peakMemoryBytes = 0;
    long long workerTotalUs = 0;
    long long warmupTotalUs = 0;
    long long measuredHashQueryTotalUs = 0;
    OverlapHashQueryTimingStats measuredHashQuery;
    int slabIndex = 0;
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
    GpuMemoryTracker* memoryTracker = nullptr,
    PerformanceTimer* timer = nullptr,
    OverlapHashQueryTimingStats* timingStats = nullptr,
    bool verbose = true,
    bool trackHashContention = false
) {
    (void)estimated_pairs;
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
        if (timingStats) {
            timingStats->raytraceMesh1ToMesh2Us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
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
        if (timingStats) {
            timingStats->raytraceMesh2ToMesh1Us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
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
    if (timingStats) {
        timingStats->countHashTablePairsUs += std::chrono::duration_cast<std::chrono::microseconds>(t_count_end - t_count_start).count();
    }

    if (occupied_slots > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        occupied_slots = static_cast<unsigned long long>(std::numeric_limits<int>::max());
    }

    int max_output = static_cast<int>(occupied_slots);

    MeshQueryResult* d_merged_results = nullptr;
    if (max_output > 0) {
        CUDA_CHECK(cudaMalloc(&d_merged_results, static_cast<size_t>(max_output) * sizeof(MeshQueryResult)));
        if (memoryTracker) {
            memoryTracker->sample("overlap_after_result_buffer_alloc");
        }
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
    if (timingStats) {
        timingStats->compactHashTablePairsUs += std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_end - t_dedup_start).count();
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

struct OverlapResidentWorker {
    int deviceId = 0;
    int slabIndex = 0;
    int hashTableSize = 0;
    std::unique_ptr<OptixContext> context;
    GeometryUploader mesh1Uploader;
    GeometryUploader mesh2Uploader;
    std::unique_ptr<OptixAccelerationStructure> mesh1AS;
    std::unique_ptr<OptixAccelerationStructure> mesh2AS;
    EdgeMeshData mesh1EdgeData;
    EdgeMeshData mesh2EdgeData;
    std::unique_ptr<MeshOverlapEdgesLauncher> edgesLauncher;
    MeshOverlapEdgesLaunchParams edgesParams1 = {};
    MeshOverlapEdgesLaunchParams edgesParams2 = {};
    GpuMemoryTracker memoryTracker;

    OverlapResidentWorker(
        int device,
        const SlabGeometryPair& slabPair,
        const OverlapExecutionConfig& config
    )
        : deviceId(device),
          slabIndex(slabPair.slabIndex),
          hashTableSize(slabPair.hashTableSize),
          memoryTracker(config.trackGpuMemory) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        context = std::make_unique<OptixContext>();

        mesh1Uploader.upload(slabPair.mesh1.geometry);
        memoryTracker.sample("overlap_after_upload_mesh1");

        mesh2Uploader.upload(slabPair.mesh2.geometry);
        memoryTracker.sample("overlap_after_upload_mesh2");

        mesh1AS = std::make_unique<OptixAccelerationStructure>(*context, mesh1Uploader);
        mesh1AS->build(&memoryTracker, "build_mesh1_gas");

        mesh2AS = std::make_unique<OptixAccelerationStructure>(*context, mesh2Uploader);
        mesh2AS->build(&memoryTracker, "build_mesh2_gas");
        memoryTracker.sample("overlap_after_build_mesh2_index", true);

        mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh1.geometry);
        mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh2.geometry);

        edgesLauncher = std::make_unique<MeshOverlapEdgesLauncher>(*context, config.ptxPath);

        edgesParams1.edge_starts = mesh1EdgeData.d_edge_starts;
        edgesParams1.edge_ends = mesh1EdgeData.d_edge_ends;
        edgesParams1.edge_source_object_ids = mesh1EdgeData.d_source_object_ids;
        edgesParams1.num_edges = mesh1EdgeData.num_edges;
        edgesParams1.mesh2_handle = mesh2AS->getHandle();
        edgesParams1.mesh2_vertices = mesh2Uploader.getVertices();
        edgesParams1.mesh2_indices = mesh2Uploader.getIndices();
        edgesParams1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
        edgesParams1.swap_pair_order = 0;
        edgesParams1.overlap_max_iterations = config.overlapMaxIterations;

        edgesParams2.edge_starts = mesh2EdgeData.d_edge_starts;
        edgesParams2.edge_ends = mesh2EdgeData.d_edge_ends;
        edgesParams2.edge_source_object_ids = mesh2EdgeData.d_source_object_ids;
        edgesParams2.num_edges = mesh2EdgeData.num_edges;
        edgesParams2.mesh2_handle = mesh1AS->getHandle();
        edgesParams2.mesh2_vertices = mesh1Uploader.getVertices();
        edgesParams2.mesh2_indices = mesh1Uploader.getIndices();
        edgesParams2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
        edgesParams2.swap_pair_order = 1;
        edgesParams2.overlap_max_iterations = config.overlapMaxIterations;
    }

    ~OverlapResidentWorker() {
        CUDA_CHECK(cudaSetDevice(deviceId));
        edgesLauncher.reset();
        mesh2AS.reset();
        mesh1AS.reset();
        context.reset();
        PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
        PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);
        mesh1Uploader.free();
        mesh2Uploader.free();
    }

    OverlapResidentWorker(const OverlapResidentWorker&) = delete;
    OverlapResidentWorker& operator=(const OverlapResidentWorker&) = delete;
};

static OverlapGpuWorkerResult runOverlapOnResidentWorker(
    OverlapResidentWorker& resident,
    const OverlapExecutionConfig& config
) {
    const auto workerStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaSetDevice(resident.deviceId));

    const auto warmupStart = std::chrono::high_resolution_clock::now();
    for (int warmup = 0; warmup < config.warmupRuns; ++warmup) {
        unsigned long long* dWarmupHashTable = nullptr;
        CUDA_CHECK(cudaMalloc(&dWarmupHashTable, static_cast<size_t>(resident.hashTableSize) * sizeof(unsigned long long)));
        QueryResults warmupResults = executeHashQuery(
            *resident.edgesLauncher,
            resident.edgesParams1,
            resident.edgesParams2,
            resident.mesh1EdgeData.num_edges,
            resident.mesh2EdgeData.num_edges,
            dWarmupHashTable,
            resident.hashTableSize,
            0,
            config.direction,
            nullptr,
            nullptr,
            nullptr,
            false,
            false
        );
        if (warmupResults.d_merged_results) {
            CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
        CUDA_CHECK(cudaFree(dWarmupHashTable));
    }
    const long long warmupTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - warmupStart
    ).count();

    unsigned long long* dHashTable = nullptr;
    CUDA_CHECK(cudaMalloc(&dHashTable, static_cast<size_t>(resident.hashTableSize) * sizeof(unsigned long long)));
    resident.memoryTracker.sample("overlap_after_hash_table_alloc");

    OverlapHashQueryTimingStats measuredHashTiming;
    const auto measuredStart = std::chrono::high_resolution_clock::now();
    QueryResults queryResults = executeHashQuery(
        *resident.edgesLauncher,
        resident.edgesParams1,
        resident.edgesParams2,
        resident.mesh1EdgeData.num_edges,
        resident.mesh2EdgeData.num_edges,
        dHashTable,
        resident.hashTableSize,
        0,
        config.direction,
        &resident.memoryTracker,
        nullptr,
        &measuredHashTiming,
        false,
        config.trackHashContention
    );
    const long long measuredHashQueryTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - measuredStart
    ).count();

    OverlapGpuWorkerResult workerResult;
    workerResult.slabIndex = resident.slabIndex;
    workerResult.deviceId = resident.deviceId;
    workerResult.hashAccesses = queryResults.hashAccesses;
    workerResult.hashContentions = queryResults.hashContentions;
    workerResult.hashTableSize = resident.hashTableSize;
    workerResult.resultBufferCapacity = queryResults.resultBufferCapacity;
    workerResult.peakMemoryBytes = resident.memoryTracker.getPeakUsedBytes();
    workerResult.warmupTotalUs = warmupTotalUs;
    workerResult.measuredHashQueryTotalUs = measuredHashQueryTotalUs;
    workerResult.measuredHashQuery = measuredHashTiming;

    if (queryResults.numUnique > 0) {
        workerResult.dPairs = queryResults.d_merged_results;
        workerResult.numPairs = queryResults.numUnique;
        queryResults.d_merged_results = nullptr;
    }

    if (queryResults.d_merged_results) CUDA_CHECK(cudaFree(queryResults.d_merged_results));
    if (dHashTable) CUDA_CHECK(cudaFree(dHashTable));

    workerResult.workerTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - workerStart
    ).count();

    return workerResult;
}

static void printAndRecordSlabMemoryStats(
    const std::vector<SlabGeometryPair>& slabPairs,
    PerformanceTimer& timer
) {
    const std::vector<SlabMemoryStats> memoryStats = computeSlabMemoryStats(slabPairs);
    unsigned long long totalPlannedBytes = 0;
    unsigned long long minPlannedBytes = std::numeric_limits<unsigned long long>::max();
    unsigned long long maxPlannedBytes = 0;

    for (const SlabMemoryStats& stats : memoryStats) {
        totalPlannedBytes += stats.totalPlannedBytes;
        minPlannedBytes = std::min(minPlannedBytes, stats.totalPlannedBytes);
        maxPlannedBytes = std::max(maxPlannedBytes, stats.totalPlannedBytes);
    }
    if (memoryStats.empty()) {
        minPlannedBytes = 0;
    }
    const unsigned long long avgPlannedBytes = memoryStats.empty()
        ? 0
        : totalPlannedBytes / static_cast<unsigned long long>(memoryStats.size());
    const unsigned long long maxAvgRatioX1000 = avgPlannedBytes > 0
        ? (maxPlannedBytes * 1000ULL) / avgPlannedBytes
        : 0;

    const std::ios::fmtflags oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    std::cout << "\n=== Planned Per-GPU Memory Distribution ===" << std::endl;
    for (size_t slabOffset = 0; slabOffset < slabPairs.size(); ++slabOffset) {
        const SlabGeometryPair& slabPair = slabPairs[slabOffset];
        const SlabMemoryStats& stats = memoryStats[slabOffset];
        const double pct = totalPlannedBytes > 0
            ? 100.0 * static_cast<double>(stats.totalPlannedBytes) / static_cast<double>(totalPlannedBytes)
            : 0.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "Slab " << slabPair.slabIndex
                  << ": mesh1 triangles=" << slabPair.mesh1.numTriangles()
                  << ", mesh1 edges=" << slabPair.mesh1.numEdges()
                  << ", mesh2 triangles=" << slabPair.mesh2.numTriangles()
                  << ", mesh2 edges=" << slabPair.mesh2.numEdges()
                  << ", estimated_geometry_bytes=" << stats.estimatedGeometryBytes
                  << ", hash_table_bytes=" << stats.hashTableBytes
                  << ", total_planned_bytes=" << stats.totalPlannedBytes
                  << ", planned_pct=" << pct
                  << ", hash_table_size=" << slabPair.hashTableSize << std::endl;

        const std::string prefix = "Profile_Slab_" + std::to_string(slabPair.slabIndex);
        timer.addCounter(prefix + "_Mesh1_Triangles", static_cast<unsigned long long>(slabPair.mesh1.numTriangles()));
        timer.addCounter(prefix + "_Mesh1_Edges", static_cast<unsigned long long>(slabPair.mesh1.numEdges()));
        timer.addCounter(prefix + "_Mesh2_Triangles", static_cast<unsigned long long>(slabPair.mesh2.numTriangles()));
        timer.addCounter(prefix + "_Mesh2_Edges", static_cast<unsigned long long>(slabPair.mesh2.numEdges()));
        timer.addCounter(prefix + "_Estimated_Geometry_Bytes", stats.estimatedGeometryBytes);
        timer.addCounter(prefix + "_Hash_Table_Bytes", stats.hashTableBytes);
        timer.addCounter(prefix + "_Total_Planned_Bytes", stats.totalPlannedBytes);
        timer.addCounter(prefix + "_Planned_Pct_x1000", static_cast<unsigned long long>(pct * 1000.0));
    }
    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    timer.addCounter("Profile_Planned_Memory_Min_Bytes", minPlannedBytes);
    timer.addCounter("Profile_Planned_Memory_Max_Bytes", maxPlannedBytes);
    timer.addCounter("Profile_Planned_Memory_Avg_Bytes", avgPlannedBytes);
    timer.addCounter("Profile_Planned_Memory_Max_Avg_Ratio_x1000", maxAvgRatioX1000);
}

static void recordSlabPlanningTimingStats(
    const SlabPlanningTimingStats& stats,
    long long memoryStatsUs,
    PerformanceTimer& timer
) {
    timer.addMeasurement("Plan Detail Boundary CDF", stats.boundaryCdfUs);
    timer.addMeasurement("Plan Detail Boundary Search", stats.boundarySearchUs);
    timer.addMeasurement("Plan Detail Mesh1 Launch Points", stats.mesh1LaunchPointsUs);
    timer.addMeasurement("Plan Detail Mesh2 Launch Points", stats.mesh2LaunchPointsUs);
    timer.addMeasurement("Plan Detail Mesh1 Owner Assign", stats.mesh1OwnerAssignUs);
    timer.addMeasurement("Plan Detail Mesh2 Owner Assign", stats.mesh2OwnerAssignUs);
    timer.addMeasurement("Plan Detail Mesh1 Active Setup", stats.mesh1ActiveSetupUs);
    timer.addMeasurement("Plan Detail Mesh2 Active Setup", stats.mesh2ActiveSetupUs);
    timer.addMeasurement("Plan Detail Mesh1 Endpoint Sweep", stats.mesh1EndpointSweepUs);
    timer.addMeasurement("Plan Detail Mesh2 Endpoint Sweep", stats.mesh2EndpointSweepUs);
    timer.addMeasurement("Plan Detail Mesh1 Copy Sort", stats.mesh1ActiveCopySortUs);
    timer.addMeasurement("Plan Detail Mesh2 Copy Sort", stats.mesh2ActiveCopySortUs);
    timer.addMeasurement("Plan Detail Mesh1 Materialize", stats.mesh1MaterializeUs);
    timer.addMeasurement("Plan Detail Mesh2 Materialize", stats.mesh2MaterializeUs);
    timer.addMeasurement("Plan Detail Mesh1 Object Metadata", stats.mesh1ObjectMetadataUs);
    timer.addMeasurement("Plan Detail Mesh2 Object Metadata", stats.mesh2ObjectMetadataUs);
    timer.addMeasurement("Plan Detail Pair Assembly", stats.pairAssemblyUs);
    timer.addMeasurement("Plan Detail Memory Stats", memoryStatsUs);
}

static void recordOverlapPeakMemoryStats(
    const std::vector<OverlapGpuWorkerResult>& workerResults,
    PerformanceTimer& timer
) {
    unsigned long long totalPeakBytes = 0;
    unsigned long long minPeakBytes = std::numeric_limits<unsigned long long>::max();
    unsigned long long maxPeakBytes = 0;

    for (const OverlapGpuWorkerResult& workerResult : workerResults) {
        totalPeakBytes += workerResult.peakMemoryBytes;
        minPeakBytes = std::min(minPeakBytes, workerResult.peakMemoryBytes);
        maxPeakBytes = std::max(maxPeakBytes, workerResult.peakMemoryBytes);
        timer.addCounter(
            "Profile_Slab_" + std::to_string(workerResult.slabIndex) + "_GPU_Peak_Used_Bytes",
            workerResult.peakMemoryBytes
        );
    }
    if (workerResults.empty()) {
        minPeakBytes = 0;
    }

    const unsigned long long avgPeakBytes = workerResults.empty()
        ? 0
        : totalPeakBytes / static_cast<unsigned long long>(workerResults.size());
    const unsigned long long maxAvgRatioX1000 = avgPeakBytes > 0
        ? (maxPeakBytes * 1000ULL) / avgPeakBytes
        : 0;

    timer.addCounter("Profile_GPU_Peak_Min_Used_Bytes", minPeakBytes);
    timer.addCounter("Profile_GPU_Peak_Max_Used_Bytes", maxPeakBytes);
    timer.addCounter("Profile_GPU_Peak_Avg_Used_Bytes", avgPeakBytes);
    timer.addCounter("Profile_GPU_Peak_Max_Avg_Ratio_x1000", maxAvgRatioX1000);
}

static void recordOverlapWorkerTimingStats(
    const std::vector<OverlapGpuWorkerResult>& workerResults,
    PerformanceTimer& timer
) {
    unsigned long long totalWorkerUs = 0;
    unsigned long long maxWorkerUs = 0;
    unsigned long long minWorkerUs = std::numeric_limits<unsigned long long>::max();

    std::cout << "\n=== Per-GPU Worker Timing ===" << std::endl;
    for (const OverlapGpuWorkerResult& workerResult : workerResults) {
        const auto totalUs = static_cast<unsigned long long>(workerResult.workerTotalUs);
        totalWorkerUs += totalUs;
        maxWorkerUs = std::max(maxWorkerUs, totalUs);
        minWorkerUs = std::min(minWorkerUs, totalUs);

        std::cout << "Slab " << workerResult.slabIndex
                  << ": worker_total_us=" << workerResult.workerTotalUs
                  << ", measured_hash_query_us=" << workerResult.measuredHashQueryTotalUs
                  << ", warmup_total_us=" << workerResult.warmupTotalUs
                  << ", raytrace_mesh1_to_mesh2_us=" << workerResult.measuredHashQuery.raytraceMesh1ToMesh2Us
                  << ", raytrace_mesh2_to_mesh1_us=" << workerResult.measuredHashQuery.raytraceMesh2ToMesh1Us
                  << ", count_hash_table_pairs_us=" << workerResult.measuredHashQuery.countHashTablePairsUs
                  << ", compact_hash_table_pairs_us=" << workerResult.measuredHashQuery.compactHashTablePairsUs
                  << std::endl;

        const std::string prefix = "Profile_Slab_" + std::to_string(workerResult.slabIndex) + "_Worker";
        timer.addCounter(prefix + "_Total_Us", totalUs);
        timer.addCounter(prefix + "_Warmup_Total_Us", static_cast<unsigned long long>(workerResult.warmupTotalUs));
        timer.addCounter(prefix + "_Measured_Hash_Query_Total_Us", static_cast<unsigned long long>(workerResult.measuredHashQueryTotalUs));
        timer.addCounter(prefix + "_Hash_Overlap_Mesh1_To_Mesh2_Us", static_cast<unsigned long long>(workerResult.measuredHashQuery.raytraceMesh1ToMesh2Us));
        timer.addCounter(prefix + "_Hash_Overlap_Mesh2_To_Mesh1_Us", static_cast<unsigned long long>(workerResult.measuredHashQuery.raytraceMesh2ToMesh1Us));
        timer.addCounter(prefix + "_Hash_Count_Table_Pairs_Us", static_cast<unsigned long long>(workerResult.measuredHashQuery.countHashTablePairsUs));
        timer.addCounter(prefix + "_Hash_Compact_Table_Us", static_cast<unsigned long long>(workerResult.measuredHashQuery.compactHashTablePairsUs));
    }

    if (workerResults.empty()) {
        minWorkerUs = 0;
    }
    const unsigned long long avgWorkerUs = workerResults.empty()
        ? 0
        : totalWorkerUs / static_cast<unsigned long long>(workerResults.size());
    const unsigned long long maxAvgRatioX1000 = avgWorkerUs > 0
        ? (maxWorkerUs * 1000ULL) / avgWorkerUs
        : 0;

    timer.addCounter("Profile_Worker_Total_Min_Us", minWorkerUs);
    timer.addCounter("Profile_Worker_Total_Max_Us", maxWorkerUs);
    timer.addCounter("Profile_Worker_Total_Avg_Us", avgWorkerUs);
    timer.addCounter("Profile_Worker_Total_Max_Avg_Ratio_x1000", maxAvgRatioX1000);
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
    bool trackGpuMemory = false;
    int numGpus = 1;

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
        options.emplace_back("--track-gpu-memory", "Track GPU memory checkpoints and peak usage");
        options.emplace_back("--num-gpus <int>", "Number of GPUs / shared x-slabs to use (default: 1)");
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
        if (arg == "--track-gpu-memory") {
            trackGpuMemory = true;
            return true;
        }
        if (arg == "--num-gpus" && i + 1 < argc) {
            numGpus = std::stoi(argv[++i]);
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
    const bool trackGpuMemory = options.trackGpuMemory;
    const int requestedNumGpus = options.numGpus;

    if (!options.hasRequiredMeshInputs()) {
        std::cerr << "Usage: " << argv[0] << " --mesh1 <path> --mesh2 <path> [options]" << std::endl;
        return 1;
    }

    if (hashTableFreeMemFraction < 0.0f || hashTableFreeMemFraction > 1.0f) {
        std::cerr << "Error: --hash-table-free-mem-fraction must be in [0, 1]." << std::endl;
        return 1;
    }
    if (requestedNumGpus <= 0) {
        std::cerr << "Error: --num-gpus must be > 0." << std::endl;
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
    if (!mesh1.partition.hasData()) {
        std::cerr << "Error: Mesh1 does not contain partition metadata. Re-run pierce_preprocess." << std::endl;
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
    if (!mesh2.partition.hasData()) {
        std::cerr << "Error: Mesh2 does not contain partition metadata. Re-run pierce_preprocess." << std::endl;
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
        std::cout << "Requested GPUs:            " << requestedNumGpus << std::endl;
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

    std::cout << "\n=== Overlap Query Configuration ===" << std::endl;
    std::cout << "Requested GPUs:            " << requestedNumGpus << std::endl;
    std::cout << "Query direction:           " << directionToString(queryDirection) << std::endl;
    std::cout << "Overlap max iterations:    " << overlapMaxIterations << std::endl;
    std::cout << "Track hash contention:     " << (trackHashContention ? "yes" : "no") << std::endl;
    std::cout << "Track GPU memory:          " << (trackGpuMemory ? "yes" : "no") << std::endl;
    std::cout << "===============================\n" << std::endl;

    int finalNumUnique = 0;
    unsigned long long finalHashAccesses = 0;
    unsigned long long finalHashContentions = 0;
    unsigned long long finalHashTableSize = 0;
    unsigned long long finalResultBufferCapacity = 0;
    std::vector<MeshQueryResult> hostResults;

    timer.next("Selectivity Estimation");
    long long estimatedPairs = estimatePairs(false);
    unsigned long long hash_table_size = (manualHashTableSize > 0)
        ? manualHashTableSize
        : computeHashTableSize(estimatedPairs);

    int availableGpuCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&availableGpuCount));
    if (availableGpuCount <= 0) {
        std::cerr << "No CUDA devices available." << std::endl;
        return 1;
    }
    const int activeGpuCount = std::min(requestedNumGpus, availableGpuCount);
    if (activeGpuCount != requestedNumGpus) {
        std::cout << "Requested " << requestedNumGpus << " GPUs, using " << activeGpuCount
                  << " available device(s)." << std::endl;
    }

    timer.next("Plan Shared Slabs");
    SlabPlanningTimingStats slabPlanningTiming;
    const SlabPartitionPlan slabPlan = planSharedXAxisSlabs(mesh1, mesh2, activeGpuCount, &slabPlanningTiming);
    std::vector<SlabGeometryPair> slabPairs = buildSlabGeometryPairs(
        mesh1,
        mesh2,
        slabPlan,
        static_cast<int>(hash_table_size),
        &slabPlanningTiming
    );

    std::cout << "\n=== Selectivity Estimation (Overlap - Direct) ===" << std::endl;
    std::cout << "Final Estimated Pairs:     " << estimatedPairs << std::endl;
    if (manualHashTableSize > 0) {
        std::cout << "Using Manual Hash Table Size: " << hash_table_size << std::endl;
    } else if (hashTableFreeMemFraction > 0.0f) {
        std::cout << "Using Free GPU Memory Hash Table Size: " << hash_table_size << std::endl;
    } else {
        std::cout << "Using Direct Estimated Hash Table Size: " << hash_table_size << std::endl;
    }
    const auto memoryStatsStart = std::chrono::high_resolution_clock::now();
    printAndRecordSlabMemoryStats(slabPairs, timer);
    const long long memoryStatsUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - memoryStatsStart
    ).count();
    recordSlabPlanningTimingStats(slabPlanningTiming, memoryStatsUs, timer);

    OverlapExecutionConfig executionConfig;
    executionConfig.direction = queryDirection;
    executionConfig.trackHashContention = trackHashContention;
    executionConfig.overlapMaxIterations = overlapMaxIterations;
    executionConfig.trackGpuMemory = trackGpuMemory;
    executionConfig.warmupRuns = warmupRuns;
    executionConfig.ptxPath = options.ptxPath;

    timer.next("Initialize GPU Workers");
    std::vector<std::unique_ptr<OverlapResidentWorker>> residentWorkers;
    residentWorkers.resize(slabPairs.size());
    std::vector<std::thread> initThreads;
    std::mutex initErrorMutex;
    std::string initError;
    std::atomic<bool> initFailed{false};
    for (size_t workerIndex = 0; workerIndex < slabPairs.size(); ++workerIndex) {
        initThreads.emplace_back([&, workerIndex]() {
            try {
                residentWorkers[workerIndex] = std::make_unique<OverlapResidentWorker>(
                    static_cast<int>(workerIndex),
                    slabPairs[workerIndex],
                    executionConfig
                );
            } catch (const std::exception& ex) {
                initFailed.store(true);
                std::lock_guard<std::mutex> lock(initErrorMutex);
                if (initError.empty()) {
                    std::ostringstream message;
                    message << "GPU worker " << workerIndex << " resident initialization failed: " << ex.what();
                    initError = message.str();
                }
            }
        });
    }
    for (std::thread& initThread : initThreads) {
        initThread.join();
    }
    if (initFailed.load()) {
        std::cerr << (initError.empty() ? "Multi-GPU overlap resident initialization failed." : initError) << std::endl;
        return 1;
    }

    for (int run = 0; run < numberOfRuns; ++run) {
        timer.next("Execute Hash Query");
        std::vector<OverlapGpuWorkerResult> workerResults(residentWorkers.size());
        std::vector<std::thread> workerThreads;
        std::mutex errorMutex;
        std::string workerError;
        std::atomic<bool> workerFailed{false};

        for (size_t workerIndex = 0; workerIndex < residentWorkers.size(); ++workerIndex) {
            workerThreads.emplace_back([&, workerIndex]() {
                try {
                    workerResults[workerIndex] = runOverlapOnResidentWorker(
                        *residentWorkers[workerIndex],
                        executionConfig
                    );
                } catch (const std::exception& ex) {
                    workerFailed.store(true);
                    std::lock_guard<std::mutex> lock(errorMutex);
                    if (workerError.empty()) {
                        std::ostringstream message;
                        message << "GPU worker " << workerIndex << " failed: " << ex.what();
                        workerError = message.str();
                    }
                }
            });
        }
        for (std::thread& workerThread : workerThreads) {
            workerThread.join();
        }
        if (workerFailed.load()) {
            std::cerr << (workerError.empty() ? "Multi-GPU overlap execution failed." : workerError) << std::endl;
            return 1;
        }

        recordOverlapWorkerTimingStats(workerResults, timer);

        GpuGlobalDedupResult globalDedupResult;
        bool usedGlobalDedup = false;
        if (activeGpuCount == 1 && workerResults.size() == 1) {
            globalDedupResult.aggregatorDeviceId = workerResults[0].deviceId;
            globalDedupResult.inputCount = workerResults[0].numPairs;
            globalDedupResult.uniqueCount = workerResults[0].numPairs;
            globalDedupResult.d_uniquePairs = workerResults[0].dPairs;
            workerResults[0].dPairs = nullptr;
        } else {
            timer.next("Global GPU Deduplication");
            std::vector<DevicePairBuffer> workerBuffers;
            workerBuffers.reserve(workerResults.size());
            for (const auto& workerResult : workerResults) {
                workerBuffers.push_back({workerResult.deviceId, workerResult.dPairs, workerResult.numPairs});
            }
            try {
                GpuMemoryTracker* dedupMemoryTracker = (!residentWorkers.empty() && residentWorkers[0])
                    ? &residentWorkers[0]->memoryTracker
                    : nullptr;
                globalDedupResult = gather_and_deduplicate_pairs_gpu(workerBuffers, 0, dedupMemoryTracker);
                usedGlobalDedup = true;
            } catch (const std::exception& ex) {
                std::cerr << "Global GPU deduplication failed: " << ex.what() << std::endl;
                return 1;
            }
        }
        hostResults.clear();
        finalNumUnique = static_cast<int>(globalDedupResult.uniqueCount);
        finalHashAccesses = 0;
        finalHashContentions = 0;
        finalResultBufferCapacity = 0;
        finalHashTableSize = 0;
        unsigned long long peakGpuMemoryBytes = 0;
        for (const auto& workerResult : workerResults) {
            finalHashAccesses += workerResult.hashAccesses;
            finalHashContentions += workerResult.hashContentions;
            finalResultBufferCapacity += workerResult.resultBufferCapacity;
            finalHashTableSize += workerResult.hashTableSize;
            peakGpuMemoryBytes = std::max(peakGpuMemoryBytes, workerResult.peakMemoryBytes);
        }
        for (const auto& residentWorker : residentWorkers) {
            peakGpuMemoryBytes = std::max(
                peakGpuMemoryBytes,
                static_cast<unsigned long long>(residentWorker->memoryTracker.getPeakUsedBytes())
            );
        }

        timer.addCounter("Profile_Active_GPU_Count", static_cast<unsigned long long>(activeGpuCount));
        timer.addCounter("Profile_Global_Dedup_Mode", usedGlobalDedup ? 1ULL : 0ULL);
        timer.addCounter("Profile_Global_Dedup_Aggregator_Device", static_cast<unsigned long long>(globalDedupResult.aggregatorDeviceId));
        timer.addCounter("Profile_Global_Dedup_Input_Pairs", static_cast<unsigned long long>(globalDedupResult.inputCount));
        timer.addCounter("Profile_Global_Dedup_Unique_Pairs", static_cast<unsigned long long>(globalDedupResult.uniqueCount));
        timer.addCounter("Profile_Global_Dedup_Gather_Us", static_cast<unsigned long long>(globalDedupResult.gatherUs));
        timer.addCounter("Profile_Global_Dedup_SortUnique_Us", static_cast<unsigned long long>(globalDedupResult.dedupUs));
        timer.addCounter("Profile_Global_Dedup_Total_Us", static_cast<unsigned long long>(globalDedupResult.gatherUs + globalDedupResult.dedupUs));
        timer.addCounter(
            "Profile_Global_Dedup_Buffer_Bytes",
            usedGlobalDedup
                ? static_cast<unsigned long long>(globalDedupResult.inputCount) * static_cast<unsigned long long>(sizeof(MeshQueryResult))
                : 0ULL
        );

        if (!pairsOutputPath.empty() && globalDedupResult.uniqueCount > 0) {
            timer.next("Download Results");
            hostResults.resize(static_cast<size_t>(globalDedupResult.uniqueCount));
            CUDA_CHECK(cudaSetDevice(globalDedupResult.aggregatorDeviceId));
            CUDA_CHECK(cudaMemcpy(
                hostResults.data(),
                globalDedupResult.d_uniquePairs,
                static_cast<size_t>(globalDedupResult.uniqueCount) * sizeof(MeshQueryResult),
                cudaMemcpyDeviceToHost
            ));
        }
        if (globalDedupResult.d_uniquePairs) {
            CUDA_CHECK(cudaSetDevice(globalDedupResult.aggregatorDeviceId));
            CUDA_CHECK(cudaFree(globalDedupResult.d_uniquePairs));
        }

        if (trackGpuMemory) {
            for (size_t workerIndex = 0; workerIndex < workerResults.size(); ++workerIndex) {
                workerResults[workerIndex].peakMemoryBytes = std::max(
                    workerResults[workerIndex].peakMemoryBytes,
                    static_cast<unsigned long long>(residentWorkers[workerIndex]->memoryTracker.getPeakUsedBytes())
                );
            }
            timer.addCounter("Profile_GPU_Peak_Used_Bytes", peakGpuMemoryBytes);
            if (run == 0) {
                recordOverlapPeakMemoryStats(workerResults, timer);
            }
        }

        if (trackHashContention) {
            double contentionPct = (finalHashAccesses > 0)
                ? (100.0 * static_cast<double>(finalHashContentions) / static_cast<double>(finalHashAccesses))
                : 0.0;
            std::cout << std::fixed << std::setprecision(2)
                      << "Hash contention (run " << (run + 1) << "): "
                      << finalHashContentions << "/" << finalHashAccesses
                      << " accesses (" << contentionPct << "%)" << std::endl;
            std::cout.unsetf(std::ios::floatfield);
        }
    }

    timer.next("Cleanup");
    std::set<int> mesh1UniqueObjects(mesh1.triangleToObject.begin(), mesh1.triangleToObject.end());
    int mesh1NumObjects = mesh1UniqueObjects.size();
    std::set<int> mesh2UniqueObjects(mesh2.triangleToObject.begin(), mesh2.triangleToObject.end());
    int mesh2NumObjects = mesh2UniqueObjects.size();
    int mesh1NumTriangles = static_cast<int>(mesh1.indices.size());
    int mesh2NumTriangles = static_cast<int>(mesh2.indices.size());

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

    residentWorkers.clear();

    timer.finish(outputJsonPath);
    
    std::cout << "\nQuery completed in " << (double)timer.getTotalDuration() / 1000.0 << " ms." << std::endl;
    std::cout << "Results saved to: " << outputJsonPath << std::endl;

    return 0;
}
