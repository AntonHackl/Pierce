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
#include <thread>
#include <mutex>
#include <atomic>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "MultiGpuPartitioning.h"
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
#include "../utilities/GpuMemoryTracker.h"
#include "../utilities/PairHitTracking.h"
#include "../utilities/ContainmentTracking.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    int numUnique;
    unsigned long long hashInsertFailures = 0;
};

static unsigned long long packIntersectionPairKey(int mesh1ObjectId, int mesh2ObjectId);

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

static constexpr int kIntersectionAnyhitLegacyDefaultMaxTargetsPerSource = 256;
static constexpr int kIntersectionAnyhitHardMaxTargetsPerSource = 256;
static constexpr int kContainmentHitHistogramMaxBucket = 1024;

struct IntersectionAnyhitUsageSummary {
    unsigned int maxCandidateCount = 0;
    unsigned long long overflowEvents = 0;
    unsigned int overflowSources = 0;
};

static IntersectionAnyhitUsageSummary summarizeIntersectionAnyhitUsage(
    const std::vector<unsigned int>& mesh1CandidateCounts,
    const std::vector<unsigned int>& mesh1OverflowEvents,
    const std::vector<unsigned int>& mesh2CandidateCounts,
    const std::vector<unsigned int>& mesh2OverflowEvents
) {
    IntersectionAnyhitUsageSummary summary;
    auto accumulateDirection = [&](const std::vector<unsigned int>& counts, const std::vector<unsigned int>& overflows) {
        for (size_t i = 0; i < counts.size(); ++i) {
            summary.maxCandidateCount = std::max(summary.maxCandidateCount, counts[i]);
            summary.overflowEvents += overflows[i];
            if (overflows[i] > 0) {
                summary.overflowSources++;
            }
        }
    };
    accumulateDirection(mesh1CandidateCounts, mesh1OverflowEvents);
    accumulateDirection(mesh2CandidateCounts, mesh2OverflowEvents);
    return summary;
}

struct IntersectionExecutionConfig {
    QueryDirection queryDirection = QueryDirection::Both;
    int overlapMaxIterations = 100;
    int anyhitMaxTargetsPerSource = kIntersectionAnyhitLegacyDefaultMaxTargetsPerSource;
    bool enableProfilingStats = false;
    bool trackOverflow = false;
    bool trackGpuMemory = false;
    int warmupRuns = 0;
    std::string ptxPath;
    std::string containmentFingerprintOutputPath;
    std::string containmentHitHistogramOutputPath;
};

struct ContainmentFingerprintRow {
    int slabIndex = 0;
    int direction = 0;
    int sourceObjectId = -1;
    unsigned int candidateCount = 0;
    unsigned int oddCandidateCount = 0;
    unsigned int overflowEvents = 0;
    unsigned long long hitTotal = 0;
    unsigned long long targetXor = 0;
    unsigned long long targetSum = 0;
    unsigned long long fingerprintXor = 0;
    unsigned long long fingerprintSum = 0;
};

struct IntersectionGpuWorkerResult {
    std::vector<MeshQueryResult> pairs;
    std::vector<ContainmentFingerprintRow> containmentFingerprints;
    std::vector<unsigned long long> mesh1ToMesh2HitHistogram;
    std::vector<unsigned long long> mesh2ToMesh1HitHistogram;
    IntersectionAnyhitUsageSummary anyhitSummary;
    MeshIntersectionProfilingStats profilingStats = {};
    unsigned long long hashInsertFailures = 0;
    unsigned long long peakMemoryBytes = 0;
    int slabIndex = 0;
};

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

void writeContainmentFingerprintCsv(
    const std::string& outputPath,
    const std::vector<IntersectionGpuWorkerResult>& workerResults
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open containment fingerprint output: " + outputPath);
    }

    out << "slab,direction,source_object_id,candidate_count,odd_candidate_count,hit_total,"
        << "candidate_overflow_events,target_xor,target_sum,fingerprint_xor,fingerprint_sum\n";
    for (const auto& worker : workerResults) {
        for (const ContainmentFingerprintRow& row : worker.containmentFingerprints) {
            out << row.slabIndex << ','
                << (row.direction == 0 ? "mesh1_to_mesh2" : "mesh2_to_mesh1") << ','
                << row.sourceObjectId << ','
                << row.candidateCount << ','
                << row.oddCandidateCount << ','
                << row.hitTotal << ','
                << row.overflowEvents << ','
                << row.targetXor << ','
                << row.targetSum << ','
                << row.fingerprintXor << ','
                << row.fingerprintSum << "\n";
        }
    }
}

void writeContainmentHitHistogramCsv(
    const std::string& outputPath,
    const std::vector<IntersectionGpuWorkerResult>& workerResults,
    int maxBucket
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open containment hit histogram output: " + outputPath);
    }

    out << "slab,direction,hit_count,candidate_records\n";
    auto writeDirection = [&](const IntersectionGpuWorkerResult& worker,
                              const std::vector<unsigned long long>& histogram,
                              const char* direction) {
        for (int bucket = 0; bucket <= maxBucket && bucket < static_cast<int>(histogram.size()); ++bucket) {
            const unsigned long long count = histogram[bucket];
            if (count == 0) {
                continue;
            }
            out << worker.slabIndex << ',' << direction << ',' << bucket << ',' << count << "\n";
        }
    };

    for (const auto& worker : workerResults) {
        writeDirection(worker, worker.mesh1ToMesh2HitHistogram, "mesh1_to_mesh2");
        writeDirection(worker, worker.mesh2ToMesh1HitHistogram, "mesh2_to_mesh1");
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
    GpuMemoryTracker* memoryTracker = nullptr,
    PerformanceTimer* timer = nullptr,
    bool verbose = true
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    if (params1.hash_insert_failure_counter) {
        CUDA_CHECK(cudaMemset(params1.hash_insert_failure_counter, 0, sizeof(unsigned long long)));
    }
    
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

    if (runMesh1ToMesh2 && mesh1NumEdges > 0) {
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

    if (runMesh2ToMesh1 && mesh2NumEdges > 0) {
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

    if (runMesh1ToMesh2 && mesh1NumObjects > 0) {
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

    if (runMesh2ToMesh1 && mesh2NumObjects > 0) {
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
    if (memoryTracker) {
        memoryTracker->sample("intersection_after_result_buffer_alloc");
    }

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

    unsigned long long hashInsertFailures = 0;
    if (params1.hash_insert_failure_counter) {
        CUDA_CHECK(cudaMemcpy(
            &hashInsertFailures,
            params1.hash_insert_failure_counter,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        ));
    }

    return {d_merged_results, numUnique, hashInsertFailures};
}

static MeshIntersectionProfilingStats combineProfilingStats(
    const MeshIntersectionProfilingStats& lhs,
    const MeshIntersectionProfilingStats& rhs
) {
    MeshIntersectionProfilingStats combined = lhs;
    combined.overlap_trace_calls += rhs.overlap_trace_calls;
    combined.overlap_iterations_total += rhs.overlap_iterations_total;
    combined.overlap_hits_total += rhs.overlap_hits_total;
    combined.overlap_max_iterations_per_trace = std::max(combined.overlap_max_iterations_per_trace, rhs.overlap_max_iterations_per_trace);
    combined.containment_rays_total += rhs.containment_rays_total;
    combined.containment_iterations_total += rhs.containment_iterations_total;
    combined.containment_hits_total += rhs.containment_hits_total;
    combined.containment_max_iterations_per_ray = std::max(combined.containment_max_iterations_per_ray, rhs.containment_max_iterations_per_ray);
    combined.containment_same_hit_suppressed += rhs.containment_same_hit_suppressed;
    combined.containment_candidate_additions += rhs.containment_candidate_additions;
    combined.containment_candidate_toggles += rhs.containment_candidate_toggles;
    combined.containment_candidate_overflow += rhs.containment_candidate_overflow;
    combined.containment_targets_total += rhs.containment_targets_total;
    combined.hash_insert_failures += rhs.hash_insert_failures;
    return combined;
}

static IntersectionGpuWorkerResult runIntersectionOnCurrentDevice(
    int deviceId,
    const SlabGeometryPair& slabPair,
    const IntersectionExecutionConfig& config
) {
    CUDA_CHECK(cudaSetDevice(deviceId));

    GpuMemoryTracker memoryTracker(config.trackGpuMemory);
    OptixContext context;
    MeshIntersectionLauncher intersectionLauncher(context, config.ptxPath);

    GeometryUploader mesh1Uploader;
    mesh1Uploader.upload(slabPair.mesh1.geometry);
    memoryTracker.sample("intersection_after_upload_mesh1");

    GeometryUploader mesh2Uploader;
    mesh2Uploader.upload(slabPair.mesh2.geometry);
    memoryTracker.sample("intersection_after_upload_mesh2");

    OptixAccelerationStructure mesh1AS(context, mesh1Uploader);
    mesh1AS.build(&memoryTracker, "build_mesh1_gas");

    OptixAccelerationStructure mesh2AS(context, mesh2Uploader);
    mesh2AS.build(&memoryTracker, "build_mesh2_gas");

    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh1.geometry);
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh2.geometry);

    const int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    const int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());
    const int mesh1NumEdges = mesh1EdgeData.num_edges;
    const int mesh2NumEdges = mesh2EdgeData.num_edges;
    const int mesh1NumObjects = static_cast<int>(slabPair.mesh1.localObjectToGlobalObject.size());
    const int mesh2NumObjects = static_cast<int>(slabPair.mesh2.localObjectToGlobalObject.size());

    unsigned long long* d_hash_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_table, static_cast<size_t>(slabPair.hashTableSize) * sizeof(unsigned long long)));
    memoryTracker.sample("intersection_after_hash_table_alloc");

    unsigned long long* d_hash_insert_failures = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_insert_failures, sizeof(unsigned long long)));

    MeshIntersectionProfilingStats* d_profiling_stats = nullptr;
    if (config.enableProfilingStats) {
        CUDA_CHECK(cudaMalloc(&d_profiling_stats, sizeof(MeshIntersectionProfilingStats)));
        CUDA_CHECK(cudaMemset(d_profiling_stats, 0, sizeof(MeshIntersectionProfilingStats)));
    }

    int* d_first_triangle_mesh1 = nullptr;
    int* d_first_triangle_mesh2 = nullptr;
    int* d_source_object_ids_mesh1 = nullptr;
    int* d_source_object_ids_mesh2 = nullptr;
    float3* d_launch_points_mesh1 = nullptr;
    float3* d_launch_points_mesh2 = nullptr;

    if (mesh1NumObjects > 0) {
        CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_first_triangle_mesh1,
            slabPair.mesh1.firstTriangleIndexPerLocalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&d_source_object_ids_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_source_object_ids_mesh1,
            slabPair.mesh1.localObjectToGlobalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&d_launch_points_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(
            d_launch_points_mesh1,
            slabPair.mesh1.launchPointPerLocalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(float3),
            cudaMemcpyHostToDevice
        ));
    }

    if (mesh2NumObjects > 0) {
        CUDA_CHECK(cudaMalloc(&d_first_triangle_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_first_triangle_mesh2,
            slabPair.mesh2.firstTriangleIndexPerLocalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&d_source_object_ids_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            d_source_object_ids_mesh2,
            slabPair.mesh2.localObjectToGlobalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&d_launch_points_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(
            d_launch_points_mesh2,
            slabPair.mesh2.launchPointPerLocalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(float3),
            cudaMemcpyHostToDevice
        ));
    }

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

    unsigned int* d_fingerprint_odd_count_mesh1 = nullptr;
    unsigned long long* d_fingerprint_hit_total_mesh1 = nullptr;
    unsigned long long* d_fingerprint_target_xor_mesh1 = nullptr;
    unsigned long long* d_fingerprint_target_sum_mesh1 = nullptr;
    unsigned long long* d_fingerprint_xor_mesh1 = nullptr;
    unsigned long long* d_fingerprint_sum_mesh1 = nullptr;
    unsigned int* d_fingerprint_odd_count_mesh2 = nullptr;
    unsigned long long* d_fingerprint_hit_total_mesh2 = nullptr;
    unsigned long long* d_fingerprint_target_xor_mesh2 = nullptr;
    unsigned long long* d_fingerprint_target_sum_mesh2 = nullptr;
    unsigned long long* d_fingerprint_xor_mesh2 = nullptr;
    unsigned long long* d_fingerprint_sum_mesh2 = nullptr;
    unsigned long long* d_hit_histogram_mesh1 = nullptr;
    unsigned long long* d_hit_histogram_mesh2 = nullptr;

    auto allocateAnyhitBuffers = [&](int sourceObjects,
                                     int*& d_candidateIds,
                                     unsigned int*& d_candidateParity,
                                     unsigned int*& d_candidateHits,
                                     unsigned int*& d_candidateCounts,
                                     unsigned int*& d_candidateOverflow) {
        if (sourceObjects <= 0) {
            return;
        }
        const size_t slots = static_cast<size_t>(sourceObjects) * static_cast<size_t>(config.anyhitMaxTargetsPerSource);
        CUDA_CHECK(cudaMalloc(&d_candidateIds, slots * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_candidateParity, slots * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_candidateHits, slots * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_candidateCounts, static_cast<size_t>(sourceObjects) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_candidateOverflow, static_cast<size_t>(sourceObjects) * sizeof(unsigned int)));
    };

    allocateAnyhitBuffers(
        mesh1NumObjects,
        d_anyhit_candidate_object_ids_mesh1,
        d_anyhit_candidate_parity_mesh1,
        d_anyhit_candidate_hit_counts_mesh1,
        d_anyhit_candidate_count_mesh1,
        d_anyhit_candidate_overflow_mesh1
    );
    allocateAnyhitBuffers(
        mesh2NumObjects,
        d_anyhit_candidate_object_ids_mesh2,
        d_anyhit_candidate_parity_mesh2,
        d_anyhit_candidate_hit_counts_mesh2,
        d_anyhit_candidate_count_mesh2,
        d_anyhit_candidate_overflow_mesh2
    );
    memoryTracker.sample("intersection_after_anyhit_buffers_alloc");

    const bool enableContainmentFingerprints = !config.containmentFingerprintOutputPath.empty();
    auto allocateFingerprintBuffers = [&](int sourceObjects,
                                          unsigned int*& d_oddCounts,
                                          unsigned long long*& d_hitTotals,
                                          unsigned long long*& d_targetXors,
                                          unsigned long long*& d_targetSums,
                                          unsigned long long*& d_xors,
                                          unsigned long long*& d_sums) {
        if (!enableContainmentFingerprints || sourceObjects <= 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(&d_oddCounts, static_cast<size_t>(sourceObjects) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_hitTotals, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_targetXors, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_targetSums, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_xors, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_sums, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_oddCounts, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_hitTotals, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_targetXors, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_targetSums, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_xors, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_sums, 0, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long)));
    };

    allocateFingerprintBuffers(
        mesh1NumObjects,
        d_fingerprint_odd_count_mesh1,
        d_fingerprint_hit_total_mesh1,
        d_fingerprint_target_xor_mesh1,
        d_fingerprint_target_sum_mesh1,
        d_fingerprint_xor_mesh1,
        d_fingerprint_sum_mesh1
    );
    allocateFingerprintBuffers(
        mesh2NumObjects,
        d_fingerprint_odd_count_mesh2,
        d_fingerprint_hit_total_mesh2,
        d_fingerprint_target_xor_mesh2,
        d_fingerprint_target_sum_mesh2,
        d_fingerprint_xor_mesh2,
        d_fingerprint_sum_mesh2
    );
    memoryTracker.sample("intersection_after_fingerprint_buffers_alloc");

    const bool enableHitHistogram = !config.containmentHitHistogramOutputPath.empty();
    const int hitHistogramBuckets = kContainmentHitHistogramMaxBucket + 1;
    if (enableHitHistogram) {
        CUDA_CHECK(cudaMalloc(&d_hit_histogram_mesh1, static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_hit_histogram_mesh2, static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_hit_histogram_mesh1, 0, static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_hit_histogram_mesh2, 0, static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long)));
    }
    memoryTracker.sample("intersection_after_hit_histogram_alloc");

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
    params1.mesh2_num_objects = static_cast<int>(slabPair.mesh2.localObjectToGlobalObject.size());
    params1.hash_table = d_hash_table;
    params1.hash_table_size = slabPair.hashTableSize;
    params1.use_hash_table = true;
    params1.hash_insert_failure_counter = d_hash_insert_failures;
    params1.first_triangle_index_per_object = d_first_triangle_mesh1;
    params1.source_object_ids_by_launch_index = d_source_object_ids_mesh1;
    params1.launch_points_per_object = d_launch_points_mesh1;
    params1.overlap_max_iterations = config.overlapMaxIterations;
    params1.profiling_enabled = config.enableProfilingStats ? 1 : 0;
    params1.profiling_stats = d_profiling_stats;
    params1.anyhit_max_pair_targets_per_source = config.anyhitMaxTargetsPerSource;
    params1.anyhit_candidate_object_ids = d_anyhit_candidate_object_ids_mesh1;
    params1.anyhit_candidate_parity = d_anyhit_candidate_parity_mesh1;
    params1.anyhit_candidate_hit_counts = d_anyhit_candidate_hit_counts_mesh1;
    params1.anyhit_candidate_count_per_source = d_anyhit_candidate_count_mesh1;
    params1.anyhit_candidate_overflow_per_source = d_anyhit_candidate_overflow_mesh1;
    params1.enable_containment_fingerprints = enableContainmentFingerprints ? 1 : 0;
    params1.containment_fingerprint_odd_candidate_count_per_source = d_fingerprint_odd_count_mesh1;
    params1.containment_fingerprint_hit_total_per_source = d_fingerprint_hit_total_mesh1;
    params1.containment_fingerprint_target_xor_per_source = d_fingerprint_target_xor_mesh1;
    params1.containment_fingerprint_target_sum_per_source = d_fingerprint_target_sum_mesh1;
    params1.containment_fingerprint_xor_per_source = d_fingerprint_xor_mesh1;
    params1.containment_fingerprint_sum_per_source = d_fingerprint_sum_mesh1;
    params1.enable_containment_hit_histogram = enableHitHistogram ? 1 : 0;
    params1.containment_hit_histogram_max_bucket = kContainmentHitHistogramMaxBucket;
    params1.containment_hit_histogram = d_hit_histogram_mesh1;

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
    params2.mesh2_num_objects = static_cast<int>(slabPair.mesh1.localObjectToGlobalObject.size());
    params2.hash_table = d_hash_table;
    params2.hash_table_size = slabPair.hashTableSize;
    params2.use_hash_table = true;
    params2.hash_insert_failure_counter = d_hash_insert_failures;
    params2.first_triangle_index_per_object = d_first_triangle_mesh2;
    params2.source_object_ids_by_launch_index = d_source_object_ids_mesh2;
    params2.launch_points_per_object = d_launch_points_mesh2;
    params2.overlap_max_iterations = config.overlapMaxIterations;
    params2.profiling_enabled = config.enableProfilingStats ? 1 : 0;
    params2.profiling_stats = d_profiling_stats;
    params2.anyhit_max_pair_targets_per_source = config.anyhitMaxTargetsPerSource;
    params2.anyhit_candidate_object_ids = d_anyhit_candidate_object_ids_mesh2;
    params2.anyhit_candidate_parity = d_anyhit_candidate_parity_mesh2;
    params2.anyhit_candidate_hit_counts = d_anyhit_candidate_hit_counts_mesh2;
    params2.anyhit_candidate_count_per_source = d_anyhit_candidate_count_mesh2;
    params2.anyhit_candidate_overflow_per_source = d_anyhit_candidate_overflow_mesh2;
    params2.enable_containment_fingerprints = enableContainmentFingerprints ? 1 : 0;
    params2.containment_fingerprint_odd_candidate_count_per_source = d_fingerprint_odd_count_mesh2;
    params2.containment_fingerprint_hit_total_per_source = d_fingerprint_hit_total_mesh2;
    params2.containment_fingerprint_target_xor_per_source = d_fingerprint_target_xor_mesh2;
    params2.containment_fingerprint_target_sum_per_source = d_fingerprint_target_sum_mesh2;
    params2.containment_fingerprint_xor_per_source = d_fingerprint_xor_mesh2;
    params2.containment_fingerprint_sum_per_source = d_fingerprint_sum_mesh2;
    params2.enable_containment_hit_histogram = enableHitHistogram ? 1 : 0;
    params2.containment_hit_histogram_max_bucket = kContainmentHitHistogramMaxBucket;
    params2.containment_hit_histogram = d_hit_histogram_mesh2;

    for (int warmup = 0; warmup < config.warmupRuns; ++warmup) {
        QueryResults warmupResults = executeHashQuery(
            intersectionLauncher,
            params1,
            params2,
            mesh1NumEdges,
            mesh2NumEdges,
            mesh1NumObjects,
            mesh2NumObjects,
            d_hash_table,
            slabPair.hashTableSize,
            config.queryDirection,
            nullptr,
            nullptr,
            false
        );
        if (warmupResults.d_merged_results) {
            CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }

    QueryResults results = executeHashQuery(
        intersectionLauncher,
        params1,
        params2,
        mesh1NumEdges,
        mesh2NumEdges,
        mesh1NumObjects,
        mesh2NumObjects,
        d_hash_table,
        slabPair.hashTableSize,
        config.queryDirection,
        &memoryTracker,
        nullptr,
        false
    );

    IntersectionGpuWorkerResult workerResult;
    workerResult.slabIndex = slabPair.slabIndex;
    if (results.numUnique > 0) {
        workerResult.pairs.resize(results.numUnique);
        CUDA_CHECK(cudaMemcpy(
            workerResult.pairs.data(),
            results.d_merged_results,
            static_cast<size_t>(results.numUnique) * sizeof(MeshQueryResult),
            cudaMemcpyDeviceToHost
        ));
    }
    workerResult.hashInsertFailures = results.hashInsertFailures;

    if (config.trackOverflow) {
        std::vector<unsigned int> mesh1CandidateCounts(mesh1NumObjects, 0);
        std::vector<unsigned int> mesh1OverflowEvents(mesh1NumObjects, 0);
        std::vector<unsigned int> mesh2CandidateCounts(mesh2NumObjects, 0);
        std::vector<unsigned int> mesh2OverflowEvents(mesh2NumObjects, 0);

        if (mesh1NumObjects > 0) {
            CUDA_CHECK(cudaMemcpy(mesh1CandidateCounts.data(), d_anyhit_candidate_count_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(mesh1OverflowEvents.data(), d_anyhit_candidate_overflow_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (mesh2NumObjects > 0) {
            CUDA_CHECK(cudaMemcpy(mesh2CandidateCounts.data(), d_anyhit_candidate_count_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(mesh2OverflowEvents.data(), d_anyhit_candidate_overflow_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        workerResult.anyhitSummary = summarizeIntersectionAnyhitUsage(
            mesh1CandidateCounts,
            mesh1OverflowEvents,
            mesh2CandidateCounts,
            mesh2OverflowEvents
        );
    }

    if (enableContainmentFingerprints) {
        const bool recordMesh1ToMesh2 =
            (config.queryDirection == QueryDirection::Both || config.queryDirection == QueryDirection::Mesh1ToMesh2);
        const bool recordMesh2ToMesh1 =
            (config.queryDirection == QueryDirection::Both || config.queryDirection == QueryDirection::Mesh2ToMesh1);

        auto appendFingerprintRows = [&](int direction,
                                         int sourceObjects,
                                         const std::vector<int>& localToGlobal,
                                         unsigned int* d_candidateCounts,
                                         unsigned int* d_overflowEvents,
                                         unsigned int* d_oddCounts,
                                         unsigned long long* d_hitTotals,
                                         unsigned long long* d_targetXors,
                                         unsigned long long* d_targetSums,
                                         unsigned long long* d_xors,
                                         unsigned long long* d_sums) {
            if (sourceObjects <= 0) {
                return;
            }

            std::vector<unsigned int> candidateCounts(sourceObjects, 0);
            std::vector<unsigned int> overflowEvents(sourceObjects, 0);
            std::vector<unsigned int> oddCounts(sourceObjects, 0);
            std::vector<unsigned long long> hitTotals(sourceObjects, 0);
            std::vector<unsigned long long> targetXors(sourceObjects, 0);
            std::vector<unsigned long long> targetSums(sourceObjects, 0);
            std::vector<unsigned long long> xors(sourceObjects, 0);
            std::vector<unsigned long long> sums(sourceObjects, 0);

            CUDA_CHECK(cudaMemcpy(candidateCounts.data(), d_candidateCounts, static_cast<size_t>(sourceObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(overflowEvents.data(), d_overflowEvents, static_cast<size_t>(sourceObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(oddCounts.data(), d_oddCounts, static_cast<size_t>(sourceObjects) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hitTotals.data(), d_hitTotals, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(targetXors.data(), d_targetXors, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(targetSums.data(), d_targetSums, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(xors.data(), d_xors, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, static_cast<size_t>(sourceObjects) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            workerResult.containmentFingerprints.reserve(
                workerResult.containmentFingerprints.size() + static_cast<size_t>(sourceObjects)
            );
            for (int localSource = 0; localSource < sourceObjects; ++localSource) {
                ContainmentFingerprintRow row;
                row.slabIndex = slabPair.slabIndex;
                row.direction = direction;
                row.sourceObjectId = localToGlobal[localSource];
                row.candidateCount = candidateCounts[localSource];
                row.oddCandidateCount = oddCounts[localSource];
                row.overflowEvents = overflowEvents[localSource];
                row.hitTotal = hitTotals[localSource];
                row.targetXor = targetXors[localSource];
                row.targetSum = targetSums[localSource];
                row.fingerprintXor = xors[localSource];
                row.fingerprintSum = sums[localSource];
                workerResult.containmentFingerprints.push_back(row);
            }
        };

        if (recordMesh1ToMesh2) {
            appendFingerprintRows(
                0,
                mesh1NumObjects,
                slabPair.mesh1.localObjectToGlobalObject,
                d_anyhit_candidate_count_mesh1,
                d_anyhit_candidate_overflow_mesh1,
                d_fingerprint_odd_count_mesh1,
                d_fingerprint_hit_total_mesh1,
                d_fingerprint_target_xor_mesh1,
                d_fingerprint_target_sum_mesh1,
                d_fingerprint_xor_mesh1,
                d_fingerprint_sum_mesh1
            );
        }
        if (recordMesh2ToMesh1) {
            appendFingerprintRows(
                1,
                mesh2NumObjects,
                slabPair.mesh2.localObjectToGlobalObject,
                d_anyhit_candidate_count_mesh2,
                d_anyhit_candidate_overflow_mesh2,
                d_fingerprint_odd_count_mesh2,
                d_fingerprint_hit_total_mesh2,
                d_fingerprint_target_xor_mesh2,
                d_fingerprint_target_sum_mesh2,
                d_fingerprint_xor_mesh2,
                d_fingerprint_sum_mesh2
            );
        }
    }

    if (enableHitHistogram) {
        const bool recordMesh1ToMesh2 =
            (config.queryDirection == QueryDirection::Both || config.queryDirection == QueryDirection::Mesh1ToMesh2);
        const bool recordMesh2ToMesh1 =
            (config.queryDirection == QueryDirection::Both || config.queryDirection == QueryDirection::Mesh2ToMesh1);

        workerResult.mesh1ToMesh2HitHistogram.assign(static_cast<size_t>(hitHistogramBuckets), 0ULL);
        workerResult.mesh2ToMesh1HitHistogram.assign(static_cast<size_t>(hitHistogramBuckets), 0ULL);
        if (recordMesh1ToMesh2) {
            CUDA_CHECK(cudaMemcpy(
                workerResult.mesh1ToMesh2HitHistogram.data(),
                d_hit_histogram_mesh1,
                static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost
            ));
        }
        if (recordMesh2ToMesh1) {
            CUDA_CHECK(cudaMemcpy(
                workerResult.mesh2ToMesh1HitHistogram.data(),
                d_hit_histogram_mesh2,
                static_cast<size_t>(hitHistogramBuckets) * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost
            ));
        }
    }

    if (config.enableProfilingStats && d_profiling_stats) {
        CUDA_CHECK(cudaMemcpy(&workerResult.profilingStats, d_profiling_stats, sizeof(MeshIntersectionProfilingStats), cudaMemcpyDeviceToHost));
    }

    workerResult.peakMemoryBytes = memoryTracker.getPeakUsedBytes();

    if (d_profiling_stats) CUDA_CHECK(cudaFree(d_profiling_stats));
    if (results.d_merged_results) CUDA_CHECK(cudaFree(results.d_merged_results));
    if (d_hash_insert_failures) CUDA_CHECK(cudaFree(d_hash_insert_failures));
    if (d_hash_table) CUDA_CHECK(cudaFree(d_hash_table));
    if (d_first_triangle_mesh1) CUDA_CHECK(cudaFree(d_first_triangle_mesh1));
    if (d_first_triangle_mesh2) CUDA_CHECK(cudaFree(d_first_triangle_mesh2));
    if (d_source_object_ids_mesh1) CUDA_CHECK(cudaFree(d_source_object_ids_mesh1));
    if (d_source_object_ids_mesh2) CUDA_CHECK(cudaFree(d_source_object_ids_mesh2));
    if (d_launch_points_mesh1) CUDA_CHECK(cudaFree(d_launch_points_mesh1));
    if (d_launch_points_mesh2) CUDA_CHECK(cudaFree(d_launch_points_mesh2));
    if (d_anyhit_candidate_object_ids_mesh1) CUDA_CHECK(cudaFree(d_anyhit_candidate_object_ids_mesh1));
    if (d_anyhit_candidate_parity_mesh1) CUDA_CHECK(cudaFree(d_anyhit_candidate_parity_mesh1));
    if (d_anyhit_candidate_hit_counts_mesh1) CUDA_CHECK(cudaFree(d_anyhit_candidate_hit_counts_mesh1));
    if (d_anyhit_candidate_count_mesh1) CUDA_CHECK(cudaFree(d_anyhit_candidate_count_mesh1));
    if (d_anyhit_candidate_overflow_mesh1) CUDA_CHECK(cudaFree(d_anyhit_candidate_overflow_mesh1));
    if (d_anyhit_candidate_object_ids_mesh2) CUDA_CHECK(cudaFree(d_anyhit_candidate_object_ids_mesh2));
    if (d_anyhit_candidate_parity_mesh2) CUDA_CHECK(cudaFree(d_anyhit_candidate_parity_mesh2));
    if (d_anyhit_candidate_hit_counts_mesh2) CUDA_CHECK(cudaFree(d_anyhit_candidate_hit_counts_mesh2));
    if (d_anyhit_candidate_count_mesh2) CUDA_CHECK(cudaFree(d_anyhit_candidate_count_mesh2));
    if (d_anyhit_candidate_overflow_mesh2) CUDA_CHECK(cudaFree(d_anyhit_candidate_overflow_mesh2));
    if (d_fingerprint_odd_count_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_odd_count_mesh1));
    if (d_fingerprint_hit_total_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_hit_total_mesh1));
    if (d_fingerprint_target_xor_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_target_xor_mesh1));
    if (d_fingerprint_target_sum_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_target_sum_mesh1));
    if (d_fingerprint_xor_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_xor_mesh1));
    if (d_fingerprint_sum_mesh1) CUDA_CHECK(cudaFree(d_fingerprint_sum_mesh1));
    if (d_fingerprint_odd_count_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_odd_count_mesh2));
    if (d_fingerprint_hit_total_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_hit_total_mesh2));
    if (d_fingerprint_target_xor_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_target_xor_mesh2));
    if (d_fingerprint_target_sum_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_target_sum_mesh2));
    if (d_fingerprint_xor_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_xor_mesh2));
    if (d_fingerprint_sum_mesh2) CUDA_CHECK(cudaFree(d_fingerprint_sum_mesh2));
    if (d_hit_histogram_mesh1) CUDA_CHECK(cudaFree(d_hit_histogram_mesh1));
    if (d_hit_histogram_mesh2) CUDA_CHECK(cudaFree(d_hit_histogram_mesh2));

    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);
    mesh1Uploader.free();
    mesh2Uploader.free();

    return workerResult;
}

static std::vector<MeshQueryResult> mergeAndDeduplicateHostPairs(
    const std::vector<IntersectionGpuWorkerResult>& workerResults
) {
    std::unordered_set<unsigned long long> seen;
    size_t totalPairs = 0;
    for (const auto& worker : workerResults) {
        totalPairs += worker.pairs.size();
    }
    seen.reserve(totalPairs);

    std::vector<MeshQueryResult> mergedPairs;
    mergedPairs.reserve(totalPairs);
    for (const auto& worker : workerResults) {
        for (const MeshQueryResult& pair : worker.pairs) {
            const unsigned long long key = packIntersectionPairKey(pair.object_id_mesh1, pair.object_id_mesh2);
            if (seen.insert(key).second) {
                mergedPairs.push_back(pair);
            }
        }
    }

    std::sort(mergedPairs.begin(), mergedPairs.end(), [](const MeshQueryResult& lhs, const MeshQueryResult& rhs) {
        if (lhs.object_id_mesh1 != rhs.object_id_mesh1) {
            return lhs.object_id_mesh1 < rhs.object_id_mesh1;
        }
        return lhs.object_id_mesh2 < rhs.object_id_mesh2;
    });
    return mergedPairs;
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
    IntersectionEstimatedCliOptions() : BenchmarkMeshPairCliOptions("estimated_intersection_timing.json") {
        allowNoExportFlag = true;
    }

    std::string queryDirectionArg = "both";
    bool estimateOnly = false;
    bool trackOverflow = false;
    bool enableProfilingStats = false;
    std::string pairsOutputPath;
    std::string containmentFingerprintOutputPath;
    std::string containmentHitHistogramOutputPath;
    float gamma = 0.8f;
    float epsilon = 0.001f;
    float hashLoadFactor = 0.5f;
    int overlapMaxIterations = 100;
    int anyhitMaxTargetsPerSource = -1;
    bool trackGpuMemory = false;
    int numGpus = 1;

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(options);
        appendBenchmarkRunHelp(options);
        options.emplace_back("--gamma <float>", "Estimation gamma (default: 0.8)");
        options.emplace_back("--epsilon <float>", "Estimation epsilon (default: 0.001)");
        options.emplace_back("--estimate-only", "Run only selectivity estimation");
        options.emplace_back("--query-direction <both|mesh1_to_mesh2|mesh2_to_mesh1>", "Control query direction (default: both)");
        options.emplace_back("--overlap-max-iterations <int>", "Overlap ray iteration cap (default: 100)");
        options.emplace_back("--containment-anyhit-max-targets <int>", "Containment any-hit scratch cap per source object (default: 256)");
        options.emplace_back("--hash-load-factor <float>", "Hash load factor in (0,1] (default: 0.5)");
        options.emplace_back("--track-overflow", "Enable containment any-hit overflow summary diagnostics");
        options.emplace_back("--enable-profiling-stats", "Enable device-side profiling counters");
        options.emplace_back("--track-gpu-memory", "Track GPU memory checkpoints and peak usage");
        options.emplace_back("--num-gpus <int>", "Number of GPUs / shared x-slabs to use (default: 1)");
        options.emplace_back("--pairs-output <path>", "Intersection pairs CSV path (default: intersection_pairs.csv)");
        options.emplace_back("--containment-fingerprint-output <path>", "Write compact per-source containment candidate fingerprints");
        options.emplace_back("--containment-hit-histogram-output <path>", "Write containment candidate hit-count histogram");
        appendNoExportHelp(options);
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
        if (arg == "--containment-anyhit-max-targets" && i + 1 < argc) {
            anyhitMaxTargetsPerSource = std::stoi(argv[++i]);
            return true;
        }
        if (arg == "--hash-load-factor" && i + 1 < argc) {
            hashLoadFactor = std::stof(argv[++i]);
            return true;
        }
        if (arg == "--track-overflow") {
            trackOverflow = true;
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
        if (arg == "--containment-fingerprint-output" && i + 1 < argc) {
            containmentFingerprintOutputPath = argv[++i];
            return true;
        }
        if (arg == "--containment-hit-histogram-output" && i + 1 < argc) {
            containmentHitHistogramOutputPath = argv[++i];
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
    const bool trackOverflow = options.trackOverflow;
    const bool enableProfilingStats = options.enableProfilingStats;
    const bool exportResults = options.exportResults;
    const std::string pairsOutputPath = options.pairsOutputPath.empty()
        ? "intersection_pairs.csv"
        : options.pairsOutputPath;
    const std::string containmentFingerprintOutputPath = options.containmentFingerprintOutputPath;
    const std::string containmentHitHistogramOutputPath = options.containmentHitHistogramOutputPath;
    const float gamma = options.gamma;
    const float epsilon = options.epsilon;
    const float hashLoadFactor = options.hashLoadFactor;
    const int overlapMaxIterations = options.overlapMaxIterations;
    const int warmupRuns = options.warmupRuns;
    const bool trackGpuMemory = options.trackGpuMemory;
    const int requestedNumGpus = options.numGpus;
    const bool anyhitCapExplicitlyConfigured = options.anyhitMaxTargetsPerSource > 0;
    int anyhitMaxTargetsPerSource = anyhitCapExplicitlyConfigured
        ? options.anyhitMaxTargetsPerSource
        : kIntersectionAnyhitLegacyDefaultMaxTargetsPerSource;

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
    if (anyhitMaxTargetsPerSource <= 0 ||
        anyhitMaxTargetsPerSource > kIntersectionAnyhitHardMaxTargetsPerSource) {
        std::cerr << "--containment-anyhit-max-targets must be in [1, "
                  << kIntersectionAnyhitHardMaxTargetsPerSource << "]." << std::endl;
        return 1;
    }
    if (requestedNumGpus <= 0) {
        std::cerr << "--num-gpus must be > 0." << std::endl;
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
        std::cout << "Requested GPUs:     " << requestedNumGpus << std::endl;
        std::cout << "Query Direction:    " << queryDirectionArg << std::endl;
        std::cout << "Overlap Max Iter:   " << overlapMaxIterations << std::endl;
        std::cout << "Containment AnyHit: " << anyhitMaxTargetsPerSource
                  << " per source (" << (anyhitCapExplicitlyConfigured ? "explicit" : "default") << ")" << std::endl;
        std::cout << "Overflow Tracking:  " << (trackOverflow ? "enabled" : "disabled") << std::endl;
        std::cout << "Export Results:     " << (exportResults ? "enabled" : "disabled") << std::endl;
        std::cout << "Profiling Stats:    " << (enableProfilingStats ? "enabled" : "disabled") << std::endl;
        std::cout << "===========================\n" << std::endl;
        timer.finish(outputJsonPath);
        return 0;
    }

    std::cout << "\n=== Query Configuration ===" << std::endl;
    std::cout << "Estimated Pairs:    " << estimatedPairs << std::endl;
    std::cout << "Hash Table Size:    " << hash_table_size << " (Load Factor ~" << hashLoadFactor << ")" << std::endl;
    std::cout << "Requested GPUs:     " << requestedNumGpus << std::endl;
    std::cout << "Query Direction:    " << queryDirectionArg << std::endl;
    std::cout << "Overlap Max Iter:   " << overlapMaxIterations << std::endl;
    std::cout << "Containment AnyHit: " << anyhitMaxTargetsPerSource
              << " per source (" << (anyhitCapExplicitlyConfigured ? "explicit" : "default") << ")" << std::endl;
    std::cout << "Overflow Tracking:  " << (trackOverflow ? "enabled" : "disabled") << std::endl;
    std::cout << "Fingerprints:       " << (containmentFingerprintOutputPath.empty() ? "disabled" : "enabled") << std::endl;
    std::cout << "Hit Histogram:      " << (containmentHitHistogramOutputPath.empty() ? "disabled" : "enabled") << std::endl;
    std::cout << "Export Results:     " << (exportResults ? "enabled" : "disabled") << std::endl;
    std::cout << "Profiling Stats:    " << (enableProfilingStats ? "enabled" : "disabled") << std::endl;
    std::cout << "===========================\n" << std::endl;

    timer.next("Plan Shared Slabs");
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

    const SlabPartitionPlan slabPlan = planSharedXAxisSlabs(mesh1, mesh2, activeGpuCount);
    std::vector<SlabGeometryPair> slabPairs = buildSlabGeometryPairs(mesh1, mesh2, slabPlan, hash_table_size);

    for (const SlabGeometryPair& slabPair : slabPairs) {
        std::cout << "Slab " << slabPair.slabIndex
                  << ": mesh1 triangles=" << slabPair.mesh1.geometry.indices.size()
                  << ", mesh1 edges=" << slabPair.mesh1.geometry.edges.edgeStarts.size()
                  << ", mesh2 triangles=" << slabPair.mesh2.geometry.indices.size()
                  << ", mesh2 edges=" << slabPair.mesh2.geometry.edges.edgeStarts.size()
                  << ", hash_table_size=" << slabPair.hashTableSize << std::endl;
    }

    IntersectionExecutionConfig executionConfig;
    executionConfig.queryDirection = queryDirection;
    executionConfig.overlapMaxIterations = overlapMaxIterations;
    executionConfig.anyhitMaxTargetsPerSource = anyhitMaxTargetsPerSource;
    executionConfig.enableProfilingStats = enableProfilingStats;
    executionConfig.trackOverflow = trackOverflow;
    executionConfig.trackGpuMemory = trackGpuMemory;
    executionConfig.warmupRuns = warmupRuns;
    executionConfig.ptxPath = options.ptxPath;
    executionConfig.containmentFingerprintOutputPath = containmentFingerprintOutputPath;
    executionConfig.containmentHitHistogramOutputPath = containmentHitHistogramOutputPath;

    timer.next("Query");
    std::cout << "Running Intersection Query..." << std::endl;

    std::vector<IntersectionGpuWorkerResult> workerResults(slabPairs.size());
    std::vector<std::thread> workerThreads;
    std::mutex errorMutex;
    std::string workerError;
    std::atomic<bool> workerFailed{false};

    for (size_t workerIndex = 0; workerIndex < slabPairs.size(); ++workerIndex) {
        workerThreads.emplace_back([&, workerIndex]() {
            try {
                workerResults[workerIndex] = runIntersectionOnCurrentDevice(
                    static_cast<int>(workerIndex),
                    slabPairs[workerIndex],
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
        std::cerr << (workerError.empty() ? "Multi-GPU execution failed." : workerError) << std::endl;
        return 1;
    }

    std::vector<MeshQueryResult> h_pairs = mergeAndDeduplicateHostPairs(workerResults);
    std::cout << "Actual Intersection Pairs: " << h_pairs.size() << std::endl;
    timer.addCounter("Profile_Actual_Intersection_Pairs", static_cast<unsigned long long>(h_pairs.size()));
    timer.addCounter("Profile_Active_GPU_Count", static_cast<unsigned long long>(activeGpuCount));

    unsigned long long aggregateHashInsertFailures = 0;
    for (const auto& workerResult : workerResults) {
        aggregateHashInsertFailures += workerResult.hashInsertFailures;
    }
    std::cout << "Hash insert failures: " << aggregateHashInsertFailures << std::endl;
    timer.addCounter("Profile_Hash_Insert_Failures", aggregateHashInsertFailures);

    if (exportResults) {
        writeIntersectionPairsCsv(pairsOutputPath, h_pairs);
        std::cout << "Intersection pairs CSV: " << pairsOutputPath << std::endl;
    }

    if (!containmentFingerprintOutputPath.empty()) {
        writeContainmentFingerprintCsv(containmentFingerprintOutputPath, workerResults);
        std::cout << "Containment fingerprints CSV: " << containmentFingerprintOutputPath << std::endl;
    }

    if (!containmentHitHistogramOutputPath.empty()) {
        writeContainmentHitHistogramCsv(
            containmentHitHistogramOutputPath,
            workerResults,
            kContainmentHitHistogramMaxBucket
        );
        std::cout << "Containment hit histogram CSV: " << containmentHitHistogramOutputPath << std::endl;
    }

    if (trackOverflow) {
        IntersectionAnyhitUsageSummary aggregateAnyhitSummary;
        for (const auto& workerResult : workerResults) {
            aggregateAnyhitSummary.maxCandidateCount =
                std::max(aggregateAnyhitSummary.maxCandidateCount, workerResult.anyhitSummary.maxCandidateCount);
            aggregateAnyhitSummary.overflowEvents += workerResult.anyhitSummary.overflowEvents;
            aggregateAnyhitSummary.overflowSources += workerResult.anyhitSummary.overflowSources;
        }
        std::cout << "Containment any-hit usage: max_candidates=" << aggregateAnyhitSummary.maxCandidateCount
                  << ", overflow_events=" << aggregateAnyhitSummary.overflowEvents
                  << ", overflow_sources=" << aggregateAnyhitSummary.overflowSources << std::endl;
        timer.addCounter("Profile_Containment_Anyhit_Max_Candidates", aggregateAnyhitSummary.maxCandidateCount);
        timer.addCounter("Profile_Containment_Anyhit_Overflow_Events", aggregateAnyhitSummary.overflowEvents);
        timer.addCounter("Profile_Containment_Anyhit_Overflow_Sources", aggregateAnyhitSummary.overflowSources);
    }

    if (enableProfilingStats) {
        MeshIntersectionProfilingStats aggregateProfilingStats = {};
        for (const auto& workerResult : workerResults) {
            aggregateProfilingStats = combineProfilingStats(aggregateProfilingStats, workerResult.profilingStats);
        }

        timer.addCounter("Profile_Overlap_Trace_Calls", aggregateProfilingStats.overlap_trace_calls);
        timer.addCounter("Profile_Overlap_Iterations_Total", aggregateProfilingStats.overlap_iterations_total);
        timer.addCounter("Profile_Overlap_Hits_Total", aggregateProfilingStats.overlap_hits_total);
        timer.addCounter("Profile_Overlap_Max_Iterations_Per_Trace", aggregateProfilingStats.overlap_max_iterations_per_trace);
        timer.addCounter("Profile_Containment_Rays_Total", aggregateProfilingStats.containment_rays_total);
        timer.addCounter("Profile_Containment_Iterations_Total", aggregateProfilingStats.containment_iterations_total);
        timer.addCounter("Profile_Containment_Hits_Total", aggregateProfilingStats.containment_hits_total);
        timer.addCounter("Profile_Containment_Max_Iterations_Per_Ray", aggregateProfilingStats.containment_max_iterations_per_ray);
        timer.addCounter("Profile_Containment_Same_Hit_Suppressed", aggregateProfilingStats.containment_same_hit_suppressed);
        timer.addCounter("Profile_Containment_Candidate_Additions", aggregateProfilingStats.containment_candidate_additions);
        timer.addCounter("Profile_Containment_Candidate_Toggles", aggregateProfilingStats.containment_candidate_toggles);
        timer.addCounter("Profile_Containment_Candidate_Overflow", aggregateProfilingStats.containment_candidate_overflow);
        timer.addCounter("Profile_Containment_Targets_Total", aggregateProfilingStats.containment_targets_total);
        timer.addCounter("Profile_Hash_Insert_Failures_Profiling", aggregateProfilingStats.hash_insert_failures);

        const unsigned long long overlapCalls = aggregateProfilingStats.overlap_trace_calls;
        const unsigned long long containmentRays = aggregateProfilingStats.containment_rays_total;
        const unsigned long long avgOverlapIterScaled = overlapCalls ? (aggregateProfilingStats.overlap_iterations_total * 1000ULL / overlapCalls) : 0ULL;
        const unsigned long long avgContainmentIterScaled = containmentRays ? (aggregateProfilingStats.containment_iterations_total * 1000ULL / containmentRays) : 0ULL;
        timer.addCounter("Profile_Overlap_Avg_Iterations_x1000", avgOverlapIterScaled);
        timer.addCounter("Profile_Containment_Avg_Iterations_x1000", avgContainmentIterScaled);
    }

    if (trackGpuMemory) {
        unsigned long long peakMemoryBytes = 0;
        for (const auto& workerResult : workerResults) {
            peakMemoryBytes = std::max(peakMemoryBytes, workerResult.peakMemoryBytes);
        }
        timer.addCounter("Profile_GPU_Peak_Used_Bytes", peakMemoryBytes);
        std::cout << "Peak GPU memory across workers: " << peakMemoryBytes << " bytes" << std::endl;
    }

    timer.next("Cleanup");
    timer.finish(outputJsonPath);
    return 0;
}
