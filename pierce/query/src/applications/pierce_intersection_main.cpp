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
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
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

struct HashQueryTimingStats {
    long long clearHashUs = 0;
    long long overlapMesh1ToMesh2Us = 0;
    long long overlapMesh2ToMesh1Us = 0;
    long long containmentMesh1ToMesh2Us = 0;
    long long containmentMesh2ToMesh1Us = 0;
    long long resultBufferAllocUs = 0;
    long long compactHashTableUs = 0;
    long long hashFailureCopyUs = 0;
};

struct IntersectionWorkerTimingStats {
    long long totalUs = 0;
    long long setDeviceUs = 0;
    long long contextCreateUs = 0;
    long long uploadMesh1Us = 0;
    long long uploadMesh2Us = 0;
    long long buildMesh1AsUs = 0;
    long long buildMesh2AsUs = 0;
    long long uploadMesh1EdgesUs = 0;
    long long uploadMesh2EdgesUs = 0;
    long long allocHashAndProfilingUs = 0;
    long long uploadObjectMetadataUs = 0;
    long long allocAnyhitBuffersUs = 0;
    long long allocOptionalDiagnosticsUs = 0;
    long long warmupTotalUs = 0;
    long long measuredHashQueryTotalUs = 0;
    HashQueryTimingStats measuredHashQuery;
    long long downloadPairsUs = 0;
    long long overflowDownloadUs = 0;
    long long fingerprintDownloadUs = 0;
    long long hitHistogramDownloadUs = 0;
    long long profilingStatsDownloadUs = 0;
    long long cleanupUs = 0;
};

struct IntersectionGpuWorkerRuntime {
    int deviceId = 0;
    long long setDeviceUs = 0;
    long long contextCreateUs = 0;
    std::unique_ptr<OptixContext> context;
    std::unique_ptr<MeshIntersectionLauncher> launcher;

    IntersectionGpuWorkerRuntime(int device, const std::string& ptxPath)
        : deviceId(device) {
        auto phaseStart = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaSetDevice(deviceId));
        setDeviceUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - phaseStart
        ).count();

        phaseStart = std::chrono::high_resolution_clock::now();
        context = std::make_unique<OptixContext>();
        launcher = std::make_unique<MeshIntersectionLauncher>(*context, ptxPath);
        contextCreateUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - phaseStart
        ).count();
    }
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

struct IntersectionResidentWorkerState {
    int deviceId = 0;
    std::unique_ptr<GeometryUploader> mesh1Uploader;
    std::unique_ptr<GeometryUploader> mesh2Uploader;
    std::unique_ptr<OptixAccelerationStructure> mesh1AS;
    std::unique_ptr<OptixAccelerationStructure> mesh2AS;
    EdgeMeshData mesh1EdgeData;
    EdgeMeshData mesh2EdgeData;
    int* d_first_triangle_mesh1 = nullptr;
    int* d_first_triangle_mesh2 = nullptr;
    int* d_source_object_ids_mesh1 = nullptr;
    int* d_source_object_ids_mesh2 = nullptr;
    float3* d_launch_points_mesh1 = nullptr;
    float3* d_launch_points_mesh2 = nullptr;
    GpuMemoryTracker memoryTracker;

    explicit IntersectionResidentWorkerState(int device, bool trackGpuMemory)
        : deviceId(device), memoryTracker(trackGpuMemory) {}

    ~IntersectionResidentWorkerState() {
        CUDA_CHECK(cudaSetDevice(deviceId));
        mesh2AS.reset();
        mesh1AS.reset();
        PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
        PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);
        if (d_first_triangle_mesh1) CUDA_CHECK(cudaFree(d_first_triangle_mesh1));
        if (d_first_triangle_mesh2) CUDA_CHECK(cudaFree(d_first_triangle_mesh2));
        if (d_source_object_ids_mesh1) CUDA_CHECK(cudaFree(d_source_object_ids_mesh1));
        if (d_source_object_ids_mesh2) CUDA_CHECK(cudaFree(d_source_object_ids_mesh2));
        if (d_launch_points_mesh1) CUDA_CHECK(cudaFree(d_launch_points_mesh1));
        if (d_launch_points_mesh2) CUDA_CHECK(cudaFree(d_launch_points_mesh2));
        if (mesh1Uploader) mesh1Uploader->free();
        if (mesh2Uploader) mesh2Uploader->free();
    }

    IntersectionResidentWorkerState(const IntersectionResidentWorkerState&) = delete;
    IntersectionResidentWorkerState& operator=(const IntersectionResidentWorkerState&) = delete;
};

struct IntersectionGpuWorkerResult {
    MeshQueryResult* dPairs = nullptr;
    long long numPairs = 0;
    int deviceId = 0;
    std::vector<ContainmentFingerprintRow> containmentFingerprints;
    std::vector<unsigned long long> mesh1ToMesh2HitHistogram;
    std::vector<unsigned long long> mesh2ToMesh1HitHistogram;
    IntersectionAnyhitUsageSummary anyhitSummary;
    MeshIntersectionProfilingStats profilingStats = {};
    unsigned long long hashInsertFailures = 0;
    unsigned long long peakMemoryBytes = 0;
    IntersectionWorkerTimingStats timing;
    int slabIndex = 0;
    std::unique_ptr<IntersectionResidentWorkerState> residentState;
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
    bool verbose = true,
    HashQueryTimingStats* timingStats = nullptr
) {
    // Clear hash table (set to 0xFF which is our sentinel for empty)
    auto phaseStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemset(d_hash_table, 0xFF, hash_table_size * sizeof(unsigned long long)));
    if (params1.hash_insert_failure_counter) {
        CUDA_CHECK(cudaMemset(params1.hash_insert_failure_counter, 0, sizeof(unsigned long long)));
    }
    if (timingStats) {
        timingStats->clearHashUs += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - phaseStart
        ).count();
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
        const long long elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (timingStats) {
            timingStats->overlapMesh1ToMesh2Us += elapsedUs;
        }
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh1ToMesh2",
                elapsedUs
            );
        }
    }

    if (runMesh2ToMesh1 && mesh2NumEdges > 0) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchOverlapMesh2ToMesh1(params2, mesh2NumEdges);
        t1 = std::chrono::high_resolution_clock::now();
        const long long elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (timingStats) {
            timingStats->overlapMesh2ToMesh1Us += elapsedUs;
        }
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Overlap_Hash_Mesh2ToMesh1",
                elapsedUs
            );
        }
    }

    if (runMesh1ToMesh2 && mesh1NumObjects > 0) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh1ToMesh2(params1, mesh1NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        const long long elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (timingStats) {
            timingStats->containmentMesh1ToMesh2Us += elapsedUs;
        }
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh1ToMesh2",
                elapsedUs
            );
        }
    }

    if (runMesh2ToMesh1 && mesh2NumObjects > 0) {
        t0 = std::chrono::high_resolution_clock::now();
        intersectionLauncher.launchContainmentMesh2ToMesh1(params2, mesh2NumObjects);
        t1 = std::chrono::high_resolution_clock::now();
        const long long elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (timingStats) {
            timingStats->containmentMesh2ToMesh1Us += elapsedUs;
        }
        if (timer) {
            timer->addMeasurement(
                "Raytrace_Containment_Hash_Mesh2ToMesh1",
                elapsedUs
            );
        }
    }

    int max_output = hash_table_size; 

    MeshQueryResult* d_merged_results = nullptr;
    phaseStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_merged_results, max_output * sizeof(MeshQueryResult)));
    if (timingStats) {
        timingStats->resultBufferAllocUs += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - phaseStart
        ).count();
    }
    if (memoryTracker) {
        memoryTracker->sample("intersection_after_result_buffer_alloc");
    }

    auto t_dedup_start = std::chrono::high_resolution_clock::now();
    int numUnique = compact_hash_table_pairs(d_hash_table, hash_table_size, d_merged_results, max_output);
    auto t_dedup_end = std::chrono::high_resolution_clock::now();
    const long long compactUs = std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_end - t_dedup_start).count();
    if (timingStats) {
        timingStats->compactHashTableUs += compactUs;
    }
    if (timer) {
        timer->addMeasurement(
            "compact_hash_table_pairs",
            compactUs
        );
    }
    
    if (verbose) {
         std::cout << "Hash Table Query found " << numUnique << " unique pairs." << std::endl;
    }

    unsigned long long hashInsertFailures = 0;
    if (params1.hash_insert_failure_counter) {
        phaseStart = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(
            &hashInsertFailures,
            params1.hash_insert_failure_counter,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        ));
        if (timingStats) {
            timingStats->hashFailureCopyUs += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - phaseStart
            ).count();
        }
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
    const IntersectionExecutionConfig& config,
    IntersectionGpuWorkerRuntime& runtime
) {
    IntersectionWorkerTimingStats workerTiming;
    const auto workerStart = std::chrono::high_resolution_clock::now();
    auto phaseStart = workerStart;
    auto elapsedSince = [](std::chrono::high_resolution_clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start
        ).count();
    };

    phaseStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaSetDevice(deviceId));
    workerTiming.setDeviceUs += elapsedSince(phaseStart);

    auto resident = std::make_unique<IntersectionResidentWorkerState>(deviceId, config.trackGpuMemory);
    GpuMemoryTracker& memoryTracker = resident->memoryTracker;
    OptixContext& context = *runtime.context;
    MeshIntersectionLauncher& intersectionLauncher = *runtime.launcher;

    resident->mesh1Uploader = std::make_unique<GeometryUploader>();
    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh1Uploader->upload(slabPair.mesh1.geometry);
    memoryTracker.sample("intersection_after_upload_mesh1");
    workerTiming.uploadMesh1Us += elapsedSince(phaseStart);

    resident->mesh2Uploader = std::make_unique<GeometryUploader>();
    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh2Uploader->upload(slabPair.mesh2.geometry);
    memoryTracker.sample("intersection_after_upload_mesh2");
    workerTiming.uploadMesh2Us += elapsedSince(phaseStart);

    resident->mesh1AS = std::make_unique<OptixAccelerationStructure>(context, *resident->mesh1Uploader);
    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh1AS->build(&memoryTracker, "build_mesh1_gas");
    workerTiming.buildMesh1AsUs += elapsedSince(phaseStart);

    resident->mesh2AS = std::make_unique<OptixAccelerationStructure>(context, *resident->mesh2Uploader);
    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh2AS->build(&memoryTracker, "build_mesh2_gas");
    workerTiming.buildMesh2AsUs += elapsedSince(phaseStart);

    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh1.geometry);
    workerTiming.uploadMesh1EdgesUs += elapsedSince(phaseStart);
    phaseStart = std::chrono::high_resolution_clock::now();
    resident->mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(slabPair.mesh2.geometry);
    workerTiming.uploadMesh2EdgesUs += elapsedSince(phaseStart);

    const int mesh1NumTriangles = static_cast<int>(resident->mesh1Uploader->getNumIndices());
    const int mesh2NumTriangles = static_cast<int>(resident->mesh2Uploader->getNumIndices());
    const int mesh1NumEdges = resident->mesh1EdgeData.num_edges;
    const int mesh2NumEdges = resident->mesh2EdgeData.num_edges;
    const int mesh1NumObjects = static_cast<int>(slabPair.mesh1.localObjectToGlobalObject.size());
    const int mesh2NumObjects = static_cast<int>(slabPair.mesh2.localObjectToGlobalObject.size());

    unsigned long long* d_hash_table = nullptr;
    phaseStart = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_hash_table, static_cast<size_t>(slabPair.hashTableSize) * sizeof(unsigned long long)));
    memoryTracker.sample("intersection_after_hash_table_alloc");

    unsigned long long* d_hash_insert_failures = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hash_insert_failures, sizeof(unsigned long long)));

    MeshIntersectionProfilingStats* d_profiling_stats = nullptr;
    if (config.enableProfilingStats) {
        CUDA_CHECK(cudaMalloc(&d_profiling_stats, sizeof(MeshIntersectionProfilingStats)));
        CUDA_CHECK(cudaMemset(d_profiling_stats, 0, sizeof(MeshIntersectionProfilingStats)));
    }
    workerTiming.allocHashAndProfilingUs += elapsedSince(phaseStart);

    phaseStart = std::chrono::high_resolution_clock::now();
    if (mesh1NumObjects > 0) {
        CUDA_CHECK(cudaMalloc(&resident->d_first_triangle_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_first_triangle_mesh1,
            slabPair.mesh1.firstTriangleIndexPerLocalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&resident->d_source_object_ids_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_source_object_ids_mesh1,
            slabPair.mesh1.localObjectToGlobalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&resident->d_launch_points_mesh1, static_cast<size_t>(mesh1NumObjects) * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_launch_points_mesh1,
            slabPair.mesh1.launchPointPerLocalObject.data(),
            static_cast<size_t>(mesh1NumObjects) * sizeof(float3),
            cudaMemcpyHostToDevice
        ));
    }

    if (mesh2NumObjects > 0) {
        CUDA_CHECK(cudaMalloc(&resident->d_first_triangle_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_first_triangle_mesh2,
            slabPair.mesh2.firstTriangleIndexPerLocalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&resident->d_source_object_ids_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_source_object_ids_mesh2,
            slabPair.mesh2.localObjectToGlobalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMalloc(&resident->d_launch_points_mesh2, static_cast<size_t>(mesh2NumObjects) * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(
            resident->d_launch_points_mesh2,
            slabPair.mesh2.launchPointPerLocalObject.data(),
            static_cast<size_t>(mesh2NumObjects) * sizeof(float3),
            cudaMemcpyHostToDevice
        ));
    }
    workerTiming.uploadObjectMetadataUs += elapsedSince(phaseStart);

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

    phaseStart = std::chrono::high_resolution_clock::now();
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
    workerTiming.allocAnyhitBuffersUs += elapsedSince(phaseStart);

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

    phaseStart = std::chrono::high_resolution_clock::now();
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
    workerTiming.allocOptionalDiagnosticsUs += elapsedSince(phaseStart);

    MeshIntersectionLaunchParams params1{};
    params1.mesh1_vertices = resident->mesh1Uploader->getVertices();
    params1.mesh1_indices = resident->mesh1Uploader->getIndices();
    params1.mesh1_triangle_to_object = resident->mesh1Uploader->getTriangleToObject();
    params1.mesh1_num_triangles = mesh1NumTriangles;
    params1.mesh1_num_objects = mesh1NumObjects;
    params1.edge_starts = resident->mesh1EdgeData.d_edge_starts;
    params1.edge_ends = resident->mesh1EdgeData.d_edge_ends;
    params1.edge_source_object_ids = resident->mesh1EdgeData.d_source_object_ids;
    params1.num_edges = mesh1NumEdges;
    params1.mesh2_handle = resident->mesh2AS->getHandle();
    params1.mesh2_vertices = resident->mesh2Uploader->getVertices();
    params1.mesh2_indices = resident->mesh2Uploader->getIndices();
    params1.mesh2_triangle_to_object = resident->mesh2Uploader->getTriangleToObject();
    params1.mesh2_num_objects = static_cast<int>(slabPair.mesh2.localObjectToGlobalObject.size());
    params1.hash_table = d_hash_table;
    params1.hash_table_size = slabPair.hashTableSize;
    params1.use_hash_table = true;
    params1.hash_insert_failure_counter = d_hash_insert_failures;
    params1.first_triangle_index_per_object = resident->d_first_triangle_mesh1;
    params1.source_object_ids_by_launch_index = resident->d_source_object_ids_mesh1;
    params1.launch_points_per_object = resident->d_launch_points_mesh1;
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
    params2.mesh1_vertices = resident->mesh2Uploader->getVertices();
    params2.mesh1_indices = resident->mesh2Uploader->getIndices();
    params2.mesh1_triangle_to_object = resident->mesh2Uploader->getTriangleToObject();
    params2.mesh1_num_triangles = mesh2NumTriangles;
    params2.mesh1_num_objects = mesh2NumObjects;
    params2.edge_starts = resident->mesh2EdgeData.d_edge_starts;
    params2.edge_ends = resident->mesh2EdgeData.d_edge_ends;
    params2.edge_source_object_ids = resident->mesh2EdgeData.d_source_object_ids;
    params2.num_edges = mesh2NumEdges;
    params2.mesh2_handle = resident->mesh1AS->getHandle();
    params2.mesh2_vertices = resident->mesh1Uploader->getVertices();
    params2.mesh2_indices = resident->mesh1Uploader->getIndices();
    params2.mesh2_triangle_to_object = resident->mesh1Uploader->getTriangleToObject();
    params2.mesh2_num_objects = static_cast<int>(slabPair.mesh1.localObjectToGlobalObject.size());
    params2.hash_table = d_hash_table;
    params2.hash_table_size = slabPair.hashTableSize;
    params2.use_hash_table = true;
    params2.hash_insert_failure_counter = d_hash_insert_failures;
    params2.first_triangle_index_per_object = resident->d_first_triangle_mesh2;
    params2.source_object_ids_by_launch_index = resident->d_source_object_ids_mesh2;
    params2.launch_points_per_object = resident->d_launch_points_mesh2;
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

    phaseStart = std::chrono::high_resolution_clock::now();
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
            false,
            nullptr
        );
        if (warmupResults.d_merged_results) {
            CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }
    workerTiming.warmupTotalUs += elapsedSince(phaseStart);

    HashQueryTimingStats measuredHashTiming;
    phaseStart = std::chrono::high_resolution_clock::now();
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
        false,
        &measuredHashTiming
    );
    workerTiming.measuredHashQueryTotalUs += elapsedSince(phaseStart);
    workerTiming.measuredHashQuery = measuredHashTiming;

    IntersectionGpuWorkerResult workerResult;
    workerResult.slabIndex = slabPair.slabIndex;
    workerResult.deviceId = deviceId;
    if (results.numUnique > 0) {
        workerResult.dPairs = results.d_merged_results;
        workerResult.numPairs = results.numUnique;
        results.d_merged_results = nullptr;
    }
    workerResult.hashInsertFailures = results.hashInsertFailures;

    if (config.trackOverflow) {
        phaseStart = std::chrono::high_resolution_clock::now();
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
        workerTiming.overflowDownloadUs += elapsedSince(phaseStart);
    }

    if (enableContainmentFingerprints) {
        phaseStart = std::chrono::high_resolution_clock::now();
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
        workerTiming.fingerprintDownloadUs += elapsedSince(phaseStart);
    }

    if (enableHitHistogram) {
        phaseStart = std::chrono::high_resolution_clock::now();
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
        workerTiming.hitHistogramDownloadUs += elapsedSince(phaseStart);
    }

    if (config.enableProfilingStats && d_profiling_stats) {
        phaseStart = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(&workerResult.profilingStats, d_profiling_stats, sizeof(MeshIntersectionProfilingStats), cudaMemcpyDeviceToHost));
        workerTiming.profilingStatsDownloadUs += elapsedSince(phaseStart);
    }

    workerResult.peakMemoryBytes = memoryTracker.getPeakUsedBytes();

    phaseStart = std::chrono::high_resolution_clock::now();
    if (d_profiling_stats) CUDA_CHECK(cudaFree(d_profiling_stats));
    if (results.d_merged_results) CUDA_CHECK(cudaFree(results.d_merged_results));
    if (d_hash_insert_failures) CUDA_CHECK(cudaFree(d_hash_insert_failures));
    if (d_hash_table) CUDA_CHECK(cudaFree(d_hash_table));
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

    workerTiming.cleanupUs += elapsedSince(phaseStart);
    workerTiming.totalUs = elapsedSince(workerStart);
    workerResult.timing = workerTiming;
    workerResult.residentState = std::move(resident);

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

static void recordIntersectionPeakMemoryStats(
    const std::vector<IntersectionGpuWorkerResult>& workerResults,
    PerformanceTimer& timer
) {
    unsigned long long totalPeakBytes = 0;
    unsigned long long minPeakBytes = std::numeric_limits<unsigned long long>::max();
    unsigned long long maxPeakBytes = 0;

    for (const IntersectionGpuWorkerResult& workerResult : workerResults) {
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

static void recordIntersectionWorkerTimingStats(
    const std::vector<IntersectionGpuWorkerResult>& workerResults,
    PerformanceTimer& timer
) {
    unsigned long long totalWorkerUs = 0;
    unsigned long long maxWorkerUs = 0;
    unsigned long long minWorkerUs = std::numeric_limits<unsigned long long>::max();

    std::cout << "\n=== Per-GPU Worker Timing ===" << std::endl;
    for (const IntersectionGpuWorkerResult& workerResult : workerResults) {
        const IntersectionWorkerTimingStats& timing = workerResult.timing;
        const HashQueryTimingStats& hashTiming = timing.measuredHashQuery;
        const auto totalUs = static_cast<unsigned long long>(timing.totalUs);
        totalWorkerUs += totalUs;
        maxWorkerUs = std::max(maxWorkerUs, totalUs);
        minWorkerUs = std::min(minWorkerUs, totalUs);

        std::cout << "Slab " << workerResult.slabIndex
                  << ": worker_total_us=" << timing.totalUs
                  << ", measured_hash_query_us=" << timing.measuredHashQueryTotalUs
                  << ", warmup_total_us=" << timing.warmupTotalUs
                  << ", upload_mesh1_us=" << timing.uploadMesh1Us
                  << ", upload_mesh2_us=" << timing.uploadMesh2Us
                  << ", build_mesh1_as_us=" << timing.buildMesh1AsUs
                  << ", build_mesh2_as_us=" << timing.buildMesh2AsUs
                  << ", download_pairs_us=" << timing.downloadPairsUs
                  << ", cleanup_us=" << timing.cleanupUs
                  << std::endl;

        const std::string prefix = "Profile_Slab_" + std::to_string(workerResult.slabIndex) + "_Worker";
        timer.addCounter(prefix + "_Total_Us", totalUs);
        timer.addCounter(prefix + "_Set_Device_Us", static_cast<unsigned long long>(timing.setDeviceUs));
        timer.addCounter(prefix + "_Context_Create_Us", static_cast<unsigned long long>(timing.contextCreateUs));
        timer.addCounter(prefix + "_Upload_Mesh1_Us", static_cast<unsigned long long>(timing.uploadMesh1Us));
        timer.addCounter(prefix + "_Upload_Mesh2_Us", static_cast<unsigned long long>(timing.uploadMesh2Us));
        timer.addCounter(prefix + "_Build_Mesh1_AS_Us", static_cast<unsigned long long>(timing.buildMesh1AsUs));
        timer.addCounter(prefix + "_Build_Mesh2_AS_Us", static_cast<unsigned long long>(timing.buildMesh2AsUs));
        timer.addCounter(prefix + "_Upload_Mesh1_Edges_Us", static_cast<unsigned long long>(timing.uploadMesh1EdgesUs));
        timer.addCounter(prefix + "_Upload_Mesh2_Edges_Us", static_cast<unsigned long long>(timing.uploadMesh2EdgesUs));
        timer.addCounter(prefix + "_Alloc_Hash_And_Profiling_Us", static_cast<unsigned long long>(timing.allocHashAndProfilingUs));
        timer.addCounter(prefix + "_Upload_Object_Metadata_Us", static_cast<unsigned long long>(timing.uploadObjectMetadataUs));
        timer.addCounter(prefix + "_Alloc_Anyhit_Buffers_Us", static_cast<unsigned long long>(timing.allocAnyhitBuffersUs));
        timer.addCounter(prefix + "_Alloc_Optional_Diagnostics_Us", static_cast<unsigned long long>(timing.allocOptionalDiagnosticsUs));
        timer.addCounter(prefix + "_Warmup_Total_Us", static_cast<unsigned long long>(timing.warmupTotalUs));
        timer.addCounter(prefix + "_Measured_Hash_Query_Total_Us", static_cast<unsigned long long>(timing.measuredHashQueryTotalUs));
        timer.addCounter(prefix + "_Hash_Clear_Us", static_cast<unsigned long long>(hashTiming.clearHashUs));
        timer.addCounter(prefix + "_Hash_Overlap_Mesh1_To_Mesh2_Us", static_cast<unsigned long long>(hashTiming.overlapMesh1ToMesh2Us));
        timer.addCounter(prefix + "_Hash_Overlap_Mesh2_To_Mesh1_Us", static_cast<unsigned long long>(hashTiming.overlapMesh2ToMesh1Us));
        timer.addCounter(prefix + "_Hash_Containment_Mesh1_To_Mesh2_Us", static_cast<unsigned long long>(hashTiming.containmentMesh1ToMesh2Us));
        timer.addCounter(prefix + "_Hash_Containment_Mesh2_To_Mesh1_Us", static_cast<unsigned long long>(hashTiming.containmentMesh2ToMesh1Us));
        timer.addCounter(prefix + "_Hash_Result_Buffer_Alloc_Us", static_cast<unsigned long long>(hashTiming.resultBufferAllocUs));
        timer.addCounter(prefix + "_Hash_Compact_Table_Us", static_cast<unsigned long long>(hashTiming.compactHashTableUs));
        timer.addCounter(prefix + "_Hash_Failure_Copy_Us", static_cast<unsigned long long>(hashTiming.hashFailureCopyUs));
        timer.addCounter(prefix + "_Download_Pairs_Us", static_cast<unsigned long long>(timing.downloadPairsUs));
        timer.addCounter(prefix + "_Overflow_Download_Us", static_cast<unsigned long long>(timing.overflowDownloadUs));
        timer.addCounter(prefix + "_Fingerprint_Download_Us", static_cast<unsigned long long>(timing.fingerprintDownloadUs));
        timer.addCounter(prefix + "_Hit_Histogram_Download_Us", static_cast<unsigned long long>(timing.hitHistogramDownloadUs));
        timer.addCounter(prefix + "_Profiling_Stats_Download_Us", static_cast<unsigned long long>(timing.profilingStatsDownloadUs));
        timer.addCounter(prefix + "_Cleanup_Us", static_cast<unsigned long long>(timing.cleanupUs));
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

    SlabPlanningTimingStats slabPlanningTiming;
    const SlabPartitionPlan slabPlan = planSharedXAxisSlabs(mesh1, mesh2, activeGpuCount, &slabPlanningTiming);
    std::vector<SlabGeometryPair> slabPairs = buildSlabGeometryPairs(mesh1, mesh2, slabPlan, hash_table_size, &slabPlanningTiming);

    const auto memoryStatsStart = std::chrono::high_resolution_clock::now();
    printAndRecordSlabMemoryStats(slabPairs, timer);
    const long long memoryStatsUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - memoryStatsStart
    ).count();
    recordSlabPlanningTimingStats(slabPlanningTiming, memoryStatsUs, timer);

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

    timer.next("Initialize GPU Workers");
    std::vector<std::unique_ptr<IntersectionGpuWorkerRuntime>> workerRuntimes(slabPairs.size());
    std::vector<std::thread> initThreads;
    std::mutex initErrorMutex;
    std::string initError;
    std::atomic<bool> initFailed{false};

    for (size_t workerIndex = 0; workerIndex < slabPairs.size(); ++workerIndex) {
        initThreads.emplace_back([&, workerIndex]() {
            try {
                workerRuntimes[workerIndex] = std::make_unique<IntersectionGpuWorkerRuntime>(
                    static_cast<int>(workerIndex),
                    executionConfig.ptxPath
                );
            } catch (const std::exception& ex) {
                initFailed.store(true);
                std::lock_guard<std::mutex> lock(initErrorMutex);
                if (initError.empty()) {
                    std::ostringstream message;
                    message << "GPU worker runtime " << workerIndex << " initialization failed: " << ex.what();
                    initError = message.str();
                }
            }
        });
    }

    for (std::thread& initThread : initThreads) {
        initThread.join();
    }

    if (initFailed.load()) {
        std::cerr << (initError.empty() ? "GPU worker initialization failed." : initError) << std::endl;
        return 1;
    }

    std::cout << "\n=== GPU Worker Initialization Timing ===" << std::endl;
    for (size_t workerIndex = 0; workerIndex < workerRuntimes.size(); ++workerIndex) {
        const IntersectionGpuWorkerRuntime& runtime = *workerRuntimes[workerIndex];
        std::cout << "Worker " << workerIndex
                  << ": set_device_us=" << runtime.setDeviceUs
                  << ", context_create_us=" << runtime.contextCreateUs
                  << std::endl;
        const std::string prefix = "Profile_Worker_" + std::to_string(workerIndex) + "_Init";
        timer.addCounter(prefix + "_Set_Device_Us", static_cast<unsigned long long>(runtime.setDeviceUs));
        timer.addCounter(prefix + "_Context_Create_Us", static_cast<unsigned long long>(runtime.contextCreateUs));
    }

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
                    executionConfig,
                    *workerRuntimes[workerIndex]
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

    recordIntersectionWorkerTimingStats(workerResults, timer);

    GpuGlobalDedupResult globalDedupResult;
    bool usedGlobalDedup = false;
    long long globalDedupTotalUs = 0;
    if (activeGpuCount == 1 && workerResults.size() == 1) {
        globalDedupResult.aggregatorDeviceId = workerResults[0].deviceId;
        globalDedupResult.inputCount = workerResults[0].numPairs;
        globalDedupResult.uniqueCount = workerResults[0].numPairs;
        globalDedupResult.d_uniquePairs = workerResults[0].dPairs;
        workerResults[0].dPairs = nullptr;
    } else {
        std::vector<DevicePairBuffer> workerBuffers;
        workerBuffers.reserve(workerResults.size());
        for (const auto& workerResult : workerResults) {
            workerBuffers.push_back({workerResult.deviceId, workerResult.dPairs, workerResult.numPairs});
        }

        const auto globalDedupStart = std::chrono::high_resolution_clock::now();
        try {
            GpuMemoryTracker* dedupMemoryTracker =
                (!workerResults.empty() && workerResults[0].residentState)
                    ? &workerResults[0].residentState->memoryTracker
                    : nullptr;
            globalDedupResult = gather_and_deduplicate_pairs_gpu(workerBuffers, 0, dedupMemoryTracker);
            usedGlobalDedup = true;
        } catch (const std::exception& ex) {
            std::cerr << "Global GPU deduplication failed: " << ex.what() << std::endl;
            return 1;
        }
        globalDedupTotalUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - globalDedupStart
        ).count();
    }

    std::cout << "Actual Intersection Pairs: " << globalDedupResult.uniqueCount << std::endl;
    timer.addCounter("Profile_Actual_Intersection_Pairs", static_cast<unsigned long long>(globalDedupResult.uniqueCount));
    timer.addCounter("Profile_Active_GPU_Count", static_cast<unsigned long long>(activeGpuCount));
    timer.addCounter("Profile_Global_Dedup_Mode", usedGlobalDedup ? 1ULL : 0ULL);
    timer.addCounter("Profile_Global_Dedup_Aggregator_Device", static_cast<unsigned long long>(globalDedupResult.aggregatorDeviceId));
    timer.addCounter("Profile_Global_Dedup_Input_Pairs", static_cast<unsigned long long>(globalDedupResult.inputCount));
    timer.addCounter("Profile_Global_Dedup_Unique_Pairs", static_cast<unsigned long long>(globalDedupResult.uniqueCount));
    timer.addCounter("Profile_Global_Dedup_Gather_Us", static_cast<unsigned long long>(globalDedupResult.gatherUs));
    timer.addCounter("Profile_Global_Dedup_SortUnique_Us", static_cast<unsigned long long>(globalDedupResult.dedupUs));
    timer.addCounter("Profile_Global_Dedup_Total_Us", static_cast<unsigned long long>(globalDedupTotalUs));
    timer.addCounter(
        "Profile_Global_Dedup_Buffer_Bytes",
        usedGlobalDedup
            ? static_cast<unsigned long long>(globalDedupResult.inputCount) * static_cast<unsigned long long>(sizeof(MeshQueryResult))
            : 0ULL
    );

    unsigned long long aggregateHashInsertFailures = 0;
    for (const auto& workerResult : workerResults) {
        aggregateHashInsertFailures += workerResult.hashInsertFailures;
    }
    std::cout << "Hash insert failures: " << aggregateHashInsertFailures << std::endl;
    timer.addCounter("Profile_Hash_Insert_Failures", aggregateHashInsertFailures);

    if (exportResults) {
        std::vector<MeshQueryResult> h_pairs;
        if (globalDedupResult.uniqueCount > 0) {
            h_pairs.resize(static_cast<size_t>(globalDedupResult.uniqueCount));
            CUDA_CHECK(cudaSetDevice(globalDedupResult.aggregatorDeviceId));
            CUDA_CHECK(cudaMemcpy(
                h_pairs.data(),
                globalDedupResult.d_uniquePairs,
                static_cast<size_t>(globalDedupResult.uniqueCount) * sizeof(MeshQueryResult),
                cudaMemcpyDeviceToHost
            ));
        }
        writeIntersectionPairsCsv(pairsOutputPath, h_pairs);
        std::cout << "Intersection pairs CSV: " << pairsOutputPath << std::endl;
    }

    if (globalDedupResult.d_uniquePairs) {
        CUDA_CHECK(cudaSetDevice(globalDedupResult.aggregatorDeviceId));
        CUDA_CHECK(cudaFree(globalDedupResult.d_uniquePairs));
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
        for (auto& workerResult : workerResults) {
            if (workerResult.residentState) {
                workerResult.peakMemoryBytes = std::max(
                    workerResult.peakMemoryBytes,
                    static_cast<unsigned long long>(workerResult.residentState->memoryTracker.getPeakUsedBytes())
                );
            }
            peakMemoryBytes = std::max(peakMemoryBytes, workerResult.peakMemoryBytes);
        }
        timer.addCounter("Profile_GPU_Peak_Used_Bytes", peakMemoryBytes);
        recordIntersectionPeakMemoryStats(workerResults, timer);
        std::cout << "Peak GPU memory across workers: " << peakMemoryBytes << " bytes" << std::endl;
    }

    timer.next("Cleanup");
    workerResults.clear();
    workerRuntimes.clear();
    timer.finish(outputJsonPath);
    return 0;
}
