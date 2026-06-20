#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>

// Forward declaration
struct MeshIntersectionLaunchParams;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

/**
 * Utility functions and structures for pair-hit tracking and CSV export.
 * These handle the allocation, setup, and export of per-source/per-target hit counts
 * from containment ray tracing.
 */

/**
 * Encapsulates all buffers and setup for pair-hit tracking during containment queries.
 */
struct PairHitTrackingBuffers {
    static constexpr int kMaxPairTargetsPerSource = 256;

    // Device pointers
    int* d_mesh1_pair_target_ids = nullptr;
    unsigned int* d_mesh1_pair_target_hits = nullptr;
    int* d_mesh2_pair_target_ids = nullptr;
    unsigned int* d_mesh2_pair_target_hits = nullptr;

    // Host mirrors
    std::vector<int> h_mesh1_pair_target_ids;
    std::vector<unsigned int> h_mesh1_pair_target_hits;
    std::vector<int> h_mesh2_pair_target_ids;
    std::vector<unsigned int> h_mesh2_pair_target_hits;

    /**
     * Allocate and initialize pair-hit tracking buffers.
     * @param mesh1NumObjects Number of objects in mesh1
     * @param mesh2NumObjects Number of objects in mesh2
     */
    void allocate(int mesh1NumObjects, int mesh2NumObjects) {
        CUDA_CHECK(cudaMalloc(&d_mesh1_pair_target_ids, mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mesh1_pair_target_hits, mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh2_pair_target_ids, mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_mesh2_pair_target_hits, mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int)));

        CUDA_CHECK(cudaMemset(d_mesh1_pair_target_ids, 0xFF, mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_mesh1_pair_target_hits, 0, mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh2_pair_target_ids, 0xFF, mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_mesh2_pair_target_hits, 0, mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int)));

        h_mesh1_pair_target_ids.resize(mesh1NumObjects * kMaxPairTargetsPerSource, -1);
        h_mesh1_pair_target_hits.resize(mesh1NumObjects * kMaxPairTargetsPerSource, 0);
        h_mesh2_pair_target_ids.resize(mesh2NumObjects * kMaxPairTargetsPerSource, -1);
        h_mesh2_pair_target_hits.resize(mesh2NumObjects * kMaxPairTargetsPerSource, 0);
    }

    /**
     * Configure launch parameters with pair-hit tracking buffers.
     * @param params1 Launch parameters for mesh1->mesh2
     * @param params2 Launch parameters for mesh2->mesh1
     */
    void setupLaunchParams(MeshIntersectionLaunchParams& params1, MeshIntersectionLaunchParams& params2) {
        params1.enable_pair_hit_tracking = 1;
        params1.max_pair_targets_per_source = kMaxPairTargetsPerSource;
        params1.pair_target_object_ids = d_mesh1_pair_target_ids;
        params1.pair_target_hit_counts = d_mesh1_pair_target_hits;

        params2.enable_pair_hit_tracking = 1;
        params2.max_pair_targets_per_source = kMaxPairTargetsPerSource;
        params2.pair_target_object_ids = d_mesh2_pair_target_ids;
        params2.pair_target_hit_counts = d_mesh2_pair_target_hits;
    }

    /**
     * Copy pair-hit data from device to host.
     * @param mesh1NumObjects Number of objects in mesh1
     * @param mesh2NumObjects Number of objects in mesh2
     */
    void copyFromDevice(int mesh1NumObjects, int mesh2NumObjects) {
        CUDA_CHECK(cudaMemcpy(h_mesh1_pair_target_ids.data(), d_mesh1_pair_target_ids, 
                              mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh1_pair_target_hits.data(), d_mesh1_pair_target_hits, 
                              mesh1NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh2_pair_target_ids.data(), d_mesh2_pair_target_ids, 
                              mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh2_pair_target_hits.data(), d_mesh2_pair_target_hits, 
                              mesh2NumObjects * kMaxPairTargetsPerSource * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    /**
     * Free all allocated device memory.
     */
    void free() {
        if (d_mesh1_pair_target_ids) CUDA_CHECK(cudaFree(d_mesh1_pair_target_ids));
        if (d_mesh1_pair_target_hits) CUDA_CHECK(cudaFree(d_mesh1_pair_target_hits));
        if (d_mesh2_pair_target_ids) CUDA_CHECK(cudaFree(d_mesh2_pair_target_ids));
        if (d_mesh2_pair_target_hits) CUDA_CHECK(cudaFree(d_mesh2_pair_target_hits));
    }
};

/**
 * Write pair-hit tracking data to a CSV file.
 * 
 * Format:
 *   direction,source_object_id,target_object_id,target_ray_hits
 *
 * @param outputPath Path to write CSV file
 * @param maxTargetsPerSource Maximum targets tracked per source object
 * @param mesh1TargetIds Vector of target object IDs from mesh1 to mesh2 (size: mesh1NumObjects * maxTargetsPerSource)
 * @param mesh1TargetHits Vector of hit counts for mesh1 targets (size: mesh1NumObjects * maxTargetsPerSource)
 * @param mesh2TargetIds Vector of target object IDs from mesh2 to mesh1 (size: mesh2NumObjects * maxTargetsPerSource)
 * @param mesh2TargetHits Vector of hit counts for mesh2 targets (size: mesh2NumObjects * maxTargetsPerSource)
 * @throws std::runtime_error if file cannot be opened
 */
inline void writePairHitTrackingCsv(
    const std::string& outputPath,
    int maxTargetsPerSource,
    const std::vector<int>& mesh1TargetIds,
    const std::vector<unsigned int>& mesh1TargetHits,
    const std::vector<int>& mesh2TargetIds,
    const std::vector<unsigned int>& mesh2TargetHits
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open pair hit tracking output: " + outputPath);
    }

    out << "direction,source_object_id,target_object_id,target_ray_hits\n";
    
    const int mesh1Sources = static_cast<int>(mesh1TargetIds.size()) / maxTargetsPerSource;
    for (int src = 0; src < mesh1Sources; ++src) {
        const int base = src * maxTargetsPerSource;
        for (int i = 0; i < maxTargetsPerSource; ++i) {
            const int tgt = mesh1TargetIds[base + i];
            if (tgt < 0) {
                continue;
            }
            out << "mesh1_to_mesh2," << src << ',' << tgt << ',' << mesh1TargetHits[base + i] << "\n";
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
            out << "mesh2_to_mesh1," << src << ',' << tgt << ',' << mesh2TargetHits[base + i] << "\n";
        }
    }
}
