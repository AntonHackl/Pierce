#include "mesh_intersection.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"

extern "C" __constant__ MeshIntersectionLaunchParams mesh_intersection_params;

__device__ __forceinline__ void update_max_u32_anyhit(unsigned int* addr, unsigned int value) {
    atomicMax(addr, value);
}

__device__ void insert_hash_table_anyhit(int id1, int id2) {
    unsigned long long key = (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);

    unsigned long long k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;

    const int size = mesh_intersection_params.hash_table_size;
    if (size <= 0) return;

    unsigned int h = k % size;
    for (int i = 0; i < 1000; ++i) {
        unsigned long long old = atomicCAS(&mesh_intersection_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }
        h = (h + 1) % size;
    }
}

__device__ __forceinline__ void reset_anyhit_candidates_for_source(int sourceObjectId, int maxTargets) {
    const int base = sourceObjectId * maxTargets;
    mesh_intersection_params.anyhit_candidate_count_per_source[sourceObjectId] = 0U;
    mesh_intersection_params.anyhit_candidate_overflow_per_source[sourceObjectId] = 0U;
    for (int i = 0; i < maxTargets; ++i) {
        mesh_intersection_params.anyhit_candidate_object_ids[base + i] = -1;
        mesh_intersection_params.anyhit_candidate_parity[base + i] = 0U;
        mesh_intersection_params.anyhit_candidate_hit_counts[base + i] = 0U;
    }
}

extern "C" __global__ void __raygen__mesh_containment_anyhit() {
    const uint3 idx = optixGetLaunchIndex();
    const int sourceObjectId = static_cast<int>(idx.x);
    if (sourceObjectId >= mesh_intersection_params.mesh1_num_objects) {
        return;
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[sourceObjectId] = 0;
    }

    if (mesh_intersection_params.enable_pair_hit_tracking &&
        mesh_intersection_params.pair_target_object_ids &&
        mesh_intersection_params.pair_target_hit_counts &&
        mesh_intersection_params.max_pair_targets_per_source > 0) {
        const int base = sourceObjectId * mesh_intersection_params.max_pair_targets_per_source;
        for (int i = 0; i < mesh_intersection_params.max_pair_targets_per_source; ++i) {
            mesh_intersection_params.pair_target_object_ids[base + i] = -1;
            mesh_intersection_params.pair_target_hit_counts[base + i] = 0U;
        }
    }

    const int firstTri = mesh_intersection_params.first_triangle_index_per_object[sourceObjectId];
    if (firstTri < 0 || firstTri >= mesh_intersection_params.mesh1_num_triangles) {
        return;
    }

    const int maxTargets = mesh_intersection_params.anyhit_max_pair_targets_per_source;
    if (maxTargets <= 0 ||
        !mesh_intersection_params.anyhit_candidate_object_ids ||
        !mesh_intersection_params.anyhit_candidate_parity ||
        !mesh_intersection_params.anyhit_candidate_hit_counts ||
        !mesh_intersection_params.anyhit_candidate_count_per_source ||
        !mesh_intersection_params.anyhit_candidate_overflow_per_source) {
        return;
    }

    reset_anyhit_candidates_for_source(sourceObjectId, maxTargets);

    const uint3 triIndices = mesh_intersection_params.mesh1_indices[firstTri];
    const float3 origin = mesh_intersection_params.mesh1_vertices[triIndices.x];

    unsigned int payload0 = static_cast<unsigned int>(sourceObjectId);
    unsigned int payload1 = 0U;
    unsigned int payload2 = 0U;

    optixTrace(
        mesh_intersection_params.mesh2_handle,
        origin,
        make_float3(0.0f, 0.0f, 1.0f),
        nextafterf(0.0f, 1e10f),
        1e10f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        payload0,
        payload1,
        payload2);

    const int base = sourceObjectId * maxTargets;
    const int candidateCount = static_cast<int>(mesh_intersection_params.anyhit_candidate_count_per_source[sourceObjectId]);
    const int candidateOverflow = static_cast<int>(mesh_intersection_params.anyhit_candidate_overflow_per_source[sourceObjectId]);

    int targetObjects[256];
    unsigned int targetHitCounts[256];
    int numTargets = 0;

    const int scanCount = (candidateCount < maxTargets) ? candidateCount : maxTargets;
    for (int i = 0; i < scanCount; ++i) {
        if ((mesh_intersection_params.anyhit_candidate_parity[base + i] & 1U) != 0U) {
            if (numTargets < 256) {
                targetObjects[numTargets] = mesh_intersection_params.anyhit_candidate_object_ids[base + i];
                targetHitCounts[numTargets] = mesh_intersection_params.anyhit_candidate_hit_counts[base + i];
            }
            numTargets++;
        }
    }

    if (mesh_intersection_params.enable_containment_tracking &&
        mesh_intersection_params.containment_iterations_per_source &&
        mesh_intersection_params.containment_candidate_count_per_source &&
        mesh_intersection_params.containment_candidate_overflow_per_source) {
        mesh_intersection_params.containment_iterations_per_source[sourceObjectId] = 1U;
        mesh_intersection_params.containment_candidate_count_per_source[sourceObjectId] = static_cast<unsigned int>(scanCount);
        mesh_intersection_params.containment_candidate_overflow_per_source[sourceObjectId] = static_cast<unsigned int>(candidateOverflow);
    }

    if (mesh_intersection_params.enable_pair_hit_tracking &&
        mesh_intersection_params.pair_target_object_ids &&
        mesh_intersection_params.pair_target_hit_counts &&
        mesh_intersection_params.max_pair_targets_per_source > 0) {
        const int maxPairTargets = mesh_intersection_params.max_pair_targets_per_source;
        const int outBase = sourceObjectId * maxPairTargets;
        const int numTracked = (numTargets < maxPairTargets) ? numTargets : maxPairTargets;
        for (int i = 0; i < numTracked; ++i) {
            mesh_intersection_params.pair_target_object_ids[outBase + i] = targetObjects[i];
            mesh_intersection_params.pair_target_hit_counts[outBase + i] = targetHitCounts[i];
        }
    }

    const int boundedTargets = (numTargets > 256) ? 256 : numTargets;
    if (boundedTargets <= 0) {
        if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
            atomicAdd(&mesh_intersection_params.profiling_stats->containment_rays_total, 1ULL);
            atomicAdd(&mesh_intersection_params.profiling_stats->containment_iterations_total, 1ULL);
            update_max_u32_anyhit(&mesh_intersection_params.profiling_stats->containment_max_iterations_per_ray, 1U);
        }
        return;
    }

    if (mesh_intersection_params.use_hash_table) {
        for (int i = 0; i < boundedTargets; ++i) {
            if (mesh_intersection_params.swap_result_ids == 0) {
                insert_hash_table_anyhit(sourceObjectId, targetObjects[i]);
            } else {
                insert_hash_table_anyhit(targetObjects[i], sourceObjectId);
            }
        }
    } else if (mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[sourceObjectId] = boundedTargets;
    } else if (mesh_intersection_params.pass == 2) {
        const long long outIdx = mesh_intersection_params.collision_offsets[sourceObjectId];
        for (int i = 0; i < boundedTargets; ++i) {
            if (mesh_intersection_params.swap_result_ids == 0) {
                mesh_intersection_params.results[outIdx + i] = {sourceObjectId, targetObjects[i]};
            } else {
                mesh_intersection_params.results[outIdx + i] = {targetObjects[i], sourceObjectId};
            }
        }
    }

    if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_rays_total, 1ULL);
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_iterations_total, 1ULL);
        update_max_u32_anyhit(&mesh_intersection_params.profiling_stats->containment_max_iterations_per_ray, 1U);
        atomicAdd(&mesh_intersection_params.profiling_stats->containment_targets_total, static_cast<unsigned long long>(boundedTargets));
    }
}

extern "C" __global__ void __miss__ms_anyhit() {
}

extern "C" __global__ void __closesthit__ch_anyhit() {
}

extern "C" __global__ void __anyhit__ah_containment() {
    const unsigned int sourceObjectId = optixGetPayload_0();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();

    const int targetObjectId = mesh_intersection_params.mesh2_triangle_to_object[triangleIndex];
    const int maxTargets = mesh_intersection_params.anyhit_max_pair_targets_per_source;
    const int base = static_cast<int>(sourceObjectId) * maxTargets;

    unsigned int count = mesh_intersection_params.anyhit_candidate_count_per_source[sourceObjectId];
    bool updatedExisting = false;

    for (unsigned int i = 0; i < count; ++i) {
        const int idx = base + static_cast<int>(i);
        if (mesh_intersection_params.anyhit_candidate_object_ids[idx] == targetObjectId) {
            mesh_intersection_params.anyhit_candidate_hit_counts[idx] += 1U;
            mesh_intersection_params.anyhit_candidate_parity[idx] ^= 1U;
            updatedExisting = true;
            if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_hits_total, 1ULL);
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_toggles, 1ULL);
            }
            break;
        }
    }

    if (!updatedExisting) {
        if (count < static_cast<unsigned int>(maxTargets)) {
            const int idx = base + static_cast<int>(count);
            mesh_intersection_params.anyhit_candidate_object_ids[idx] = targetObjectId;
            mesh_intersection_params.anyhit_candidate_hit_counts[idx] = 1U;
            mesh_intersection_params.anyhit_candidate_parity[idx] = 1U;
            mesh_intersection_params.anyhit_candidate_count_per_source[sourceObjectId] = count + 1U;
            if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_hits_total, 1ULL);
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_additions, 1ULL);
            }
        } else {
            mesh_intersection_params.anyhit_candidate_overflow_per_source[sourceObjectId] += 1U;
            if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_hits_total, 1ULL);
                atomicAdd(&mesh_intersection_params.profiling_stats->containment_candidate_overflow, 1ULL);
            }
        }
    }

    optixIgnoreIntersection();
}
