#include "mesh_intersection.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"

extern "C" __constant__ MeshIntersectionLaunchParams mesh_intersection_params;

__device__ __forceinline__ void update_max_u32_anyhit(unsigned int* addr, unsigned int value) {
    atomicMax(addr, value);
}

__device__ __forceinline__ unsigned long long mix_containment_fingerprint(
    int targetObjectId,
    unsigned int hitCount,
    unsigned int parity
) {
    unsigned long long value = static_cast<unsigned long long>(static_cast<unsigned int>(targetObjectId));
    value ^= static_cast<unsigned long long>(hitCount) << 32;
    value ^= static_cast<unsigned long long>(parity & 1U) << 63;
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

__device__ __forceinline__ unsigned long long mix_containment_target_id(int targetObjectId) {
    unsigned long long value = static_cast<unsigned long long>(static_cast<unsigned int>(targetObjectId));
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
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

    if (mesh_intersection_params.hash_insert_failure_counter) {
        atomicAdd(mesh_intersection_params.hash_insert_failure_counter, 1ULL);
    }
    if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
        atomicAdd(&mesh_intersection_params.profiling_stats->hash_insert_failures, 1ULL);
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
    const int sourceObjectIndex = static_cast<int>(idx.x);
    if (sourceObjectIndex >= mesh_intersection_params.mesh1_num_objects) {
        return;
    }
    const int sourceObjectId = mesh_intersection_params.source_object_ids_by_launch_index
        ? mesh_intersection_params.source_object_ids_by_launch_index[sourceObjectIndex]
        : sourceObjectIndex;

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[sourceObjectIndex] = 0;
    }

    if (mesh_intersection_params.enable_pair_hit_tracking &&
        mesh_intersection_params.pair_target_object_ids &&
        mesh_intersection_params.pair_target_hit_counts &&
        mesh_intersection_params.max_pair_targets_per_source > 0) {
        const int base = sourceObjectIndex * mesh_intersection_params.max_pair_targets_per_source;
        for (int i = 0; i < mesh_intersection_params.max_pair_targets_per_source; ++i) {
            mesh_intersection_params.pair_target_object_ids[base + i] = -1;
            mesh_intersection_params.pair_target_hit_counts[base + i] = 0U;
        }
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

    reset_anyhit_candidates_for_source(sourceObjectIndex, maxTargets);

    float3 origin;
    if (mesh_intersection_params.launch_points_per_object) {
        origin = mesh_intersection_params.launch_points_per_object[sourceObjectIndex];
    } else {
        const int firstTri = mesh_intersection_params.first_triangle_index_per_object[sourceObjectIndex];
        if (firstTri < 0 || firstTri >= mesh_intersection_params.mesh1_num_triangles) {
            return;
        }
        const uint3 triIndices = mesh_intersection_params.mesh1_indices[firstTri];
        origin = mesh_intersection_params.mesh1_vertices[triIndices.x];
    }

    unsigned int payload0 = static_cast<unsigned int>(sourceObjectIndex);
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

    const int base = sourceObjectIndex * maxTargets;
    const int candidateCount = static_cast<int>(mesh_intersection_params.anyhit_candidate_count_per_source[sourceObjectIndex]);
    const int candidateOverflow = static_cast<int>(mesh_intersection_params.anyhit_candidate_overflow_per_source[sourceObjectIndex]);

    int targetObjects[256];
    unsigned int targetHitCounts[256];
    int numTargets = 0;
    unsigned int oddCandidateCount = 0U;
    unsigned long long hitTotal = 0ULL;
    unsigned long long targetXor = 0ULL;
    unsigned long long targetSum = 0ULL;
    unsigned long long fingerprintXor = 0ULL;
    unsigned long long fingerprintSum = 0ULL;

    const int scanCount = (candidateCount < maxTargets) ? candidateCount : maxTargets;
    for (int i = 0; i < scanCount; ++i) {
        const int targetObjectId = mesh_intersection_params.anyhit_candidate_object_ids[base + i];
        const unsigned int hitCount = mesh_intersection_params.anyhit_candidate_hit_counts[base + i];
        const unsigned int parity = mesh_intersection_params.anyhit_candidate_parity[base + i] & 1U;
        const unsigned long long mixed = mix_containment_fingerprint(targetObjectId, hitCount, parity);
        const unsigned long long mixedTarget = mix_containment_target_id(targetObjectId);
        if (mesh_intersection_params.enable_containment_hit_histogram &&
            mesh_intersection_params.containment_hit_histogram &&
            mesh_intersection_params.containment_hit_histogram_max_bucket >= 0) {
            const unsigned int maxBucket = static_cast<unsigned int>(mesh_intersection_params.containment_hit_histogram_max_bucket);
            const unsigned int bucket = (hitCount > maxBucket) ? maxBucket : hitCount;
            atomicAdd(&mesh_intersection_params.containment_hit_histogram[bucket], 1ULL);
        }
        hitTotal += static_cast<unsigned long long>(hitCount);
        targetXor ^= mixedTarget;
        targetSum += mixedTarget;
        fingerprintXor ^= mixed;
        fingerprintSum += mixed;

        if (parity != 0U) {
            oddCandidateCount++;
            if (numTargets < 256) {
                targetObjects[numTargets] = targetObjectId;
                targetHitCounts[numTargets] = hitCount;
            }
            numTargets++;
        }
    }

    if (mesh_intersection_params.enable_containment_fingerprints &&
        mesh_intersection_params.containment_fingerprint_odd_candidate_count_per_source &&
        mesh_intersection_params.containment_fingerprint_hit_total_per_source &&
        mesh_intersection_params.containment_fingerprint_target_xor_per_source &&
        mesh_intersection_params.containment_fingerprint_target_sum_per_source &&
        mesh_intersection_params.containment_fingerprint_xor_per_source &&
        mesh_intersection_params.containment_fingerprint_sum_per_source) {
        mesh_intersection_params.containment_fingerprint_odd_candidate_count_per_source[sourceObjectIndex] = oddCandidateCount;
        mesh_intersection_params.containment_fingerprint_hit_total_per_source[sourceObjectIndex] = hitTotal;
        mesh_intersection_params.containment_fingerprint_target_xor_per_source[sourceObjectIndex] = targetXor;
        mesh_intersection_params.containment_fingerprint_target_sum_per_source[sourceObjectIndex] = targetSum;
        mesh_intersection_params.containment_fingerprint_xor_per_source[sourceObjectIndex] = fingerprintXor;
        mesh_intersection_params.containment_fingerprint_sum_per_source[sourceObjectIndex] = fingerprintSum;
    }

    if (mesh_intersection_params.enable_containment_tracking &&
        mesh_intersection_params.containment_iterations_per_source &&
        mesh_intersection_params.containment_candidate_count_per_source &&
        mesh_intersection_params.containment_candidate_overflow_per_source) {
        mesh_intersection_params.containment_iterations_per_source[sourceObjectIndex] = 1U;
        mesh_intersection_params.containment_candidate_count_per_source[sourceObjectIndex] = static_cast<unsigned int>(scanCount);
        mesh_intersection_params.containment_candidate_overflow_per_source[sourceObjectIndex] = static_cast<unsigned int>(candidateOverflow);
    }

    if (mesh_intersection_params.enable_pair_hit_tracking &&
        mesh_intersection_params.pair_target_object_ids &&
        mesh_intersection_params.pair_target_hit_counts &&
        mesh_intersection_params.max_pair_targets_per_source > 0) {
        const int maxPairTargets = mesh_intersection_params.max_pair_targets_per_source;
        const int outBase = sourceObjectIndex * maxPairTargets;
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
        mesh_intersection_params.collision_counts[sourceObjectIndex] = boundedTargets;
    } else if (mesh_intersection_params.pass == 2) {
        const long long outIdx = mesh_intersection_params.collision_offsets[sourceObjectIndex];
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
