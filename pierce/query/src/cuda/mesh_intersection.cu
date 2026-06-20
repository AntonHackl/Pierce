#include "mesh_intersection.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>
#include "optix_common_shaders.cuh"

extern "C" __constant__ MeshIntersectionLaunchParams mesh_intersection_params;

__device__ void insert_hash_table(int id1, int id2);

__device__ __forceinline__ void update_max_u32(unsigned int* addr, unsigned int value) {
    atomicMax(addr, value);
}

static __forceinline__ __device__ int trace_edge_multi_hits(
    const float3& edgeStart,
    const float3& dirNormalized,
    float edgeLength,
    int objectIdSource,
    bool swapPairOrder,
    long long& writeCursor
) {
    int kMaxIterations = mesh_intersection_params.overlap_max_iterations;
    if (kMaxIterations <= 0) {
        kMaxIterations = 100;
    }
    const float ray_tmax = nextafterf(edgeLength, INFINITY);
    float current_t_min = nextafterf(0.0f, ray_tmax);
    int hitsFound = 0;
    int iterations = 0;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        iterations++;
        if (current_t_min > ray_tmax) break;

        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int triangleIndex = 0;

        optixTrace(
            mesh_intersection_params.mesh2_handle,
            edgeStart,
            dirNormalized,
            current_t_min,
            ray_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            hitFlag, distance, triangleIndex);

        if (!hitFlag) break;

        const float t = __uint_as_float(distance);
        if (t > ray_tmax) break;

        const int objectIdTarget = mesh_intersection_params.mesh2_triangle_to_object[triangleIndex];
        hitsFound++;

        if (mesh_intersection_params.use_hash_table) {
            if (swapPairOrder) {
                insert_hash_table(objectIdTarget, objectIdSource);
            } else {
                insert_hash_table(objectIdSource, objectIdTarget);
            }
        } else if (mesh_intersection_params.pass == 2) {
            const long long outIdx = writeCursor++;
            if (swapPairOrder) {
                mesh_intersection_params.results[outIdx] = {objectIdTarget, objectIdSource};
            } else {
                mesh_intersection_params.results[outIdx] = {objectIdSource, objectIdTarget};
            }
        }

        float next_t_min = nextafterf(t, ray_tmax);
        if (next_t_min <= current_t_min) {
            next_t_min = nextafterf(current_t_min, ray_tmax);
        }
        if (next_t_min <= current_t_min) {
            break;
        }
        current_t_min = next_t_min;
    }

    if (mesh_intersection_params.profiling_enabled && mesh_intersection_params.profiling_stats) {
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_trace_calls, 1ULL);
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_iterations_total, static_cast<unsigned long long>(iterations));
        atomicAdd(&mesh_intersection_params.profiling_stats->overlap_hits_total, static_cast<unsigned long long>(hitsFound));
        update_max_u32(&mesh_intersection_params.profiling_stats->overlap_max_iterations_per_trace, static_cast<unsigned int>(iterations));
    }

    return hitsFound;
}

__device__ float distance3f(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float3 normalize3f(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ void insert_hash_table(int id1, int id2) {
    unsigned long long key = (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);
    
    unsigned long long k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    
    int size = mesh_intersection_params.hash_table_size;
    if (size <= 0) return;
    unsigned int h = k % size;
    
    for (int i = 0; i < 1000; ++i) {
        unsigned long long old = atomicCAS(&mesh_intersection_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        
        // Success if slot was empty or already contained our key (deduplication!)
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }
        
        h = (h + 1) % size;
    }
}

extern "C" __global__ void __raygen__mesh_overlap() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int edgeIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (edgeIdx >= mesh_intersection_params.num_edges) {
        return;
    }

    const float3 edgeStart = mesh_intersection_params.edge_starts[edgeIdx];
    const float3 edgeEnd = mesh_intersection_params.edge_ends[edgeIdx];
    const int sourceObjectId = mesh_intersection_params.edge_source_object_ids[edgeIdx];
    
    int totalHits = 0;
    long long writeCursor = 0;
    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 2) {
        writeCursor = mesh_intersection_params.collision_offsets[edgeIdx];
    }

    float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x,
                                 edgeEnd.y - edgeStart.y,
                                 edgeEnd.z - edgeStart.z);
    float edgeLength = distance3f(edgeStart, edgeEnd);

    if (edgeLength > 0.0f) {
        float3 normalizedDir = normalize3f(edgeDir);
        int hitsFound = trace_edge_multi_hits(
            edgeStart,
            normalizedDir,
            edgeLength,
            sourceObjectId,
            mesh_intersection_params.swap_result_ids != 0,
            writeCursor);
        totalHits += hitsFound;
    }

    if (!mesh_intersection_params.use_hash_table && mesh_intersection_params.pass == 1) {
        mesh_intersection_params.collision_counts[edgeIdx] = totalHits;
    }
}
