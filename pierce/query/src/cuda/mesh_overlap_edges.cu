#include <optix_device.h>
#include <cuda_runtime.h>
#include <math.h>
#include "optix_common_shaders.cuh"

// Forward declaration of result type
struct MeshQueryResult {
    int object_id_mesh1;
    int object_id_mesh2;
};

// Structure for edge-based launch parameters (must match MeshOverlapEdgesLaunchParams)
struct MeshOverlapEdgesLaunchParams {
    // Mesh1 edge data
    float3* edge_starts;
    float3* edge_ends;
    int* edge_source_object_ids;
    int num_edges;
    
    // Mesh2 acceleration structure
    OptixTraversableHandle mesh2_handle;
    float3* mesh2_vertices;
    uint3* mesh2_indices;
    int* mesh2_triangle_to_object;
    
    // Hash Table for on-the-fly deduplication
    unsigned long long* hash_table;
    unsigned long long hash_table_size;
    int use_hash_table;
    int use_bitwise_hash;
    unsigned long long* hash_access_counter;
    unsigned long long* hash_contention_counter;
    int track_hash_contention;
    
    // Two-pass results
    int* collision_counts;            // Per-edge collision counts
    long long* collision_offsets;     // Exclusive scan of counts
    MeshQueryResult* results;         // Actual collision pairs
    int pass;                         // 1 = count only, 2 = write results
    int swap_pair_order;              // 0: (source,target), 1: (target,source)
    int overlap_max_iterations;
};

extern "C" __constant__ MeshOverlapEdgesLaunchParams mesh_overlap_edges_params;

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

__device__ void insert_hash_table_edges(int id1, int id2) {
    unsigned long long key = (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);

    unsigned long long k = key;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;

    unsigned long long size = mesh_overlap_edges_params.hash_table_size;
    if (size <= 0) return;

    unsigned int h;
    if (mesh_overlap_edges_params.use_bitwise_hash) {
        h = (unsigned int)(k & (size - 1));
    } else {
        h = (unsigned int)(k % size);
    }

    for (int i = 0; i < 5000; ++i) {
        if (mesh_overlap_edges_params.track_hash_contention &&
            mesh_overlap_edges_params.hash_access_counter != nullptr) {
            atomicAdd(mesh_overlap_edges_params.hash_access_counter, 1ULL);
        }

        unsigned long long old = atomicCAS(&mesh_overlap_edges_params.hash_table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) {
            return;
        }

        if (mesh_overlap_edges_params.track_hash_contention &&
            mesh_overlap_edges_params.hash_contention_counter != nullptr) {
            atomicAdd(mesh_overlap_edges_params.hash_contention_counter, 1ULL);
        }

        if (mesh_overlap_edges_params.use_bitwise_hash) {
            h = (h + 1) & (unsigned int)(size - 1);
        } else {
            h = (h + 1) % (unsigned int)size;
        }
    }
}

static __forceinline__ __device__ int trace_edge_multi_hits_edges(
    const float3& edgeStart,
    const float3& edgeEnd,
    int sourceObjectId,
    long long& writeCursor
) {
    int kMaxIterations = mesh_overlap_edges_params.overlap_max_iterations;
    if (kMaxIterations <= 0) kMaxIterations = 100;
    
    float3 edgeDir = make_float3(edgeEnd.x - edgeStart.x,
                                 edgeEnd.y - edgeStart.y,
                                 edgeEnd.z - edgeStart.z);
    float edgeLength = distance3f(edgeStart, edgeEnd);
    
    if (edgeLength <= 0.0f) return 0;
    
    float3 dirNormalized = normalize3f(edgeDir);
    
    const float ray_tmax = nextafterf(edgeLength, INFINITY);
    float current_t_min = nextafterf(0.0f, ray_tmax);
    int hitsFound = 0;
    
    for (int iter = 0; iter < kMaxIterations; ++iter) {
        if (current_t_min > ray_tmax) break;
        
        unsigned int hitFlag = 0;
        unsigned int distance = 0;
        unsigned int primitiveIndex = 0;

        optixTrace(
            mesh_overlap_edges_params.mesh2_handle,
            edgeStart,
            dirNormalized,
            current_t_min,
            ray_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0,
            hitFlag, distance, primitiveIndex);

        if (!hitFlag) break;
        
        const float t = __uint_as_float(distance);
        if (t > ray_tmax) break;

        const int objectIdTarget = mesh_overlap_edges_params.mesh2_triangle_to_object[primitiveIndex];
        hitsFound++;
        
        int outMesh1 = sourceObjectId;
        int outMesh2 = objectIdTarget;
        if (mesh_overlap_edges_params.swap_pair_order) {
            outMesh1 = objectIdTarget;
            outMesh2 = sourceObjectId;
        }

        if (mesh_overlap_edges_params.use_hash_table) {
            insert_hash_table_edges(outMesh1, outMesh2);
        } else if (mesh_overlap_edges_params.pass == 2) {
            const long long outIdx = writeCursor++;
            mesh_overlap_edges_params.results[outIdx] = {outMesh1, outMesh2};
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

    return hitsFound;
}

extern "C" __global__ void __raygen__mesh_overlap_edges() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int edgeIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
    
    if (edgeIdx >= mesh_overlap_edges_params.num_edges) {
        return;
    }
    
    float3 edgeStart = mesh_overlap_edges_params.edge_starts[edgeIdx];
    float3 edgeEnd = mesh_overlap_edges_params.edge_ends[edgeIdx];
    int sourceObjectId = mesh_overlap_edges_params.edge_source_object_ids[edgeIdx];
    
    long long writeCursor = 0;
    if (mesh_overlap_edges_params.pass == 2) {
        writeCursor = mesh_overlap_edges_params.collision_offsets[edgeIdx];
    }
    
    int totalHits = trace_edge_multi_hits_edges(
        edgeStart,
        edgeEnd,
        sourceObjectId,
        writeCursor);
    
    if (!mesh_overlap_edges_params.use_hash_table && mesh_overlap_edges_params.pass == 1) {
        mesh_overlap_edges_params.collision_counts[edgeIdx] = totalHits;
    }
}
