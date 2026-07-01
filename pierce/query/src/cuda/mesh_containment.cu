#include "mesh_containment.h"
#include <optix_device.h>
#include <cuda_runtime.h>
#include "../optix/OptixHelpers.h"
#include <math.h>

extern "C" __constant__ MeshContainmentLaunchParams containment_params;

// -----------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------

__device__ float cont_distance3f(const float3& a, const float3& b) {
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ float3 cont_normalize3f(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) return make_float3(0.0f, 0.0f, 0.0f);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

// -----------------------------------------------------------------------
// Hash table helpers
// -----------------------------------------------------------------------

__device__ unsigned long long cont_hash_key(unsigned long long key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

__device__ unsigned long long cont_pack_pair(int id1, int id2) {
    return (static_cast<unsigned long long>(id1) << 32) |
            static_cast<unsigned long long>(id2);
}

// Insert (id1, id2) into a hash table. id1 = A obj, id2 = B obj.
__device__ void cont_insert_table(unsigned long long* table, int size,
                                  int id1, int id2) {
    unsigned long long key = cont_pack_pair(id1, id2);
    unsigned int h = cont_hash_key(key) % size;
    for (int i = 0; i < 1000; ++i) {
        unsigned long long old = atomicCAS(&table[h], 0xFFFFFFFFFFFFFFFFULL, key);
        if (old == 0xFFFFFFFFFFFFFFFFULL || old == key) return;
        h = (h + 1) % size;
    }
}

// Lookup whether (id1, id2) exists in a hash table.
__device__ bool cont_lookup_table(const unsigned long long* table, int size,
                                  int id1, int id2) {
    unsigned long long key = cont_pack_pair(id1, id2);
    unsigned int h = cont_hash_key(key) % size;
    for (int i = 0; i < 1000; ++i) {
        unsigned long long val = table[h];
        if (val == key) return true;
        if (val == 0xFFFFFFFFFFFFFFFFULL) return false;
        h = (h + 1) % size;
    }
    return false;
}

// -----------------------------------------------------------------------
// Phase 1 – Edge intersection check
// Launched once for B→A (swap_ids=0) and once for A→B (swap_ids=1).
// Each thread handles one source triangle (3 edge rays).
// -----------------------------------------------------------------------

extern "C" __global__ void __raygen__check_edges_no_samehit_suppression() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int edgeIdx = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (edgeIdx >= containment_params.src_num_edges) return;

    const float3 edgeStart = containment_params.src_edge_starts[edgeIdx];
    const float3 edgeEnd = containment_params.src_edge_ends[edgeIdx];
    const int sourceObjectId = containment_params.src_edge_source_object_ids[edgeIdx];

    float3 dir = make_float3(edgeEnd.x - edgeStart.x,
                             edgeEnd.y - edgeStart.y,
                             edgeEnd.z - edgeStart.z);
    float edgeLen = cont_distance3f(edgeStart, edgeEnd);
    if (edgeLen <= 0.0f) return;

    float3 normDir = cont_normalize3f(dir);

    // Traverse all intersections along the edge segment, not only the first hit.
    // Otherwise overlap pairs can be missed and later misclassified as containment.
    const float ray_tmax = nextafterf(edgeLen, INFINITY);
    float current_t_min = nextafterf(0.0f, ray_tmax);
    const int kMaxEdgeTraceIterations = containment_params.overlap_max_iterations;

    for (int iter = 0; iter < kMaxEdgeTraceIterations; ++iter) {
        if (current_t_min > ray_tmax) {
            break;
        }

        unsigned int hitFlag = 0;
        unsigned int distBits = __float_as_uint(ray_tmax);
        unsigned int triangleIndex = 0;

        optixTrace(
            containment_params.target_handle,
            edgeStart,
            normDir,
            current_t_min,
            ray_tmax,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0,
            hitFlag, distBits, triangleIndex);

        if (!hitFlag) {
            break;
        }

        const float t = __uint_as_float(distBits);
        if (t < current_t_min || t > ray_tmax) {
            break;
        }

        const int hit_obj = containment_params.target_triangle_to_object[triangleIndex];

        // Always store as (A_obj, B_obj) regardless of direction.
        if (containment_params.swap_ids == 0) {
            // B->A: src=B, hit=A -> (A=hit, B=src)
            cont_insert_table(containment_params.intersection_hash_table,
                              containment_params.intersection_hash_table_size,
                              hit_obj, sourceObjectId);
        } else {
            // A->B: src=A, hit=B -> (A=src, B=hit)
            cont_insert_table(containment_params.intersection_hash_table,
                              containment_params.intersection_hash_table_size,
                              sourceObjectId, hit_obj);
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
}

// -----------------------------------------------------------------------
// Phase 2 – Point-in-mesh test
// One thread per B object.  Casts a single ray from the first vertex
// of each B object through A's acceleration structure.  Counts hits
// per A-object (parity/odd-even rule).  If odd AND pair is NOT in the
// intersection table → containment.
// -----------------------------------------------------------------------

#define MAX_UNIQUE_A_OBJECTS 512

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();

    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
}

extern "C" __global__ void __anyhit__ah() {
    if (containment_params.trace_phase != 1) {
        return;
    }

    const unsigned int b_obj = optixGetPayload_0();
    if (b_obj >= static_cast<unsigned int>(containment_params.b_num_objects)) {
        optixIgnoreIntersection();
        return;
    }

    const int maxUnique = containment_params.anyhit_max_unique_a_objects;
    if (maxUnique <= 0 ||
        containment_params.anyhit_a_ids == nullptr ||
        containment_params.anyhit_a_parity == nullptr ||
        containment_params.anyhit_num_unique == nullptr) {
        optixIgnoreIntersection();
        return;
    }

    const unsigned int tri = optixGetPrimitiveIndex();
    const int a_obj = containment_params.target_triangle_to_object[tri];

    const int base = static_cast<int>(b_obj) * maxUnique;
    unsigned int count = containment_params.anyhit_num_unique[b_obj];

    for (unsigned int i = 0; i < count; ++i) {
        const int idx = base + static_cast<int>(i);
        if (containment_params.anyhit_a_ids[idx] == a_obj) {
            containment_params.anyhit_a_parity[idx] ^= 1U;
            optixIgnoreIntersection();
            return;
        }
    }

    if (count < static_cast<unsigned int>(maxUnique)) {
        const int idx = base + static_cast<int>(count);
        containment_params.anyhit_a_ids[idx] = a_obj;
        containment_params.anyhit_a_parity[idx] = 1U;
        containment_params.anyhit_num_unique[b_obj] = count + 1U;
    } else if (containment_params.anyhit_overflow_events != nullptr) {
        containment_params.anyhit_overflow_events[b_obj] += 1U;
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__point_in_mesh() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const int b_obj = idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;

    if (b_obj >= containment_params.b_num_objects) return;

    float3 origin = containment_params.b_first_vertices[b_obj];
    float3 dir    = make_float3(0.0f, 0.0f, 1.0f);   // +Z ray

    int a_ids[MAX_UNIQUE_A_OBJECTS];
    int a_par[MAX_UNIQUE_A_OBJECTS];
    int num_unique = 0;

    const int maxUnique = containment_params.anyhit_max_unique_a_objects;
    if (maxUnique <= 0 ||
        containment_params.anyhit_a_ids == nullptr ||
        containment_params.anyhit_a_parity == nullptr ||
        containment_params.anyhit_num_unique == nullptr) {
        return;
    }

    const int base = b_obj * maxUnique;
    containment_params.anyhit_num_unique[b_obj] = 0U;
    if (containment_params.anyhit_overflow_events != nullptr) {
        containment_params.anyhit_overflow_events[b_obj] = 0U;
    }
    for (int i = 0; i < maxUnique; ++i) {
        containment_params.anyhit_a_ids[base + i] = -1;
        containment_params.anyhit_a_parity[base + i] = 0U;
    }

    unsigned int payload0 = static_cast<unsigned int>(b_obj);
    unsigned int payload1 = 0U;
    unsigned int payload2 = 0U;
    optixTrace(
        containment_params.target_handle,
        origin,
        dir,
        nextafterf(0.0f, 1e10f),
        1e10f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        payload0, payload1, payload2);

    num_unique = static_cast<int>(containment_params.anyhit_num_unique[b_obj]);
    if (num_unique > maxUnique) {
        num_unique = maxUnique;
    }
    if (num_unique > MAX_UNIQUE_A_OBJECTS) {
        num_unique = MAX_UNIQUE_A_OBJECTS;
    }
    for (int i = 0; i < num_unique; ++i) {
        a_ids[i] = containment_params.anyhit_a_ids[base + i];
        a_par[i] = static_cast<int>(containment_params.anyhit_a_parity[base + i] & 1U);
    }

    // Emit containment pairs
    for (int i = 0; i < num_unique; ++i) {
        if (a_par[i] == 1) {
            // B vertex is inside A object – check it's not an intersection pair
            if (!cont_lookup_table(containment_params.intersection_hash_table,
                                   containment_params.intersection_hash_table_size,
                                   a_ids[i], b_obj)) {
                cont_insert_table(containment_params.containment_hash_table,
                                  containment_params.containment_hash_table_size,
                                  a_ids[i], b_obj);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Miss / AnyHit / ClosestHit  (shared by all raygen programs)
// -----------------------------------------------------------------------
