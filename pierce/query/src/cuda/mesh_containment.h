#pragma once

#include "common.h"
#include <optix.h>
#include <cuda_runtime.h>

struct MeshContainmentLaunchParams {
    // Source precomputed edges (Phase 1 rays)
    float3* src_edge_starts;
    float3* src_edge_ends;
    int*    src_edge_source_object_ids;
    int     src_num_edges;

    // Target mesh acceleration structure (to trace against)
    OptixTraversableHandle target_handle;
    int*    target_triangle_to_object;

    // Intersection hash table (Phase 1: populated by edge checks)
    unsigned long long* intersection_hash_table;
    int                 intersection_hash_table_size;

    // Direction flag for edge kernel:
    //   0 = B edges vs A AS  → insert(hit_obj, src_obj) i.e. (A_obj, B_obj)
    //   1 = A edges vs B AS  → insert(src_obj, hit_obj) i.e. (A_obj, B_obj)
    int swap_ids;

    // Phase 2: Point-in-mesh data
    float3* b_first_vertices;       // One vertex per B object (pre-computed on host)
    int     b_num_objects;

    // Containment result hash table (Phase 2 output)
    unsigned long long* containment_hash_table;
    int                 containment_hash_table_size;

    int trace_phase; // 0=edge phase, 1=point-in-mesh phase

    // AnyHit point-in-mesh scratch buffers (indexed by B object).
    int anyhit_max_unique_a_objects;
    int* anyhit_a_ids;
    unsigned int* anyhit_a_parity;
    unsigned int* anyhit_num_unique;
    unsigned int* anyhit_overflow_events;

    // Iteration limit for edge overlap scans.
    int overlap_max_iterations;
};
