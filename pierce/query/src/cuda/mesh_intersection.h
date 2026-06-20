#pragma once

#include "common.h"
#include <optix.h>
#include <cuda_runtime.h>

struct MeshIntersectionProfilingStats {
    unsigned long long overlap_trace_calls;
    unsigned long long overlap_iterations_total;
    unsigned long long overlap_hits_total;
    unsigned int overlap_max_iterations_per_trace;

    unsigned long long containment_rays_total;
    unsigned long long containment_iterations_total;
    unsigned long long containment_hits_total;
    unsigned int containment_max_iterations_per_ray;
    unsigned long long containment_same_hit_suppressed;
    unsigned long long containment_candidate_additions;
    unsigned long long containment_candidate_toggles;
    unsigned long long containment_candidate_overflow;
    unsigned long long containment_targets_total;
};

struct MeshIntersectionLaunchParams {
    // Mesh1 data (plain on GPU)
    float3* mesh1_vertices;
    uint3* mesh1_indices;
    int* mesh1_triangle_to_object;
    int mesh1_num_triangles;
    int mesh1_num_objects;

    // Precomputed source edge rays (for overlap phase)
    float3* edge_starts;
    float3* edge_ends;
    int* edge_source_object_ids;
    int num_edges;
    
    // Mesh2 acceleration structure
    OptixTraversableHandle mesh2_handle;
    float3* mesh2_vertices;
    uint3* mesh2_indices;
    int* mesh2_triangle_to_object;
    int mesh2_num_objects;
    
    // Hash Table for on-the-fly deduplication
    unsigned long long* hash_table;
    int hash_table_size;
    int use_hash_table; // Use int instead of bool
    
    // Exactly one launch triangle per source object for containment.
    int* first_triangle_index_per_object;
    
    // Two-pass results (legacy)
    int* collision_counts;
    long long* collision_offsets;
    MeshQueryResult* results;
    int pass;
    int swap_result_ids; // 0 = (src_obj, hit_obj), 1 = (hit_obj, src_obj)

    // Runtime controls and optional profiling.
    int overlap_max_iterations;
    int profiling_enabled;
    MeshIntersectionProfilingStats* profiling_stats;

    // Optional per-source/per-target containment hit tracking.
    int enable_pair_hit_tracking;
    int max_pair_targets_per_source;
    int* pair_target_object_ids;
    unsigned int* pair_target_hit_counts;

    // Optional per-source containment traversal diagnostics.
    int enable_containment_tracking;
    unsigned int* containment_iterations_per_source;
    unsigned int* containment_candidate_count_per_source;
    unsigned int* containment_candidate_overflow_per_source;

    // Optional AnyHit containment scratch buffers.
    int anyhit_max_pair_targets_per_source;
    int* anyhit_candidate_object_ids;
    unsigned int* anyhit_candidate_parity;
    unsigned int* anyhit_candidate_hit_counts;
    unsigned int* anyhit_candidate_count_per_source;
    unsigned int* anyhit_candidate_overflow_per_source;
};
