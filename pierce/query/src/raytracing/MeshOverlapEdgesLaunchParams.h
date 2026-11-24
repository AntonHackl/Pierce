#pragma once

#include "common.h"

constexpr unsigned int overlapRawHitShards = 16384;

// Extended launch parameters for edge-based processing
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
    int swap_pair_order;              // 0: (source,target), 1: (target,source) -> canonical (mesh1,mesh2)
    int overlap_max_iterations;
    unsigned int* raw_hit_counters;
    unsigned int* hash_insert_failed;
};
