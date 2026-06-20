#pragma once

#include "../optix/OptixContext.h"
#include "../optix/OptixHelpers.h"
#include "../../../common/include/common.h"
#include <optix.h>
#include <string>

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
};

class MeshOverlapEdgesLauncher {
public:
    MeshOverlapEdgesLauncher(OptixContext& context, const std::string& ptxPathHint = "");
    ~MeshOverlapEdgesLauncher();
    
    MeshOverlapEdgesLauncher(const MeshOverlapEdgesLauncher&) = delete;
    MeshOverlapEdgesLauncher& operator=(const MeshOverlapEdgesLauncher&) = delete;
    
    void launchMesh1ToMesh2(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    void launchMesh2ToMesh1(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    bool isValid() const { return pipeline_ != nullptr; }
    
private:
    OptixContext& context_;
    std::string ptxPathHint_;
    OptixModule module_;
    OptixPipeline pipeline_;
    OptixProgramGroup raygenPG_;
    OptixProgramGroup missPG_;
    OptixProgramGroup hitPG_;
    OptixShaderBindingTable sbt_;
    CUdeviceptr d_rg_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_lp_;

    void launchInternal(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};
