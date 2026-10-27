#include "OverlapQuery.h"
#include "../cuda/mesh_query_deduplication.h"
#include "../cuda/scan_utils.h"
#include <chrono>
#include <iostream>
#include <memory>
namespace {
struct DeviceRelease {
    template<class T> void operator()(T* p) const { if (p) CUDA_CHECK(cudaFree(p)); }
};
}

// Execute the overlap query using precomputed unique edges in both directions.
QueryResults executeTwoPassQueryEdgesOptimized(
    MeshOverlapEdgesLauncher& edgesLauncher,
    MeshOverlapEdgesLaunchParams& edgesParams1,
    MeshOverlapEdgesLaunchParams& edgesParams2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    PerformanceTimer* timer,
    bool verbose
) {
    // PASS 1: Count collisions for edges
    int* d_edge_collision_counts_int = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_collision_counts_int, (size_t)mesh1NumEdges * sizeof(int)));
    std::unique_ptr<int, DeviceRelease> own_d_edge_collision_counts_int(d_edge_collision_counts_int);
    
    edgesParams1.collision_counts = d_edge_collision_counts_int;
    edgesParams1.pass = 1;
    auto t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh1ToMesh2(edgesParams1, mesh1NumEdges);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
            timer->addMeasurement(
            "Raytrace_Mesh1ToMesh2_Pass1",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }
    
    // Scan edge counts
    long long* d_edge_collision_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_collision_offsets, (size_t)mesh1NumEdges * sizeof(long long)));
    std::unique_ptr<long long, DeviceRelease> own_d_edge_collision_offsets(d_edge_collision_offsets);
    long long total_edge_results = exclusive_scan_gpu(d_edge_collision_counts_int, d_edge_collision_offsets, mesh1NumEdges);
    
    // PASS 1b: Count collisions for Mesh2->Mesh1 (edge-based)
    int* d_collision_counts2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_counts2, (size_t)mesh2NumEdges * sizeof(int)));
    std::unique_ptr<int, DeviceRelease> own_d_collision_counts2(d_collision_counts2);
    
    edgesParams2.collision_counts = d_collision_counts2;
    edgesParams2.pass = 1;
    t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh2ToMesh1(edgesParams2, mesh2NumEdges);
    t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "Raytrace_Mesh2ToMesh1_Pass1",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }
    
    // Scan Mesh2 counts
    long long* d_collision_offsets2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_offsets2, (size_t)mesh2NumEdges * sizeof(long long)));
    std::unique_ptr<long long, DeviceRelease> own_d_collision_offsets2(d_collision_offsets2);
    long long total_mesh2_results = exclusive_scan_gpu(d_collision_counts2, d_collision_offsets2, mesh2NumEdges);
    
    own_d_collision_counts2.reset();
    
    if (verbose) {
        std::cout << "Pass 1 (Optimized): Found " << total_edge_results << " (edges) + " 
                  << total_mesh2_results << " (edges mesh2->mesh1) = " 
                  << (total_edge_results + total_mesh2_results) << " potential overlaps." << std::endl;
    }
    
    // PASS 2: Store results into merged buffer
    long long total_all = total_edge_results + total_mesh2_results;
    MeshQueryResult* d_merged_results = nullptr;
    std::unique_ptr<MeshQueryResult, DeviceRelease> own_results;
    if (total_all > 0) {
        CUDA_CHECK(cudaMalloc(&d_merged_results, (size_t)total_all * sizeof(MeshQueryResult)));
        own_results.reset(d_merged_results);
    }
    
    edgesParams1.collision_offsets = d_edge_collision_offsets;
    edgesParams1.results = d_merged_results;
    edgesParams1.pass = 2;
    t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh1ToMesh2(edgesParams1, mesh1NumEdges);
    t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "Raytrace_Mesh1ToMesh2_Pass2",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }
    
    edgesParams2.collision_offsets = d_collision_offsets2;
    edgesParams2.results = (d_merged_results ? d_merged_results + total_edge_results : nullptr);
    edgesParams2.pass = 2;
    t0 = std::chrono::high_resolution_clock::now();
    edgesLauncher.launchMesh2ToMesh1(edgesParams2, mesh2NumEdges);
    t1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "Raytrace_Mesh2ToMesh1_Pass2",
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );
    }
    
    own_d_edge_collision_counts_int.reset();
    own_d_edge_collision_offsets.reset();
    own_d_collision_offsets2.reset();
    
    // Deduplicate
    auto t_dedup_0 = std::chrono::high_resolution_clock::now();
    long long numUnique = merge_and_deduplicate_pairs_gpu(nullptr, total_all, nullptr, 0, d_merged_results);
    auto t_dedup_1 = std::chrono::high_resolution_clock::now();
    if (timer) {
        timer->addMeasurement(
            "gpu deduplication",
            std::chrono::duration_cast<std::chrono::microseconds>(t_dedup_1 - t_dedup_0).count()
        );
    }
    
    if (verbose) {
        std::cout << "Deduplication: Found " << numUnique << " unique object pairs." << std::endl;
    }
    
    unsigned long long resultBufferCapacity = total_edge_results + total_mesh2_results;
    // We allocate total_all for d_merged_results.
    // Total allocated bytes for result buffers in this pass:
    unsigned long long resultBufferAllocatedBytes = (total_edge_results + total_mesh2_results) * sizeof(MeshQueryResult);
    
    return {own_results.release(), numUnique, resultBufferCapacity, resultBufferAllocatedBytes};
}

