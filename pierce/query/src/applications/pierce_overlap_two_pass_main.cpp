// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <map>
#include <iomanip>
#include <chrono>
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "GeometryUploader.h"
#include "Geometry.h"
#include "GeometryIO.h"
#include "../cuda/mesh_query_deduplication.h"
#include "scan_utils.h"
#include "common.h"
#include "../optix/OptixHelpers.h"
#include "../raytracing/MeshOverlapEdgesLauncher.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "../timer.h"
#include "../ptx_utils.h"
#include "app_cli_options.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    long long numUnique;
    unsigned long long resultBufferCapacity;
    unsigned long long resultBufferAllocatedBytes;
};

// Execute the overlap query using precomputed unique edges in both directions.
QueryResults executeTwoPassQueryEdgesOptimized(
    MeshOverlapEdgesLauncher& edgesLauncher,
    MeshOverlapEdgesLaunchParams& edgesParams1,
    MeshOverlapEdgesLaunchParams& edgesParams2,
    int mesh1NumEdges,
    int mesh2NumEdges,
    PerformanceTimer* timer = nullptr,
    bool verbose = true
) {
    // PASS 1: Count collisions for edges
    int* d_edge_collision_counts_int = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edge_collision_counts_int, (size_t)mesh1NumEdges * sizeof(int)));
    
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
    long long total_edge_results = exclusive_scan_gpu(d_edge_collision_counts_int, d_edge_collision_offsets, mesh1NumEdges);
    
    // PASS 1b: Count collisions for Mesh2->Mesh1 (edge-based)
    int* d_collision_counts2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_collision_counts2, (size_t)mesh2NumEdges * sizeof(int)));
    
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
    long long total_mesh2_results = exclusive_scan_gpu(d_collision_counts2, d_collision_offsets2, mesh2NumEdges);
    
    CUDA_CHECK(cudaFree(d_collision_counts2));
    
    if (verbose) {
        std::cout << "Pass 1 (Optimized): Found " << total_edge_results << " (edges) + " 
                  << total_mesh2_results << " (edges mesh2->mesh1) = " 
                  << (total_edge_results + total_mesh2_results) << " potential overlaps." << std::endl;
    }
    
    // PASS 2: Store results into merged buffer
    long long total_all = total_edge_results + total_mesh2_results;
    MeshQueryResult* d_merged_results = nullptr;
    if (total_all > 0) {
        CUDA_CHECK(cudaMalloc(&d_merged_results, (size_t)total_all * sizeof(MeshQueryResult)));
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
    
    CUDA_CHECK(cudaFree(d_edge_collision_counts_int));
    CUDA_CHECK(cudaFree(d_edge_collision_offsets));
    CUDA_CHECK(cudaFree(d_collision_offsets2));
    
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
    
    return {d_merged_results, numUnique, resultBufferCapacity, resultBufferAllocatedBytes};
}

class MeshOverlapCliOptions : public BenchmarkMeshPairCliOptions {
public:
    MeshOverlapCliOptions() : BenchmarkMeshPairCliOptions("mesh_overlap_timing.json") {
        allowNoExportFlag = true;
    }

    void printHelp(const char* exeName) const {
        std::vector<HelpEntry> options;
        appendMeshPairHelp(options);
        appendBenchmarkRunHelp(options);
        appendNoExportHelp(options);
        appendHelpFlag(options);

        printHelpMessage(
            exeName,
            "[--mesh1 <path>] [--mesh2 <path>] [--output <json_output_file>] [--runs <number>] [--ptx <ptx_file>]",
            "Mesh overlap join query for reporting overlapping object pairs between two datasets.",
            options
        );
    }
};


int main(int argc, char* argv[]) {
    PerformanceTimer timer;
    timer.start("Data Reading");
    MeshOverlapCliOptions options;
    options.ptxPath = detectPTXPath("mesh_overlap_edges.ptx");
    options.parse(argc, argv);

    if (options.helpRequested) {
        options.printHelp(argv[0]);
        return 0;
    }

    options.sanitizeRunCounts();

    const std::string& mesh1Path = options.mesh1Path;
    const std::string& mesh2Path = options.mesh2Path;
    const std::string& outputJsonPath = options.outputJsonPath;
    const int numberOfRuns = options.numberOfRuns;
    const bool exportResults = options.exportResults;
    const int warmupRuns = options.warmupRuns;
    
    std::cout << "Mesh-to-Mesh Overlap Join" << std::endl;
    
    if (mesh1Path.empty()) {
        std::cerr << "Error: Mesh1 file path is required. Use --mesh1 <path_to_mesh1_file>" << std::endl;
        return 1;
    }
    
    if (mesh2Path.empty()) {
        std::cerr << "Error: Mesh2 file path is required. Use --mesh2 <path_to_mesh2_file>" << std::endl;
        return 1;
    }
    
    timer.next("Application Creation");
    
    OptixContext context;
    
    timer.next("Load Mesh1");
    std::cout << "Loading Mesh1 from: " << mesh1Path << std::endl;
    GeometryData mesh1Data = loadGeometryFromFile(mesh1Path);
    if (mesh1Data.vertices.empty()) {
        std::cerr << "Error: Failed to load Mesh1 from " << mesh1Path << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh1Data, mesh1Path, "Mesh1")) {
        return 1;
    }
    std::cout << "Mesh1 loaded: " << mesh1Data.vertices.size() << " vertices, " 
              << mesh1Data.indices.size() << " triangles" << std::endl;
    
    timer.next("Load Mesh2");
    std::cout << "Loading Mesh2 from: " << mesh2Path << std::endl;
    GeometryData mesh2Data = loadGeometryFromFile(mesh2Path);
    if (mesh2Data.vertices.empty()) {
        std::cerr << "Error: Failed to load Mesh2 from " << mesh2Path << std::endl;
        return 1;
    }
    if (!requirePrecomputedEdges(mesh2Data, mesh2Path, "Mesh2")) {
        return 1;
    }
    std::cout << "Mesh2 loaded: " << mesh2Data.vertices.size() << " vertices, " 
              << mesh2Data.indices.size() << " triangles" << std::endl;
    
    timer.next("Upload Mesh1");
    GeometryUploader mesh1Uploader;
    mesh1Uploader.upload(mesh1Data);
    std::cout << "Mesh1 uploaded to GPU" << std::endl;
    
    timer.next("Upload Mesh2");
    GeometryUploader mesh2Uploader;
    mesh2Uploader.upload(mesh2Data);
    std::cout << "Mesh2 uploaded to GPU" << std::endl;
    
    timer.next("Build Mesh1 Index");
    OptixAccelerationStructure mesh1AS(context, mesh1Uploader);
    mesh1AS.build();
    std::cout << "Mesh1 acceleration structure built" << std::endl;
    
    timer.next("Build Mesh2 Index");
    OptixAccelerationStructure mesh2AS(context, mesh2Uploader);
    mesh2AS.build();
    std::cout << "Mesh2 acceleration structure built" << std::endl;
    
    timer.next("Upload Mesh1");
    EdgeMeshData mesh1EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh1Data);
    int mesh1NumEdges = mesh1EdgeData.num_edges;
    std::cout << "Mesh1 edges uploaded: " << mesh1NumEdges << " unique edges" << std::endl;

    timer.next("Upload Mesh2");
    EdgeMeshData mesh2EdgeData = PrecomputedEdgeData::uploadFromGeometry(mesh2Data);
    int mesh2NumEdges = mesh2EdgeData.num_edges;
    std::cout << "Mesh2 edges uploaded: " << mesh2NumEdges << " unique edges" << std::endl;
    
    timer.next("Create Launcher");
    MeshOverlapEdgesLauncher edgesLauncher(context, options.ptxPath);
    
    timer.next("Prepare Kernel Parameters");
    
    int mesh1NumTriangles = static_cast<int>(mesh1Uploader.getNumIndices());
    int mesh2NumTriangles = static_cast<int>(mesh2Uploader.getNumIndices());
    
    // Prepare edge parameters for optimized query
    MeshOverlapEdgesLaunchParams edgesParams1;
    edgesParams1.edge_starts = mesh1EdgeData.d_edge_starts;
    edgesParams1.edge_ends = mesh1EdgeData.d_edge_ends;
    edgesParams1.edge_source_object_ids = mesh1EdgeData.d_source_object_ids;
    edgesParams1.num_edges = mesh1NumEdges;
    edgesParams1.mesh2_handle = mesh2AS.getHandle();
    edgesParams1.mesh2_vertices = mesh2Uploader.getVertices();
    edgesParams1.mesh2_indices = mesh2Uploader.getIndices();
    edgesParams1.mesh2_triangle_to_object = mesh2Uploader.getTriangleToObject();
    edgesParams1.hash_table = nullptr;
    edgesParams1.hash_table_size = 0;
    edgesParams1.use_hash_table = 0;
    edgesParams1.use_bitwise_hash = 0;
    edgesParams1.collision_counts = nullptr;
    edgesParams1.collision_offsets = nullptr;
    edgesParams1.results = nullptr;
    edgesParams1.pass = 0;
    edgesParams1.swap_pair_order = 0;

    MeshOverlapEdgesLaunchParams edgesParams2;
    edgesParams2.edge_starts = mesh2EdgeData.d_edge_starts;
    edgesParams2.edge_ends = mesh2EdgeData.d_edge_ends;
    edgesParams2.edge_source_object_ids = mesh2EdgeData.d_source_object_ids;
    edgesParams2.num_edges = mesh2NumEdges;
    edgesParams2.mesh2_handle = mesh1AS.getHandle();
    edgesParams2.mesh2_vertices = mesh1Uploader.getVertices();
    edgesParams2.mesh2_indices = mesh1Uploader.getIndices();
    edgesParams2.mesh2_triangle_to_object = mesh1Uploader.getTriangleToObject();
    edgesParams2.hash_table = nullptr;
    edgesParams2.hash_table_size = 0;
    edgesParams2.use_hash_table = 0;
    edgesParams2.use_bitwise_hash = 0;
    edgesParams2.collision_counts = nullptr;
    edgesParams2.collision_offsets = nullptr;
    edgesParams2.results = nullptr;
    edgesParams2.pass = 0;
    edgesParams2.swap_pair_order = 1;
    
    timer.next("Warmup");
    if (warmupRuns > 0) {
        std::cout << "Running " << warmupRuns << " warmup iterations (edge-optimized)..." << std::endl;
        for (int warmup = 0; warmup < warmupRuns; ++warmup) {
            QueryResults warmupResults = executeTwoPassQueryEdgesOptimized(
                edgesLauncher,
                edgesParams1, edgesParams2,
                mesh1NumEdges, mesh2NumEdges,
                nullptr,
                false
            );
            if (warmupResults.d_merged_results) CUDA_CHECK(cudaFree(warmupResults.d_merged_results));
        }
    }
    
    timer.next("Query");
    
    std::cout << "\n=== Executing mesh overlap detection (edge-optimized) ===" << std::endl;
    QueryResults queryResults = executeTwoPassQueryEdgesOptimized(
        edgesLauncher,
        edgesParams1, edgesParams2,
        mesh1NumEdges, mesh2NumEdges,
        &timer,
        true
    );
    
    MeshQueryResult* d_merged_results = queryResults.d_merged_results;
    long long numUnique = queryResults.numUnique;
    
    timer.next("Download Results");
    
    std::vector<MeshQueryResult> uniqueResults(numUnique);
    if (numUnique > 0) {
        CUDA_CHECK(cudaMemcpy(uniqueResults.data(), d_merged_results,
                              static_cast<size_t>(numUnique) * sizeof(MeshQueryResult),
                              cudaMemcpyDeviceToHost));
    }
    
    timer.next("Output");
    
    std::set<int> mesh1UniqueObjects(mesh1Data.triangleToObject.begin(), mesh1Data.triangleToObject.end());
    int mesh1NumObjects = mesh1UniqueObjects.size();
    std::set<int> mesh2UniqueObjects(mesh2Data.triangleToObject.begin(), mesh2Data.triangleToObject.end());
    int mesh2NumObjects = mesh2UniqueObjects.size();

    std::cout << "\n=== Mesh Overlap Join Summary ===" << std::endl;
    std::cout << "Mesh1 triangles: " << mesh1NumTriangles << std::endl;
    std::cout << "Mesh1 objects: " << mesh1NumObjects << std::endl;
    std::cout << "Mesh2 triangles: " << mesh2NumTriangles << std::endl;
    std::cout << "Mesh2 objects: " << mesh2NumObjects << std::endl;

    std::cout << "Unique object pairs: " << numUnique << std::endl;
    
    std::cout << "Result Buffer Capacity: " << queryResults.resultBufferCapacity << std::endl;
    std::cout << "Result Buffer Allocated Bytes: " << queryResults.resultBufferAllocatedBytes << std::endl;
    std::cout << "Result Buffer Used Bytes: " << (static_cast<unsigned long long>(numUnique) * sizeof(MeshQueryResult)) << std::endl;
    
    if (exportResults) {
        std::cout << "Exporting results to mesh_overlap_results.csv" << std::endl;
        std::ofstream csvFile("mesh_overlap_results.csv");
        csvFile << "object_id_mesh1,object_id_mesh2\n";
        for (const auto& result : uniqueResults) {
            csvFile << result.object_id_mesh1 << "," << result.object_id_mesh2 << "\n";
        }
        csvFile.close();
    }
    
    timer.next("Cleanup");
    
    if (d_merged_results) CUDA_CHECK(cudaFree(d_merged_results));
    PrecomputedEdgeData::freeEdgeData(mesh1EdgeData);
    PrecomputedEdgeData::freeEdgeData(mesh2EdgeData);
    
    timer.finish(outputJsonPath);
    return 0;
}
