#include "QueryRuntime.h"
#include "GeometryIO.h"
#include <chrono>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
long long elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
MeshOverlapEdgesLaunchParams parameters(pierce::ResidentDataset& source,
                                       pierce::ResidentDataset& target, int swap) {
    MeshOverlapEdgesLaunchParams p{};
    p.edge_starts = source.edges.d_edge_starts;
    p.edge_ends = source.edges.d_edge_ends;
    p.edge_source_object_ids = source.edges.d_source_object_ids;
    p.num_edges = source.edges.num_edges;
    p.mesh2_handle = target.index.getHandle();
    p.mesh2_vertices = target.uploader.getVertices();
    p.mesh2_indices = target.uploader.getIndices();
    p.mesh2_triangle_to_object = target.uploader.getTriangleToObject();
    p.swap_pair_order = swap;
    return p;
}
}

namespace pierce {
ResidentDataset::ResidentDataset(OptixContext& context, const std::string& path)
    : geometry(loadGeometryFromFile(path)), index(context, uploader) {
    if (geometry.vertices.empty() || geometry.indices.empty() ||
        geometry.triangleToObject.size() != geometry.indices.size() ||
        !requirePrecomputedEdges(geometry, path, "Dataset"))
        throw std::runtime_error("Invalid prepared dataset: " + path);
    uploader.upload(geometry);
    index.build();
    edges = PrecomputedEdgeData::uploadFromGeometry(geometry);
}
ResidentDataset::~ResidentDataset() { PrecomputedEdgeData::freeEdgeData(edges); }
size_t ResidentDataset::allocatedBytes() const {
    return geometry.vertices.size() * sizeof(float3) + geometry.indices.size() * sizeof(uint3) +
        geometry.triangleToObject.size() * sizeof(int) + index.allocatedBytes() +
        geometry.edges.edgeStarts.size() * (2 * sizeof(float3) + sizeof(int));
}
long long QueryRuntime::load(const std::string& a, const std::string& b) {
    if (loaded()) throw std::runtime_error("Unload the current pair before loading another");
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    ++reads_;
    auto first = std::make_unique<ResidentDataset>(context_, a);
    ++uploads_; ++builds_;
    ++reads_;
    auto second = std::make_unique<ResidentDataset>(context_, b);
    ++uploads_; ++builds_;
    CUDA_CHECK(cudaDeviceSynchronize());
    a_ = std::move(first); b_ = std::move(second);
    ++generation_;
    return elapsed(start);
}
void QueryRuntime::unload() {
    CUDA_CHECK(cudaDeviceSynchronize());
    a_.reset(); b_.reset();
}
size_t QueryRuntime::datasetBytes() const {
    return (a_ ? a_->allocatedBytes() : 0) + (b_ ? b_->allocatedBytes() : 0);
}
RuntimeResult QueryRuntime::query(const std::string& a, const std::string& b) {
    if (!loaded() || a != handleA() || b != handleB())
        throw std::runtime_error("Datasets are unloaded or handles are invalid");
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    auto p1 = parameters(*a_, *b_, 0), p2 = parameters(*b_, *a_, 1);
    PerformanceTimer measurements;
    auto raw = executeTwoPassQueryEdgesOptimized(launcher_, p1, p2,
        a_->edges.num_edges, b_->edges.num_edges, &measurements, false);
    // Own the result even if allocating the host vector throws.
    auto release = [](MeshQueryResult* p) { if (p) CUDA_CHECK(cudaFree(p)); };
    std::unique_ptr<MeshQueryResult, decltype(release)> buffer(raw.d_merged_results, release);
    const auto downloadStart = Clock::now();
    RuntimeResult result;
    result.pairs.resize(static_cast<size_t>(raw.numUnique));
    if (raw.numUnique) CUDA_CHECK(cudaMemcpy(result.pairs.data(), buffer.get(),
        result.pairs.size() * sizeof(MeshQueryResult), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    result.benchmarkPhaseSumUs = elapsed(downloadStart);
    for (const char* name : {"Raytrace_Mesh1ToMesh2_Pass1", "Raytrace_Mesh2ToMesh1_Pass1",
                            "Raytrace_Mesh1ToMesh2_Pass2", "Raytrace_Mesh2ToMesh1_Pass2", "gpu deduplication"})
        result.benchmarkPhaseSumUs += measurements.getMeasurementDuration(name);
    buffer.reset();
    CUDA_CHECK(cudaDeviceSynchronize());
    result.queryExecutionUs = elapsed(start);
    ++queries_;
    return result;
}
}
