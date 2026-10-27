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
ResidentDataset::ResidentDataset(OptixContext& context, const std::string& path, MemoryBudget& budget)
    : geometry(loadGeometryFromFile(path)), index(context, uploader) {
    if (geometry.vertices.empty() || geometry.indices.empty() ||
        geometry.triangleToObject.size() != geometry.indices.size() ||
        !requirePrecomputedEdges(geometry, path, "Dataset"))
        throw std::runtime_error("Invalid prepared dataset: " + path);
    const auto meshBytes = checkedAdd(checkedMultiply(geometry.vertices.size(), sizeof(float3)),
        checkedAdd(checkedMultiply(geometry.indices.size(), sizeof(uint3)),
                   checkedMultiply(geometry.triangleToObject.size(), sizeof(int))));
    const auto edgeBytes = checkedMultiply(geometry.edges.edgeStarts.size(), 2 * sizeof(float3) + sizeof(int));
    budget.check(checkedAdd(meshBytes, edgeBytes), "dataset_upload");
    uploader.upload(geometry);
    index.build(&budget);
    budget.check(edgeBytes, "edge_upload");
    edges = PrecomputedEdgeData::uploadFromGeometry(geometry);
}
ResidentDataset::~ResidentDataset() { PrecomputedEdgeData::freeEdgeData(edges); }
size_t ResidentDataset::allocatedBytes() const {
    return checkedAdd(checkedAdd(checkedMultiply(geometry.vertices.size(), sizeof(float3)),
        checkedMultiply(geometry.indices.size(), sizeof(uint3))),
        checkedAdd(checkedMultiply(geometry.triangleToObject.size(), sizeof(int)),
        checkedAdd(index.allocatedBytes(), checkedMultiply(geometry.edges.edgeStarts.size(), 2 * sizeof(float3) + sizeof(int)))));
}
const ResidentPair& QueryRuntime::loadPair(const std::string& key, const std::string& a, const std::string& b) {
    if (key.empty()) throw std::runtime_error("A nonempty pair key is required");
    for (const auto& entry : pairs_) {
        const auto& existing = *entry.second;
        if (existing.key == key) {
            if (existing.pathA != a || existing.pathB != b) throw std::runtime_error("Unload key before changing its paths");
            return existing;
        }
    }
    budget_.begin();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    auto pending = std::make_unique<ResidentPair>();
    pending->key = key; pending->pathA = a; pending->pathB = b;
    pending->handle = identity_ + "-" + std::to_string(++generation_);
    ++reads_;
    pending->a = std::make_unique<ResidentDataset>(context_, a, budget_);
    ++uploads_; ++builds_;
    ++reads_;
    pending->b = std::make_unique<ResidentDataset>(context_, b, budget_);
    ++uploads_; ++builds_;
    CUDA_CHECK(cudaDeviceSynchronize());
    pending->loadUs = elapsed(start);
    const auto handle = pending->handle;
    pairs_.emplace(handle, std::move(pending));
    return *pairs_.at(handle);
}
const ResidentPair& QueryRuntime::pair(const std::string& handle) const {
    auto found = pairs_.find(handle);
    if (found == pairs_.end()) throw std::runtime_error("Pair is unloaded or handle is invalid");
    return *found->second;
}
void QueryRuntime::unloadPair(const std::string& handle) {
    pair(handle); // Explicitly reject stale handles.
    budget_.begin();
    CUDA_CHECK(cudaDeviceSynchronize());
    pairs_.erase(handle);
}
size_t QueryRuntime::datasetBytes() const {
    size_t bytes = 0;
    for (const auto& entry : pairs_) bytes = checkedAdd(bytes, entry.second->allocatedBytes());
    return bytes;
}
RuntimeResult QueryRuntime::queryPair(const std::string& handle) {
    pair(handle);
    auto& resident = *pairs_.at(handle);
    budget_.begin();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    auto p1 = parameters(*resident.a, *resident.b, 0), p2 = parameters(*resident.b, *resident.a, 1);
    PerformanceTimer measurements;
    auto raw = executeTwoPassQueryEdgesOptimized(launcher_, p1, p2,
        resident.a->edges.num_edges, resident.b->edges.num_edges, &measurements, false, &budget_);
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
    result.rawHitCount = raw.resultBufferCapacity;
    result.resultBufferBytes = raw.resultBufferAllocatedBytes;
    ++queries_; ++resident.completedQueries;
    resident.resultCount = result.pairs.size();
    resident.queryExecutionUs = result.queryExecutionUs;
    resident.benchmarkPhaseSumUs = result.benchmarkPhaseSumUs;
    resident.rawHitCount = result.rawHitCount;
    resident.resultBufferBytes = result.resultBufferBytes;
    return result;
}
}
