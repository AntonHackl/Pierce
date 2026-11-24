#include "QueryRuntime.h"
#include "GeometryIO.h"
#include "OverlapEstimation.h"
#include "../cuda/mesh_query_deduplication.h"
#include <chrono>
#include <numeric>
#include <set>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
long long elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
struct DeviceRelease {
    template<class T> void operator()(T* pointer) const { if (pointer) CUDA_CHECK(cudaFree(pointer)); }
};
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
static DatasetVersion datasetVersion(const std::string& input) {
    auto path = std::filesystem::canonical(input);
    return {path, std::filesystem::last_write_time(path), std::filesystem::file_size(path)};
}

ResidentDataset::ResidentDataset(OptixContext& context, DatasetVersion dataset, MemoryBudget& budget, bool retainHostGeometry)
    : version(std::move(dataset)), index(context, uploader) {
    const auto path = version.path.string();
    auto loaded = loadGeometryFromFile(path);
    if (loaded.vertices.empty() || loaded.indices.empty() ||
        loaded.triangleToObject.size() != loaded.indices.size() ||
        !requirePrecomputedEdges(loaded, path, "Dataset"))
        throw std::runtime_error("Invalid prepared dataset: " + path);
    const auto meshBytes = checkedAdd(checkedMultiply(loaded.vertices.size(), sizeof(float3)),
        checkedAdd(checkedMultiply(loaded.indices.size(), sizeof(uint3)),
                   checkedMultiply(loaded.triangleToObject.size(), sizeof(int))));
    const auto edgeBytes = checkedMultiply(loaded.edges.edgeStarts.size(), 2 * sizeof(float3) + sizeof(int));
    budget.check(checkedAdd(meshBytes, edgeBytes), "dataset_upload");
    uploader.upload(loaded);
    index.build(&budget);
    budget.check(edgeBytes, "edge_upload");
    edges = PrecomputedEdgeData::uploadFromGeometry(loaded);
    if (retainHostGeometry) geometry = std::move(loaded);
    else geometry.grid = std::move(loaded.grid);
}
ResidentDataset::~ResidentDataset() { PrecomputedEdgeData::freeEdgeData(edges); }
size_t ResidentDataset::allocatedBytes() const {
    const auto mesh = checkedAdd(checkedMultiply(uploader.getNumVertices(), sizeof(float3)),
        checkedAdd(checkedMultiply(uploader.getNumIndices(), sizeof(uint3)),
                   checkedMultiply(uploader.getNumTrianglesToObject(), sizeof(int))));
    return checkedAdd(mesh, checkedAdd(index.allocatedBytes(),
        checkedMultiply(edges.num_edges, 2 * sizeof(float3) + sizeof(int))));
}

std::shared_ptr<ResidentDataset> QueryRuntime::loadDataset(DatasetVersion version) {
    for (const auto& entry : pairs_) {
        const auto& pair = *entry.second;
        if (pair.a->version == version) return pair.a;
        if (pair.b->version == version) return pair.b;
    }
    ++reads_;
    auto dataset = std::make_shared<ResidentDataset>(context_, std::move(version), budget_, retainHostGeometry_);
    ++uploads_;
    ++builds_;
    return dataset;
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
    auto versionA = datasetVersion(a);
    auto versionB = datasetVersion(b);
    const bool sameDataset = versionA == versionB;
    pending->a = loadDataset(std::move(versionA));
    pending->b = sameDataset ? pending->a : loadDataset(std::move(versionB));
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
    std::set<const ResidentDataset*> counted;
    for (const auto& entry : pairs_) {
        for (const auto* dataset : {entry.second->a.get(), entry.second->b.get()}) {
            if (counted.insert(dataset).second) bytes = checkedAdd(bytes, dataset->allocatedBytes());
        }
    }
    return bytes;
}
RuntimeResult QueryRuntime::queryPair(const std::string& handle) {
    pair(handle);
    auto& resident = *pairs_.at(handle);
    budget_.begin();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    auto p1 = parameters(*resident.a, *resident.b, 0), p2 = parameters(*resident.b, *resident.a, 1);
    RuntimeResult result;
    const auto estimateStart = Clock::now();
    const auto estimated = estimateOverlapPairs(resident.a->geometry.grid, resident.b->geometry.grid,
                                               0.001f, 0.8f, true, &budget_);
    result.estimatedPairCount = estimated;
    result.phaseTimings.estimationUs = elapsed(estimateStart);

    constexpr Bytes maxSlots = std::numeric_limits<int>::max();
    constexpr Bytes statsBytes = (overlapRawHitShards + 1) * sizeof(unsigned int);
    auto slots = estimated > 0 ? std::min<Bytes>(estimatedOverlapHashSlots(estimated), maxSlots) : 1025;
    p1.use_hash_table = p2.use_hash_table = 1;
    for (;;) {
        result.hashTableBytes = checkedMultiply(slots, sizeof(unsigned long long));
        budget_.check(checkedAdd(result.hashTableBytes, statsBytes), "query_hash_table");
        unsigned long long* table = nullptr;
        CUDA_CHECK(cudaMalloc(&table, result.hashTableBytes));
        std::unique_ptr<unsigned long long, DeviceRelease> ownTable(table);
        CUDA_CHECK(cudaMemset(table, 0xff, result.hashTableBytes));
        unsigned int* stats = nullptr;
        CUDA_CHECK(cudaMalloc(&stats, statsBytes));
        std::unique_ptr<unsigned int, DeviceRelease> ownStats(stats);
        CUDA_CHECK(cudaMemset(stats, 0, statsBytes));
        p1.hash_table = p2.hash_table = table;
        p1.hash_table_size = p2.hash_table_size = slots;
        p1.hash_insert_failed = p2.hash_insert_failed = stats;
        p1.raw_hit_counters = p2.raw_hit_counters = stats + 1;

        const auto traceStart = Clock::now();
        launcher_.launchMesh1ToMesh2(p1, p1.num_edges);
        launcher_.launchMesh2ToMesh1(p2, p2.num_edges);
        unsigned int failed = 0;
        CUDA_CHECK(cudaMemcpy(&failed, stats, sizeof(failed), cudaMemcpyDeviceToHost));
        result.phaseTimings.traceUs += elapsed(traceStart);
        if (failed) {
            if (slots == maxSlots) throw std::runtime_error("Overlap hash table exhausted its supported capacity");
            slots = std::min(maxSlots, 2 * slots + 1);
            continue;
        }

        // One output slot per hash slot guarantees space for every inserted pair.
        result.resultBufferBytes = checkedMultiply(slots, sizeof(MeshQueryResult));
        budget_.check(checkedAdd(result.resultBufferBytes, sizeof(int)), "query_compaction");
        MeshQueryResult* output = nullptr;
        CUDA_CHECK(cudaMalloc(&output, result.resultBufferBytes));
        std::unique_ptr<MeshQueryResult, DeviceRelease> ownOutput(output);
        const auto compactStart = Clock::now();
        const int count = compact_hash_table_pairs(table, slots, output, static_cast<int>(slots));
        result.phaseTimings.compactionUs = elapsed(compactStart);

        const auto downloadStart = Clock::now();
        result.pairs.resize(count);
        if (count) CUDA_CHECK(cudaMemcpy(result.pairs.data(), output,
            result.pairs.size() * sizeof(MeshQueryResult), cudaMemcpyDeviceToHost));
        std::vector<unsigned int> rawHits(overlapRawHitShards);
        CUDA_CHECK(cudaMemcpy(rawHits.data(), stats + 1, statsBytes - sizeof(unsigned int), cudaMemcpyDeviceToHost));
        result.rawHitCount = std::accumulate(rawHits.begin(), rawHits.end(), Bytes{0});
        result.phaseTimings.downloadUs = elapsed(downloadStart);
        break;
    }

    result.benchmarkPhaseSumUs = result.phaseTimings.estimationUs + result.phaseTimings.traceUs
                               + result.phaseTimings.compactionUs + result.phaseTimings.downloadUs;
    CUDA_CHECK(cudaDeviceSynchronize());
    result.queryExecutionUs = elapsed(start);
    ++queries_; ++resident.completedQueries;
    resident.resultCount = result.pairs.size();
    resident.queryExecutionUs = result.queryExecutionUs;
    resident.benchmarkPhaseSumUs = result.benchmarkPhaseSumUs;
    resident.phaseTimings = result.phaseTimings;
    resident.rawHitCount = result.rawHitCount;
    resident.resultBufferBytes = result.resultBufferBytes;
    resident.estimatedPairCount = result.estimatedPairCount;
    resident.hashTableBytes = result.hashTableBytes;
    return result;
}
}
