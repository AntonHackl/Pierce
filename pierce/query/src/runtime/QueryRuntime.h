#pragma once
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "OverlapQuery.h"
#include <memory>
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace pierce {
struct ResidentDataset {
    GeometryData geometry;
    GeometryUploader uploader;
    OptixAccelerationStructure index;
    EdgeMeshData edges;
    ResidentDataset(OptixContext& context, const std::string& path, MemoryBudget& budget);
    ~ResidentDataset();
    size_t allocatedBytes() const;
};
struct RuntimeResult {
    std::vector<MeshQueryResult> pairs;
    long long queryExecutionUs = 0, benchmarkPhaseSumUs = 0;
    Bytes rawHitCount = 0, resultBufferBytes = 0;
};
struct ResidentPair {
    std::string key, handle, pathA, pathB;
    std::unique_ptr<ResidentDataset> a, b;
    long long loadUs = 0;
    Bytes completedQueries = 0, resultCount = 0, rawHitCount = 0, resultBufferBytes = 0;
    long long queryExecutionUs = 0, benchmarkPhaseSumUs = 0;
    size_t allocatedBytes() const { return checkedAdd(a->allocatedBytes(), b->allocatedBytes()); }
};
// All operations run on one owning thread. Declaration order destroys resident
// pairs before the shared pipeline/context. Pair objects stay at stable addresses.
class QueryRuntime {
    OptixContext context_;
    MeshOverlapEdgesLauncher launcher_;
    MemoryBudget budget_;
    std::map<std::string, std::unique_ptr<ResidentPair>> pairs_;
    const std::string identity_ = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    Bytes generation_ = 0, reads_ = 0, uploads_ = 0, builds_ = 0, queries_ = 0;
public:
    explicit QueryRuntime(const std::string& ptx, MemoryProvider provider = {})
        : launcher_(context_, ptx), budget_(std::move(provider)) {}
    const ResidentPair& loadPair(const std::string& key, const std::string& a, const std::string& b);
    void unloadPair(const std::string& handle);
    RuntimeResult queryPair(const std::string& handle);
    const ResidentPair& pair(const std::string& handle) const;
    const auto& pairs() const { return pairs_; }
    size_t datasetBytes() const;
    size_t runtimeBytes() const { return launcher_.allocatedBytes(); }
    MemorySnapshot memory() const { return budget_.snapshot(); }
    const auto& checkpoints() const { return budget_.checkpoints(); }
    Bytes reads() const { return reads_; }
    Bytes uploads() const { return uploads_; }
    Bytes builds() const { return builds_; }
    Bytes queries() const { return queries_; }

};
}
