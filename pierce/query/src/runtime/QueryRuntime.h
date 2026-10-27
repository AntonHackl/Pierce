#pragma once
#include "../optix/OptixContext.h"
#include "../optix/OptixAccelerationStructure.h"
#include "../geometry/PrecomputedEdgeData.h"
#include "OverlapQuery.h"
#include <memory>
#include <chrono>
#include <string>
#include <vector>

namespace pierce {
struct ResidentDataset {
    GeometryData geometry;
    GeometryUploader uploader;
    OptixAccelerationStructure index;
    EdgeMeshData edges;
    ResidentDataset(OptixContext& context, const std::string& path);
    ~ResidentDataset();
    size_t allocatedBytes() const;
};

struct RuntimeResult {
    std::vector<MeshQueryResult> pairs;
    long long queryExecutionUs = 0;
    long long benchmarkPhaseSumUs = 0;
};

// Non-copyable GPU owners remain at stable addresses. Member declaration order
// guarantees datasets and launcher are destroyed before their context.
class QueryRuntime {
    OptixContext context_;
    MeshOverlapEdgesLauncher launcher_;
    std::unique_ptr<ResidentDataset> a_, b_;
    const std::string identity_ = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    unsigned long long generation_ = 0;
    unsigned long long reads_ = 0, uploads_ = 0, builds_ = 0, queries_ = 0;
public:
    explicit QueryRuntime(const std::string& ptx) : launcher_(context_, ptx) {}
    long long load(const std::string& a, const std::string& b);
    void unload();
    bool loaded() const { return a_ && b_; }
    std::string handleA() const { return loaded() ? identity_ + "-" + std::to_string(generation_) + "-a" : ""; }
    std::string handleB() const { return loaded() ? identity_ + "-" + std::to_string(generation_) + "-b" : ""; }
    RuntimeResult query(const std::string& a, const std::string& b);
    size_t datasetBytes() const;
    size_t runtimeBytes() const { return launcher_.allocatedBytes(); }
    unsigned long long reads() const { return reads_; }
    unsigned long long uploads() const { return uploads_; }
    unsigned long long builds() const { return builds_; }
    unsigned long long queries() const { return queries_; }
};
}
