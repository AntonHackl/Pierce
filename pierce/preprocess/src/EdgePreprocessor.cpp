#include "EdgePreprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>

namespace {

struct QuantizedPoint {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;

    bool operator<(const QuantizedPoint& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct EdgeKey {
    QuantizedPoint a;
    QuantizedPoint b;
    int objectId;

    bool operator<(const EdgeKey& other) const {
        if (a < other.a) return true;
        if (other.a < a) return false;
        if (b < other.b) return true;
        if (other.b < b) return false;
        return objectId < other.objectId;
    }
};

struct EdgeInfo {
    float3 p0;
    float3 p1;
    int sourceObject;
};

} // namespace

EdgeData EdgePreprocessor::extractEdges(
    const std::vector<uint3, PinnedAllocator<uint3>>& indices,
    const std::vector<int, PinnedAllocator<int>>& triangleToObject,
    const std::vector<float3, PinnedAllocator<float3>>& vertices) {
    EdgeData edgeData;
    if (indices.empty() || vertices.empty() || triangleToObject.size() != indices.size()) {
        return edgeData;
    }

    const double quantScale = 1e5;
    auto quantizePoint = [&](const float3& p) -> QuantizedPoint {
        return {
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.x) * quantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.y) * quantScale)),
            static_cast<std::int64_t>(std::llround(static_cast<double>(p.z) * quantScale))
        };
    };

    std::map<EdgeKey, EdgeInfo> uniqueEdgesMap;

    for (size_t triIdx = 0; triIdx < indices.size(); ++triIdx) {
        const uint3 tri = indices[triIdx];
        if (tri.x >= vertices.size() || tri.y >= vertices.size() || tri.z >= vertices.size()) {
            continue;
        }

        const int sourceObject = triangleToObject[triIdx];
        const unsigned int verts[3] = {tri.x, tri.y, tri.z};

        for (int edgeIdx = 0; edgeIdx < 3; ++edgeIdx) {
            const unsigned int v0 = verts[edgeIdx];
            const unsigned int v1 = verts[(edgeIdx + 1) % 3];

            const float3 p0 = vertices[v0];
            const float3 p1 = vertices[v1];
            QuantizedPoint q0 = quantizePoint(p0);
            QuantizedPoint q1 = quantizePoint(p1);

            if (q1 < q0) {
                std::swap(q0, q1);
            }

            const EdgeKey edgeKey{q0, q1, sourceObject};
            auto it = uniqueEdgesMap.find(edgeKey);
            if (it == uniqueEdgesMap.end()) {
                EdgeInfo info;
                info.p0 = p0;
                info.p1 = p1;
                info.sourceObject = sourceObject;
                uniqueEdgesMap.emplace(edgeKey, std::move(info));
                continue;
            }
        }
    }

    edgeData.edgeStarts.reserve(uniqueEdgesMap.size());
    edgeData.edgeEnds.reserve(uniqueEdgesMap.size());
    edgeData.sourceObjectIds.reserve(uniqueEdgesMap.size());

    for (const auto& kv : uniqueEdgesMap) {
        const EdgeInfo& edge = kv.second;
        edgeData.edgeStarts.push_back(edge.p0);
        edgeData.edgeEnds.push_back(edge.p1);
        edgeData.sourceObjectIds.push_back(edge.sourceObject);
    }

    return edgeData;
}
