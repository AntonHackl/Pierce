#include "MultiGpuPartitioning.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr double kTriangleWeightBytes = sizeof(uint3) + sizeof(int) + sizeof(float) * 3.0;
constexpr double kEdgeWeightBytes = sizeof(float3) * 2.0 + sizeof(int);

struct WeightedCenter {
    float center;
    double weight;
};

int findOwnerSlab(const std::vector<float>& boundaries, float center) {
    if (boundaries.size() < 2) {
        return 0;
    }

    auto upper = std::upper_bound(boundaries.begin() + 1, boundaries.end() - 1, center);
    int slab = static_cast<int>(upper - boundaries.begin()) - 1;
    if (slab < 0) {
        slab = 0;
    }
    const int maxSlab = static_cast<int>(boundaries.size()) - 2;
    if (slab > maxSlab) {
        slab = maxSlab;
    }
    return slab;
}

bool intervalOverlapsSlab(float intervalMin, float intervalMax, float slabMin, float slabMax, bool isLastSlab) {
    if (isLastSlab) {
        return intervalMax >= slabMin && intervalMin <= slabMax;
    }
    return intervalMax >= slabMin && intervalMin < slabMax;
}

void appendWeightedCenters(
    const GeometryData& geometry,
    std::vector<WeightedCenter>& weightedCenters
) {
    if (!geometry.partition.hasData()) {
        throw std::runtime_error("Missing partition metadata. Re-run pierce_preprocess.");
    }

    weightedCenters.reserve(
        weightedCenters.size() +
        geometry.partition.triangleSortedByCenter.size() +
        geometry.partition.edgeSortedByCenter.size()
    );

    for (uint32_t triangleIndex : geometry.partition.triangleSortedByCenter) {
        weightedCenters.push_back({geometry.partition.triangles.centers[triangleIndex], kTriangleWeightBytes});
    }
    for (uint32_t edgeIndex : geometry.partition.edgeSortedByCenter) {
        weightedCenters.push_back({geometry.partition.edges.centers[edgeIndex], kEdgeWeightBytes});
    }
}

GeometryData buildSlabGeometry(
    const GeometryData& source,
    const std::vector<int>& slabTriangles,
    const std::vector<int>& slabEdges
) {
    GeometryData slabGeometry;
    slabGeometry.indices.reserve(slabTriangles.size());
    slabGeometry.triangleToObject.reserve(slabTriangles.size());
    slabGeometry.edges.edgeStarts.reserve(slabEdges.size());
    slabGeometry.edges.edgeEnds.reserve(slabEdges.size());
    slabGeometry.edges.sourceObjectIds.reserve(slabEdges.size());

    std::unordered_map<unsigned int, unsigned int> vertexMap;
    vertexMap.reserve(slabTriangles.size() * 3);

    for (int triangleIndex : slabTriangles) {
        const uint3& triangle = source.indices[triangleIndex];
        const unsigned int globalVertices[3] = {triangle.x, triangle.y, triangle.z};
        unsigned int localVertices[3];

        for (int i = 0; i < 3; ++i) {
            const unsigned int globalVertexIndex = globalVertices[i];
            auto it = vertexMap.find(globalVertexIndex);
            if (it == vertexMap.end()) {
                const unsigned int localVertexIndex = static_cast<unsigned int>(slabGeometry.vertices.size());
                slabGeometry.vertices.push_back(source.vertices[globalVertexIndex]);
                vertexMap.emplace(globalVertexIndex, localVertexIndex);
                localVertices[i] = localVertexIndex;
            } else {
                localVertices[i] = it->second;
            }
        }

        slabGeometry.indices.push_back({localVertices[0], localVertices[1], localVertices[2]});
        slabGeometry.triangleToObject.push_back(source.triangleToObject[triangleIndex]);
    }

    for (int edgeIndex : slabEdges) {
        slabGeometry.edges.edgeStarts.push_back(source.edges.edgeStarts[edgeIndex]);
        slabGeometry.edges.edgeEnds.push_back(source.edges.edgeEnds[edgeIndex]);
        slabGeometry.edges.sourceObjectIds.push_back(source.edges.sourceObjectIds[edgeIndex]);
    }

    slabGeometry.totalTriangles = slabGeometry.indices.size();
    return slabGeometry;
}

struct ObjectLaunchPoint {
    bool valid = false;
    float3 point{};
};

std::vector<ObjectLaunchPoint> buildObjectLaunchPoints(const GeometryData& source) {
    std::vector<ObjectLaunchPoint> launchPoints(source.partition.objects.size());
    for (size_t triangleIndex = 0; triangleIndex < source.indices.size(); ++triangleIndex) {
        const int objectId = source.triangleToObject[triangleIndex];
        if (objectId < 0 || static_cast<size_t>(objectId) >= launchPoints.size() || launchPoints[objectId].valid) {
            continue;
        }

        const uint3& triangle = source.indices[triangleIndex];
        if (triangle.x >= source.vertices.size()) {
            continue;
        }
        launchPoints[objectId].valid = true;
        launchPoints[objectId].point = source.vertices[triangle.x];
    }

    return launchPoints;
}

std::vector<int> assignObjectOwnerSlabs(
    const std::vector<ObjectLaunchPoint>& launchPoints,
    const std::vector<float>& boundaries
) {
    std::vector<int> ownerSlabPerObject(launchPoints.size(), -1);
    for (size_t objectId = 0; objectId < ownerSlabPerObject.size(); ++objectId) {
        if (!launchPoints[objectId].valid) {
            continue;
        }
        ownerSlabPerObject[objectId] = findOwnerSlab(boundaries, launchPoints[objectId].point.x);
    }

    return ownerSlabPerObject;
}

SlabGeometry buildSingleSlabGeometry(
    const GeometryData& source,
    const std::vector<float>& boundaries,
    const std::vector<int>& ownerSlabPerObject,
    const std::vector<ObjectLaunchPoint>& launchPoints,
    int slabIndex
) {
    const int numSlabs = static_cast<int>(boundaries.size()) - 1;
    const bool isLastSlab = (slabIndex == numSlabs - 1);
    const float slabMin = boundaries[slabIndex];
    const float slabMax = boundaries[slabIndex + 1];

    std::vector<int> slabTriangles;
    slabTriangles.reserve(source.indices.size() / std::max(1, numSlabs) + 16);
    std::vector<int> slabEdges;
    slabEdges.reserve(source.edges.edgeStarts.size() / std::max(1, numSlabs) + 16);
    std::vector<int> ownedObjects;

    std::unordered_set<int> uniqueTriangles;
    uniqueTriangles.reserve(source.indices.size() / std::max(1, numSlabs) + 16);

    for (size_t objectId = 0; objectId < ownerSlabPerObject.size(); ++objectId) {
        if (ownerSlabPerObject[objectId] == slabIndex && launchPoints[objectId].valid) {
            ownedObjects.push_back(static_cast<int>(objectId));
        }
    }

    for (size_t triangleIndex = 0; triangleIndex < source.indices.size(); ++triangleIndex) {
        if (intervalOverlapsSlab(
                source.partition.triangles.mins[triangleIndex],
                source.partition.triangles.maxs[triangleIndex],
                slabMin,
                slabMax,
                isLastSlab)) {
            if (uniqueTriangles.insert(static_cast<int>(triangleIndex)).second) {
                slabTriangles.push_back(static_cast<int>(triangleIndex));
            }
        }
    }

    std::sort(slabTriangles.begin(), slabTriangles.end());

    for (size_t edgeIndex = 0; edgeIndex < source.edges.edgeStarts.size(); ++edgeIndex) {
        if (intervalOverlapsSlab(
                source.partition.edges.mins[edgeIndex],
                source.partition.edges.maxs[edgeIndex],
                slabMin,
                slabMax,
                isLastSlab)) {
            slabEdges.push_back(static_cast<int>(edgeIndex));
        }
    }

    SlabGeometry slab;
    slab.geometry = buildSlabGeometry(source, slabTriangles, slabEdges);
    slab.localObjectToGlobalObject = ownedObjects;
    slab.firstTriangleIndexPerLocalObject.assign(ownedObjects.size(), -1);
    slab.launchPointPerLocalObject.reserve(ownedObjects.size());

    std::unordered_map<int, int> globalObjectToLocal;
    globalObjectToLocal.reserve(ownedObjects.size());
    for (size_t localObject = 0; localObject < ownedObjects.size(); ++localObject) {
        globalObjectToLocal.emplace(ownedObjects[localObject], static_cast<int>(localObject));
        slab.launchPointPerLocalObject.push_back(launchPoints[ownedObjects[localObject]].point);
    }

    for (size_t localTriangleIndex = 0; localTriangleIndex < slab.geometry.triangleToObject.size(); ++localTriangleIndex) {
        const int globalObjectId = slab.geometry.triangleToObject[localTriangleIndex];
        auto it = globalObjectToLocal.find(globalObjectId);
        if (it != globalObjectToLocal.end() && slab.firstTriangleIndexPerLocalObject[it->second] < 0) {
            slab.firstTriangleIndexPerLocalObject[it->second] = static_cast<int>(localTriangleIndex);
        }
    }

    return slab;
}

int scaledHashTableSize(int globalHashTableSize, double localWeight, double totalWeight) {
    if (globalHashTableSize <= 0) {
        return 1025;
    }
    if (totalWeight <= 0.0) {
        return globalHashTableSize;
    }

    const long long scaledEstimate = std::llround(globalHashTableSize * (localWeight / totalWeight));
    int scaled = static_cast<int>(std::max(1024LL, scaledEstimate));
    if ((scaled % 2) == 0) {
        scaled += 1;
    }
    return scaled;
}

} // namespace

SlabPartitionPlan planSharedXAxisSlabs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    int numSlabs
) {
    if (numSlabs <= 0) {
        throw std::invalid_argument("numSlabs must be positive");
    }
    if (!mesh1.partition.hasData() || !mesh2.partition.hasData()) {
        throw std::runtime_error("Missing partition metadata. Re-run pierce_preprocess.");
    }

    SlabPartitionPlan plan;
    plan.boundaries.resize(static_cast<size_t>(numSlabs) + 1U, 0.0f);

    std::vector<WeightedCenter> weightedCenters;
    appendWeightedCenters(mesh1, weightedCenters);
    appendWeightedCenters(mesh2, weightedCenters);
    if (weightedCenters.empty()) {
        return plan;
    }

    std::sort(weightedCenters.begin(), weightedCenters.end(), [](const WeightedCenter& lhs, const WeightedCenter& rhs) {
        if (lhs.center != rhs.center) {
            return lhs.center < rhs.center;
        }
        return lhs.weight < rhs.weight;
    });

    const float globalMin = std::min(
        *std::min_element(mesh1.partition.triangles.mins.begin(), mesh1.partition.triangles.mins.end()),
        *std::min_element(mesh2.partition.triangles.mins.begin(), mesh2.partition.triangles.mins.end())
    );
    const float globalMax = std::max(
        *std::max_element(mesh1.partition.triangles.maxs.begin(), mesh1.partition.triangles.maxs.end()),
        *std::max_element(mesh2.partition.triangles.maxs.begin(), mesh2.partition.triangles.maxs.end())
    );
    plan.boundaries.front() = globalMin;
    plan.boundaries.back() = globalMax;

    double totalWeight = 0.0;
    for (const WeightedCenter& entry : weightedCenters) {
        totalWeight += entry.weight;
    }

    double cumulativeWeight = 0.0;
    int nextCut = 1;
    for (const WeightedCenter& entry : weightedCenters) {
        cumulativeWeight += entry.weight;
        while (nextCut < numSlabs && cumulativeWeight >= (totalWeight * nextCut) / static_cast<double>(numSlabs)) {
            plan.boundaries[nextCut] = entry.center;
            ++nextCut;
        }
    }

    for (int i = 1; i < numSlabs; ++i) {
        if (plan.boundaries[i] < plan.boundaries[i - 1]) {
            plan.boundaries[i] = plan.boundaries[i - 1];
        }
        if (plan.boundaries[i] > plan.boundaries.back()) {
            plan.boundaries[i] = plan.boundaries.back();
        }
    }

    return plan;
}

std::vector<SlabGeometryPair> buildSlabGeometryPairs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    const SlabPartitionPlan& plan,
    int globalHashTableSize
) {
    const int numSlabs = static_cast<int>(plan.boundaries.size()) - 1;
    if (numSlabs <= 0) {
        return {};
    }

    std::vector<SlabGeometryPair> pairs;
    pairs.reserve(numSlabs);

    const std::vector<ObjectLaunchPoint> mesh1LaunchPoints = buildObjectLaunchPoints(mesh1);
    const std::vector<ObjectLaunchPoint> mesh2LaunchPoints = buildObjectLaunchPoints(mesh2);
    const std::vector<int> mesh1OwnerSlabs = assignObjectOwnerSlabs(mesh1LaunchPoints, plan.boundaries);
    const std::vector<int> mesh2OwnerSlabs = assignObjectOwnerSlabs(mesh2LaunchPoints, plan.boundaries);

    const double totalWeight =
        mesh1.partition.triangles.size() * kTriangleWeightBytes +
        mesh1.partition.edges.size() * kEdgeWeightBytes +
        mesh2.partition.triangles.size() * kTriangleWeightBytes +
        mesh2.partition.edges.size() * kEdgeWeightBytes;

    for (int slabIndex = 0; slabIndex < numSlabs; ++slabIndex) {
        SlabGeometryPair pair;
        pair.slabIndex = slabIndex;
        pair.mesh1 = buildSingleSlabGeometry(mesh1, plan.boundaries, mesh1OwnerSlabs, mesh1LaunchPoints, slabIndex);
        pair.mesh2 = buildSingleSlabGeometry(mesh2, plan.boundaries, mesh2OwnerSlabs, mesh2LaunchPoints, slabIndex);

        const double localWeight =
            pair.mesh1.geometry.indices.size() * kTriangleWeightBytes +
            pair.mesh1.geometry.edges.edgeStarts.size() * kEdgeWeightBytes +
            pair.mesh2.geometry.indices.size() * kTriangleWeightBytes +
            pair.mesh2.geometry.edges.edgeStarts.size() * kEdgeWeightBytes;
        pair.hashTableSize = scaledHashTableSize(globalHashTableSize, localWeight, totalWeight);
        pairs.push_back(std::move(pair));
    }

    return pairs;
}
