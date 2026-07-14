#include "MultiGpuPartitioning.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr double kTriangleWeightBytes = PARTITION_TRIANGLE_WEIGHT_BYTES;
constexpr double kEdgeWeightBytes = PARTITION_EDGE_WEIGHT_BYTES;

using Clock = std::chrono::high_resolution_clock;

long long elapsedUs(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

struct PartitionWeightCdf {
    const PartitionWeightSummary* summary = nullptr;
    std::vector<double> prefixWeights;
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

    const unsigned int unmappedVertex = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> vertexMap(source.vertices.size(), unmappedVertex);

    for (int triangleIndex : slabTriangles) {
        const uint3& triangle = source.indices[triangleIndex];
        const unsigned int globalVertices[3] = {triangle.x, triangle.y, triangle.z};
        unsigned int localVertices[3];

        for (int i = 0; i < 3; ++i) {
            const unsigned int globalVertexIndex = globalVertices[i];
            unsigned int& mappedVertexIndex = vertexMap[globalVertexIndex];
            if (mappedVertexIndex == unmappedVertex) {
                const unsigned int localVertexIndex = static_cast<unsigned int>(slabGeometry.vertices.size());
                slabGeometry.vertices.push_back(source.vertices[globalVertexIndex]);
                mappedVertexIndex = localVertexIndex;
                localVertices[i] = localVertexIndex;
            } else {
                localVertices[i] = mappedVertexIndex;
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

std::vector<SlabGeometry> buildSlabGeometries(
    const GeometryData& source,
    const std::vector<float>& boundaries,
    const std::vector<int>& ownerSlabPerObject,
    const std::vector<ObjectLaunchPoint>& launchPoints,
    long long* activeSetupUs,
    long long* endpointSweepUs,
    long long* activeCopySortUs,
    long long* materializeUs,
    long long* objectMetadataUs
) {
    const auto activeSetupStart = Clock::now();
    const int numSlabs = static_cast<int>(boundaries.size()) - 1;
    std::vector<SlabGeometry> slabs(static_cast<size_t>(numSlabs));

    std::vector<int> activeTriangles;
    std::vector<int> activeEdges;
    std::vector<int> activeTrianglePositions(source.indices.size(), -1);
    std::vector<int> activeEdgePositions(source.edges.edgeStarts.size(), -1);
    activeTriangles.reserve(source.indices.size() / std::max(1, numSlabs) + 16);
    activeEdges.reserve(source.edges.edgeStarts.size() / std::max(1, numSlabs) + 16);

    auto addActive = [](std::vector<int>& active, std::vector<int>& positions, int index) {
        if (index < 0 || static_cast<size_t>(index) >= positions.size() || positions[static_cast<size_t>(index)] >= 0) {
            return;
        }
        positions[static_cast<size_t>(index)] = static_cast<int>(active.size());
        active.push_back(index);
    };

    auto removeActive = [](std::vector<int>& active, std::vector<int>& positions, int index) {
        if (index < 0 || static_cast<size_t>(index) >= positions.size()) {
            return;
        }
        const int position = positions[static_cast<size_t>(index)];
        if (position < 0) {
            return;
        }
        const int replacement = active.back();
        active[static_cast<size_t>(position)] = replacement;
        positions[static_cast<size_t>(replacement)] = position;
        active.pop_back();
        positions[static_cast<size_t>(index)] = -1;
    };
    if (activeSetupUs) {
        *activeSetupUs += elapsedUs(activeSetupStart);
    }

    size_t nextTriangleMin = 0;
    size_t nextTriangleMax = 0;
    size_t nextEdgeMin = 0;
    size_t nextEdgeMax = 0;

    for (int slabIndex = 0; slabIndex < numSlabs; ++slabIndex) {
        const bool isLastSlab = (slabIndex == numSlabs - 1);
        const float slabMin = boundaries[slabIndex];
        const float slabMax = boundaries[slabIndex + 1];

        const auto sweepStart = Clock::now();
        while (nextTriangleMin < source.partition.triangleSortedByMin.size()) {
            const int triangleIndex = static_cast<int>(source.partition.triangleSortedByMin[nextTriangleMin]);
            const float triMin = source.partition.triangles.mins[triangleIndex];
            if (isLastSlab ? triMin > slabMax : triMin >= slabMax) {
                break;
            }
            addActive(activeTriangles, activeTrianglePositions, triangleIndex);
            ++nextTriangleMin;
        }

        while (nextTriangleMax < source.partition.triangleSortedByMax.size()) {
            const int triangleIndex = static_cast<int>(source.partition.triangleSortedByMax[nextTriangleMax]);
            if (source.partition.triangles.maxs[triangleIndex] >= slabMin) {
                break;
            }
            removeActive(activeTriangles, activeTrianglePositions, triangleIndex);
            ++nextTriangleMax;
        }

        while (nextEdgeMin < source.partition.edgeSortedByMin.size()) {
            const int edgeIndex = static_cast<int>(source.partition.edgeSortedByMin[nextEdgeMin]);
            const float edgeMin = source.partition.edges.mins[edgeIndex];
            if (isLastSlab ? edgeMin > slabMax : edgeMin >= slabMax) {
                break;
            }
            addActive(activeEdges, activeEdgePositions, edgeIndex);
            ++nextEdgeMin;
        }

        while (nextEdgeMax < source.partition.edgeSortedByMax.size()) {
            const int edgeIndex = static_cast<int>(source.partition.edgeSortedByMax[nextEdgeMax]);
            if (source.partition.edges.maxs[edgeIndex] >= slabMin) {
                break;
            }
            removeActive(activeEdges, activeEdgePositions, edgeIndex);
            ++nextEdgeMax;
        }
        if (endpointSweepUs) {
            *endpointSweepUs += elapsedUs(sweepStart);
        }

        SlabGeometry& slab = slabs[static_cast<size_t>(slabIndex)];
        const auto copySortStart = Clock::now();
        std::vector<int> slabTriangles = activeTriangles;
        std::vector<int> slabEdges = activeEdges;
        if (activeCopySortUs) {
            *activeCopySortUs += elapsedUs(copySortStart);
        }

        const auto materializeStart = Clock::now();
        slab.geometry = buildSlabGeometry(source, slabTriangles, slabEdges);
        if (materializeUs) {
            *materializeUs += elapsedUs(materializeStart);
        }

        const auto objectMetadataStart = Clock::now();
        for (size_t objectId = 0; objectId < ownerSlabPerObject.size(); ++objectId) {
            if (ownerSlabPerObject[objectId] == slabIndex && launchPoints[objectId].valid) {
                slab.localObjectToGlobalObject.push_back(static_cast<int>(objectId));
            }
        }
        slab.firstTriangleIndexPerLocalObject.assign(slab.localObjectToGlobalObject.size(), -1);
        slab.launchPointPerLocalObject.reserve(slab.localObjectToGlobalObject.size());

        std::unordered_map<int, int> globalObjectToLocal;
        globalObjectToLocal.reserve(slab.localObjectToGlobalObject.size());
        for (size_t localObject = 0; localObject < slab.localObjectToGlobalObject.size(); ++localObject) {
            const int globalObject = slab.localObjectToGlobalObject[localObject];
            globalObjectToLocal.emplace(globalObject, static_cast<int>(localObject));
            slab.launchPointPerLocalObject.push_back(launchPoints[globalObject].point);
        }

        for (size_t localTriangleIndex = 0; localTriangleIndex < slab.geometry.triangleToObject.size(); ++localTriangleIndex) {
            const int globalObjectId = slab.geometry.triangleToObject[localTriangleIndex];
            auto it = globalObjectToLocal.find(globalObjectId);
            if (it != globalObjectToLocal.end() && slab.firstTriangleIndexPerLocalObject[it->second] < 0) {
                slab.firstTriangleIndexPerLocalObject[it->second] = static_cast<int>(localTriangleIndex);
            }
        }
        if (objectMetadataUs) {
            *objectMetadataUs += elapsedUs(objectMetadataStart);
        }
    }

    return slabs;
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

PartitionWeightCdf buildPartitionWeightCdf(const PartitionWeightSummary& summary) {
    if (!summary.hasData()) {
        throw std::runtime_error("Missing partition weight summary. Re-run pierce_preprocess.");
    }
    PartitionWeightCdf cdf;
    cdf.summary = &summary;
    cdf.prefixWeights.resize(summary.binWeights.size() + 1, 0.0);
    for (size_t i = 0; i < summary.binWeights.size(); ++i) {
        cdf.prefixWeights[i + 1] = cdf.prefixWeights[i] + summary.binWeights[i];
    }
    return cdf;
}

double cumulativeWeightAtX(const PartitionWeightCdf& cdf, float x) {
    const PartitionWeightSummary& summary = *cdf.summary;
    if (x <= summary.centerMinX) {
        return 0.0;
    }
    if (x >= summary.centerMaxX) {
        return summary.totalWeight;
    }

    const size_t binCount = summary.binWeights.size();
    const double normalized = static_cast<double>(x - summary.centerMinX) /
        static_cast<double>(summary.centerMaxX - summary.centerMinX);
    const double scaled = normalized * static_cast<double>(binCount);
    const size_t bin = std::min(binCount - 1, static_cast<size_t>(scaled));
    const double inBinFraction = std::max(0.0, std::min(1.0, scaled - static_cast<double>(bin)));
    return cdf.prefixWeights[bin] + summary.binWeights[bin] * inBinFraction;
}

double combinedCumulativeWeightAtX(
    const PartitionWeightCdf& mesh1Cdf,
    const PartitionWeightCdf& mesh2Cdf,
    float x
) {
    return cumulativeWeightAtX(mesh1Cdf, x) + cumulativeWeightAtX(mesh2Cdf, x);
}

unsigned long long geometryMemoryBytes(const SlabGeometry& slab) {
    const GeometryData& geometry = slab.geometry;
    unsigned long long bytes = 0;
    bytes += static_cast<unsigned long long>(geometry.vertices.size() * sizeof(float3));
    bytes += static_cast<unsigned long long>(geometry.indices.size() * sizeof(uint3));
    bytes += static_cast<unsigned long long>(geometry.triangleToObject.size() * sizeof(int));
    bytes += static_cast<unsigned long long>(geometry.edges.edgeStarts.size() * sizeof(float3));
    bytes += static_cast<unsigned long long>(geometry.edges.edgeEnds.size() * sizeof(float3));
    bytes += static_cast<unsigned long long>(geometry.edges.sourceObjectIds.size() * sizeof(int));
    bytes += static_cast<unsigned long long>(slab.localObjectToGlobalObject.size() * sizeof(int));
    bytes += static_cast<unsigned long long>(slab.firstTriangleIndexPerLocalObject.size() * sizeof(int));
    bytes += static_cast<unsigned long long>(slab.launchPointPerLocalObject.size() * sizeof(float3));
    return bytes;
}

} // namespace

SlabPartitionPlan planSharedXAxisSlabs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    int numSlabs,
    SlabPlanningTimingStats* timingStats
) {
    if (numSlabs <= 0) {
        throw std::invalid_argument("numSlabs must be positive");
    }
    if (!mesh1.partition.hasData() || !mesh2.partition.hasData()) {
        throw std::runtime_error("Missing partition metadata. Re-run pierce_preprocess.");
    }

    SlabPartitionPlan plan;
    plan.boundaries.resize(static_cast<size_t>(numSlabs) + 1U, 0.0f);

    const PartitionWeightSummary& mesh1Summary = mesh1.partition.weightSummary;
    const PartitionWeightSummary& mesh2Summary = mesh2.partition.weightSummary;
    if (!mesh1Summary.hasData() || !mesh2Summary.hasData()) {
        throw std::runtime_error("Missing partition weight summary. Re-run pierce_preprocess.");
    }

    const double totalWeight = mesh1Summary.totalWeight + mesh2Summary.totalWeight;
    if (totalWeight <= 0.0) {
        return plan;
    }

    auto phaseStart = Clock::now();
    const PartitionWeightCdf mesh1Cdf = buildPartitionWeightCdf(mesh1Summary);
    const PartitionWeightCdf mesh2Cdf = buildPartitionWeightCdf(mesh2Summary);
    if (timingStats) {
        timingStats->boundaryCdfUs += elapsedUs(phaseStart);
    }

    const float globalMin = std::min(
        mesh1Summary.triangleMinX,
        mesh2Summary.triangleMinX
    );
    const float globalMax = std::max(
        mesh1Summary.triangleMaxX,
        mesh2Summary.triangleMaxX
    );
    plan.boundaries.front() = globalMin;
    plan.boundaries.back() = globalMax;

    phaseStart = Clock::now();
    constexpr int kBinarySearchIterations = 32;
    for (int cut = 1; cut < numSlabs; ++cut) {
        const double targetWeight = (totalWeight * cut) / static_cast<double>(numSlabs);
        float low = globalMin;
        float high = globalMax;
        for (int iteration = 0; iteration < kBinarySearchIterations; ++iteration) {
            const float mid = (low + high) * 0.5f;
            if (combinedCumulativeWeightAtX(mesh1Cdf, mesh2Cdf, mid) < targetWeight) {
                low = mid;
            } else {
                high = mid;
            }
        }
        plan.boundaries[cut] = high;
    }
    if (timingStats) {
        timingStats->boundarySearchUs += elapsedUs(phaseStart);
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
    int globalHashTableSize,
    SlabPlanningTimingStats* timingStats
) {
    const int numSlabs = static_cast<int>(plan.boundaries.size()) - 1;
    if (numSlabs <= 0) {
        return {};
    }

    std::vector<SlabGeometryPair> pairs;
    pairs.reserve(numSlabs);

    auto phaseStart = Clock::now();
    const std::vector<ObjectLaunchPoint> mesh1LaunchPoints = buildObjectLaunchPoints(mesh1);
    if (timingStats) {
        timingStats->mesh1LaunchPointsUs += elapsedUs(phaseStart);
    }

    phaseStart = Clock::now();
    const std::vector<ObjectLaunchPoint> mesh2LaunchPoints = buildObjectLaunchPoints(mesh2);
    if (timingStats) {
        timingStats->mesh2LaunchPointsUs += elapsedUs(phaseStart);
    }

    phaseStart = Clock::now();
    const std::vector<int> mesh1OwnerSlabs = assignObjectOwnerSlabs(mesh1LaunchPoints, plan.boundaries);
    if (timingStats) {
        timingStats->mesh1OwnerAssignUs += elapsedUs(phaseStart);
    }

    phaseStart = Clock::now();
    const std::vector<int> mesh2OwnerSlabs = assignObjectOwnerSlabs(mesh2LaunchPoints, plan.boundaries);
    if (timingStats) {
        timingStats->mesh2OwnerAssignUs += elapsedUs(phaseStart);
    }

    std::vector<SlabGeometry> mesh1Slabs = buildSlabGeometries(
        mesh1,
        plan.boundaries,
        mesh1OwnerSlabs,
        mesh1LaunchPoints,
        timingStats ? &timingStats->mesh1ActiveSetupUs : nullptr,
        timingStats ? &timingStats->mesh1EndpointSweepUs : nullptr,
        timingStats ? &timingStats->mesh1ActiveCopySortUs : nullptr,
        timingStats ? &timingStats->mesh1MaterializeUs : nullptr,
        timingStats ? &timingStats->mesh1ObjectMetadataUs : nullptr
    );
    std::vector<SlabGeometry> mesh2Slabs = buildSlabGeometries(
        mesh2,
        plan.boundaries,
        mesh2OwnerSlabs,
        mesh2LaunchPoints,
        timingStats ? &timingStats->mesh2ActiveSetupUs : nullptr,
        timingStats ? &timingStats->mesh2EndpointSweepUs : nullptr,
        timingStats ? &timingStats->mesh2ActiveCopySortUs : nullptr,
        timingStats ? &timingStats->mesh2MaterializeUs : nullptr,
        timingStats ? &timingStats->mesh2ObjectMetadataUs : nullptr
    );

    const double totalWeight =
        mesh1.partition.triangles.size() * kTriangleWeightBytes +
        mesh1.partition.edges.size() * kEdgeWeightBytes +
        mesh2.partition.triangles.size() * kTriangleWeightBytes +
        mesh2.partition.edges.size() * kEdgeWeightBytes;

    phaseStart = Clock::now();
    for (int slabIndex = 0; slabIndex < numSlabs; ++slabIndex) {
        SlabGeometryPair pair;
        pair.slabIndex = slabIndex;
        pair.mesh1 = std::move(mesh1Slabs[static_cast<size_t>(slabIndex)]);
        pair.mesh2 = std::move(mesh2Slabs[static_cast<size_t>(slabIndex)]);

        const double localWeight =
            pair.mesh1.geometry.indices.size() * kTriangleWeightBytes +
            pair.mesh1.geometry.edges.edgeStarts.size() * kEdgeWeightBytes +
            pair.mesh2.geometry.indices.size() * kTriangleWeightBytes +
            pair.mesh2.geometry.edges.edgeStarts.size() * kEdgeWeightBytes;
        pair.hashTableSize = scaledHashTableSize(globalHashTableSize, localWeight, totalWeight);
        pairs.push_back(std::move(pair));
    }
    if (timingStats) {
        timingStats->pairAssemblyUs += elapsedUs(phaseStart);
    }

    return pairs;
}

std::vector<SlabMemoryStats> computeSlabMemoryStats(
    const std::vector<SlabGeometryPair>& slabPairs
) {
    std::vector<SlabMemoryStats> stats;
    stats.reserve(slabPairs.size());
    for (const SlabGeometryPair& slabPair : slabPairs) {
        SlabMemoryStats slabStats;
        slabStats.slabIndex = slabPair.slabIndex;
        slabStats.mesh1GeometryBytes = geometryMemoryBytes(slabPair.mesh1);
        slabStats.mesh2GeometryBytes = geometryMemoryBytes(slabPair.mesh2);
        slabStats.estimatedGeometryBytes = slabStats.mesh1GeometryBytes + slabStats.mesh2GeometryBytes;
        slabStats.hashTableBytes = static_cast<unsigned long long>(slabPair.hashTableSize) *
            static_cast<unsigned long long>(sizeof(unsigned long long));
        slabStats.totalPlannedBytes = slabStats.estimatedGeometryBytes + slabStats.hashTableBytes;
        stats.push_back(slabStats);
    }
    return stats;
}
