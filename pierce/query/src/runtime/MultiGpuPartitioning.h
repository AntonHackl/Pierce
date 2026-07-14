#pragma once

#include "Geometry.h"

#include <vector>

struct SlabPartitionPlan {
    std::vector<float> boundaries;
};

struct SlabGeometry {
    GeometryData geometry;
    std::vector<int> localObjectToGlobalObject;
    std::vector<int> firstTriangleIndexPerLocalObject;
    std::vector<float3> launchPointPerLocalObject;

    size_t numTriangles() const { return geometry.indices.size(); }
    size_t numEdges() const { return geometry.edges.edgeStarts.size(); }
};

struct SlabGeometryPair {
    SlabGeometry mesh1;
    SlabGeometry mesh2;
    int slabIndex = 0;
    int hashTableSize = 0;
};

struct SlabMemoryStats {
    int slabIndex = 0;
    unsigned long long mesh1GeometryBytes = 0;
    unsigned long long mesh2GeometryBytes = 0;
    unsigned long long estimatedGeometryBytes = 0;
    unsigned long long hashTableBytes = 0;
    unsigned long long totalPlannedBytes = 0;
};

struct SlabPlanningTimingStats {
    long long boundaryCdfUs = 0;
    long long boundarySearchUs = 0;
    long long mesh1LaunchPointsUs = 0;
    long long mesh2LaunchPointsUs = 0;
    long long mesh1OwnerAssignUs = 0;
    long long mesh2OwnerAssignUs = 0;
    long long mesh1ActiveSetupUs = 0;
    long long mesh2ActiveSetupUs = 0;
    long long mesh1EndpointSweepUs = 0;
    long long mesh2EndpointSweepUs = 0;
    long long mesh1ActiveCopySortUs = 0;
    long long mesh2ActiveCopySortUs = 0;
    long long mesh1MaterializeUs = 0;
    long long mesh2MaterializeUs = 0;
    long long mesh1ObjectMetadataUs = 0;
    long long mesh2ObjectMetadataUs = 0;
    long long pairAssemblyUs = 0;
};

SlabPartitionPlan planSharedXAxisSlabs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    int numSlabs,
    SlabPlanningTimingStats* timingStats = nullptr
);

std::vector<SlabGeometryPair> buildSlabGeometryPairs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    const SlabPartitionPlan& plan,
    int globalHashTableSize,
    SlabPlanningTimingStats* timingStats = nullptr
);

std::vector<SlabMemoryStats> computeSlabMemoryStats(
    const std::vector<SlabGeometryPair>& slabPairs
);
