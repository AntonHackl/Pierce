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
};

struct SlabGeometryPair {
    SlabGeometry mesh1;
    SlabGeometry mesh2;
    int slabIndex = 0;
    int hashTableSize = 0;
};

SlabPartitionPlan planSharedXAxisSlabs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    int numSlabs
);

std::vector<SlabGeometryPair> buildSlabGeometryPairs(
    const GeometryData& mesh1,
    const GeometryData& mesh2,
    const SlabPartitionPlan& plan,
    int globalHashTableSize
);
