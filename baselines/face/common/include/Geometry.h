#pragma once

#include <vector>
#include "vec_types.h"
#include "GridCell.h"

struct GridData {
    float3 minBound;
    float3 maxBound;
    uint3 resolution;
    std::vector<GridCell> cells;
    bool hasGrid = false;
};

struct GeometryData {
    // Standard std::vector for CPU baselines
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<int> triangleToObject;
    size_t totalTriangles = 0;
    
    // Grid statistics for selectivity estimation
    GridData grid;
};

struct PointData {
    std::vector<float3> positions;
    size_t numPoints = 0;
};
