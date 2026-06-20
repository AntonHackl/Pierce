#pragma once

#include <vector>

// PinnedMemory.h includes cuda_runtime.h, which defines float3/uint3
// So we must include it first to avoid redefinition conflicts
#include "PinnedMemory.h"
#include "GridCell.h"

#ifdef INCLUDE_OPTIX
#include <optix.h>
// cuda_runtime.h already included via PinnedMemory.h
#endif

struct SparseGridEntry {
    int3 index;
    GridCell stats;
};

struct GridData {
    float cellSize = 0.0f;
    std::vector<SparseGridEntry> sparseCells;
    bool hasGrid = false;
};

struct EdgeData {
    std::vector<float3, PinnedAllocator<float3>> edgeStarts;
    std::vector<float3, PinnedAllocator<float3>> edgeEnds;
    std::vector<int, PinnedAllocator<int>> sourceObjectIds;

    bool hasEdges() const {
        return !edgeStarts.empty() &&
               edgeStarts.size() == edgeEnds.size() &&
               edgeStarts.size() == sourceObjectIds.size();
    }

    size_t numEdges() const {
        return edgeStarts.size();
    }
};

struct GeometryData {
    std::vector<float3, PinnedAllocator<float3>> vertices;
    std::vector<uint3, PinnedAllocator<uint3>> indices;
    std::vector<int, PinnedAllocator<int>> triangleToObject;
    size_t totalTriangles = 0;

    EdgeData edges;
    
    GridData grid;
};

struct PointData {
    std::vector<float3, PinnedAllocator<float3>> positions;
    size_t numPoints = 0;
};
