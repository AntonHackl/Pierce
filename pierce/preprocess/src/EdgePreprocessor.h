#pragma once

#include "../../common/include/Geometry.h"

class EdgePreprocessor {
public:
    static EdgeData extractEdges(
        const std::vector<uint3, PinnedAllocator<uint3>>& indices,
        const std::vector<int, PinnedAllocator<int>>& triangleToObject,
        const std::vector<float3, PinnedAllocator<float3>>& vertices);
};
