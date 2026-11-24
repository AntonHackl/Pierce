#pragma once

#include "Geometry.h"
#include "MemoryBudget.h"

namespace pierce {
struct EstimationMemory {
    size_t gpuScratchBytes = 0;
    size_t hostMatchedCellBytes = 0;
};

long long estimateOverlapPairs(const GridData& a, const GridData& b, float epsilon = 0.001f,
                               float gamma = 0.8f, bool useAlphaCorrection = true,
                               MemoryBudget* budget = nullptr, EstimationMemory* memory = nullptr);
unsigned long long estimatedOverlapHashSlots(long long estimatedPairs);
}
