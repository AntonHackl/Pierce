#pragma once

#include "Geometry.h"
#include "MemoryBudget.h"

namespace pierce {
long long estimateOverlapPairs(const GridData& a, const GridData& b, float epsilon = 0.001f,
                               float gamma = 0.8f, bool useAlphaCorrection = true,
                               MemoryBudget* budget = nullptr);
unsigned long long estimatedOverlapHashSlots(long long estimatedPairs);
}
