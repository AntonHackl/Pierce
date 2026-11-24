#include "OverlapEstimation.h"
#include "../cuda/estimated_overlap.h"
#include <cmath>
#include <unordered_map>

namespace pierce {
// Helper to calculate global average size of objects from grid statistics
static float calculateGlobalAvgSize(const std::vector<SparseGridEntry>& sparseCells) {
    double totalSize = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            totalSize += (double)entry.stats.AvgSizeMean * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 0.0f;
    return (float)(totalSize / totalCount);
}

// Helper to calculate global average VolRatio from grid statistics
static float calculateGlobalAvgVolRatio(const std::vector<SparseGridEntry>& sparseCells) {
    double totalRatio = 0.0;
    long long totalCount = 0;
    
    for (const auto& entry : sparseCells) {
        if (entry.stats.TouchCount > 0) {
            totalRatio += (double)entry.stats.VolRatio * (double)entry.stats.TouchCount;
            totalCount += entry.stats.TouchCount;
        }
    }
    
    if (totalCount == 0) return 1.0f;  // Default to 1.0 (no correction)
    return (float)(totalRatio / totalCount);
}

long long estimateOverlapPairs(const GridData& a, const GridData& b, float epsilon,
                               float gamma, bool useAlphaCorrection, MemoryBudget* budget) {
    if (!a.hasGrid || !b.hasGrid) return 0;

    struct Int3Hash {
        size_t operator()(const int3& k) const {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
        }
    };
    struct Int3Equal {
        bool operator()(const int3& a, const int3& b) const {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }
    };

    std::unordered_map<int3, GridCell, Int3Hash, Int3Equal> mapA;
    mapA.reserve(a.sparseCells.size());
    for (const auto& entry : a.sparseCells) mapA[entry.index] = entry.stats;
    std::vector<GridCell> matchedA, matchedB;
    for (const auto& entry : b.sparseCells) {
        const auto it = mapA.find(entry.index);
        if (it == mapA.end()) continue;
        matchedA.push_back(it->second);
        matchedB.push_back(entry.stats);
    }
    if (matchedA.empty()) return 0;
    if (matchedA.size() > std::numeric_limits<int>::max() - 255)
        throw std::overflow_error("Too many matched cells for overlap estimation");
    if (budget) budget->check(checkedMultiply(matchedA.size(), 2 * sizeof(GridCell) + sizeof(float)),
                              "query_estimation");

    const float cellVolume = a.cellSize * a.cellSize * a.cellSize;
    const float estimated = estimateOverlapSelectivity(matchedA.data(), matchedB.data(),
        static_cast<int>(matchedA.size()), cellVolume, epsilon, gamma);
    float alpha = 1.0f;
    if (useAlphaCorrection) {
        const float sizeA = calculateGlobalAvgSize(a.sparseCells) * std::cbrt(calculateGlobalAvgVolRatio(a.sparseCells));
        const float sizeB = calculateGlobalAvgSize(b.sparseCells) * std::cbrt(calculateGlobalAvgVolRatio(b.sparseCells));
        const float combined = sizeA + sizeB;
        alpha = std::max(1.0f, combined * combined * combined / std::max(cellVolume, 1e-9f));
    }
    const double corrected = estimated / alpha;
    if (!std::isfinite(corrected) || corrected < 0 || corrected >= static_cast<double>(std::numeric_limits<long long>::max()))
        throw std::runtime_error("Invalid overlap selectivity estimate");
    return static_cast<long long>(corrected);
}

unsigned long long estimatedOverlapHashSlots(long long estimatedPairs) {
    if (estimatedPairs <= 0) return 16777216;
    const auto slots = static_cast<unsigned long long>(std::clamp(estimatedPairs, 512LL, 1073741824LL)) * 2;
    return slots + 1;
}
}
