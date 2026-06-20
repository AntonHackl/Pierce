#pragma once

#include <cstdint>

struct GridCell {
    uint32_t CenterCount;    // Count of objects whose CENTER is in this cell
    uint32_t TouchCount;     // Count of objects TOUCHING this cell
    float AvgSizeMean;       // Mean of (Width + Height + Depth) / 3
    float VolRatio;          // Mean of (MeshVolume / AABBVolume)
};
