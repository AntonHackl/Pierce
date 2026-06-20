#pragma once
#include <cuda_runtime.h>
#include "../../common/include/GridCell.h"

// Returns the estimated total number of overlapping pairs (boundary overlap only)
float estimateOverlapSelectivity(
    const GridCell* h_gridA,
    const GridCell* h_gridB,
    int numCells,
    float cellVolume,
    float epsilon = 0.001f,
    float gamma = 0.8f
);
