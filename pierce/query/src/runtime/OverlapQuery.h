#pragma once
#include "../raytracing/MeshOverlapEdgesLauncher.h"
#include "../timer.h"

struct QueryResults {
    MeshQueryResult* d_merged_results;
    long long numUnique;
    unsigned long long resultBufferCapacity;
    unsigned long long resultBufferAllocatedBytes;
};

QueryResults executeTwoPassQueryEdgesOptimized(
    MeshOverlapEdgesLauncher&, MeshOverlapEdgesLaunchParams&, MeshOverlapEdgesLaunchParams&,
    int, int, PerformanceTimer* timer = nullptr, bool verbose = true);
