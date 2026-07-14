#pragma once

#include "common.h"
#include <cuda_runtime.h>
#include <vector>

struct DevicePairBuffer {
    int deviceId = 0;
    MeshQueryResult* d_pairs = nullptr;
    long long count = 0;
};

struct GpuGlobalDedupResult {
    int aggregatorDeviceId = 0;
    MeshQueryResult* d_uniquePairs = nullptr;
    long long inputCount = 0;
    long long uniqueCount = 0;
    long long gatherUs = 0;
    long long dedupUs = 0;
};

GpuGlobalDedupResult gather_and_deduplicate_pairs_gpu(
    const std::vector<DevicePairBuffer>& worker_buffers,
    int aggregator_device_id
);

extern "C" {
long long merge_and_deduplicate_pairs_gpu(
    const MeshQueryResult* d_results1, long long num_results1,
    const MeshQueryResult* d_results2, long long num_results2,
    MeshQueryResult* d_merged_output  // Should be allocated with size >= (num_results1 + num_results2)
);

int compact_hash_table_pairs(
    const unsigned long long* d_hash_table, unsigned long long table_size,
    MeshQueryResult* d_output, int max_output_size,
    bool* overflowed_out = nullptr
);

unsigned long long count_hash_table_pairs(
    const unsigned long long* d_hash_table,
    unsigned long long table_size
);

} // extern "C"
