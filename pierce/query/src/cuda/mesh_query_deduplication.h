#pragma once

#include "common.h"
#include <cuda_runtime.h>

extern "C" {
long long merge_and_deduplicate_pairs_gpu(
    const MeshQueryResult* d_results1, long long num_results1,
    const MeshQueryResult* d_results2, long long num_results2,
    MeshQueryResult* d_merged_output  // Should be allocated with size >= (num_results1 + num_results2)
);

int compact_hash_table_pairs(
    const unsigned long long* d_hash_table, unsigned long long table_size,
    MeshQueryResult* d_output, int max_output_size
);

} // extern "C"
