#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Count how many hits are in the results array (GPU operation)
int count_hits_gpu(const RayResult* d_results, int num_results);

// Compact hits from input to output (GPU operation)
// d_input: device pointer to full results array
// d_output: device pointer to output array (must be pre-allocated, size >= num hits)
// num_results: number of total results
// h_num_hits: output parameter - number of hits written to d_output
void compact_hits_gpu(const RayResult* d_input, RayResult* d_output, 
                      int num_results, int* h_num_hits);

#ifdef __cplusplus
}
#endif

