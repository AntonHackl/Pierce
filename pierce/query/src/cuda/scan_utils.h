#pragma once

#include <cuda_runtime.h>

// Performs exclusive scan (prefix sum) on device array
// Returns total sum
extern "C" {
    long long exclusive_scan_gpu(const int* d_input, long long* d_output, int num_elements);
}
