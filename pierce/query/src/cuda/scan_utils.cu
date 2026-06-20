#include "scan_utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>

extern "C" {

long long exclusive_scan_gpu(const int* d_input, long long* d_output, int num_elements) {
    if (num_elements == 0) {
        return 0;
    }
    
    thrust::device_ptr<const int> input_begin(d_input);
    thrust::device_ptr<const int> input_end = input_begin + num_elements;
    thrust::device_ptr<long long> output_begin(d_output);
    
    // Perform exclusive scan (prefix sum) with long long initial value to prevent overflow
    thrust::exclusive_scan(thrust::device, input_begin, input_end, output_begin, 0LL);
    cudaDeviceSynchronize();
    
    long long total = thrust::reduce(thrust::device, input_begin, input_end, 0LL);
    cudaDeviceSynchronize();
    
    return total;
}

} // extern "C"
