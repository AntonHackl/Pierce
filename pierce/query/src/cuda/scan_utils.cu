#include "scan_utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>
#include "../optix/OptixHelpers.h"

extern "C" {

long long exclusive_scan_gpu(const int* d_input, long long* d_output, int num_elements) {
    if (num_elements == 0) {
        return 0;
    }
    
    thrust::device_ptr<const int> input_begin(d_input);
    thrust::device_ptr<const int> input_end = input_begin + num_elements;
    thrust::device_ptr<long long> output_begin(d_output);
    
    thrust::exclusive_scan(thrust::device, input_begin, input_end, output_begin, 0LL);

    long long last_offset = 0;
    int last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_output + num_elements - 1, sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_input + num_elements - 1, sizeof(int), cudaMemcpyDeviceToHost));

    return last_offset + static_cast<long long>(last_count);
}

} // extern "C"
