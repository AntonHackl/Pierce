#include <cuda_runtime.h>
#include "common.h"

// Kernel to count hits (hit = polygon_index != -1)
__global__ void count_hits_kernel(const RayResult* results, int num_results, int* hit_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_results) {
        if (results[idx].polygon_index != -1) {
            atomicAdd(hit_count, 1);
        }
    }
}

__global__ void compact_hits_kernel(const RayResult* input, RayResult* output, 
                                     int num_results, int* output_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_results) {
        if (input[idx].polygon_index != -1) {
            int out_idx = atomicAdd(output_indices, 1);
            output[out_idx] = input[idx];
        }
    }
}

extern "C" {

int count_hits_gpu(const RayResult* d_results, int num_results) {
    int* d_hit_count;
    cudaMalloc(&d_hit_count, sizeof(int));
    cudaMemset(d_hit_count, 0, sizeof(int));
    
    int threads = 256;
    int blocks = (num_results + threads - 1) / threads;
    
    count_hits_kernel<<<blocks, threads>>>(d_results, num_results, d_hit_count);
    
    int h_hit_count = 0;
    cudaMemcpy(&h_hit_count, d_hit_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_hit_count);
    
    return h_hit_count;
}

void compact_hits_gpu(const RayResult* d_input, RayResult* d_output, 
                      int num_results, int* h_num_hits) {
    int* d_output_idx;
    cudaMalloc(&d_output_idx, sizeof(int));
    cudaMemset(d_output_idx, 0, sizeof(int));
    
    int threads = 256;
    int blocks = (num_results + threads - 1) / threads;
    
    compact_hits_kernel<<<blocks, threads>>>(d_input, d_output, num_results, d_output_idx);
    
    cudaMemcpy(h_num_hits, d_output_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output_idx);
    
    cudaDeviceSynchronize();
}

} // extern "C"

