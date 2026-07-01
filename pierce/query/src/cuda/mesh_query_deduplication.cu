#include "mesh_query_deduplication.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "../optix/OptixHelpers.h"

__device__ __host__ inline unsigned long long pair_to_key(int id1, int id2) {
    return (static_cast<unsigned long long>(id1) << 32) | static_cast<unsigned long long>(id2);
}

__device__ __host__ inline void key_to_pair(unsigned long long key, int& id1, int& id2) {
    id1 = static_cast<int>(key >> 32);
    id2 = static_cast<int>(key & 0xFFFFFFFF);
}

__global__ void pairs_to_keys_kernel(const MeshQueryResult* pairs, unsigned long long* keys, long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        keys[idx] = pair_to_key(pairs[idx].object_id_mesh1, pairs[idx].object_id_mesh2);
    }
}

__global__ void keys_to_pairs_kernel(const unsigned long long* keys, MeshQueryResult* pairs, long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        key_to_pair(keys[idx], pairs[idx].object_id_mesh1, pairs[idx].object_id_mesh2);
    }
}

__global__ void compact_hash_pairs_kernel(
    const unsigned long long* hash_table, 
    unsigned long long capacity, 
    MeshQueryResult* output, 
    int* count_out, 
    int max_output_size,
    int* overflow_out) 
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < capacity) {
        unsigned long long key = hash_table[idx];
        if (key != 0xFFFFFFFFFFFFFFFFULL) {
            int pos = atomicAdd(count_out, 1);
            if (pos < max_output_size) {
                 int id1 = static_cast<int>(key >> 32);
                 int id2 = static_cast<int>(key & 0xFFFFFFFF);
                 output[pos].object_id_mesh1 = id1;
                 output[pos].object_id_mesh2 = id2;
            } else if (overflow_out != nullptr) {
                atomicExch(overflow_out, 1);
            }
        }
    }
}

__global__ void count_hash_pairs_kernel(
    const unsigned long long* hash_table,
    unsigned long long capacity,
    unsigned long long* count_out)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < capacity) {
        if (hash_table[idx] != 0xFFFFFFFFFFFFFFFFULL) {
            atomicAdd(count_out, 1ULL);
        }
    }
}

// Sort+unique in-place with automatic batching for large datasets.
// For normal datasets: single sort+unique (zero overhead).
// For huge datasets: recursively splits in half until each chunk fits in GPU memory.
static long long deduplicate_inplace(unsigned long long* d_keys, long long count) {
    if (count <= 1) return count;
    
    // Check if we have enough free memory for thrust::sort temp buffer
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    size_t data_bytes = (size_t)count * sizeof(unsigned long long);
    // thrust::sort (radix sort) needs roughly 2x the data as temp workspace
    size_t sort_needs = data_bytes * 2 + (512ULL << 20); // 2x data + 512MB headroom
    
    if (sort_needs <= free_mem) {
        // Normal path: single sort+unique
        thrust::device_ptr<unsigned long long> begin(d_keys);
        thrust::device_ptr<unsigned long long> end = begin + count;
        thrust::sort(thrust::device, begin, end);
        thrust::device_ptr<unsigned long long> new_end = thrust::unique(thrust::device, begin, end);
        return new_end - begin;
    }
    
    // Batched path: split in half, deduplicate each, merge
    long long half = count / 2;
    
    long long unique1 = deduplicate_inplace(d_keys, half);
    long long unique2 = deduplicate_inplace(d_keys + half, count - half);
    
    // Compact: move unique2 right after unique1
    if (unique2 > 0 && unique1 < half) {
        CUDA_CHECK(cudaMemcpy(d_keys + unique1, d_keys + half, 
                   (size_t)unique2 * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));
    }
    
    // Final sort+unique on the merged unique sets (much smaller than original)
    return deduplicate_inplace(d_keys, unique1 + unique2);
}

extern "C" {

long long merge_and_deduplicate_pairs_gpu(
    const MeshQueryResult* d_results1, long long num_results1,
    const MeshQueryResult* d_results2, long long num_results2,
    MeshQueryResult* d_merged_output
) {
    long long total_results = num_results1 + num_results2;
    
    if (total_results == 0) {
        return 0;
    }
    
    // Copy result arrays to merged output (if not already in place)
    if (d_results1 != nullptr && num_results1 > 0) {
        CUDA_CHECK(cudaMemcpy(d_merged_output, d_results1, 
                   num_results1 * sizeof(MeshQueryResult), 
                   cudaMemcpyDeviceToDevice));
    }
    if (d_results2 != nullptr && num_results2 > 0) {
        CUDA_CHECK(cudaMemcpy(d_merged_output + num_results1, d_results2, 
                   num_results2 * sizeof(MeshQueryResult), 
                   cudaMemcpyDeviceToDevice));
    }
    
    // Sort and unique in-place (auto-batches if needed)
    unsigned long long* d_merged_keys = reinterpret_cast<unsigned long long*>(d_merged_output);
    return deduplicate_inplace(d_merged_keys, total_results);
}


int compact_hash_table_pairs(
    const unsigned long long* d_hash_table, unsigned long long table_size,
    MeshQueryResult* d_output, int max_output_size,
    bool* overflowed_out
) {
    int* d_count;
    int* d_overflow = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    if (overflowed_out != nullptr) {
        cudaMalloc(&d_overflow, sizeof(int));
        cudaMemset(d_overflow, 0, sizeof(int));
    }
    
    int threads = 256;
    int blocks = (table_size + threads - 1) / threads;
    
    compact_hash_pairs_kernel<<<blocks, threads>>>(
        d_hash_table, table_size, d_output, d_count, max_output_size, d_overflow
    );
    cudaDeviceSynchronize();
    
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (overflowed_out != nullptr) {
        int h_overflow = 0;
        cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
        *overflowed_out = (h_overflow != 0);
        cudaFree(d_overflow);
    }
    cudaFree(d_count);
    
    return (h_count < max_output_size) ? h_count : max_output_size;
}

unsigned long long count_hash_table_pairs(
    const unsigned long long* d_hash_table,
    unsigned long long table_size
) {
    unsigned long long* d_count;
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    int threads = 256;
    int blocks = static_cast<int>((table_size + threads - 1) / threads);

    count_hash_pairs_kernel<<<blocks, threads>>>(d_hash_table, table_size, d_count);
    cudaDeviceSynchronize();

    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return h_count;
}

} // extern "C"
