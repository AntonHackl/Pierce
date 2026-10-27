#include "MemoryBudget.h"
#include "../optix/OptixHelpers.h"
namespace pierce {
MemorySnapshot deviceMemory(const std::string&) {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {properties.name, static_cast<Bytes>(total), static_cast<Bytes>(free)};
}
}
