#pragma once

// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define OPTIX_CHECK(call) do { \
    OptixResult res = call; \
    if(res != OPTIX_SUCCESS) { \
        std::cerr << "OptiX error " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

inline std::vector<char> readPTX(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open PTX file: " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

