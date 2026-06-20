#include "estimated_overlap.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cmath>

__global__ void estimateSelectivityKernel(
    const GridCell* __restrict__ gridA,
    const GridCell* __restrict__ gridB,
    float* __restrict__ resultBuffer,
    int numCells,
    float cellVolume,
    float epsilon,
    float gamma
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCells) return;

    GridCell A = gridA[idx];
    GridCell B = gridB[idx];

    if (A.TouchCount == 0 || B.TouchCount == 0) {
        resultBuffer[idx] = 0.0f;
        return;
    }

    // Probabilistic Density Estimation: Touch * Touch * Probability
    float combinedSize = A.AvgSizeMean + B.AvgSizeMean + epsilon;
    float minkowskiVol = combinedSize * combinedSize * combinedSize;
    float prob = minkowskiVol / cellVolume;

    // Use geometric mean for shape correction
    float combinedRatio = sqrtf(A.VolRatio * B.VolRatio);
    float shapeCorrection = powf(combinedRatio, gamma);

    prob *= shapeCorrection;
    prob = fminf(prob, 1.0f);

    resultBuffer[idx] = (float)A.TouchCount * (float)B.TouchCount * prob;
}

float estimateOverlapSelectivity(
    const GridCell* h_gridA,
    const GridCell* h_gridB,
    int numCells,
    float cellVolume,
    float epsilon,
    float gamma
) {
    GridCell* d_gridA = nullptr;
    GridCell* d_gridB = nullptr;
    float* d_result = nullptr;
    
    cudaError_t err = cudaMalloc(&d_gridA, numCells * sizeof(GridCell));
    if (err != cudaSuccess) return -1.0f;
    cudaMalloc(&d_gridB, numCells * sizeof(GridCell));
    cudaMalloc(&d_result, numCells * sizeof(float));
    
    cudaMemcpy(d_gridA, h_gridA, numCells * sizeof(GridCell), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridB, h_gridB, numCells * sizeof(GridCell), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (numCells + blockSize - 1) / blockSize;
    
    estimateSelectivityKernel<<<numBlocks, blockSize>>>(d_gridA, d_gridB, d_result, numCells, cellVolume, epsilon, gamma);
    cudaDeviceSynchronize();
    
    thrust::device_ptr<float> ptr(d_result);
    float total = thrust::reduce(ptr, ptr + numCells, 0.0f, thrust::plus<float>());
    
    cudaFree(d_gridA);
    cudaFree(d_gridB);
    cudaFree(d_result);
    
    return total;
}
