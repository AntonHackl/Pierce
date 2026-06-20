#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>

// Forward declaration
struct MeshIntersectionLaunchParams;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

struct ContainmentTrackingBuffers {
    unsigned int* d_mesh1_iterations = nullptr;
    unsigned int* d_mesh1_candidate_counts = nullptr;
    unsigned int* d_mesh1_candidate_overflows = nullptr;
    unsigned int* d_mesh2_iterations = nullptr;
    unsigned int* d_mesh2_candidate_counts = nullptr;
    unsigned int* d_mesh2_candidate_overflows = nullptr;

    std::vector<unsigned int> h_mesh1_iterations;
    std::vector<unsigned int> h_mesh1_candidate_counts;
    std::vector<unsigned int> h_mesh1_candidate_overflows;
    std::vector<unsigned int> h_mesh2_iterations;
    std::vector<unsigned int> h_mesh2_candidate_counts;
    std::vector<unsigned int> h_mesh2_candidate_overflows;

    void allocate(int mesh1NumObjects, int mesh2NumObjects) {
        CUDA_CHECK(cudaMalloc(&d_mesh1_iterations, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh1_candidate_counts, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh1_candidate_overflows, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh2_iterations, mesh2NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh2_candidate_counts, mesh2NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_mesh2_candidate_overflows, mesh2NumObjects * sizeof(unsigned int)));

        CUDA_CHECK(cudaMemset(d_mesh1_iterations, 0, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh1_candidate_counts, 0, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh1_candidate_overflows, 0, mesh1NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh2_iterations, 0, mesh2NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh2_candidate_counts, 0, mesh2NumObjects * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_mesh2_candidate_overflows, 0, mesh2NumObjects * sizeof(unsigned int)));

        h_mesh1_iterations.resize(mesh1NumObjects, 0);
        h_mesh1_candidate_counts.resize(mesh1NumObjects, 0);
        h_mesh1_candidate_overflows.resize(mesh1NumObjects, 0);
        h_mesh2_iterations.resize(mesh2NumObjects, 0);
        h_mesh2_candidate_counts.resize(mesh2NumObjects, 0);
        h_mesh2_candidate_overflows.resize(mesh2NumObjects, 0);
    }

    void setupLaunchParams(MeshIntersectionLaunchParams& params1, MeshIntersectionLaunchParams& params2) {
        params1.enable_containment_tracking = 1;
        params1.containment_iterations_per_source = d_mesh1_iterations;
        params1.containment_candidate_count_per_source = d_mesh1_candidate_counts;
        params1.containment_candidate_overflow_per_source = d_mesh1_candidate_overflows;

        params2.enable_containment_tracking = 1;
        params2.containment_iterations_per_source = d_mesh2_iterations;
        params2.containment_candidate_count_per_source = d_mesh2_candidate_counts;
        params2.containment_candidate_overflow_per_source = d_mesh2_candidate_overflows;
    }

    void copyFromDevice(int mesh1NumObjects, int mesh2NumObjects) {
        CUDA_CHECK(cudaMemcpy(h_mesh1_iterations.data(), d_mesh1_iterations,
            mesh1NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh1_candidate_counts.data(), d_mesh1_candidate_counts,
            mesh1NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh1_candidate_overflows.data(), d_mesh1_candidate_overflows,
            mesh1NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh2_iterations.data(), d_mesh2_iterations,
            mesh2NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh2_candidate_counts.data(), d_mesh2_candidate_counts,
            mesh2NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mesh2_candidate_overflows.data(), d_mesh2_candidate_overflows,
            mesh2NumObjects * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    void free() {
        if (d_mesh1_iterations) CUDA_CHECK(cudaFree(d_mesh1_iterations));
        if (d_mesh1_candidate_counts) CUDA_CHECK(cudaFree(d_mesh1_candidate_counts));
        if (d_mesh1_candidate_overflows) CUDA_CHECK(cudaFree(d_mesh1_candidate_overflows));
        if (d_mesh2_iterations) CUDA_CHECK(cudaFree(d_mesh2_iterations));
        if (d_mesh2_candidate_counts) CUDA_CHECK(cudaFree(d_mesh2_candidate_counts));
        if (d_mesh2_candidate_overflows) CUDA_CHECK(cudaFree(d_mesh2_candidate_overflows));
    }
};

inline void writeContainmentTrackingCsv(
    const std::string& outputPath,
    const std::vector<unsigned int>& mesh1Iterations,
    const std::vector<unsigned int>& mesh1CandidateCounts,
    const std::vector<unsigned int>& mesh1CandidateOverflows,
    const std::vector<unsigned int>& mesh2Iterations,
    const std::vector<unsigned int>& mesh2CandidateCounts,
    const std::vector<unsigned int>& mesh2CandidateOverflows
) {
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open containment tracking output: " + outputPath);
    }

    out << "direction,source_object_id,iterations,candidate_count,candidate_overflow_events\n";

    for (size_t i = 0; i < mesh1Iterations.size(); ++i) {
        out << "mesh1_to_mesh2," << i << ',' << mesh1Iterations[i] << ','
            << mesh1CandidateCounts[i] << ',' << mesh1CandidateOverflows[i] << "\n";
    }
    for (size_t i = 0; i < mesh2Iterations.size(); ++i) {
        out << "mesh2_to_mesh1," << i << ',' << mesh2Iterations[i] << ','
            << mesh2CandidateCounts[i] << ',' << mesh2CandidateOverflows[i] << "\n";
    }
}
