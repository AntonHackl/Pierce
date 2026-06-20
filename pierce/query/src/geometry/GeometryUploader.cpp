#include "GeometryUploader.h"

GeometryUploader::GeometryUploader()
    : d_vertices_(nullptr), d_indices_(nullptr), d_triangle_to_object_(nullptr),
      num_vertices_(0), num_indices_(0), num_triangle_to_object_(0) {
}

GeometryUploader::~GeometryUploader() {
    freeInternal();
}

void GeometryUploader::upload(const GeometryData& geometry) {
    freeInternal();
    
    num_vertices_ = geometry.vertices.size();
    num_indices_ = geometry.indices.size();
    num_triangle_to_object_ = geometry.triangleToObject.size();
    
    if (num_vertices_ == 0 || num_indices_ == 0) {
        return;
    }
    
    size_t vbytes = num_vertices_ * sizeof(float3);
    size_t ibytes = num_indices_ * sizeof(uint3);
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), vbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), ibytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_to_object_), num_triangle_to_object_ * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_vertices_, geometry.vertices.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices_, geometry.indices.data(), ibytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triangle_to_object_, geometry.triangleToObject.data(), 
                          num_triangle_to_object_ * sizeof(int), cudaMemcpyHostToDevice));
}

void GeometryUploader::upload(const std::vector<float3>& vertices,
                               const std::vector<uint3>& indices,
                               const std::vector<int>& triangleToObject) {
    freeInternal();
    
    num_vertices_ = vertices.size();
    num_indices_ = indices.size();
    num_triangle_to_object_ = triangleToObject.size();
    
    if (num_vertices_ == 0 || num_indices_ == 0) {
        return;
    }
    
    size_t vbytes = num_vertices_ * sizeof(float3);
    size_t ibytes = num_indices_ * sizeof(uint3);
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), vbytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), ibytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangle_to_object_), num_triangle_to_object_ * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_vertices_, vertices.data(), vbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices_, indices.data(), ibytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triangle_to_object_, triangleToObject.data(), 
                          num_triangle_to_object_ * sizeof(int), cudaMemcpyHostToDevice));
}

void GeometryUploader::free() {
    freeInternal();
}

void GeometryUploader::freeInternal() {
    if (d_vertices_) {
        CUDA_CHECK(cudaFree(d_vertices_));
        d_vertices_ = nullptr;
    }
    if (d_indices_) {
        CUDA_CHECK(cudaFree(d_indices_));
        d_indices_ = nullptr;
    }
    if (d_triangle_to_object_) {
        CUDA_CHECK(cudaFree(d_triangle_to_object_));
        d_triangle_to_object_ = nullptr;
    }
    num_vertices_ = 0;
    num_indices_ = 0;
    num_triangle_to_object_ = 0;
}

