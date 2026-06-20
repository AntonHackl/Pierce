#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <new>

// CUDA error checking macro with fail-fast behavior
#define CUDA_CHECK_PINNED(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "FATAL: Pinned memory allocation failed at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "This typically means the system has run out of pinned memory resources." << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

template<typename T>
class PinnedAllocator {
public:
    using value_type = T;

    PinnedAllocator() noexcept {}
    template<typename U> PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr;
        cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        cudaFreeHost(ptr);
    }
};

template<typename T, typename U>
bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<U>&) { return true; }

template<typename T, typename U>
bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<U>&) { return false; }

/**
 * RAII wrapper for pinned memory buffers used in geometry data transfers.
 * Automatically manages allocation and deallocation of page-locked memory
 * for faster CPU-GPU transfers via DMA.
 */
struct PinnedGeometryBuffers {
    float3* vertices_pinned = nullptr;
    uint3* indices_pinned = nullptr;
    int* triangleToObject_pinned = nullptr;
    
    size_t vertices_size = 0;
    size_t indices_size = 0;
    size_t triangleToObject_size = 0;
    
    bool allocated = false;
    
    /**
     * Allocate pinned memory buffers for geometry data.
     * Fails fast with clear error message if allocation fails.
     */
    void allocate(size_t num_vertices, size_t num_indices, size_t num_triangleToObject) {
        if (allocated) {
            std::cerr << "Warning: PinnedGeometryBuffers already allocated, freeing first" << std::endl;
            free();
        }
        
        vertices_size = num_vertices;
        indices_size = num_indices;
        triangleToObject_size = num_triangleToObject;
        
        if (num_vertices > 0) {
            CUDA_CHECK_PINNED(cudaMallocHost(&vertices_pinned, num_vertices * sizeof(float3)));
        }
        if (num_indices > 0) {
            CUDA_CHECK_PINNED(cudaMallocHost(&indices_pinned, num_indices * sizeof(uint3)));
        }
        if (num_triangleToObject > 0) {
            CUDA_CHECK_PINNED(cudaMallocHost(&triangleToObject_pinned, num_triangleToObject * sizeof(int)));
        }
        
        allocated = true;
    }
    
    /**
     * Free all pinned memory buffers.
     */
    void free() {
        if (!allocated) return;
        
        if (vertices_pinned) {
            cudaFreeHost(vertices_pinned);
            vertices_pinned = nullptr;
        }
        if (indices_pinned) {
            cudaFreeHost(indices_pinned);
            indices_pinned = nullptr;
        }
        if (triangleToObject_pinned) {
            cudaFreeHost(triangleToObject_pinned);
            triangleToObject_pinned = nullptr;
        }
        
        vertices_size = 0;
        indices_size = 0;
        triangleToObject_size = 0;
        allocated = false;
    }
    
    /**
     * Copy data from std::vectors to pinned buffers.
     * Must be called after allocate() and before GPU transfer.
     */
    template<typename VertexVec, typename IndexVec, typename TriToObjVec>
    void copyFrom(const VertexVec& vertices, const IndexVec& indices, const TriToObjVec& triangleToObject) {
        if (!allocated) {
            std::cerr << "FATAL: Attempting to copy to unallocated PinnedGeometryBuffers" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (vertices.size() != vertices_size || indices.size() != indices_size || 
            triangleToObject.size() != triangleToObject_size) {
            std::cerr << "FATAL: Size mismatch in PinnedGeometryBuffers::copyFrom" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (vertices_size > 0) {
            std::copy(vertices.begin(), vertices.end(), vertices_pinned);
        }
        if (indices_size > 0) {
            std::copy(indices.begin(), indices.end(), indices_pinned);
        }
        if (triangleToObject_size > 0) {
            std::copy(triangleToObject.begin(), triangleToObject.end(), triangleToObject_pinned);
        }
    }
    
    // RAII: Destructor automatically frees pinned memory
    ~PinnedGeometryBuffers() {
        free();
    }
    
    // Prevent copying (pinned memory should not be copied)
    PinnedGeometryBuffers(const PinnedGeometryBuffers&) = delete;
    PinnedGeometryBuffers& operator=(const PinnedGeometryBuffers&) = delete;
    
    // Allow moving
    PinnedGeometryBuffers(PinnedGeometryBuffers&& other) noexcept {
        vertices_pinned = other.vertices_pinned;
        indices_pinned = other.indices_pinned;
        triangleToObject_pinned = other.triangleToObject_pinned;
        vertices_size = other.vertices_size;
        indices_size = other.indices_size;
        triangleToObject_size = other.triangleToObject_size;
        allocated = other.allocated;
        
        other.vertices_pinned = nullptr;
        other.indices_pinned = nullptr;
        other.triangleToObject_pinned = nullptr;
        other.allocated = false;
    }
    
    PinnedGeometryBuffers& operator=(PinnedGeometryBuffers&& other) noexcept {
        if (this != &other) {
            free();
            
            vertices_pinned = other.vertices_pinned;
            indices_pinned = other.indices_pinned;
            triangleToObject_pinned = other.triangleToObject_pinned;
            vertices_size = other.vertices_size;
            indices_size = other.indices_size;
            triangleToObject_size = other.triangleToObject_size;
            allocated = other.allocated;
            
            other.vertices_pinned = nullptr;
            other.indices_pinned = nullptr;
            other.triangleToObject_pinned = nullptr;
            other.allocated = false;
        }
        return *this;
    }
    
    PinnedGeometryBuffers() = default;
};

/**
 * RAII wrapper for pinned memory buffers used in point data transfers.
 */
struct PinnedPointBuffers {
    float3* positions_pinned = nullptr;
    size_t positions_size = 0;
    bool allocated = false;
    
    /**
     * Allocate pinned memory buffer for point positions.
     */
    void allocate(size_t num_points) {
        if (allocated) {
            std::cerr << "Warning: PinnedPointBuffers already allocated, freeing first" << std::endl;
            free();
        }
        
        positions_size = num_points;
        
        if (num_points > 0) {
            CUDA_CHECK_PINNED(cudaMallocHost(&positions_pinned, num_points * sizeof(float3)));
        }
        
        allocated = true;
    }
    
    /**
     * Free pinned memory buffer.
     */
    void free() {
        if (!allocated) return;
        
        if (positions_pinned) {
            cudaFreeHost(positions_pinned);
            positions_pinned = nullptr;
        }
        
        positions_size = 0;
        allocated = false;
    }
    
    /**
     * Copy data from std::vector to pinned buffer.
     */
    template<typename PositionVec>
    void copyFrom(const PositionVec& positions) {
        if (!allocated) {
            std::cerr << "FATAL: Attempting to copy to unallocated PinnedPointBuffers" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (positions.size() != positions_size) {
            std::cerr << "FATAL: Size mismatch in PinnedPointBuffers::copyFrom" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (positions_size > 0) {
            std::copy(positions.begin(), positions.end(), positions_pinned);
        }
    }
    
    // RAII: Destructor automatically frees pinned memory
    ~PinnedPointBuffers() {
        free();
    }
    
    // Prevent copying
    PinnedPointBuffers(const PinnedPointBuffers&) = delete;
    PinnedPointBuffers& operator=(const PinnedPointBuffers&) = delete;
    
    // Allow moving
    PinnedPointBuffers(PinnedPointBuffers&& other) noexcept {
        positions_pinned = other.positions_pinned;
        positions_size = other.positions_size;
        allocated = other.allocated;
        
        other.positions_pinned = nullptr;
        other.allocated = false;
    }
    
    PinnedPointBuffers& operator=(PinnedPointBuffers&& other) noexcept {
        if (this != &other) {
            free();
            
            positions_pinned = other.positions_pinned;
            positions_size = other.positions_size;
            allocated = other.allocated;
            
            other.positions_pinned = nullptr;
            other.allocated = false;
        }
        return *this;
    }
    
    PinnedPointBuffers() = default;
};

/**
 * RAII wrapper for pinned memory buffer used to download results from GPU.
 */
template<typename T>
struct PinnedResultBuffer {
    T* buffer_pinned = nullptr;
    size_t buffer_size = 0;
    bool allocated = false;
    
    /**
     * Allocate pinned memory buffer for results.
     */
    void allocate(size_t num_elements) {
        if (allocated) {
            std::cerr << "Warning: PinnedResultBuffer already allocated, freeing first" << std::endl;
            free();
        }
        
        buffer_size = num_elements;
        
        if (num_elements > 0) {
            CUDA_CHECK_PINNED(cudaMallocHost(&buffer_pinned, num_elements * sizeof(T)));
        }
        
        allocated = true;
    }
    
    /**
     * Free pinned memory buffer.
     */
    void free() {
        if (!allocated) return;
        
        if (buffer_pinned) {
            cudaFreeHost(buffer_pinned);
            buffer_pinned = nullptr;
        }
        
        buffer_size = 0;
        allocated = false;
    }
    
    /**
     * Copy from pinned buffer to std::vector after GPU download.
     */
    template<typename Vec>
    void copyTo(Vec& destination) {
        if (!allocated) {
            std::cerr << "FATAL: Attempting to copy from unallocated PinnedResultBuffer" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        if (buffer_size > 0) {
            destination.resize(buffer_size);
            std::copy(buffer_pinned, buffer_pinned + buffer_size, destination.begin());
        }
    }
    
    // RAII: Destructor automatically frees pinned memory
    ~PinnedResultBuffer() {
        free();
    }
    
    // Prevent copying
    PinnedResultBuffer(const PinnedResultBuffer&) = delete;
    PinnedResultBuffer& operator=(const PinnedResultBuffer&) = delete;
    
    // Allow moving
    PinnedResultBuffer(PinnedResultBuffer&& other) noexcept {
        buffer_pinned = other.buffer_pinned;
        buffer_size = other.buffer_size;
        allocated = other.allocated;
        
        other.buffer_pinned = nullptr;
        other.allocated = false;
    }
    
    PinnedResultBuffer& operator=(PinnedResultBuffer&& other) noexcept {
        if (this != &other) {
            free();
            
            buffer_pinned = other.buffer_pinned;
            buffer_size = other.buffer_size;
            allocated = other.allocated;
            
            other.buffer_pinned = nullptr;
            other.allocated = false;
        }
        return *this;
    }
    
    PinnedResultBuffer() = default;
};
