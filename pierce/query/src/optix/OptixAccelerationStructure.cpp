#include "OptixAccelerationStructure.h"
#include "OptixHelpers.h"
#include <iostream>

OptixAccelerationStructure::OptixAccelerationStructure(OptixContext& context, GeometryUploader& geometry)
    : context_(context), geometry_(geometry), gasHandle_(0),
      d_tempBuffer_(0), d_gasOutput_(0), tempSize_(0), outputSize_(0) {
}

OptixAccelerationStructure::~OptixAccelerationStructure() {
    freeInternal();
}

void OptixAccelerationStructure::build() {
    if (!geometry_.isUploaded()) {
        std::cerr << "Error: Geometry must be uploaded before building acceleration structure" << std::endl;
        return;
    }
    buildInternal();
}

void OptixAccelerationStructure::buildInternal() {
    if (gasHandle_ != 0) {
        // Already built, free old resources first
        freeInternal();
    }
    
    CUdeviceptr d_vertices_ptr = reinterpret_cast<CUdeviceptr>(geometry_.getVertices());
    CUdeviceptr d_indices_ptr = reinterpret_cast<CUdeviceptr>(geometry_.getIndices());
    
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &d_vertices_ptr;
    buildInput.triangleArray.numVertices = static_cast<unsigned int>(geometry_.getNumVertices());
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    buildInput.triangleArray.indexBuffer = d_indices_ptr;
    buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(geometry_.getNumIndices());
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    unsigned int triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
    buildInput.triangleArray.flags = &triangle_input_flags;
    buildInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes gasSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_.getContext(), &accelOptions, &buildInput, 1, &gasSizes));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer_), gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput_), gasSizes.outputSizeInBytes));
    
    tempSize_ = gasSizes.tempSizeInBytes;
    outputSize_ = gasSizes.outputSizeInBytes;
    
    OPTIX_CHECK(optixAccelBuild(context_.getContext(), 0, &accelOptions, &buildInput, 1,
                                 d_tempBuffer_, gasSizes.tempSizeInBytes,
                                 d_gasOutput_, gasSizes.outputSizeInBytes,
                                 &gasHandle_, nullptr, 0));
    
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer_)));
    d_tempBuffer_ = 0;
    tempSize_ = 0;
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void OptixAccelerationStructure::freeInternal() {
    if (d_gasOutput_) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasOutput_)));
        d_gasOutput_ = 0;
        outputSize_ = 0;
    }
    if (d_tempBuffer_) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer_)));
        d_tempBuffer_ = 0;
        tempSize_ = 0;
    }
    gasHandle_ = 0;
}

