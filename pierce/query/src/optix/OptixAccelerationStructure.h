#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "GeometryUploader.h"
#include "OptixContext.h"

class OptixAccelerationStructure {
public:
    OptixAccelerationStructure(OptixContext& context, GeometryUploader& geometry);
    ~OptixAccelerationStructure();
    
    OptixAccelerationStructure(const OptixAccelerationStructure&) = delete;
    OptixAccelerationStructure& operator=(const OptixAccelerationStructure&) = delete;
    
    OptixTraversableHandle getHandle() const { return gasHandle_; }
    bool isValid() const { return gasHandle_ != 0; }
    
    // Build from geometry uploader (already uploaded to GPU)
    void build();
    
private:
    OptixContext& context_;
    GeometryUploader& geometry_;
    OptixTraversableHandle gasHandle_;
    CUdeviceptr d_tempBuffer_;
    CUdeviceptr d_gasOutput_;
    size_t tempSize_;
    size_t outputSize_;
    
    void buildInternal();
    void freeInternal();
};

