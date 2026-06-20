#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "OptixHelpers.h"

class OptixContext {
public:
    OptixContext();
    ~OptixContext();
    
    OptixContext(const OptixContext&) = delete;
    OptixContext& operator=(const OptixContext&) = delete;
    
    OptixDeviceContext getContext() const { return context_; }
    bool isValid() const { return context_ != nullptr; }
    
private:
    OptixDeviceContext context_;
};

