#include "OptixContext.h"
#include <optix_function_table_definition.h>

OptixContext::OptixContext() : context_(nullptr) {
    CUDA_CHECK(cudaFree(0)); // Initialize CUDA
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(0, nullptr, &context_));
}

OptixContext::~OptixContext() {
    if (context_ != nullptr) {
        optixDeviceContextDestroy(context_);
        context_ = nullptr;
    }
}

