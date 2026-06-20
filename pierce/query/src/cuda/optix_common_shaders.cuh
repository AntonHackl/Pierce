#pragma once

#include <optix_device.h>

// Shared OptiX shader entry points for simple ray-hit payload capture.
extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __anyhit__ah() {
}

extern "C" __global__ void __closesthit__ch() {
    const float t = optixGetRayTmax();
    const unsigned int triangleIndex = optixGetPrimitiveIndex();

    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(t));
    optixSetPayload_2(triangleIndex);
}
