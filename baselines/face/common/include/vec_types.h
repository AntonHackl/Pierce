#pragma once

#include <cmath>
#include <cstdint>

struct float3 {
    float x, y, z;
};

struct uint3 {
    uint32_t x, y, z;
};

struct int3 {
    int32_t x, y, z;
};

inline float3 make_float3(float x, float y, float z) {
    return {x, y, z};
}
