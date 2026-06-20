#pragma once
#include <optix.h>
#include <cuda_runtime.h>

struct RayGenData {
    float3 origin;
    float3 direction;
};

struct HitGroupData {
    float3 color;
};

struct RayResult {
    int ray_id;        // Original ray index for mapping back to point
    int polygon_index; // Polygon that was hit (-1 for miss)
};

struct LaunchParams {
    RayGenData ray_gen;
    HitGroupData hit_group;
    OptixTraversableHandle handle;
    RayResult* result;

    float3* ray_origins;
    uint3* indices;
    int* triangle_to_object;
    int num_rays;
    
    RayResult* compact_result;
    int* hit_counter;
};

struct MeshQueryResult {
    int object_id_mesh1;       // Object ID from Mesh1's triangleToObject
    int object_id_mesh2;       // Object ID from Mesh2's triangleToObject
};
