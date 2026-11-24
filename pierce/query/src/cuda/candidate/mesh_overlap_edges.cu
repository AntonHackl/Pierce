#include "../optix_common_shaders.cuh"
#include "../../raytracing/CandidatePairInputs.h"

extern "C" __constant__ MeshOverlapEdgesLaunchParams mesh_overlap_edges_params;

static __forceinline__ __device__ bool clipAxis(float start, float delta, float low,
                                               float high, float& from, float& to) {
    if (delta == 0.0f) return start >= low && start <= high;
    const float a = (low-start)/delta, b = (high-start)/delta;
    from = fmaxf(from, fminf(a,b));
    to = fminf(to, fmaxf(a,b));
    return from <= to;
}

static __forceinline__ __device__ bool crosses(const CandidateObject& source,
    const CandidateObject& target, const CandidateEdge* edges, int& calls) {
    for (int i = 0; i < source.edgeCount; ++i) {
        const auto edge = edges[source.firstEdge+i];
        const float3 p = edge.start;
        const float3 v = make_float3(edge.end.x-p.x, edge.end.y-p.y, edge.end.z-p.z);
        float from = 0.0f, to = 1.0f;
        if (!clipAxis(p.x,v.x,target.low.x,target.high.x,from,to) ||
            !clipAxis(p.y,v.y,target.low.y,target.high.y,from,to) ||
            !clipAxis(p.z,v.z,target.low.z,target.high.z,from,to)) continue;
        const float length = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        if (length == 0.0f) continue;
        const float3 direction = make_float3(v.x/length,v.y/length,v.z/length);
        unsigned int hit = 0, distance = 0, primitive = 0;
        ++calls;
        optixTrace(target.handle, p, direction, nextafterf(0.0f,length),
                   nextafterf(length,INFINITY), 0.0f, OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                   0,1,0,hit,distance,primitive);
        if (hit) return true;
    }
    return false;
}

extern "C" __global__ void __raygen__mesh_overlap_edges() {
    const auto& params = mesh_overlap_edges_params;
    const auto lane = optixGetLaunchIndex().x;
    const auto pair = reinterpret_cast<const int2*>(params.edge_source_object_ids)[lane];
    const auto* objects = reinterpret_cast<const CandidateObject*>(params.mesh2_vertices);
    const auto* edges = reinterpret_cast<const CandidateEdge*>(params.edge_starts);
    int calls = 0;
    const bool hit = crosses(objects[pair.x],objects[pair.y],edges,calls) ||
                     crosses(objects[pair.y],objects[pair.x],edges,calls);
    // Sign encodes the predicate, magnitude minus one is the executed trace count.
    params.collision_counts[lane] = hit ? calls+1 : -calls-1;
}
