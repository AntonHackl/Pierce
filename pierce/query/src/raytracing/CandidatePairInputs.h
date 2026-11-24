#pragma once
#include "MeshOverlapEdgesLaunchParams.h"

struct CandidateObject {
    OptixTraversableHandle handle;
    float3 low, high;
    int firstEdge, edgeCount;
};

struct CandidateEdge {
    float3 start, end;
};

// The experimental module shares the production launcher's parameter ABI.
// Its pointer fields carry the batched inputs listed here.
inline MeshOverlapEdgesLaunchParams candidateParams(
    CandidateObject* objects, CandidateEdge* edges, int2* pairs, int* results, int count) {
    MeshOverlapEdgesLaunchParams params{};
    params.mesh2_vertices = reinterpret_cast<float3*>(objects);
    params.edge_starts = reinterpret_cast<float3*>(edges);
    params.edge_source_object_ids = reinterpret_cast<int*>(pairs);
    params.collision_counts = results;
    params.num_edges = count;
    return params;
}
