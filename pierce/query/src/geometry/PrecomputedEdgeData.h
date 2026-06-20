#pragma once

#include "Geometry.h"

struct EdgeMeshData {
    float3* d_edge_starts;
    float3* d_edge_ends;
    int* d_source_object_ids;
    int num_edges;

    EdgeMeshData()
                : d_edge_starts(nullptr),
                    d_edge_ends(nullptr),
                    d_source_object_ids(nullptr),
                    num_edges(0) {}
};

class PrecomputedEdgeData {
public:
    static EdgeMeshData uploadFromGeometry(const GeometryData& geometry);
    static void freeEdgeData(EdgeMeshData& edgeData);
};
