#include "PrecomputedEdgeData.h"

#include "../optix/OptixHelpers.h"

EdgeMeshData PrecomputedEdgeData::uploadFromGeometry(const GeometryData& geometry) {
    EdgeMeshData result;

    if (!geometry.edges.hasEdges()) {
        return result;
    }

    result.num_edges = static_cast<int>(geometry.edges.edgeStarts.size());
    const size_t edge_bytes = geometry.edges.edgeStarts.size() * sizeof(float3);
    const size_t source_object_ids_bytes = geometry.edges.sourceObjectIds.size() * sizeof(int);

    CUDA_CHECK(cudaMalloc(&result.d_edge_starts, edge_bytes));
    CUDA_CHECK(cudaMalloc(&result.d_edge_ends, edge_bytes));
    if (source_object_ids_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&result.d_source_object_ids, source_object_ids_bytes));
    }

    CUDA_CHECK(cudaMemcpy(result.d_edge_starts, geometry.edges.edgeStarts.data(), edge_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_edge_ends, geometry.edges.edgeEnds.data(), edge_bytes, cudaMemcpyHostToDevice));
    if (source_object_ids_bytes > 0) {
        CUDA_CHECK(cudaMemcpy(result.d_source_object_ids, geometry.edges.sourceObjectIds.data(), source_object_ids_bytes, cudaMemcpyHostToDevice));
    }

    return result;
}

void PrecomputedEdgeData::freeEdgeData(EdgeMeshData& edgeData) {
    if (edgeData.d_edge_starts) {
        CUDA_CHECK(cudaFree(edgeData.d_edge_starts));
        edgeData.d_edge_starts = nullptr;
    }
    if (edgeData.d_edge_ends) {
        CUDA_CHECK(cudaFree(edgeData.d_edge_ends));
        edgeData.d_edge_ends = nullptr;
    }
    if (edgeData.d_source_object_ids) {
        CUDA_CHECK(cudaFree(edgeData.d_source_object_ids));
        edgeData.d_source_object_ids = nullptr;
    }
    edgeData.num_edges = 0;
}
