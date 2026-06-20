#pragma once

#include "../optix/OptixContext.h"
#include "../cuda/mesh_containment.h"
#include "../optix/OptixHelpers.h"
#include <optix.h>
#include <string>

/**
 * MeshContainmentLauncher
 *
 * Manages two OptiX pipelines compiled from mesh_containment.cu:
 *   1. edge_pipeline   – __raygen__check_edges   (Phase 1, used for both B→A and A→B)
 *   2. pim_pipeline    – __raygen__point_in_mesh  (Phase 2, one thread per B object)
 */
class MeshContainmentLauncher {
public:
    MeshContainmentLauncher(OptixContext& context, const std::string& ptxPathHint = "");
    ~MeshContainmentLauncher();

    MeshContainmentLauncher(const MeshContainmentLauncher&) = delete;
    MeshContainmentLauncher& operator=(const MeshContainmentLauncher&) = delete;

    /// Launch the edge-intersection kernel (Phase 1).
    /// Call once with B mesh as source (swap_ids=0) and once with A mesh as source (swap_ids=1).
    void launchEdgeCheck(const MeshContainmentLaunchParams& params, int numTriangles);

    /// Launch the point-in-mesh kernel (Phase 2).
    /// numBObjects threads are launched.
    void launchPointInMesh(const MeshContainmentLaunchParams& params, int numBObjects);

    bool isValid() const { return edgePipeline_ != nullptr && pimPipeline_ != nullptr; }

private:
    OptixContext& context_;
    std::string ptxPathHint_;

    OptixModule  module_;

    // Edge-check pipeline
    OptixPipeline      edgePipeline_;
    OptixProgramGroup  edgeRaygenPG_;

    // Point-in-mesh pipeline
    OptixPipeline      pimPipeline_;
    OptixProgramGroup  pimRaygenPG_;

    // Shared miss + hit program groups
    OptixProgramGroup  missPG_;
    OptixProgramGroup  hitPG_;

    // SBTs
    OptixShaderBindingTable edgeSBT_;
    OptixShaderBindingTable pimSBT_;

    // Device-side records
    CUdeviceptr d_rgEdge_;
    CUdeviceptr d_rgPIM_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;

    // Device-side launch-params buffers (one per launch function)
    CUdeviceptr d_lpEdge_;
    CUdeviceptr d_lpPIM_;

    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBTs();
    void freeInternal();
};
