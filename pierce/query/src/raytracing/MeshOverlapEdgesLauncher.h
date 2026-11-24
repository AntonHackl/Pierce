#pragma once

#include "../optix/OptixContext.h"
#include "../optix/OptixHelpers.h"
#include "MeshOverlapEdgesLaunchParams.h"
#include <optix.h>
#include <string>

class MeshOverlapEdgesLauncher {
public:
    MeshOverlapEdgesLauncher(OptixContext& context, const std::string& ptxPathHint = "");
    ~MeshOverlapEdgesLauncher();
    
    MeshOverlapEdgesLauncher(const MeshOverlapEdgesLauncher&) = delete;
    MeshOverlapEdgesLauncher& operator=(const MeshOverlapEdgesLauncher&) = delete;
    
    void launchMesh1ToMesh2(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    void launchMesh2ToMesh1(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    size_t allocatedBytes() const { return allocatedBytes_; }
    bool isValid() const { return pipeline_ != nullptr; }
    
private:
    OptixContext& context_;
    std::string ptxPathHint_;
    OptixModule module_;
    OptixPipeline pipeline_;
    OptixProgramGroup raygenPG_;
    OptixProgramGroup missPG_;
    OptixProgramGroup hitPG_;
    OptixShaderBindingTable sbt_;
    CUdeviceptr d_rg_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_lp_;
    size_t allocatedBytes_ = 0;

    void launchInternal(const MeshOverlapEdgesLaunchParams& params, int numEdges);
    
    void createModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};
