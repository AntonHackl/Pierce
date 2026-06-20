#pragma once

#include "../optix/OptixContext.h"
#include "../cuda/mesh_intersection.h"
#include "../optix/OptixHelpers.h"
#include <optix.h>
#include <string>

class MeshIntersectionLauncher {
public:
    MeshIntersectionLauncher(OptixContext& context, const std::string& ptxPathHint = "");
    ~MeshIntersectionLauncher();
    
    MeshIntersectionLauncher(const MeshIntersectionLauncher&) = delete;
    MeshIntersectionLauncher& operator=(const MeshIntersectionLauncher&) = delete;
    
    void launchOverlapMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchOverlapMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchContainmentMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchContainmentMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize);
    void launchMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize);
    
    bool isValid() const { return pipeline_ != nullptr && containmentAnyhitPipeline_ != nullptr; }
    
private:
    OptixContext& context_;
    std::string ptxPathHint_;
    OptixModule module_;
    OptixModule containmentAnyhitModule_;
    OptixPipeline pipeline_;
    OptixPipeline containmentAnyhitPipeline_;
    OptixProgramGroup raygenOverlapPG_;
    OptixProgramGroup raygenContainmentAnyhitPG_;
    OptixProgramGroup missPG_;
    OptixProgramGroup missContainmentAnyhitPG_;
    OptixProgramGroup hitPG_;
    OptixProgramGroup hitContainmentAnyhitPG_;
    OptixShaderBindingTable sbt_;
    OptixShaderBindingTable containmentAnyhitSbt_;
    CUdeviceptr d_rg_overlap_;
    CUdeviceptr d_rg_containment_anyhit_;
    CUdeviceptr d_ms_;
    CUdeviceptr d_ms_containment_anyhit_;
    CUdeviceptr d_hg_;
    CUdeviceptr d_hg_containment_anyhit_;
    CUdeviceptr d_lp_;

    void launchInternal(const MeshIntersectionLaunchParams& params, int launchSize, CUdeviceptr raygenRecord);
    void launchInternalWithSbt(const MeshIntersectionLaunchParams& params, int launchSize, OptixPipeline pipeline, OptixShaderBindingTable& sbt, CUdeviceptr raygenRecord);
    
    void createModule();
    void createContainmentAnyhitModule();
    void createProgramGroups();
    void createPipelines();
    void createSBT();
    void freeInternal();
};
