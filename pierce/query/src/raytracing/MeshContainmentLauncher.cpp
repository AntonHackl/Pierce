#include "MeshContainmentLauncher.h"
#include "../ptx_utils.h"
#include <iostream>

// -----------------------------------------------------------------------
// Construction / Destruction
// -----------------------------------------------------------------------

MeshContainmentLauncher::MeshContainmentLauncher(OptixContext& context,
                                                 const std::string& ptxPathHint)
    : context_(context), ptxPathHint_(ptxPathHint),
      module_(nullptr),
      edgePipeline_(nullptr), pimPipeline_(nullptr),
      edgeRaygenPG_(nullptr), pimRaygenPG_(nullptr),
      missPG_(nullptr), hitPG_(nullptr),
      d_rgEdge_(0), d_rgPIM_(0), d_ms_(0), d_hg_(0),
      d_lpEdge_(0), d_lpPIM_(0)
{
    createModule();
    createProgramGroups();
    createPipelines();
    createSBTs();
}

MeshContainmentLauncher::~MeshContainmentLauncher() {
    freeInternal();
}

// -----------------------------------------------------------------------
// Module – load mesh_containment.ptx
// -----------------------------------------------------------------------

void MeshContainmentLauncher::createModule() {
    std::string ptxPath = detectPTXPath("mesh_containment.ptx", ptxPathHint_);

    std::vector<char> ptxData = readPTX(ptxPath.c_str());

    OptixModuleCompileOptions moduleOpts = {};
    moduleOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur                  = false;
    pipeOpts.traversableGraphFlags           = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues                = 3;
    pipeOpts.numAttributeValues              = 2;
    pipeOpts.exceptionFlags                  = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "containment_params";

    char   log[8192];
    size_t logSize = sizeof(log);

    OptixResult result = optixModuleCreate(
        context_.getContext(), &moduleOpts, &pipeOpts,
        ptxData.data(), ptxData.size(),
        log, &logSize, &module_);

    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed (mesh_containment): " << result << std::endl;
        if (logSize > 1) std::cerr << log << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------
// Program groups
// -----------------------------------------------------------------------

void MeshContainmentLauncher::createProgramGroups() {
    OptixProgramGroupOptions pgOpts = {};

    // Raygen: edge check
    OptixProgramGroupDesc edgeRGDesc = {};
    edgeRGDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    edgeRGDesc.raygen.module          = module_;
    edgeRGDesc.raygen.entryFunctionName = "__raygen__check_edges_no_samehit_suppression";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &edgeRGDesc, 1,
                                        &pgOpts, nullptr, nullptr, &edgeRaygenPG_));

    // Raygen: point-in-mesh
    OptixProgramGroupDesc pimRGDesc = {};
    pimRGDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pimRGDesc.raygen.module           = module_;
    pimRGDesc.raygen.entryFunctionName = "__raygen__point_in_mesh";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &pimRGDesc, 1,
                                        &pgOpts, nullptr, nullptr, &pimRaygenPG_));

    // Miss
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind           = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module    = module_;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &missDesc, 1,
                                        &pgOpts, nullptr, nullptr, &missPG_));

    // Closest-hit + any-hit
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH        = module_;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitDesc.hitgroup.moduleAH        = module_;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &hitDesc, 1,
                                        &pgOpts, nullptr, nullptr, &hitPG_));
}

// -----------------------------------------------------------------------
// Pipelines (one per raygen entry point)
// -----------------------------------------------------------------------

void MeshContainmentLauncher::createPipelines() {
    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur                  = false;
    pipeOpts.traversableGraphFlags           = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues                = 3;
    pipeOpts.numAttributeValues              = 2;
    pipeOpts.exceptionFlags                  = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "containment_params";

    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 1;

    // Edge pipeline
    std::vector<OptixProgramGroup> edgePGs = { edgeRaygenPG_, missPG_, hitPG_ };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipeOpts, &linkOpts,
                                    edgePGs.data(), edgePGs.size(),
                                    nullptr, nullptr, &edgePipeline_));

    // Point-in-mesh pipeline
    std::vector<OptixProgramGroup> pimPGs = { pimRaygenPG_, missPG_, hitPG_ };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipeOpts, &linkOpts,
                                    pimPGs.data(), pimPGs.size(),
                                    nullptr, nullptr, &pimPipeline_));
}

// -----------------------------------------------------------------------
// Shader Binding Tables
// -----------------------------------------------------------------------

void MeshContainmentLauncher::createSBTs() {
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord   { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord    { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

    RaygenRecord rgEdge{}, rgPIM{};
    OPTIX_CHECK(optixSbtRecordPackHeader(edgeRaygenPG_, &rgEdge));
    OPTIX_CHECK(optixSbtRecordPackHeader(pimRaygenPG_,  &rgPIM));

    MissRecord ms{};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &ms));

    HitRecord hg{};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_, &hg));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgEdge_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgPIM_),  sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_),     sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_),     sizeof(HitRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rgEdge_), &rgEdge, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rgPIM_),  &rgPIM,  sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_),     &ms,     sizeof(MissRecord),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_),     &hg,     sizeof(HitRecord),    cudaMemcpyHostToDevice));

    // Edge SBT
    edgeSBT_ = {};
    edgeSBT_.raygenRecord                = d_rgEdge_;
    edgeSBT_.missRecordBase              = d_ms_;
    edgeSBT_.missRecordStrideInBytes     = sizeof(MissRecord);
    edgeSBT_.missRecordCount             = 1;
    edgeSBT_.hitgroupRecordBase          = d_hg_;
    edgeSBT_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    edgeSBT_.hitgroupRecordCount         = 1;

    // Point-in-mesh SBT
    pimSBT_ = {};
    pimSBT_.raygenRecord                = d_rgPIM_;
    pimSBT_.missRecordBase              = d_ms_;
    pimSBT_.missRecordStrideInBytes     = sizeof(MissRecord);
    pimSBT_.missRecordCount             = 1;
    pimSBT_.hitgroupRecordBase          = d_hg_;
    pimSBT_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    pimSBT_.hitgroupRecordCount         = 1;

    // Allocate device-side launch-params buffers
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lpEdge_), sizeof(MeshContainmentLaunchParams)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lpPIM_),  sizeof(MeshContainmentLaunchParams)));
}

// -----------------------------------------------------------------------
// Launch helpers
// -----------------------------------------------------------------------

void MeshContainmentLauncher::launchEdgeCheck(const MeshContainmentLaunchParams& params,
                                              int numTriangles) {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_lpEdge_), &params,
                          sizeof(MeshContainmentLaunchParams), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(edgePipeline_, 0, d_lpEdge_,
                            sizeof(MeshContainmentLaunchParams), &edgeSBT_,
                            numTriangles, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshContainmentLauncher::launchPointInMesh(const MeshContainmentLaunchParams& params,
                                                int numBObjects) {
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_lpPIM_), &params,
                          sizeof(MeshContainmentLaunchParams), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pimPipeline_, 0, d_lpPIM_,
                            sizeof(MeshContainmentLaunchParams), &pimSBT_,
                            numBObjects, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------
// Teardown
// -----------------------------------------------------------------------

void MeshContainmentLauncher::freeInternal() {
    if (d_lpPIM_)  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lpPIM_)));
    if (d_lpEdge_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lpEdge_)));
    if (d_hg_)     CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_)));
    if (d_ms_)     CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_)));
    if (d_rgPIM_)  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rgPIM_)));
    if (d_rgEdge_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rgEdge_)));
    if (pimPipeline_)   optixPipelineDestroy(pimPipeline_);
    if (edgePipeline_)  optixPipelineDestroy(edgePipeline_);
    if (hitPG_)         optixProgramGroupDestroy(hitPG_);
    if (missPG_)        optixProgramGroupDestroy(missPG_);
    if (pimRaygenPG_)   optixProgramGroupDestroy(pimRaygenPG_);
    if (edgeRaygenPG_)  optixProgramGroupDestroy(edgeRaygenPG_);
    if (module_)        optixModuleDestroy(module_);
}
