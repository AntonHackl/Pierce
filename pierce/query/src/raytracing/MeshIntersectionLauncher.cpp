#include "MeshIntersectionLauncher.h"
#include "../ptx_utils.h"
#include <iostream>

MeshIntersectionLauncher::MeshIntersectionLauncher(OptixContext& context, const std::string& ptxPathHint)
    : context_(context), ptxPathHint_(ptxPathHint),
    module_(nullptr), containmentAnyhitModule_(nullptr),
    pipeline_(nullptr), containmentAnyhitPipeline_(nullptr),
    raygenOverlapPG_(nullptr), raygenContainmentAnyhitPG_(nullptr),
    missPG_(nullptr), missContainmentAnyhitPG_(nullptr),
    hitPG_(nullptr), hitContainmentAnyhitPG_(nullptr),
    d_rg_overlap_(0), d_rg_containment_anyhit_(0),
    d_ms_(0), d_ms_containment_anyhit_(0),
    d_hg_(0), d_hg_containment_anyhit_(0), d_lp_(0) {
    
    createModule();
    createContainmentAnyhitModule();
    createProgramGroups();
    createPipelines();
    createSBT();
}

MeshIntersectionLauncher::~MeshIntersectionLauncher() {
    freeInternal();
}

void MeshIntersectionLauncher::createModule() {
    std::string ptxPath = detectPTXPath("mesh_intersection.ptx", ptxPathHint_);
    std::vector<char> ptxData = readPTX(ptxPath.c_str());
    
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
    
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_intersection_params";
    
    char log[8192];
    size_t sizeof_log = sizeof(log);
    
    OptixResult result = optixModuleCreate(context_.getContext(),
                                          &moduleCompileOptions,
                                          &pipelineCompileOptions,
                                          ptxData.data(),
                                          ptxData.size(),
                                          log,
                                          &sizeof_log,
                                          &module_);
    
    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed for mesh intersection: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
}

void MeshIntersectionLauncher::createContainmentAnyhitModule() {
    std::string ptxPath = detectPTXPath("mesh_intersection_anyhit.ptx", ptxPathHint_);
    std::vector<char> ptxData = readPTX(ptxPath.c_str());

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_intersection_params";

    char log[8192];
    size_t sizeof_log = sizeof(log);

    OptixResult result = optixModuleCreate(context_.getContext(),
                                          &moduleCompileOptions,
                                          &pipelineCompileOptions,
                                          ptxData.data(),
                                          ptxData.size(),
                                          log,
                                          &sizeof_log,
                                          &containmentAnyhitModule_);

    if (result != OPTIX_SUCCESS) {
        std::cerr << "OptiX module creation failed for mesh intersection anyhit: " << result << std::endl;
        if (sizeof_log > 1) {
            std::cerr << "OptiX module log:\n" << log << std::endl;
        }
        std::exit(EXIT_FAILURE);
    }
}

void MeshIntersectionLauncher::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    
    OptixProgramGroupDesc rgOverlap = {};
    rgOverlap.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgOverlap.raygen.module = module_;
    rgOverlap.raygen.entryFunctionName = "__raygen__mesh_overlap";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &rgOverlap, 1, &pgOptions, nullptr, nullptr, &raygenOverlapPG_));

    OptixProgramGroupDesc rgContainAnyhit = {};
    rgContainAnyhit.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgContainAnyhit.raygen.module = containmentAnyhitModule_;
    rgContainAnyhit.raygen.entryFunctionName = "__raygen__mesh_containment_anyhit";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &rgContainAnyhit, 1, &pgOptions, nullptr, nullptr, &raygenContainmentAnyhitPG_));
    
    // Miss program group
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module_;
    missDesc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &missDesc, 1, &pgOptions, nullptr, nullptr, &missPG_));

    OptixProgramGroupDesc missAnyhitDesc = {};
    missAnyhitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missAnyhitDesc.miss.module = containmentAnyhitModule_;
    missAnyhitDesc.miss.entryFunctionName = "__miss__ms_anyhit";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &missAnyhitDesc, 1, &pgOptions, nullptr, nullptr, &missContainmentAnyhitPG_));
    
    // Hit program group
    OptixProgramGroupDesc hitDesc = {};
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = module_;
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitDesc.hitgroup.moduleAH = module_;
    hitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &hitDesc, 1, &pgOptions, nullptr, nullptr, &hitPG_));

    OptixProgramGroupDesc hitAnyhitDesc = {};
    hitAnyhitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitAnyhitDesc.hitgroup.moduleCH = containmentAnyhitModule_;
    hitAnyhitDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch_anyhit";
    hitAnyhitDesc.hitgroup.moduleAH = containmentAnyhitModule_;
    hitAnyhitDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah_containment";
    OPTIX_CHECK(optixProgramGroupCreate(context_.getContext(), &hitAnyhitDesc, 1, &pgOptions, nullptr, nullptr, &hitContainmentAnyhitPG_));
}

void MeshIntersectionLauncher::createPipelines() {
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "mesh_intersection_params";
    
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;
    
    std::vector<OptixProgramGroup> pgs2 = {
        raygenOverlapPG_,
        missPG_,
        hitPG_
    };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    pgs2.data(), pgs2.size(), nullptr, nullptr, &pipeline_));

    std::vector<OptixProgramGroup> containmentAnyhitPgs = {
        raygenContainmentAnyhitPG_,
        missContainmentAnyhitPG_,
        hitContainmentAnyhitPG_
    };
    OPTIX_CHECK(optixPipelineCreate(context_.getContext(), &pipelineCompileOptions, &linkOptions,
                                    containmentAnyhitPgs.data(), containmentAnyhitPgs.size(), nullptr, nullptr, &containmentAnyhitPipeline_));
}

void MeshIntersectionLauncher::createSBT() {
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord { 
        char header[OPTIX_SBT_RECORD_HEADER_SIZE]; 
    };
    
    RaygenRecord rgOverlap = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenOverlapPG_, &rgOverlap));

    RaygenRecord rgContainAnyhit = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenContainmentAnyhitPG_, &rgContainAnyhit));
    
    MissRecord msRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &msRecord));

    MissRecord msAnyhitRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missContainmentAnyhitPG_, &msAnyhitRecord));
    
    HitRecord hgRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_, &hgRecord));

    HitRecord hgAnyhitRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitContainmentAnyhitPG_, &hgAnyhitRecord));
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_overlap_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rg_containment_anyhit_), sizeof(RaygenRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_), sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ms_containment_anyhit_), sizeof(MissRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_), sizeof(HitRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hg_containment_anyhit_), sizeof(HitRecord)));
    
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_overlap_), &rgOverlap, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_rg_containment_anyhit_), &rgContainAnyhit, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_), &msRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ms_containment_anyhit_), &msAnyhitRecord, sizeof(MissRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_), &hgRecord, sizeof(HitRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hg_containment_anyhit_), &hgAnyhitRecord, sizeof(HitRecord), cudaMemcpyHostToDevice));
    
    sbt_ = {};
    sbt_.raygenRecord = d_rg_overlap_;
    sbt_.missRecordBase = d_ms_;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = d_hg_;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    sbt_.hitgroupRecordCount = 1;

    containmentAnyhitSbt_ = {};
    containmentAnyhitSbt_.raygenRecord = d_rg_containment_anyhit_;
    containmentAnyhitSbt_.missRecordBase = d_ms_containment_anyhit_;
    containmentAnyhitSbt_.missRecordStrideInBytes = sizeof(MissRecord);
    containmentAnyhitSbt_.missRecordCount = 1;
    containmentAnyhitSbt_.hitgroupRecordBase = d_hg_containment_anyhit_;
    containmentAnyhitSbt_.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    containmentAnyhitSbt_.hitgroupRecordCount = 1;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lp_), sizeof(MeshIntersectionLaunchParams)));
}

void MeshIntersectionLauncher::launchInternal(const MeshIntersectionLaunchParams& params, int launchSize, CUdeviceptr raygenRecord) {
    launchInternalWithSbt(params, launchSize, pipeline_, sbt_, raygenRecord);
}

void MeshIntersectionLauncher::launchInternalWithSbt(const MeshIntersectionLaunchParams& params, int launchSize, OptixPipeline pipeline, OptixShaderBindingTable& sbt, CUdeviceptr raygenRecord) {
    CUDA_CHECK(cudaMemcpy((void*)d_lp_, &params, sizeof(MeshIntersectionLaunchParams), cudaMemcpyHostToDevice));
    sbt.raygenRecord = raygenRecord;
    OPTIX_CHECK(optixLaunch(pipeline, 0, d_lp_, sizeof(MeshIntersectionLaunchParams), &sbt, launchSize, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MeshIntersectionLauncher::launchOverlapMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 0;
    launchInternal(launchParams, launchSize, d_rg_overlap_);
}

void MeshIntersectionLauncher::launchOverlapMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 1;
    launchInternal(launchParams, launchSize, d_rg_overlap_);
}

void MeshIntersectionLauncher::launchContainmentMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 0;
    launchInternalWithSbt(launchParams, launchSize, containmentAnyhitPipeline_, containmentAnyhitSbt_, d_rg_containment_anyhit_);
}

void MeshIntersectionLauncher::launchContainmentMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    MeshIntersectionLaunchParams launchParams = params;
    launchParams.swap_result_ids = 1;
    launchInternalWithSbt(launchParams, launchSize, containmentAnyhitPipeline_, containmentAnyhitSbt_, d_rg_containment_anyhit_);
}

void MeshIntersectionLauncher::launchMesh1ToMesh2(const MeshIntersectionLaunchParams& params, int launchSize) {
    launchOverlapMesh1ToMesh2(params, launchSize);
}

void MeshIntersectionLauncher::launchMesh2ToMesh1(const MeshIntersectionLaunchParams& params, int launchSize) {
    launchOverlapMesh2ToMesh1(params, launchSize);
}

void MeshIntersectionLauncher::freeInternal() {
    if (d_lp_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lp_)));
    if (d_hg_containment_anyhit_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_containment_anyhit_)));
    if (d_hg_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_hg_)));
    if (d_ms_containment_anyhit_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_containment_anyhit_)));
    if (d_ms_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ms_)));
    if (d_rg_containment_anyhit_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_containment_anyhit_)));
    if (d_rg_overlap_) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_rg_overlap_)));
    if (containmentAnyhitPipeline_) optixPipelineDestroy(containmentAnyhitPipeline_);
    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (hitContainmentAnyhitPG_) optixProgramGroupDestroy(hitContainmentAnyhitPG_);
    if (hitPG_) optixProgramGroupDestroy(hitPG_);
    if (missContainmentAnyhitPG_) optixProgramGroupDestroy(missContainmentAnyhitPG_);
    if (missPG_) optixProgramGroupDestroy(missPG_);
    if (raygenContainmentAnyhitPG_) optixProgramGroupDestroy(raygenContainmentAnyhitPG_);
    if (raygenOverlapPG_) optixProgramGroupDestroy(raygenOverlapPG_);
    if (containmentAnyhitModule_) optixModuleDestroy(containmentAnyhitModule_);
    if (module_) optixModuleDestroy(module_);
}
