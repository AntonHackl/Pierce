#include "../runtime/QueryRuntime.h"
#include "../raytracing/CandidatePairInputs.h"
#include "../optix/OptixHelpers.h"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <algorithm>
#include <map>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>

#include <unordered_map>

namespace {
using Clock = std::chrono::steady_clock;
using Pairs = std::vector<std::pair<int,int>>;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
using Point = bg::model::point<float, 3, bg::cs::cartesian>;
using Box = bg::model::box<Point>;
using Entry = std::pair<Box,int>;
using RTree = bgi::rtree<Entry,bgi::rstar<16>>;

Box box(const CandidateObject& object) {
    return Box(Point(object.low.x,object.low.y,object.low.z),
               Point(object.high.x,object.high.y,object.high.z));
}

void canonicalize(Pairs& pairs) {
    std::sort(pairs.begin(),pairs.end());
    if (std::adjacent_find(pairs.begin(),pairs.end()) != pairs.end()) {
        throw std::runtime_error("Duplicate pair in query output");
    }
}

void logDeviceMemory(const char* stage) {
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes,&totalBytes));
    std::cout << "memory," << stage << ',' << freeBytes << ',' << totalBytes << std::endl;
}

long long wallTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

template<class T> class Buffer {
    T* ptr_ = nullptr;
public:
    explicit Buffer(size_t n) { if (n) CUDA_CHECK(cudaMalloc(&ptr_, n*sizeof(T))); }
    ~Buffer() { if (ptr_) CUDA_CHECK(cudaFree(ptr_)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* get() const { return ptr_; }
    void upload(const std::vector<T>& rows) {
        if (!rows.empty()) CUDA_CHECK(cudaMemcpy(ptr_, rows.data(), rows.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct Object {
    size_t firstVertex = 0, firstTriangle = 0, outputOffset = 0;

    std::vector<float3> vertices;
    std::vector<uint3> triangles;
    std::unordered_map<unsigned int,unsigned int> indices;
    std::vector<CandidateEdge> edges;
    CandidateObject input{};
    int id;

    explicit Object(int key) : id(key) {
        const float inf = std::numeric_limits<float>::infinity();
        input.low = make_float3(inf,inf,inf);
        input.high = make_float3(-inf,-inf,-inf);
    }
    unsigned int vertex(const GeometryData& data, unsigned int i) {
        auto [it, inserted] = indices.emplace(i, static_cast<unsigned int>(vertices.size()));
        if (inserted) {
            const auto v = data.vertices.at(i);
            vertices.push_back(v);
            input.low = make_float3(std::min(input.low.x,v.x),std::min(input.low.y,v.y),std::min(input.low.z,v.z));
            input.high = make_float3(std::max(input.high.x,v.x),std::max(input.high.y,v.y),std::max(input.high.z,v.z));
        }
        return it->second;
    }
};

void appendObjects(const GeometryData& data,
                   std::vector<std::unique_ptr<Object>>& objects) {
    std::unordered_map<int,Object*> byId;
    for (size_t i = 0; i < data.indices.size(); ++i) {
        const int id = data.triangleToObject.at(i);
        auto& object = byId[id];
        if (!object) {
            objects.push_back(std::make_unique<Object>(id));
            object = objects.back().get();
        }
        const auto t = data.indices[i];
        object->triangles.push_back(make_uint3(object->vertex(data,t.x),object->vertex(data,t.y),object->vertex(data,t.z)));
    }
    for (size_t i = 0; i < data.edges.edgeStarts.size(); ++i) {
        byId.at(data.edges.sourceObjectIds.at(i))->edges.push_back({data.edges.edgeStarts[i],data.edges.edgeEnds[i]});
    }
}

OptixBuildInput triangleInput(const Object& object, const CUdeviceptr& vertices,
                              CUdeviceptr triangles) {
    static constexpr unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING |
                                          OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& mesh = input.triangleArray;
    mesh.vertexBuffers = &vertices;
    mesh.numVertices = static_cast<unsigned int>(object.vertices.size());
    mesh.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    mesh.vertexStrideInBytes = sizeof(float3);
    mesh.indexBuffer = triangles;
    mesh.numIndexTriplets = static_cast<unsigned int>(object.triangles.size());
    mesh.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    mesh.indexStrideInBytes = sizeof(uint3);
    mesh.flags = &flags;
    mesh.numSbtRecords = 1;
    return input;
}

std::unique_ptr<Buffer<unsigned char>> buildObjectIndexes(
        OptixContext& context, const std::vector<std::unique_ptr<Object>>& objects) {
    std::vector<float3> vertices;
    std::vector<uint3> triangles;
    for (const auto& object : objects) {
        object->firstVertex = vertices.size();
        object->firstTriangle = triangles.size();
        vertices.insert(vertices.end(),object->vertices.begin(),object->vertices.end());
        triangles.insert(triangles.end(),object->triangles.begin(),object->triangles.end());
    }
    Buffer<float3> deviceVertices(vertices.size());
    Buffer<uint3> deviceTriangles(triangles.size());
    deviceVertices.upload(vertices);
    deviceTriangles.upload(triangles);
    OptixAccelBuildOptions options{};
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    std::map<std::pair<size_t,size_t>,OptixAccelBufferSizes> sizes;
    size_t outputBytes = 0, scratchBytes = 0;
    constexpr size_t alignment = OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
    for (const auto& object : objects) {
        const auto key = std::make_pair(object->vertices.size(),object->triangles.size());
        auto [it, inserted] = sizes.try_emplace(key);
        if (inserted) {
            const auto vertexAddress = reinterpret_cast<CUdeviceptr>(deviceVertices.get()+object->firstVertex);
            const auto indexAddress = reinterpret_cast<CUdeviceptr>(deviceTriangles.get()+object->firstTriangle);
            const auto input = triangleInput(*object,vertexAddress,indexAddress);
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context.getContext(),&options,&input,1,&it->second));
        }
        object->outputOffset = outputBytes;
        outputBytes += ((it->second.outputSizeInBytes+alignment-1)/alignment)*alignment;
        scratchBytes = std::max(scratchBytes,it->second.tempSizeInBytes);
    }

    auto output = std::make_unique<Buffer<unsigned char>>(outputBytes);
    Buffer<unsigned char> scratch(scratchBytes);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    logDeviceMemory("object_build_buffers");
    size_t queued = 0;
    for (const auto& object : objects) {
        const auto vertexAddress = reinterpret_cast<CUdeviceptr>(deviceVertices.get()+object->firstVertex);
        const auto indexAddress = reinterpret_cast<CUdeviceptr>(deviceTriangles.get()+object->firstTriangle);
        const auto input = triangleInput(*object,vertexAddress,indexAddress);
        const auto& size = sizes.at({object->vertices.size(),object->triangles.size()});
        OPTIX_CHECK(optixAccelBuild(context.getContext(),stream,&options,&input,1,
            reinterpret_cast<CUdeviceptr>(scratch.get()),size.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(output->get()+object->outputOffset),size.outputSizeInBytes,
            &object->input.handle,nullptr,0));
        if (++queued % 4096 == 0) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return output;
}
void run(char** args, int threads, bool parallelOnly) {
    const int repeats = std::stoi(args[4]);
    if (repeats <= 0) throw std::runtime_error("Repetitions must be positive");
    const std::filesystem::path bin = args[3];
    logDeviceMemory("start");
    pierce::QueryRuntime shared((bin/"mesh_overlap_edges.ptx").string());
    const auto& resident = shared.loadPair("scene",args[1],args[2]);
    logDeviceMemory("shared_loaded");
    OptixContext context;
    MeshOverlapEdgesLauncher launcher(context,(bin/"candidate"/"mesh_overlap_edges.ptx").string());
    const auto setup = Clock::now();
    std::vector<std::unique_ptr<Object>> objects;
    appendObjects(resident.a->geometry,objects);
    const int split = static_cast<int>(objects.size());
    appendObjects(resident.b->geometry,objects);
    std::vector<CandidateObject> inputs;
    std::vector<CandidateEdge> edges;
    const auto objectIndexes = buildObjectIndexes(context,objects);
    std::vector<Entry> boxes;
    for (const auto& object : objects) {
        auto& input = object->input;
        input.firstEdge = static_cast<int>(edges.size());
        input.edgeCount = static_cast<int>(object->edges.size());
        edges.insert(edges.end(),object->edges.begin(),object->edges.end());
        if (inputs.size() >= static_cast<size_t>(split)) {
            boxes.emplace_back(box(input),static_cast<int>(inputs.size()));
        }
        inputs.push_back(input);
    }
    const auto treeStart = Clock::now();
    const RTree tree(boxes.begin(),boxes.end());
    const double treeMs = elapsed(treeStart);
    Buffer<CandidateObject> deviceObjects(inputs.size());
    Buffer<CandidateEdge> deviceEdges(edges.size());
    deviceObjects.upload(inputs);
    deviceEdges.upload(edges);
    constexpr int batchSize = 262144;
    Buffer<int2> devicePairs(batchSize);
    Buffer<int> deviceResults(batchSize);
    std::vector<int2> candidates;
    candidates.reserve(batchSize);
    std::vector<int> hits(batchSize);
    const double candidateSetup = elapsed(setup);
    logDeviceMemory("both_loaded");
    std::cout << "setup," << resident.loadUs/1000.0 << ',' << candidateSetup << ',' << edges.size()
              << ',' << treeMs << ',' << split << ',' << boxes.size() << std::endl;

    for (int trial = -12; trial < repeats; ++trial) {
        Pairs sharedPairs, candidatePairs, parallelPairs;
        auto sharedRun = [&] {
            const auto wallStart = wallTimeMs();
            auto result = shared.queryPair(resident.handle);
            const auto wallEnd = wallTimeMs();
            sharedPairs.reserve(result.pairs.size());
            for (const auto& p : result.pairs) sharedPairs.emplace_back(p.object_id_mesh1,p.object_id_mesh2);
            if (trial >= 0 || trial == -12) std::cout << (trial < 0 ? "first-shared," : "shared,") << trial << ',' << result.queryExecutionUs/1000.0
                << ',' << result.phaseTimings.traceUs/1000.0 << ',' << result.rawHitCount
                << ',' << result.phaseTimings.estimationUs/1000.0
                << ',' << result.phaseTimings.compactionUs/1000.0
                << ',' << result.phaseTimings.downloadUs/1000.0
                << ',' << wallStart << ',' << wallEnd << '\n';
        };
        auto candidateRun = [&](int workers, Pairs& output) {
            const auto wallStart = wallTimeMs();
            const auto start = Clock::now();
            auto filterStart = start;
            size_t candidateCount = 0;
            long long calls = 0;
            double tracing = 0, filtering = 0;
            double upload = 0, download = 0, host = 0;
            auto refine = [&] {
                filtering += elapsed(filterStart);
                const auto uploadStart = Clock::now();
                devicePairs.upload(candidates);
                upload += elapsed(uploadStart);
                const int count = static_cast<int>(candidates.size());
                const auto params = candidateParams(deviceObjects.get(),deviceEdges.get(),
                                                   devicePairs.get(),deviceResults.get(),count);
                const auto traceStart = Clock::now();
                launcher.launchMesh1ToMesh2(params,count);
                tracing += elapsed(traceStart);
                const auto downloadStart = Clock::now();
                CUDA_CHECK(cudaMemcpy(hits.data(),deviceResults.get(),count*sizeof(int),cudaMemcpyDeviceToHost));
                download += elapsed(downloadStart);
                const auto hostStart = Clock::now();
                for (int i = 0; i < count; ++i) {
                    calls += std::abs(hits[i])-1;
                    if (hits[i] > 0) {
                        output.emplace_back(objects[candidates[i].x]->id,objects[candidates[i].y]->id);
                    }
                }
                candidateCount += candidates.size();
                candidates.clear();
                host += elapsed(hostStart);
                filterStart = Clock::now();
            };
            auto enumerate = [&](int first, int last, auto&& emit) {
                for (int i = first; i < last; ++i) {
                    for (auto it = tree.qbegin(bgi::intersects(box(inputs[i]))); it != tree.qend(); ++it) {
                        emit(make_int2(i,it->second));
                    }
                }
            };
            auto enqueue = [&](int2 pair) {
                candidates.push_back(pair);
                if (candidates.size() == batchSize) refine();
            };
            if (workers == 1) {
                enumerate(0,split,enqueue);
            } else {
                // Query the immutable tree in parallel. Stage one wave of candidates
                // at a time and preserve A order for the existing GPU batches.
                constexpr int maxObjectsPerWorker = 8192;
                std::vector<std::future<std::vector<int2>>> jobs;
                jobs.reserve(workers);
                for (int first = 0; first < split;) {
                    const int waveSize = std::min(split-first,workers*maxObjectsPerWorker);
                    const int stride = (waveSize+workers-1)/workers;
                    const int end = first+waveSize;
                    for (; first < end; first += stride) {
                        jobs.push_back(std::async(std::launch::async,[&, begin = first,
                            last = std::min(first+stride,end)] {
                            std::vector<int2> found;
                            enumerate(begin,last,[&](int2 pair) { found.push_back(pair); });
                            return found;
                        }));
                    }
                    first = end;
                    for (auto& job : jobs) {
                        for (auto pair : job.get()) enqueue(pair);
                    }
                    jobs.clear();
                }
            }
            if (!candidates.empty()) refine();
            else filtering += elapsed(filterStart);
            const double total = elapsed(start);
            if (trial >= 0 || trial == -12) std::cout << (trial < 0 ? "first-" : "") << (workers == 1 ? "candidate," : "parallel,") << trial << ',' << total << ',' << tracing
                << ',' << filtering << ',' << candidateCount << ',' << calls
                << ',' << upload << ',' << download << ',' << host
                << ',' << wallStart << ',' << wallTimeMs() << '\n';
        };
        auto serialRun = [&] { if (!parallelOnly) candidateRun(1,candidatePairs); };
        auto parallelRun = [&] { if (threads > 1) candidateRun(threads,parallelPairs); };
        if (parallelOnly) {
            const bool reverse = std::getenv("PIERCE_COMPARE_PARALLEL_FIRST") != nullptr;
            if ((trial % 2 == 0) != reverse) { sharedRun(); parallelRun(); }
            else { parallelRun(); sharedRun(); }
        } else switch ((trial+12)%3) {
            case 0: sharedRun(); serialRun(); parallelRun(); break;
            case 1: serialRun(); parallelRun(); sharedRun(); break;
            case 2: parallelRun(); sharedRun(); serialRun(); break;
        }

        canonicalize(sharedPairs);
        auto verify = [&](Pairs& candidatePairs, const char* name) {
            canonicalize(candidatePairs);
            if (sharedPairs == candidatePairs) return;
            const auto [a,b] = std::mismatch(sharedPairs.begin(),sharedPairs.end(),
                                             candidatePairs.begin(),candidatePairs.end());
            std::cerr << name << " pair set mismatch on trial " << trial << ": shared size " << sharedPairs.size()
                      << ", candidate size " << candidatePairs.size();
            if (a != sharedPairs.end()) std::cerr << ", shared first difference " << a->first << ':' << a->second;
            if (b != candidatePairs.end()) std::cerr << ", candidate first difference " << b->first << ':' << b->second;
            std::cerr << '\n';
            throw std::runtime_error("Full pair verification failed");
        };
        if (!parallelOnly) verify(candidatePairs,"Serial");
        if (threads > 1) verify(parallelPairs,"Parallel");
        if (trial >= 0 || trial == -12) std::cout << (trial < 0 ? "first-verified," : "verified,") << trial << ',' << sharedPairs.size() << std::endl;
    }
}
}

int main(int argc,char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr << "Usage: pierce_bvh_compare a.pre b.pre ptx-directory repetitions [filter-threads] [--parallel-only]\n";
        return 2;
    }
    try {
        const int threads = argc >= 6 ? std::stoi(argv[5]) : 1;
        if (threads < 1 || threads > 256) throw std::runtime_error("Filter threads must be in [1, 256]");
        const bool parallelOnly = argc == 7;
        if (parallelOnly && (std::string(argv[6]) != "--parallel-only" || threads == 1)) {
            throw std::runtime_error("Parallel-only mode requires multiple filter threads");
        }
        run(argv,threads,parallelOnly);
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
