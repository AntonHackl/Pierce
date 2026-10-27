#include "../runtime/QueryRuntime.h"
#include "../ptx_utils.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

// PropertyTree is already supplied by Pierce's Boost dependency. Only request
// parsing uses it; responses explicitly encode numeric and boolean JSON types.
static std::string quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

static std::string memoryJson(const pierce::MemorySnapshot& m) {
    std::ostringstream out;
    out << "{\"device\":" << quote(m.device) << ",\"totalBytes\":" << m.total
        << ",\"freeBytes\":" << m.free << ",\"reservedBytes\":" << m.reserve() << "}";
    return out.str();
}
static std::string pairJson(const pierce::ResidentPair& p) {
    std::ostringstream out;
    out << "{\"key\":" << quote(p.key) << ",\"pairHandle\":" << quote(p.handle)
        << ",\"datasetLoadUs\":" << p.loadUs << ",\"datasetGpuBytes\":" << p.allocatedBytes()
        << ",\"completedQueries\":" << p.completedQueries << ",\"lastResult\":";
    if (!p.completedQueries) out << "null";
    else out << "{\"resultCount\":" << p.resultCount << ",\"queryExecutionUs\":" << p.queryExecutionUs
             << ",\"benchmarkPhaseSumUs\":" << p.benchmarkPhaseSumUs << ",\"rawHitCount\":" << p.rawHitCount
             << ",\"resultBufferBytes\":" << p.resultBufferBytes << "}";
    out << "}";
    return out.str();
}
static std::string statusJson(pierce::QueryRuntime* runtime, long long initializationUs) {
    std::ostringstream out;
    out << "{\"runtimeInitializationUs\":" << initializationUs << ",\"pairs\":[";
    bool first = true;
    if (runtime) for (const auto& entry : runtime->pairs()) {
        if (!first) out << ',';
        first = false;
        out << pairJson(*entry.second);
    }
    out << "],\"datasetReads\":" << (runtime ? runtime->reads() : 0)
        << ",\"datasetUploads\":" << (runtime ? runtime->uploads() : 0)
        << ",\"indexBuilds\":" << (runtime ? runtime->builds() : 0)
        << ",\"pipelineCreations\":" << (runtime ? 1 : 0)
        << ",\"completedQueries\":" << (runtime ? runtime->queries() : 0)
        << ",\"datasetGpuBytes\":" << (runtime ? runtime->datasetBytes() : 0)
        << ",\"runtimeGpuBytes\":" << (runtime ? runtime->runtimeBytes() : 0)
        << ",\"temporaryQueryGpuBytes\":0,\"memory\":" << (runtime ? memoryJson(runtime->memory()) : "null")
        << ",\"checkpoints\":[";
    first = true;
    if (runtime) for (const auto& point : runtime->checkpoints()) {
        if (!first) out << ',';
        first = false;
        out << "{\"stage\":" << quote(point.stage) << ",\"requiredBytes\":" << point.required
            << ",\"memory\":" << memoryJson(point.memory) << "}";
    }
    out << "]}";
    return out.str();
}
int main(int argc, char** argv) {
    std::string ptx = detectPTXPath("mesh_overlap_edges.ptx");
    if (argc == 3 && std::string(argv[1]) == "--ptx") ptx = argv[2];
    else if (argc != 1) { std::cerr << "Usage: pierce_worker [--ptx file]\n"; return 1; }
    std::ostream protocol(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());
    std::unique_ptr<pierce::QueryRuntime> runtime;
    long long initializationUs = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string id;
        bool shutdown = false;
        try {
            boost::property_tree::ptree request;
            std::istringstream input(line);
            boost::property_tree::read_json(input, request);
            id = request.get<std::string>("id");
            if (id.empty() || request.get<int>("version") != 2)
                throw std::runtime_error("Expected nonempty id and protocol version 2");
            const auto command = request.get<std::string>("command");
            if (command == "load") {
                auto key = request.get<std::string>("key");
                auto a = request.get<std::string>("mesh1"), b = request.get<std::string>("mesh2");
                if (key.empty() || a.empty() || b.empty()) throw std::runtime_error("Key and both paths are required");
                if (!runtime) {
                    const auto start = std::chrono::steady_clock::now();
                    runtime = std::make_unique<pierce::QueryRuntime>(ptx);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    initializationUs = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start).count();
                }
                runtime->loadPair(key, a, b);
            } else if (command == "query") {
                if (!runtime) throw std::runtime_error("Load a pair before querying");
                runtime->queryPair(request.get<std::string>("pairHandle"));
            } else if (command == "unload") {
                if (!runtime) throw std::runtime_error("Pair is unloaded");
                runtime->unloadPair(request.get<std::string>("pairHandle"));
            } else if (command == "shutdown") {
                runtime.reset();
                shutdown = true;
            } else if (command != "status") throw std::runtime_error("Unknown command");
            protocol << "{\"version\":2,\"id\":" << quote(id) << ",\"ok\":true,\"data\":"
                     << statusJson(runtime.get(), initializationUs) << "}\n" << std::flush;
        } catch (const pierce::MemoryBudgetExceeded& error) {
            const auto& point = error.checkpoint;
            protocol << "{\"version\":2,\"id\":" << quote(id)
                << ",\"ok\":false,\"error\":" << quote(error.what())
                << ",\"code\":\"memory_budget_exceeded\",\"details\":{\"stage\":" << quote(point.stage)
                << ",\"requiredBytes\":" << point.required << ",\"availableBytes\":" << point.memory.available()
                << ",\"reservedBytes\":" << point.memory.reserve() << "},\"data\":"
                << statusJson(runtime.get(), initializationUs) << "}\n" << std::flush;
        } catch (const std::exception& error) {
            protocol << "{\"version\":2,\"id\":" << quote(id) << ",\"ok\":false,\"code\":\"native_error\",\"error\":"
                     << quote(error.what()) << ",\"data\":" << statusJson(runtime.get(), initializationUs) << "}\n" << std::flush;
        }
        if (shutdown) break;
    }
    runtime.reset();
    return 0;
}
