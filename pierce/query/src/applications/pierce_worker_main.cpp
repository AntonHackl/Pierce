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

int main(int argc, char** argv) {
    std::string ptx = detectPTXPath("mesh_overlap_edges.ptx");
    if (argc == 3 && std::string(argv[1]) == "--ptx") ptx = argv[2];
    else if (argc != 1) { std::cerr << "Usage: pierce_worker [--ptx file]\n"; return 1; }
    // Existing native helpers write to cout. Keep an independent protocol stream
    // before redirecting cout, so no library diagnostic can corrupt framing.
    std::ostream protocol(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());
    std::unique_ptr<pierce::QueryRuntime> runtime;
    long long initializationUs = 0, loadUs = 0;
    auto ensureRuntime = [&]() {
        if (!runtime) {
            auto start = std::chrono::steady_clock::now();
            runtime = std::make_unique<pierce::QueryRuntime>(ptx);
            CUDA_CHECK(cudaDeviceSynchronize());
            initializationUs = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
        }
    };
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string id;
        bool shutdown = false;
        try {
            boost::property_tree::ptree request;
            std::istringstream input(line);
            boost::property_tree::read_json(input, request);
            id = request.get<std::string>("id");
            if (id.empty() || request.get<int>("version") != 1)
                throw std::runtime_error("Expected nonempty id and protocol version 1");
            auto command = request.get<std::string>("command");
            std::ostringstream data;
            if (command == "load") {
                const auto a = request.get<std::string>("mesh1");
                const auto b = request.get<std::string>("mesh2");
                if (a.empty() || b.empty()) throw std::runtime_error("Both paths are required");
                ensureRuntime();
                if (!runtime->loaded()) loadUs = runtime->load(a, b);
                data << "{\"handleA\":" << quote(runtime->handleA()) << ",\"handleB\":" << quote(runtime->handleB())
                     << ",\"datasetLoadUs\":" << loadUs << ",\"runtimeInitializationUs\":" << initializationUs << "}";
            } else if (command == "query") {
                if (!runtime) throw std::runtime_error("Load datasets before querying");
                auto result = runtime->query(request.get<std::string>("handleA"), request.get<std::string>("handleB"));
                data << "{\"resultCount\":" << result.pairs.size() << ",\"queryExecutionUs\":" << result.queryExecutionUs
                     << ",\"benchmarkPhaseSumUs\":" << result.benchmarkPhaseSumUs << "}";
            } else if (command == "unload") {
                if (runtime) runtime->unload();
                loadUs = 0;
                data << "{\"loaded\":false}";
            } else if (command == "status") {
                const bool loaded = runtime && runtime->loaded();
                data << "{\"loaded\":" << (loaded ? "true" : "false")
                     << ",\"handleA\":" << quote(runtime ? runtime->handleA() : "")
                     << ",\"handleB\":" << quote(runtime ? runtime->handleB() : "")
                     << ",\"datasetReads\":" << (runtime ? runtime->reads() : 0)
                     << ",\"datasetUploads\":" << (runtime ? runtime->uploads() : 0)
                     << ",\"indexBuilds\":" << (runtime ? runtime->builds() : 0)
                     << ",\"pipelineCreations\":" << (runtime ? 1 : 0)
                     << ",\"completedQueries\":" << (runtime ? runtime->queries() : 0)
                     << ",\"datasetGpuBytes\":" << (runtime ? runtime->datasetBytes() : 0)
                     << ",\"runtimeGpuBytes\":" << (runtime ? runtime->runtimeBytes() : 0)
                     << ",\"temporaryQueryGpuBytes\":0}";
            } else if (command == "shutdown") {
                runtime.reset();
                data << "{}";
                shutdown = true;
            } else throw std::runtime_error("Unknown command");
            protocol << "{\"version\":1,\"id\":" << quote(id) << ",\"ok\":true,\"data\":" << data.str() << "}\n" << std::flush;
        } catch (const std::exception& error) {
            protocol << "{\"version\":1,\"id\":" << quote(id) << ",\"ok\":false,\"error\":" << quote(error.what()) << "}\n" << std::flush;
        }
        if (shutdown) break;
    }
    runtime.reset();
    return 0;
}
