#include "../runtime/QueryRuntime.h"
#include "BinaryIO.h"
#include "../ptx_utils.h"
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

using Pairs = std::set<std::pair<int, int>>;
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static Pairs pairs(const pierce::RuntimeResult& result) {
    Pairs out;
    for (const auto& pair : result.pairs) out.emplace(pair.object_id_mesh1, pair.object_id_mesh2);
    require(out.size() == result.pairs.size(), "Duplicate results");
    return out;
}
static size_t freeBytes() {
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: pierce_runtime_verify a.pre b.pre baseline.csv zero.pre\n";
        return 1;
    }
    try {
        Pairs expected, baseline;
        for (int group = 0; group < 200; ++group)
            for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b)
                expected.emplace(group * 5 + a, group * 5 + b);
        std::ifstream csv(argv[3]);
        require(bool(csv), "Baseline CSV missing");
        std::string line;
        std::getline(csv, line);
        while (std::getline(csv, line)) {
            std::istringstream row(line);
            int a, b; char comma;
            require(bool(row >> a >> comma >> b) && comma == ',', "Invalid baseline CSV");
            require(baseline.emplace(a, b).second, "Duplicate baseline pair");
        }
        require(baseline == expected, "Baseline differs from analytical cube result");
        pierce::QueryRuntime runtime(detectPTXPath("mesh_overlap_edges.ptx"));
        bool rejected = false;
        try { runtime.query("bad", "bad"); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Query before load accepted");
        runtime.load(argv[1], argv[2]);
        const auto a = runtime.handleA(), b = runtime.handleB();
        const auto residentBytes = runtime.datasetBytes();
        require(residentBytes > 0, "Missing resident allocations");
        size_t afterSecond = 0;
        for (int i = 0; i < 100; ++i) {
            require(pairs(runtime.query(a, b)) == baseline, "Resident results differ from baseline");
            require(runtime.handleA() == a && runtime.handleB() == b, "Handles changed");
            require(runtime.datasetBytes() == residentBytes, "Resident allocations grew");
            if (i == 1) afterSecond = freeBytes();
        }
        require(runtime.reads() == 2 && runtime.uploads() == 2 && runtime.builds() == 2,
                "Repeated queries rebuilt datasets");
        const auto afterHundred = freeBytes();
        // Driver caches and other desktop GPU clients may move global free memory.
        require(afterHundred + (16ULL << 20) >= afterSecond, "GPU free memory declined by over 16 MiB");
        runtime.unload();
        require(runtime.datasetBytes() == 0 && !runtime.loaded(), "Unload retained dataset allocations");
        const auto unloadedBaseline = freeBytes();
        for (int cycle = 0; cycle < 5; ++cycle) {
            runtime.load(argv[1], argv[2]);
            rejected = false;
            try { runtime.query(a, b); } catch (const std::exception&) { rejected = true; }
            require(rejected, "Stale handles accepted");
            require(pairs(runtime.query(runtime.handleA(), runtime.handleB())) == expected, "Reload differs");
            runtime.unload();
            require(runtime.datasetBytes() == 0, "Unload did not release tracked allocations");
        }
        require(freeBytes() + (16ULL << 20) >= unloadedBaseline, "Reload cycles leaked GPU memory");
        rejected = false;
        try { runtime.load(argv[1], std::string(argv[2]) + ".missing"); }
        catch (const std::exception&) { rejected = true; }
        require(rejected && !runtime.loaded() && runtime.datasetBytes() == 0, "Pair load was not atomic");
        auto zero = RaySpace::IO::readBinaryFile(argv[2]);
        for (auto& vertex : zero.vertices) vertex.x += 10000;
        for (auto& edge : zero.edges.edgeStarts) edge.x += 10000;
        for (auto& edge : zero.edges.edgeEnds) edge.x += 10000;
        zero.grid.hasGrid = false;
        require(RaySpace::IO::writeBinaryFile(argv[4], zero), "Could not write zero fixture");
        runtime.load(argv[1], argv[4]);
        require(runtime.query(runtime.handleA(), runtime.handleB()).pairs.empty(), "Zero fixture returned pairs");
        runtime.unload();
        std::cout << "PASS: 100 resident queries, full baseline/analytical pair equality, stable handles and indexes, "
                     "5 reload cycles, stale handles, atomic failure, zero results.\n"
                  << "Resident dataset GPU bytes: " << residentBytes << "; free bytes after queries 2/100: "
                  << afterSecond << "/" << afterHundred << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
