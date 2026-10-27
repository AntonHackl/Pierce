#include "../runtime/QueryRuntime.h"
#include "../ptx_utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

using Pair = std::pair<int, int>;
using Pairs = std::vector<Pair>;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static Pairs canonical(const std::vector<MeshQueryResult>& input) {
    Pairs result;
    result.reserve(input.size());
    for (const auto& p : input) result.emplace_back(p.object_id_mesh1, p.object_id_mesh2);
    std::sort(result.begin(), result.end());
    require(std::adjacent_find(result.begin(), result.end()) == result.end(), "Duplicate pair");
    return result;
}
static Pairs baseline(const char* path) {
    std::ifstream file(path);
    require(bool(file), "Baseline CSV missing");
    std::string line; std::getline(file, line);
    Pairs result;
    while (std::getline(file, line)) {
        std::istringstream row(line); int a,b; char comma;
        require(bool(row >> a >> comma >> b) && comma == ',', "Malformed baseline");
        result.emplace_back(a,b);
    }
    std::sort(result.begin(), result.end());
    require(std::adjacent_find(result.begin(), result.end()) == result.end(), "Duplicate baseline");
    return result;
}
static void arithmeticTests() {
    using namespace pierce;
    MemoryBudget budget([](const std::string&) { return MemorySnapshot{"fake", 8ULL << 30, 2ULL << 30}; });
    const auto m = budget.snapshot();
    require(m.reserve() == (8ULL << 30) / 10, "Wrong proportional reserve");
    budget.check(m.available(), "boundary");
    bool denied = false;
    try { budget.check(m.available() + 1, "boundary"); } catch (const MemoryBudgetExceeded&) { denied = true; }
    require(denied, "Budget boundary allowed");
    require(MemorySnapshot{"fake", 1ULL << 30, 1ULL << 30}.reserve() == (512ULL << 20), "Wrong minimum reserve");
    require(MemorySnapshot{"fake", 1ULL << 30, 1}.available() == 0, "Available bytes underflow");
    for (bool multiply : {false, true}) {
        bool overflow = false;
        try {
            if (multiply) checkedMultiply(std::numeric_limits<Bytes>::max(), 2);
            else checkedAdd(std::numeric_limits<Bytes>::max(), 1);
        } catch (const std::overflow_error&) { overflow = true; }
        require(overflow, "Byte arithmetic overflow accepted");
    }
}
int main(int argc, char** argv) {
    if (argc != 7) { std::cerr << "Usage: pierce_multi_verify small-a small-b large-a large-b small.csv large.csv\n"; return 1; }
    try {
        arithmeticTests();
        auto smallExpected = baseline(argv[5]), largeExpected = baseline(argv[6]);
        require(smallExpected.size() == 5000, "Small baseline count changed");
        Pairs analytical;
        for (int g = 0; g < 200; ++g) for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b)
            analytical.emplace_back(g * 5 + a, g * 5 + b);
        require(analytical == smallExpected, "Small baseline differs from analytical pairs");
        std::string denyStage;
        int denyAfter = 0;
        auto provider = [&](const std::string& stage) {
            auto m = pierce::deviceMemory(stage);
            if (stage == denyStage && --denyAfter <= 0) m.free = 0;
            return m;
        };
        pierce::QueryRuntime runtime(detectPTXPath("mesh_overlap_edges.ptx"), provider);
        const auto before = runtime.memory();
        const auto small = runtime.loadPair("small", argv[1], argv[2]).handle;
        const auto large = runtime.loadPair("large", argv[3], argv[4]).handle;
        const auto owned = runtime.datasetBytes();
        const auto both = runtime.memory();
        std::cout << "Memory before loading/free: " << before.free << "; both resident/free: " << both.free
                  << "; resident owned: " << owned << '\n';
        pierce::Bytes afterWarm = 0;
        for (int i = 0; i < 100; ++i) {
            const bool useLarge = i % 2;
            auto result = runtime.queryPair(useLarge ? large : small);
            require(canonical(result.pairs) == (useLarge ? largeExpected : smallExpected), "Pair set changed while switching");
            require(runtime.datasetBytes() == owned, "Resident memory changed during switching");
            if (i == 3) afterWarm = runtime.memory().free;
            if (i == 1) for (const auto& c : runtime.checkpoints())
                std::cout << "Checkpoint " << c.stage << ": free=" << c.memory.free << ", required=" << c.required << '\n';
        }
        require(runtime.reads() == 4 && runtime.uploads() == 4 && runtime.builds() == 4, "Switching rebuilt data");
        require(runtime.pair(small).completedQueries == 50 && runtime.pair(large).completedQueries == 50, "Per-pair counts changed");
        require(runtime.memory().free + (16ULL << 20) >= afterWarm, "Memory grew during switching");
        // Fail after the first collection has loaded: pending pair must unwind.
        const auto survivorBytes = runtime.datasetBytes();
        for (const auto& failure : std::vector<std::pair<std::string,int>>{{"dataset_upload",1},{"dataset_upload",2},{"bvh_build",1},{"edge_upload",1}}) {
            denyStage = failure.first; denyAfter = failure.second;
            bool denied = false;
            try { runtime.loadPair("rejected", argv[1], argv[2]); } catch (const pierce::MemoryBudgetExceeded&) { denied = true; }
            require(denied && runtime.pairs().size() == 2 && runtime.datasetBytes() == survivorBytes, "Failed load changed residents");
            denyStage.clear();
            require(canonical(runtime.queryPair(small).pairs) == smallExpected, "Query after failed load changed");
        }
        for (const char* stage : {"query_counts", "query_results_and_sort"}) {
            denyStage = stage; denyAfter = 1;
            auto freeBefore = runtime.memory().free;
            bool denied = false;
            try { runtime.queryPair(large); } catch (const pierce::MemoryBudgetExceeded&) { denied = true; }
            require(denied && runtime.datasetBytes() == survivorBytes, "Query budget did not reject cleanly");
            denyStage.clear();
            require(runtime.memory().free + (16ULL << 20) >= freeBefore, "Rejected query leaked buffers");
            require(canonical(runtime.queryPair(large).pairs) == largeExpected, "Query after rejection changed");
        }
        runtime.unloadPair(large);
        require(canonical(runtime.queryPair(small).pairs) == smallExpected, "Unloading large changed small");
        const auto smallOnly = runtime.datasetBytes();
        for (int i = 0; i < 3; ++i) {
            auto handle = runtime.loadPair("large", argv[3], argv[4]).handle;
            require(handle != large, "Reload reused stale handle");
            require(canonical(runtime.queryPair(handle).pairs) == largeExpected, "Reload pair set changed");
            runtime.unloadPair(handle);
            require(runtime.datasetBytes() == smallOnly, "Reload retained allocations");
        }
        bool rejected = false;
        try { runtime.queryPair(large); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Stale handle accepted");
        try { runtime.loadPair("bad", argv[1], std::string(argv[2]) + ".missing"); }
        catch (const std::exception&) {}
        require(runtime.pairs().size() == 1 && runtime.datasetBytes() == smallOnly, "Failed pair published");
        runtime.unloadPair(small);
        require(runtime.datasetBytes() == 0, "Unload left resident buffers");
        std::cout << "Memory after unload/free: " << runtime.memory().free << '\n';
        std::cout << "PASS: 100 alternating full-pair comparisons, isolated reload, stable residency, budget arithmetic, "
                     "upload/BVH/edge/count/result rejections, recovery, stale handles. Large pairs: " << largeExpected.size() << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
