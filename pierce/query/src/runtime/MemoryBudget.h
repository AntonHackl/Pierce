#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pierce {
using Bytes = std::uint64_t;
inline Bytes checkedAdd(Bytes a, Bytes b) {
    if (b > std::numeric_limits<Bytes>::max() - a) throw std::overflow_error("GPU byte count overflow");
    return a + b;
}
inline Bytes checkedMultiply(Bytes a, Bytes b) {
    if (a && b > std::numeric_limits<Bytes>::max() / a) throw std::overflow_error("GPU byte count overflow");
    return a * b;
}
struct MemorySnapshot {
    std::string device;
    Bytes total = 0, free = 0;
    Bytes reserve() const { return std::max<Bytes>(512ULL << 20, total / 10); }
    Bytes available() const { return free > reserve() ? free - reserve() : 0; }
};
using MemoryProvider = std::function<MemorySnapshot(const std::string&)>;
MemorySnapshot deviceMemory(const std::string& stage);
struct MemoryCheckpoint {
    std::string stage;
    Bytes required;
    MemorySnapshot memory;
};
class MemoryBudgetExceeded : public std::runtime_error {
public:
    MemoryCheckpoint checkpoint;
    explicit MemoryBudgetExceeded(MemoryCheckpoint point)
        : std::runtime_error("Not enough GPU memory for " + point.stage +
              ". Unload another workload or use a lower-selectivity dataset."), checkpoint(std::move(point)) {}
};
class MemoryBudget {
    MemoryProvider provider_;
    std::vector<MemoryCheckpoint> checkpoints_;
public:
    explicit MemoryBudget(MemoryProvider provider = {}) : provider_(provider ? std::move(provider) : deviceMemory) {}
    MemorySnapshot snapshot() const { return provider_("status"); }
    void begin() { checkpoints_.clear(); }
    void check(Bytes required, const std::string& stage) {
        MemoryCheckpoint point{stage, required, provider_(stage)};
        checkpoints_.push_back(point);
        if (required > point.memory.available()) throw MemoryBudgetExceeded(point);
    }
    const std::vector<MemoryCheckpoint>& checkpoints() const { return checkpoints_; }
};
}
