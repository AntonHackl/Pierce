#pragma once

#include <atomic>
#include <cstdint>

namespace tdbase {

enum class preprocess_target : std::uint8_t {
    nuclei = 0,
    vessel = 1,
};

inline std::atomic<std::uint64_t> g_wrapper_total_ns{0};
inline std::atomic<std::uint64_t> g_write_ns{0};
inline std::atomic<std::uint64_t> g_finalize_ns{0};
inline std::atomic<std::uint64_t> g_wrapper_total_nuclei_ns{0};
inline std::atomic<std::uint64_t> g_write_nuclei_ns{0};
inline std::atomic<std::uint64_t> g_finalize_nuclei_ns{0};
inline std::atomic<std::uint64_t> g_wrapper_total_vessel_ns{0};
inline std::atomic<std::uint64_t> g_write_vessel_ns{0};
inline std::atomic<std::uint64_t> g_finalize_vessel_ns{0};

inline void reset_preprocess_timing() {
    g_wrapper_total_ns.store(0);
    g_write_ns.store(0);
    g_finalize_ns.store(0);
    g_wrapper_total_nuclei_ns.store(0);
    g_write_nuclei_ns.store(0);
    g_finalize_nuclei_ns.store(0);
    g_wrapper_total_vessel_ns.store(0);
    g_write_vessel_ns.store(0);
    g_finalize_vessel_ns.store(0);
}

inline double ns_to_ms(std::uint64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

inline double wrapper_total_ms() {
    return ns_to_ms(g_wrapper_total_ns.load());
}

inline double write_ms() {
    return ns_to_ms(g_write_ns.load());
}

inline double finalize_ms() {
    return ns_to_ms(g_finalize_ns.load());
}

inline std::atomic<std::uint64_t>& wrapper_target_ns(preprocess_target target) {
    return target == preprocess_target::nuclei ? g_wrapper_total_nuclei_ns : g_wrapper_total_vessel_ns;
}

inline std::atomic<std::uint64_t>& write_target_ns(preprocess_target target) {
    return target == preprocess_target::nuclei ? g_write_nuclei_ns : g_write_vessel_ns;
}

inline std::atomic<std::uint64_t>& finalize_target_ns(preprocess_target target) {
    return target == preprocess_target::nuclei ? g_finalize_nuclei_ns : g_finalize_vessel_ns;
}

inline double wrapper_total_ms(preprocess_target target) {
    return ns_to_ms(wrapper_target_ns(target).load());
}

inline double write_ms(preprocess_target target) {
    return ns_to_ms(write_target_ns(target).load());
}

inline double finalize_ms(preprocess_target target) {
    return ns_to_ms(finalize_target_ns(target).load());
}

inline double preprocessing_only_ms() {
    return wrapper_total_ms() + write_ms() + finalize_ms();
}

inline double preprocessing_only_ms(preprocess_target target) {
    return wrapper_total_ms(target) + write_ms(target) + finalize_ms(target);
}

}  // namespace tdbase
