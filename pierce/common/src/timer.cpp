#include "timer.h"
#include <iomanip>
#include <unordered_map>
#include <cctype>

PerformanceTimer::PerformanceTimer() : is_running(false) {
}

void PerformanceTimer::start(const std::string& phaseName) {
    phases.clear();
    total_start = std::chrono::high_resolution_clock::now();
    is_running = true;
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = total_start;
    phases.push_back(phase);
}

void PerformanceTimer::next(const std::string& phaseName) {
    if (!is_running) {
        std::cerr << "Timer not started! Call start() first." << std::endl;
        return;
    }
    
    endCurrentPhase();
    
    Phase phase;
    phase.name = phaseName;
    phase.start_time = std::chrono::high_resolution_clock::now();
    phases.push_back(phase);
}

void PerformanceTimer::finish(const std::string& filename) {
    if (!is_running) {
        std::cerr << "Timer not started! Call start() first." << std::endl;
        return;
    }
    
    endCurrentPhase();
    total_end = std::chrono::high_resolution_clock::now();
    is_running = false;

    printResults(filename);
}

void PerformanceTimer::endCurrentPhase() {
    if (!phases.empty() && phases.back().end_time == std::chrono::high_resolution_clock::time_point{}) {
        phases.back().end_time = std::chrono::high_resolution_clock::now();
        phases.back().duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            phases.back().end_time - phases.back().start_time).count();
    }
}

long long PerformanceTimer::getPhaseDuration(const std::string& phaseName) const {
    for (const auto& phase : phases) {
        if (phase.name == phaseName) {
            return phase.duration_us;
        }
    }
    return -1; // Phase not found
}

long long PerformanceTimer::getTotalDuration() const {
    if (total_start == std::chrono::high_resolution_clock::time_point{} || 
        total_end == std::chrono::high_resolution_clock::time_point{}) {
        return -1;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
}

void PerformanceTimer::printResults(const std::string& filename) const {
    std::cout << "\n=== Detailed Performance Summary ===" << std::endl;
    
    for (const auto& phase : phases) {
        std::cout << std::left << std::setw(30) << (phase.name + ":") 
                  << std::right << std::setw(20) << phase.duration_us << " microseconds ("
                  << std::fixed << std::setprecision(2) << (double)phase.duration_us / 1000.0 << " ms)" << std::endl;
    }
    
    long long total_us = getTotalDuration();
    if (total_us > 0) {
        std::cout << std::left << std::setw(25) << "Total Execution Time:" 
                  << std::right << std::setw(10) << total_us << " microseconds ("
                  << std::fixed << std::setprecision(2) << (double)total_us / 1000.0 << " ms)" << std::endl;
    }
    
    writeResultsToFile(filename);
}

void PerformanceTimer::writeResultsToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Warning: Could not write timing results to file." << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"phases\": {\n";

    // To support multiple occurrences of the same phase name (e.g. Query per task),
    // emit numbered, lowercase keys like "query_1", "query_2", "output_1" etc.
    std::unordered_map<std::string,int> counts;
    for (size_t i = 0; i < phases.size(); ++i) {
        const auto& phase = phases[i];
        std::string name = phase.name;
        // convert to lowercase
        std::string lname;
        lname.reserve(name.size());
        for (char c : name) lname.push_back(std::tolower(static_cast<unsigned char>(c)));

        int idx = ++counts[lname];
        std::string key = lname + "_" + std::to_string(idx);

        file << "    \"" << key << "\": {\n";
        file << "      \"duration_us\": " << phase.duration_us << ",\n";
        file << "      \"duration_ms\": " << std::fixed << std::setprecision(2) << (double)phase.duration_us / 1000.0 << "\n";
        file << "    }";
        if (i < phases.size() - 1) file << ",";
        file << "\n";
    }

    file << "  },\n";
    
    long long total_us = getTotalDuration();
    if (total_us > 0) {
        file << "  \"total\": {\n";
        file << "    \"duration_us\": " << total_us << ",\n";
        file << "    \"duration_ms\": " << std::fixed << std::setprecision(2) << (double)total_us / 1000.0 << "\n";
        file << "  }\n";
    } else {
        file << "  \"total\": null\n";
    }
    
    file << "}\n";
    
    file.close();
    std::cout << "Performance timing written to " << filename << std::endl;
}
