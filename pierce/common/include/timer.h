#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

class PerformanceTimer {
public:
    PerformanceTimer();
    
    void start(const std::string& phaseName);
    
    void next(const std::string& phaseName);
    
    void finish(const std::string& filename);
    
    long long getPhaseDuration(const std::string& phaseName) const;
    
    long long getTotalDuration() const;

private:
    struct Phase {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        long long duration_us;
    };
    
    std::vector<Phase> phases;
    std::chrono::high_resolution_clock::time_point total_start;
    std::chrono::high_resolution_clock::time_point total_end;
    bool is_running;
    
    void endCurrentPhase();
    void printResults(const std::string& filename) const;
    void writeResultsToFile(const std::string& filename) const;
};
