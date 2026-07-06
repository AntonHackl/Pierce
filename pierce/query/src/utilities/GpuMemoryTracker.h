#pragma once

#include <cstddef>
#include <string>
#include <vector>

class PerformanceTimer;

class GpuMemoryTracker {
public:
    struct Sample {
        std::string checkpoint;
        size_t usedBytes = 0;
        size_t freeBytes = 0;
        size_t totalBytes = 0;
    };

    explicit GpuMemoryTracker(bool enabled = false);

    bool isEnabled() const { return enabled_; }

    void sample(const std::string& checkpointName, bool syncBeforeSample = false);

    size_t getPeakUsedBytes() const { return peakUsedBytes_; }
    size_t getPeakFreeBytes() const { return peakFreeBytes_; }
    size_t getTotalBytes() const { return totalBytes_; }
    const std::string& getPeakCheckpoint() const { return peakCheckpoint_; }
    const std::vector<Sample>& getSamples() const { return samples_; }

    void addToTimer(PerformanceTimer& timer) const;
    void printSummary() const;

private:
    bool enabled_;
    std::vector<Sample> samples_;
    size_t peakUsedBytes_;
    size_t peakFreeBytes_;
    size_t totalBytes_;
    std::string peakCheckpoint_;

    static std::string sanitizeCheckpointName(const std::string& checkpointName);
};
