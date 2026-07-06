#include "GpuMemoryTracker.h"

#include <algorithm>
#include <cctype>
#include <cuda_runtime.h>
#include <iostream>

#include "../optix/OptixHelpers.h"
#include "../timer.h"

GpuMemoryTracker::GpuMemoryTracker(bool enabled)
    : enabled_(enabled), peakUsedBytes_(0), peakFreeBytes_(0), totalBytes_(0) {
}

void GpuMemoryTracker::sample(const std::string& checkpointName, bool syncBeforeSample) {
    if (!enabled_) {
        return;
    }

    if (syncBeforeSample) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t usedBytes = totalBytes - freeBytes;

    auto existing = std::find_if(
        samples_.begin(),
        samples_.end(),
        [&](const Sample& sample) { return sample.checkpoint == checkpointName; }
    );
    if (existing != samples_.end()) {
        existing->usedBytes = usedBytes;
        existing->freeBytes = freeBytes;
        existing->totalBytes = totalBytes;
    } else {
        samples_.push_back({checkpointName, usedBytes, freeBytes, totalBytes});
    }
    if (usedBytes >= peakUsedBytes_) {
        peakUsedBytes_ = usedBytes;
        peakFreeBytes_ = freeBytes;
        totalBytes_ = totalBytes;
        peakCheckpoint_ = checkpointName;
    } else if (totalBytes_ == 0) {
        totalBytes_ = totalBytes;
    }
}

void GpuMemoryTracker::addToTimer(PerformanceTimer& timer) const {
    if (!enabled_ || samples_.empty()) {
        return;
    }

    timer.addCounter("gpu_memory_peak_used_bytes", static_cast<unsigned long long>(peakUsedBytes_));
    timer.addCounter("gpu_memory_peak_free_bytes", static_cast<unsigned long long>(peakFreeBytes_));
    timer.addCounter("gpu_memory_total_bytes", static_cast<unsigned long long>(totalBytes_));

    for (const auto& sample : samples_) {
        const std::string sanitized = sanitizeCheckpointName(sample.checkpoint);
        timer.addCounter(
            "gpu_memory_used_" + sanitized + "_bytes",
            static_cast<unsigned long long>(sample.usedBytes)
        );
    }
}

void GpuMemoryTracker::printSummary() const {
    if (!enabled_ || samples_.empty()) {
        return;
    }

    std::cout << "\n=== GPU Memory Summary ===" << std::endl;
    std::cout << "Peak Used Bytes: " << peakUsedBytes_ << std::endl;
    std::cout << "Peak Free Bytes: " << peakFreeBytes_ << std::endl;
    std::cout << "Total GPU Bytes: " << totalBytes_ << std::endl;
    std::cout << "Peak Checkpoint: " << peakCheckpoint_ << std::endl;
}

std::string GpuMemoryTracker::sanitizeCheckpointName(const std::string& checkpointName) {
    std::string result;
    result.reserve(checkpointName.size());

    for (char c : checkpointName) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            result.push_back(static_cast<char>(std::tolower(uc)));
        } else {
            result.push_back('_');
        }
    }

    while (!result.empty() && result.front() == '_') {
        result.erase(result.begin());
    }
    while (!result.empty() && result.back() == '_') {
        result.pop_back();
    }

    std::replace(result.begin(), result.end(), '-', '_');
    if (result.empty()) {
        result = "unnamed_checkpoint";
    }
    return result;
}
