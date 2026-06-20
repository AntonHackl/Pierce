#include "DatasetUtils.h"

#include <iostream>
#include <iomanip>

void printProgressBar(std::size_t current, std::size_t total, int barWidth) {
    if (total == 0) return;
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "% (" << current << "/" << total << ")";
    std::cout.flush();

    if (current == total) {
        std::cout << std::endl;
    }
}
