#include "GeometryIO.h"
#include "../../../common/include/BinaryIO.h"

#include <iostream>

GeometryData loadGeometryFromFile(const std::string& geometryFilePath) {
    std::cout << "=== Loading Preprocessed Geometry (Binary) ===" << std::endl;
    return RaySpace::IO::readBinaryFile(geometryFilePath);
}

bool requirePrecomputedEdges(const GeometryData& geometry, const std::string& geometryPath, const char* meshLabel) {
    if (geometry.edges.hasEdges()) {
        return true;
    }

    std::cerr << "Error: " << meshLabel << " loaded from '" << geometryPath
              << "' does not contain precomputed edges." << std::endl;
    std::cerr << "Re-run pierce_preprocess to regenerate the .pre file with embedded edge payload." << std::endl;
    return false;
}
