#pragma once

#include <string>
#include <memory>

// #include "Geometry.h" // For GeometryData
#include "../../common/include/Geometry.h"

// Interface for interchangeable dataset loaders
class IDatasetLoader {
public:
    virtual ~IDatasetLoader() = default;
    // Load geometry from a dataset path (WKT file for polygons, .obj file for meshes)
    virtual GeometryData load(const std::string& path) = 0;
};
