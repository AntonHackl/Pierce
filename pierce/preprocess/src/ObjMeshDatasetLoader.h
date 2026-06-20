#pragma once

#include <string>
#include <memory>

#include "IDatasetLoader.h"
#include "Geometry.h"

class ObjMeshDatasetLoader : public IDatasetLoader {
public:
    GeometryData load(const std::string& filePath) override;
};
