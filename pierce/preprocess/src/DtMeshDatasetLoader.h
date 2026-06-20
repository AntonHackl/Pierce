#pragma once

#include <string>
#include "IDatasetLoader.h"
#include "Geometry.h"

class DtMeshDatasetLoader : public IDatasetLoader {
public:
    GeometryData load(const std::string& filePath) override;
};
