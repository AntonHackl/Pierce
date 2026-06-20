#pragma once

#include <memory>
#include <string>

#include "IDatasetLoader.h"

enum class DatasetType {
    Mesh,
    DtMesh
};

class DatasetLoaderFactory {
public:
    static std::unique_ptr<IDatasetLoader> create(DatasetType type);
    // Select a supported mesh loader from the input extension.
    static std::unique_ptr<IDatasetLoader> createFromPath(const std::string& path);
};
