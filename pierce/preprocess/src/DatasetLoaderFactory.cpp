#include "DatasetLoaderFactory.h"

#include <filesystem>

#include "ObjMeshDatasetLoader.h"
#include "DtMeshDatasetLoader.h"

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::create(DatasetType type) {
    switch (type) {
        case DatasetType::Mesh:
            return std::make_unique<ObjMeshDatasetLoader>();
        case DatasetType::DtMesh:
            return std::make_unique<DtMeshDatasetLoader>();
        default:
            return nullptr;
    }
}

std::unique_ptr<IDatasetLoader> DatasetLoaderFactory::createFromPath(const std::string& path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    
    // Convert to lowercase for robust extension matching
    for (auto& c : ext) c = std::tolower(c);

    if (ext == ".obj") {
        return std::make_unique<ObjMeshDatasetLoader>();
    } else if (ext == ".dt") {
        return std::make_unique<DtMeshDatasetLoader>();
    }
    
    return nullptr;
}
