#include "DtMeshDatasetLoader.h"
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

// Include tdbase headers
// We assume 'tdbase_lib/include' is in the include path
#include "tile.h"
#include "himesh.h"

GeometryData DtMeshDatasetLoader::load(const std::string& filePath) {
    GeometryData geometry;
    float3 minB = {1e30f, 1e30f, 1e30f};
    float3 maxB = {-1e30f, -1e30f, -1e30f};

    std::cout << "=== DT Mesh Dataset Loading ===" << std::endl;
    std::cout << "Loading DT mesh from file: " << filePath << std::endl;
    
    // Check file existence
    if (!tdbase::file_exist(filePath.c_str())) {
         std::cerr << "Error: File does not exist: " << filePath << std::endl;
         return geometry;
    }

    try {
        // Load the tile
        // tdbase::Tile constructor loads the file
        tdbase::Tile tile(filePath);
        
        // Decode to max LOD (100) to get full geometry
        tile.decode_all(100);
        
        std::cout << "Loaded tile with " << tile.num_objects() << " objects." << std::endl;

        // Iterate over wrappers in the tile
        size_t objectIndex = 0;
        for (size_t i = 0; i < tile.num_objects(); ++i) {
            tdbase::HiMesh_Wrapper* wrapper = tile.get_mesh_wrapper(i);
            if (!wrapper) continue;

            // For each voxel in the wrapper, extract triangles
            for (tdbase::Voxel* voxel : wrapper->voxels) {
                if (!voxel || !voxel->triangles || voxel->num_triangles == 0) continue;

                // Map from vertex coordinate to index
                std::map<std::tuple<float, float, float>, unsigned int> vertexMap;
                size_t vertexOffset = geometry.vertices.size();

                // Process all triangles in this voxel
                for (size_t t = 0; t < voxel->num_triangles; ++t) {
                    // Each triangle is 9 floats: v1(x,y,z), v2(x,y,z), v3(x,y,z)
                    size_t baseIdx = t * 9;
                    
                    unsigned int indices[3];
                    for (int v = 0; v < 3; ++v) {
                        float x = voxel->triangles[baseIdx + v*3 + 0];
                        float y = voxel->triangles[baseIdx + v*3 + 1];
                        float z = voxel->triangles[baseIdx + v*3 + 2];
                        
                        auto key = std::make_tuple(x, y, z);
                        if (vertexMap.find(key) == vertexMap.end()) {
                            // New vertex
                            geometry.vertices.push_back({x, y, z});
                            
                            minB.x = std::min(minB.x, x);
                            minB.y = std::min(minB.y, y);
                            minB.z = std::min(minB.z, z);
                            maxB.x = std::max(maxB.x, x);
                            maxB.y = std::max(maxB.y, y);
                            maxB.z = std::max(maxB.z, z);
                            
                            vertexMap[key] = geometry.vertices.size() - 1;
                        }
                        indices[v] = vertexMap[key];
                    }
                    
                    geometry.indices.push_back({indices[0], indices[1], indices[2]});
                    geometry.triangleToObject.push_back(objectIndex);
                }
            }
            objectIndex++;
        }
        
        geometry.totalTriangles = geometry.indices.size();
        // Legacy bounds removed from GridData
        std::cout << "Total vertices: " << geometry.vertices.size() << std::endl;
        std::cout << "Total triangles: " << geometry.totalTriangles << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading DT file: " << e.what() << std::endl;
    }

    std::cout << "=============================\n" << std::endl;
    return geometry;
}
