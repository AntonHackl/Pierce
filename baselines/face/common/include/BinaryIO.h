#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include "Geometry.h"

namespace RaySpace {
namespace IO {

constexpr uint32_t BINARY_FILE_MAGIC = 0x52334442; // "R3DB"
constexpr uint32_t BINARY_FILE_VERSION = 4;

struct FileHeaderV1 {
    uint32_t magic;
    uint32_t version;
    uint64_t numVertices;
    uint64_t numIndices;
    uint64_t numMappings;
    uint64_t totalTriangles;
    uint8_t hasGrid;
    uint8_t padding[7];
};

struct FileHeaderV2V3 {
    uint32_t magic;
    uint32_t version;
    uint64_t numVertices;
    uint64_t numIndices;
    uint64_t numMappings;
    uint64_t numEdges;
    uint64_t numEdgeSourceObjects;
    uint64_t totalTriangles;
    uint8_t hasGrid;
    uint8_t hasEdges;
    uint8_t padding[6];
};

struct FileHeaderV4 {
    uint32_t magic;
    uint32_t version;
    uint64_t numVertices;
    uint64_t numIndices;
    uint64_t numMappings;
    uint64_t numEdges;
    uint64_t totalTriangles;
    uint8_t hasGrid;
    uint8_t hasEdges;
    uint8_t padding[6];
};

struct GridParams {
    float minBound[3];
    float maxBound[3];
    uint32_t resolution[3];
    uint32_t padding; // Align
};

// Pierce v3 stores sparse grid metadata and entries instead of dense grid bounds/resolution.
struct GridParamsV3 {
    float cellSize;
    uint32_t numSparseCells;
};

struct SparseGridEntryV3 {
    int3 index;
    GridCell stats;
};

// Write geometry and grid data to a binary file
inline bool writeBinaryFile(const std::string& filename, const GeometryData& geometry) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    FileHeaderV4 header;
    header.magic = BINARY_FILE_MAGIC;
    header.version = BINARY_FILE_VERSION;
    header.numVertices = geometry.vertices.size();
    header.numIndices = geometry.indices.size();
    header.numMappings = geometry.triangleToObject.size();
    header.numEdges = 0;
    header.totalTriangles = geometry.totalTriangles;
    header.hasGrid = geometry.grid.hasGrid ? 1 : 0;
    header.hasEdges = 0;
    
    // Write Header
    out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeaderV4));

    // Write Main Data Arrays
    if (header.numVertices > 0)
        out.write(reinterpret_cast<const char*>(geometry.vertices.data()), header.numVertices * sizeof(float3));
    
    if (header.numIndices > 0)
        out.write(reinterpret_cast<const char*>(geometry.indices.data()), header.numIndices * sizeof(uint3));

    if (header.numMappings > 0)
        out.write(reinterpret_cast<const char*>(geometry.triangleToObject.data()), header.numMappings * sizeof(int));

    // Write Grid Data if present
    if (header.hasGrid) {
        GridParams gp;
        gp.minBound[0] = geometry.grid.minBound.x;
        gp.minBound[1] = geometry.grid.minBound.y;
        gp.minBound[2] = geometry.grid.minBound.z;
        gp.maxBound[0] = geometry.grid.maxBound.x;
        gp.maxBound[1] = geometry.grid.maxBound.y;
        gp.maxBound[2] = geometry.grid.maxBound.z;
        gp.resolution[0] = geometry.grid.resolution.x;
        gp.resolution[1] = geometry.grid.resolution.y;
        gp.resolution[2] = geometry.grid.resolution.z;

        out.write(reinterpret_cast<const char*>(&gp), sizeof(GridParams));

        size_t numCells = geometry.grid.cells.size();
        if (numCells != (size_t)gp.resolution[0] * gp.resolution[1] * gp.resolution[2]) {
            std::cerr << "Warning: Grid cell count mismatch in write!" << std::endl;
        }
        
        // Write cells directly
        if (numCells > 0)
            out.write(reinterpret_cast<const char*>(geometry.grid.cells.data()), numCells * sizeof(GridCell));
    }

    out.close();
    return true;
}

// Read geometry and grid data from a binary file
inline GeometryData readBinaryFile(const std::string& filename) {
    GeometryData geometry;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return geometry;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    if (magic != BINARY_FILE_MAGIC) {
        std::cerr << "Error: Invalid file format (Magic Number Mismatch)" << std::endl;
        return geometry;
    }

    in.seekg(0, std::ios::beg);

    uint64_t numVertices = 0;
    uint64_t numIndices = 0;
    uint64_t numMappings = 0;
    uint64_t totalTriangles = 0;
    uint8_t hasGrid = 0;
    uint8_t hasEdges = 0;
    uint64_t numEdges = 0;
    uint64_t numEdgeSourceObjects = 0;

    if (version == 1) {
        FileHeaderV1 header;
        in.read(reinterpret_cast<char*>(&header), sizeof(FileHeaderV1));
        numVertices = header.numVertices;
        numIndices = header.numIndices;
        numMappings = header.numMappings;
        totalTriangles = header.totalTriangles;
        hasGrid = header.hasGrid;
    } else if (version == 2 || version == 3) {
        FileHeaderV2V3 header;
        in.read(reinterpret_cast<char*>(&header), sizeof(FileHeaderV2V3));
        numVertices = header.numVertices;
        numIndices = header.numIndices;
        numMappings = header.numMappings;
        numEdges = header.numEdges;
        numEdgeSourceObjects = header.numEdgeSourceObjects;
        totalTriangles = header.totalTriangles;
        hasGrid = header.hasGrid;
        hasEdges = header.hasEdges;
    } else if (version == 4) {
        FileHeaderV4 header;
        in.read(reinterpret_cast<char*>(&header), sizeof(FileHeaderV4));
        numVertices = header.numVertices;
        numIndices = header.numIndices;
        numMappings = header.numMappings;
        numEdges = header.numEdges;
        totalTriangles = header.totalTriangles;
        hasGrid = header.hasGrid;
        hasEdges = header.hasEdges;
    } else {
        std::cerr << "Error: Unsupported binary file version: " << version << std::endl;
        return geometry;
    }

    geometry.totalTriangles = totalTriangles;

    // Resize and Read
    if (numVertices > 0) {
        geometry.vertices.resize(numVertices);
        in.read(reinterpret_cast<char*>(geometry.vertices.data()), numVertices * sizeof(float3));
    }

    if (numIndices > 0) {
        geometry.indices.resize(numIndices);
        in.read(reinterpret_cast<char*>(geometry.indices.data()), numIndices * sizeof(uint3));
    }

    if (numMappings > 0) {
        geometry.triangleToObject.resize(numMappings);
        in.read(reinterpret_cast<char*>(geometry.triangleToObject.data()), numMappings * sizeof(int));
    }

    // Versions 2/3 may contain edge sections that CPU baselines do not use.
    if ((version == 2 || version == 3) && hasEdges) {
        const std::streamoff edgeBytes =
            static_cast<std::streamoff>(numEdges) * static_cast<std::streamoff>(sizeof(float3) * 2 + sizeof(int) * 2) +
            static_cast<std::streamoff>(numEdgeSourceObjects) * static_cast<std::streamoff>(sizeof(int));
        in.seekg(edgeBytes, std::ios::cur);
    }

    if (version == 4 && hasEdges) {
        const std::streamoff edgeBytes =
            static_cast<std::streamoff>(numEdges) * static_cast<std::streamoff>(sizeof(float3) * 2 + sizeof(int));
        in.seekg(edgeBytes, std::ios::cur);
    }

    if (hasGrid) {
        if (version == 3 || version == 4) {
            // RaySpace v3/v4 use sparse-grid payload:
            //   [cellSize(float), numSparseCells(uint32)] + numSparseCells * SparseGridEntryV3
            // CPU baselines do not consume the grid for overlap/intersection/containment, so skip it.
            GridParamsV3 gp;
            in.read(reinterpret_cast<char*>(&gp), sizeof(GridParamsV3));

            const std::streamoff sparseBytes =
                static_cast<std::streamoff>(gp.numSparseCells) * static_cast<std::streamoff>(sizeof(SparseGridEntryV3));
            in.seekg(sparseBytes, std::ios::cur);
        } else {
            // Legacy dense-grid payload used by v1/v2.
            geometry.grid.hasGrid = true;
            GridParams gp;
            in.read(reinterpret_cast<char*>(&gp), sizeof(GridParams));

            geometry.grid.minBound = {gp.minBound[0], gp.minBound[1], gp.minBound[2]};
            geometry.grid.maxBound = {gp.maxBound[0], gp.maxBound[1], gp.maxBound[2]};
            geometry.grid.resolution = {gp.resolution[0], gp.resolution[1], gp.resolution[2]};
            geometry.grid.cells.resize(gp.resolution[0] * gp.resolution[1] * gp.resolution[2]);

            if (!geometry.grid.cells.empty()) {
                in.read(reinterpret_cast<char*>(geometry.grid.cells.data()), geometry.grid.cells.size() * sizeof(GridCell));
            }
        }
    }

    return geometry;
}

}
}
