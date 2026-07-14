#pragma once

#include <fstream>
#include <vector>
#include <iostream>
#include "Geometry.h"

namespace RaySpace {
namespace IO {

constexpr uint32_t BINARY_FILE_MAGIC = 0x52334442; // "R3DB"
constexpr uint32_t BINARY_FILE_VERSION = 7;

struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t numVertices;
    uint64_t numIndices;
    uint64_t numMappings;
    uint64_t numEdges;
    uint64_t numObjects;
    uint64_t totalTriangles;
    uint8_t hasGrid;
    uint8_t hasEdges;
    uint8_t hasPartitionMetadata;
    uint8_t padding[5]; // Align to 8 bytes
};

struct GridParams {
    float cellSize;
    uint32_t numSparseCells;
};

inline bool writeBinaryFile(const std::string& filename, const GeometryData& geometry) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    FileHeader header;
    header.magic = BINARY_FILE_MAGIC;
    header.version = BINARY_FILE_VERSION;
    header.numVertices = geometry.vertices.size();
    header.numIndices = geometry.indices.size();
    header.numMappings = geometry.triangleToObject.size();
    header.numEdges = geometry.edges.edgeStarts.size();
    header.numObjects = geometry.partition.objects.centers.size();
    header.totalTriangles = geometry.totalTriangles;
    header.hasGrid = geometry.grid.hasGrid ? 1 : 0;
    header.hasEdges = geometry.edges.hasEdges() ? 1 : 0;
    header.hasPartitionMetadata = geometry.partition.hasData() ? 1 : 0;
    
    out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));

    if (header.numVertices > 0)
        out.write(reinterpret_cast<const char*>(geometry.vertices.data()), header.numVertices * sizeof(float3));
    
    if (header.numIndices > 0)
        out.write(reinterpret_cast<const char*>(geometry.indices.data()), header.numIndices * sizeof(uint3));

    if (header.numMappings > 0)
        out.write(reinterpret_cast<const char*>(geometry.triangleToObject.data()), header.numMappings * sizeof(int));

    if (header.hasEdges) {
        if (header.numEdges > 0) {
            out.write(reinterpret_cast<const char*>(geometry.edges.edgeStarts.data()), header.numEdges * sizeof(float3));
            out.write(reinterpret_cast<const char*>(geometry.edges.edgeEnds.data()), header.numEdges * sizeof(float3));
            out.write(reinterpret_cast<const char*>(geometry.edges.sourceObjectIds.data()), header.numEdges * sizeof(int));
        }
    }

    if (header.hasGrid) {
        GridParams gp;
        gp.cellSize = geometry.grid.cellSize;
        gp.numSparseCells = static_cast<uint32_t>(geometry.grid.sparseCells.size());

        out.write(reinterpret_cast<const char*>(&gp), sizeof(GridParams));

        if (gp.numSparseCells > 0)
            out.write(reinterpret_cast<const char*>(geometry.grid.sparseCells.data()), gp.numSparseCells * sizeof(SparseGridEntry));
    }

    if (header.hasPartitionMetadata) {
        const auto writeAxisIntervalData = [&](const AxisIntervalData& axisData) {
            if (!axisData.mins.empty()) {
                out.write(reinterpret_cast<const char*>(axisData.mins.data()), axisData.mins.size() * sizeof(float));
                out.write(reinterpret_cast<const char*>(axisData.maxs.data()), axisData.maxs.size() * sizeof(float));
                out.write(reinterpret_cast<const char*>(axisData.centers.data()), axisData.centers.size() * sizeof(float));
            }
        };
        const auto writeOrder = [&](const std::vector<uint32_t>& order) {
            if (!order.empty()) {
                out.write(reinterpret_cast<const char*>(order.data()), order.size() * sizeof(uint32_t));
            }
        };
        const auto writeWeightSummary = [&](const PartitionWeightSummary& summary) {
            const uint32_t binCount = static_cast<uint32_t>(summary.binWeights.size());
            out.write(reinterpret_cast<const char*>(&summary.triangleMinX), sizeof(float));
            out.write(reinterpret_cast<const char*>(&summary.triangleMaxX), sizeof(float));
            out.write(reinterpret_cast<const char*>(&summary.centerMinX), sizeof(float));
            out.write(reinterpret_cast<const char*>(&summary.centerMaxX), sizeof(float));
            out.write(reinterpret_cast<const char*>(&summary.totalWeight), sizeof(double));
            out.write(reinterpret_cast<const char*>(&binCount), sizeof(uint32_t));
            if (binCount > 0) {
                out.write(reinterpret_cast<const char*>(summary.binWeights.data()), binCount * sizeof(double));
            }
        };

        writeAxisIntervalData(geometry.partition.triangles);
        writeAxisIntervalData(geometry.partition.edges);
        writeAxisIntervalData(geometry.partition.objects);
        writeOrder(geometry.partition.triangleSortedByMin);
        writeOrder(geometry.partition.triangleSortedByMax);
        writeOrder(geometry.partition.edgeSortedByMin);
        writeOrder(geometry.partition.edgeSortedByMax);
        writeWeightSummary(geometry.partition.weightSummary);
    }

    out.close();
    return true;
}

inline GeometryData readBinaryFile(const std::string& filename) {
    GeometryData geometry;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return geometry;
    }

    FileHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(FileHeader));

    if (header.magic != BINARY_FILE_MAGIC) {
        std::cerr << "Error: Invalid file format (Magic mismatch). File: " << filename << std::endl;
         // Fallback or empty return? 
         // For now return empty, could also throw exception
        return geometry;
    }

    if (header.version != BINARY_FILE_VERSION) {
        std::cerr << "Error: Unsupported binary geometry version " << header.version
                  << " in file: " << filename
                  << ". Expected version " << BINARY_FILE_VERSION
                  << ". Re-run pierce_preprocess to regenerate partition summaries." << std::endl;
        return geometry;
    }

    geometry.totalTriangles = header.totalTriangles;
    geometry.vertices.resize(header.numVertices);
    geometry.indices.resize(header.numIndices);
    geometry.triangleToObject.resize(header.numMappings);

    if (header.numVertices > 0)
        in.read(reinterpret_cast<char*>(geometry.vertices.data()), header.numVertices * sizeof(float3));

    if (header.numIndices > 0)
        in.read(reinterpret_cast<char*>(geometry.indices.data()), header.numIndices * sizeof(uint3));

    if (header.numMappings > 0)
        in.read(reinterpret_cast<char*>(geometry.triangleToObject.data()), header.numMappings * sizeof(int));

    if (header.hasEdges) {
        geometry.edges.edgeStarts.resize(header.numEdges);
        geometry.edges.edgeEnds.resize(header.numEdges);
        geometry.edges.sourceObjectIds.resize(header.numEdges);

        if (header.numEdges > 0) {
            in.read(reinterpret_cast<char*>(geometry.edges.edgeStarts.data()), header.numEdges * sizeof(float3));
            in.read(reinterpret_cast<char*>(geometry.edges.edgeEnds.data()), header.numEdges * sizeof(float3));
            in.read(reinterpret_cast<char*>(geometry.edges.sourceObjectIds.data()), header.numEdges * sizeof(int));
        }
    }

    if (header.hasGrid) {
        geometry.grid.hasGrid = true;
        GridParams gp;
        in.read(reinterpret_cast<char*>(&gp), sizeof(GridParams));

        geometry.grid.cellSize = gp.cellSize;
        
        uint32_t numSparseCells = gp.numSparseCells;
        geometry.grid.sparseCells.resize(numSparseCells);
        
        if (numSparseCells > 0)
            in.read(reinterpret_cast<char*>(geometry.grid.sparseCells.data()), numSparseCells * sizeof(SparseGridEntry));
    }

    if (header.hasPartitionMetadata) {
        auto readAxisIntervalData = [&](AxisIntervalData& axisData, uint64_t count) {
            axisData.mins.resize(count);
            axisData.maxs.resize(count);
            axisData.centers.resize(count);
            if (count > 0) {
                in.read(reinterpret_cast<char*>(axisData.mins.data()), count * sizeof(float));
                in.read(reinterpret_cast<char*>(axisData.maxs.data()), count * sizeof(float));
                in.read(reinterpret_cast<char*>(axisData.centers.data()), count * sizeof(float));
            }
        };
        auto readOrder = [&](std::vector<uint32_t>& order, uint64_t count) {
            order.resize(count);
            if (count > 0) {
                in.read(reinterpret_cast<char*>(order.data()), count * sizeof(uint32_t));
            }
        };
        auto readWeightSummary = [&](PartitionWeightSummary& summary) {
            uint32_t binCount = 0;
            in.read(reinterpret_cast<char*>(&summary.triangleMinX), sizeof(float));
            in.read(reinterpret_cast<char*>(&summary.triangleMaxX), sizeof(float));
            in.read(reinterpret_cast<char*>(&summary.centerMinX), sizeof(float));
            in.read(reinterpret_cast<char*>(&summary.centerMaxX), sizeof(float));
            in.read(reinterpret_cast<char*>(&summary.totalWeight), sizeof(double));
            in.read(reinterpret_cast<char*>(&binCount), sizeof(uint32_t));
            summary.binWeights.resize(binCount);
            if (binCount > 0) {
                in.read(reinterpret_cast<char*>(summary.binWeights.data()), binCount * sizeof(double));
            }
        };

        readAxisIntervalData(geometry.partition.triangles, header.numIndices);
        readAxisIntervalData(geometry.partition.edges, header.numEdges);
        readAxisIntervalData(geometry.partition.objects, header.numObjects);
        readOrder(geometry.partition.triangleSortedByMin, header.numIndices);
        readOrder(geometry.partition.triangleSortedByMax, header.numIndices);
        readOrder(geometry.partition.edgeSortedByMin, header.numEdges);
        readOrder(geometry.partition.edgeSortedByMax, header.numEdges);
        readWeightSummary(geometry.partition.weightSummary);
    }

    in.close();
    return geometry;
}

} // namespace IO
} // namespace RaySpace
