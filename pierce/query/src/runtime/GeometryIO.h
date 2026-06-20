#pragma once

#include <string>
#include "Geometry.h"

// Load preprocessed geometry from a binary file created by pierce_preprocess
GeometryData loadGeometryFromFile(const std::string& geometryFilePath);

// Enforce that geometry carries precomputed edges in .pre payload.
bool requirePrecomputedEdges(const GeometryData& geometry, const std::string& geometryPath, const char* meshLabel);
