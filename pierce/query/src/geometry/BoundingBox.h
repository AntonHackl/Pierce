#pragma once

#include <vector>
#include <limits>
#include "common.h"
#include "Geometry.h"

struct BoundingBox {
    float3 min;
    float3 max;
    
    BoundingBox() {
        min.x = min.y = min.z = std::numeric_limits<float>::max();
        max.x = max.y = max.z = -std::numeric_limits<float>::max();
    }
    
    void expand(const float3& point) {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);
        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }
    
    static BoundingBox computeFromGeometry(const GeometryData& geometry);
    
    // Create box mesh geometry (8 vertices, 12 triangles)
    // Returns vertices, indices, and triangle_to_object mapping
    struct BoxMesh {
        std::vector<float3> vertices;
        std::vector<uint3> indices;
        std::vector<int> triangleToObject;
    };
    
    BoxMesh createBoxMesh() const;
};

