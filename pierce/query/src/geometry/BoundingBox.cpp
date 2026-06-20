#include "BoundingBox.h"
#include <algorithm>

BoundingBox BoundingBox::computeFromGeometry(const GeometryData& geometry) {
    BoundingBox bbox;
    for (const auto& v : geometry.vertices) {
        bbox.expand(v);
    }
    return bbox;
}

BoundingBox::BoxMesh BoundingBox::createBoxMesh() const {
    BoxMesh mesh;
    mesh.vertices.reserve(8);
    mesh.indices.reserve(12);
    
    float3 v0; v0.x = min.x; v0.y = min.y; v0.z = min.z; mesh.vertices.push_back(v0); // 0
    float3 v1; v1.x = max.x; v1.y = min.y; v1.z = min.z; mesh.vertices.push_back(v1); // 1
    float3 v2; v2.x = max.x; v2.y = max.y; v2.z = min.z; mesh.vertices.push_back(v2); // 2
    float3 v3; v3.x = min.x; v3.y = max.y; v3.z = min.z; mesh.vertices.push_back(v3); // 3
    float3 v4; v4.x = min.x; v4.y = min.y; v4.z = max.z; mesh.vertices.push_back(v4); // 4
    float3 v5; v5.x = max.x; v5.y = min.y; v5.z = max.z; mesh.vertices.push_back(v5); // 5
    float3 v6; v6.x = max.x; v6.y = max.y; v6.z = max.z; mesh.vertices.push_back(v6); // 6
    float3 v7; v7.x = min.x; v7.y = max.y; v7.z = max.z; mesh.vertices.push_back(v7); // 7
    
    // 12 triangles forming the box (with outward-facing winding order)
    // Bottom face (z = min)
    uint3 t0; t0.x = 0; t0.y = 2; t0.z = 1; mesh.indices.push_back(t0);
    uint3 t1; t1.x = 0; t1.y = 3; t1.z = 2; mesh.indices.push_back(t1);
    // Top face (z = max)
    uint3 t2; t2.x = 4; t2.y = 5; t2.z = 6; mesh.indices.push_back(t2);
    uint3 t3; t3.x = 4; t3.y = 6; t3.z = 7; mesh.indices.push_back(t3);
    // Front face (y = min)
    uint3 t4; t4.x = 0; t4.y = 1; t4.z = 5; mesh.indices.push_back(t4);
    uint3 t5; t5.x = 0; t5.y = 5; t5.z = 4; mesh.indices.push_back(t5);
    // Back face (y = max)
    uint3 t6; t6.x = 3; t6.y = 6; t6.z = 2; mesh.indices.push_back(t6);
    uint3 t7; t7.x = 3; t7.y = 7; t7.z = 6; mesh.indices.push_back(t7);
    // Left face (x = min)
    uint3 t8; t8.x = 0; t8.y = 4; t8.z = 7; mesh.indices.push_back(t8);
    uint3 t9; t9.x = 0; t9.y = 7; t9.z = 3; mesh.indices.push_back(t9);
    // Right face (x = max)
    uint3 t10; t10.x = 1; t10.y = 2; t10.z = 6; mesh.indices.push_back(t10);
    uint3 t11; t11.x = 1; t11.y = 6; t11.z = 5; mesh.indices.push_back(t11);
    
    // Triangle to object mapping (all belong to object 0)
    mesh.triangleToObject.assign(mesh.indices.size(), 0);
    
    return mesh;
}

