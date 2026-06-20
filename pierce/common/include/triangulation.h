#pragma once

// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <vector>
#include <string>
#include <iostream>

// CGAL includes
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/Polygon_set_2.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

// CGAL kernel and basic types
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes_2;

// Info for triangulation faces
struct FaceInfo2 {
    FaceInfo2() {}
    int nesting_level;
    bool in_domain() { return nesting_level % 2 == 1; }
};

typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K> Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<K, Fbb> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> TDS;
typedef CGAL::Exact_predicates_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;

struct Point2D {
    float x, y;
    Point2D(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};

struct Triangle {
    Point2D vertices[3];
    Triangle(const Point2D& v0, const Point2D& v1, const Point2D& v2) {
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;
    }
};

struct PolygonWithHoles {
    std::vector<Point2D> outer;
    std::vector<std::vector<Point2D>> holes;
};

struct TriangulationStats {
    int cgal_success = 0;         // Method 0: CGAL successful
    int cgal_repaired = 0;        // Method 1: CGAL with polygon repair for self-intersecting
    int cgal_decomposed = 0;      // Method 2: CGAL with polygon decomposition
    int failed_method = 0;        // Method 3: All methods failed
    
    void print() const {
        std::cout << "Triangulation methods used: CGAL=" << cgal_success 
                  << ", Repaired=" << cgal_repaired 
                  << ", Decomposed=" << cgal_decomposed
                  << ", Failed=" << failed_method << std::endl;
    }
};

std::vector<PolygonWithHoles> readPolygonVerticesFromFile(const std::string& filepath);

std::pair<std::vector<Triangle>, int> triangulatePolygon(const PolygonWithHoles& polygon);

std::vector<Polygon_with_holes_2> repairSelfIntersectingPolygon(const Polygon_2& polygon);
std::vector<Polygon_2> decomposePolygonToSimple(const Polygon_2& polygon);

size_t countTriangles(const std::vector<std::vector<Triangle>>& triangulated_polygons);

void mark_domains(CDT& ct, CDT::Face_handle start, int index, std::list<CDT::Edge>& border);
void mark_domains(CDT& cdt);
Polygon_2 convertToPolygon2(const std::vector<Point2D>& points);
std::vector<Triangle> extractTriangles(const CDT& cdt);
