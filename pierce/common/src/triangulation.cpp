// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "triangulation.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <list>

void mark_domains(CDT& ct, CDT::Face_handle start, int index, std::list<CDT::Edge>& border) {
    if (start->info().nesting_level != -1) {
        return;
    }
    std::list<CDT::Face_handle> queue;
    queue.push_back(start);
    while (!queue.empty()) {
        CDT::Face_handle fh = queue.front();
        queue.pop_front();
        if (fh->info().nesting_level == -1) {
            fh->info().nesting_level = index;
            for (int i = 0; i < 3; i++) {
                CDT::Edge e(fh, i);
                CDT::Face_handle n = fh->neighbor(i);
                if (n->info().nesting_level == -1) {
                    if (ct.is_constrained(e)) border.push_back(e);
                    else queue.push_back(n);
                }
            }
        }
    }
}

void mark_domains(CDT& cdt) {
    for (CDT::Face_handle f : cdt.all_face_handles()) {
        f->info().nesting_level = -1;
    }
    std::list<CDT::Edge> border;
    mark_domains(cdt, cdt.infinite_face(), 0, border);
    while (!border.empty()) {
        CDT::Edge e = border.front();
        border.pop_front();
        CDT::Face_handle n = e.first->neighbor(e.second);
        if (n->info().nesting_level == -1) {
            mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
        }
    }
}

Polygon_2 convertToPolygon2(const std::vector<Point2D>& points) {
    Polygon_2 polygon;
    for (const auto& pt : points) {
        polygon.push_back(Point_2(pt.x, pt.y));
    }
    return polygon;
}

std::vector<Triangle> extractTriangles(const CDT& cdt) {
    std::vector<Triangle> triangles;
    
    for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
         fit != cdt.finite_faces_end(); ++fit) {
        if (fit->info().in_domain()) {
            Point_2 p0 = fit->vertex(0)->point();
            Point_2 p1 = fit->vertex(1)->point();
            Point_2 p2 = fit->vertex(2)->point();
            
            triangles.emplace_back(
                Point2D(static_cast<float>(p0.x()), static_cast<float>(p0.y())),
                Point2D(static_cast<float>(p1.x()), static_cast<float>(p1.y())),
                Point2D(static_cast<float>(p2.x()), static_cast<float>(p2.y()))
            );
        }
    }
    
    return triangles;
}

void extractPolygonWithHoles(const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float>>& poly,
                             std::vector<PolygonWithHoles>& out)
{
    PolygonWithHoles pwh;
    auto outer = boost::geometry::exterior_ring(poly);
    for(const auto& pt : outer) {
        pwh.outer.push_back(Point2D(pt.x(), pt.y()));
    }
    if(!pwh.outer.empty() && pwh.outer.front().x == pwh.outer.back().x &&
       pwh.outer.front().y == pwh.outer.back().y) {
        pwh.outer.pop_back();
    }
    if (pwh.outer.size() < 3) {
        std::cerr << "Polygon outer ring has less than 3 vertices after processing. Skipping polygon..." << std::endl;
        return;
    }
    size_t num_holes = boost::geometry::num_interior_rings(poly);
    const auto& holes = boost::geometry::interior_rings(poly);
    for(size_t h = 0; h < num_holes; ++h) {
        std::vector<Point2D> hole_vertices;
        try {
            const auto& hole = holes[h];
            if (hole.empty() || hole.size() < 3) {
                continue;
            }
            for(const auto& pt : hole) {
                hole_vertices.push_back(Point2D(pt.x(), pt.y()));
            }
            if(!hole_vertices.empty() && hole_vertices.front().x == hole_vertices.back().x &&
               hole_vertices.front().y == hole_vertices.back().y) {
                hole_vertices.pop_back();
            }
            pwh.holes.push_back(hole_vertices);
        } catch(const std::exception& e) {
            std::cerr << "Error accessing polygon hole: " << e.what() << std::endl;
            continue;
        }
    }
    out.push_back(pwh);
}

std::vector<PolygonWithHoles> readPolygonVerticesFromFile(const std::string& filepath)
{
    using BoostPolygon = boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<float>>;
    using BoostMultiPolygon = boost::geometry::model::multi_polygon<BoostPolygon>;
    std::ifstream file(filepath);
    std::vector<std::string> lines;
    std::string line;
    while(getline(file, line))
    {
        // Find lines containing POLYGON or MULTIPOLYGON
        if((line.find("POLYGON") != std::string::npos || line.find("MULTIPOLYGON") != std::string::npos))
        {
            lines.push_back(line);
        }
    }
    if(lines.empty())
    {
        return {};
    }
    std::vector<PolygonWithHoles> all_polygons;
    for(const auto& wkt : lines)
    {
        try {
            if(wkt.find("MULTIPOLYGON") != std::string::npos) {
                BoostMultiPolygon multipoly;
                boost::geometry::read_wkt(wkt, multipoly);
                for(const auto& poly : multipoly) {
                    extractPolygonWithHoles(poly, all_polygons);
                }
            } else if(wkt.find("POLYGON") != std::string::npos) {
                BoostPolygon poly;
                boost::geometry::read_wkt(wkt, poly);
                extractPolygonWithHoles(poly, all_polygons);
            }
        } catch(const std::exception& e) {
            std::cerr << "Error reading WKT: " << e.what() << std::endl;
        }
    }
    return all_polygons;
}

std::vector<Polygon_with_holes_2> repairSelfIntersectingPolygon(const Polygon_2& polygon) {
    std::vector<Polygon_with_holes_2> result;
    
    try {
        CGAL::Polygon_set_2<K> polygon_set;
        polygon_set.insert(polygon);
        
        std::vector<Polygon_with_holes_2> repaired_polygons;
        polygon_set.polygons_with_holes(std::back_inserter(repaired_polygons));
        
        for (const auto& poly : repaired_polygons) {
            if (poly.outer_boundary().size() >= 3) {
                result.push_back(poly);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Polygon repair failed: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<Polygon_2> decomposePolygonToSimple(const Polygon_2& polygon) {
    std::vector<Polygon_2> result;
    
    try {
        // For self-intersecting polygons, we can try to split them at intersection points
        // This is a simplified approach - in practice, this would need more sophisticated handling
        
        // First try: Use the polygon as-is if it has valid area
        if (polygon.size() >= 3) {
            result.push_back(polygon);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Polygon decomposition failed: " << e.what() << std::endl;
    }
    
    return result;
}

std::pair<std::vector<Triangle>, int> triangulatePolygon(const PolygonWithHoles& poly)
{
    try {
        CDT cdt;
        
        Polygon_2 outer_polygon = convertToPolygon2(poly.outer);
        
        if (outer_polygon.is_simple()) {
            cdt.insert_constraint(outer_polygon.vertices_begin(), outer_polygon.vertices_end(), true);
            
            for (const auto& hole : poly.holes) {
                if (hole.size() >= 3) {
                    Polygon_2 hole_polygon = convertToPolygon2(hole);
                    if (hole_polygon.is_simple()) {
                        cdt.insert_constraint(hole_polygon.vertices_begin(), hole_polygon.vertices_end(), true);
                    }
                }
            }
            
            mark_domains(cdt);
            
            std::vector<Triangle> triangles = extractTriangles(cdt);
            
            if (!triangles.empty()) {
                return std::make_pair(triangles, 0); // Standard method
            }
        }
        
        std::cout << "Attempting polygon repair for self-intersecting polygon..." << std::endl;
        
        std::vector<Polygon_with_holes_2> repaired_polygons = repairSelfIntersectingPolygon(outer_polygon);
        
        if (!repaired_polygons.empty()) {
            std::vector<Triangle> all_triangles;
            
            for (const auto& repaired_poly : repaired_polygons) {
                CDT repair_cdt;
                
                const Polygon_2& outer = repaired_poly.outer_boundary();
                repair_cdt.insert_constraint(outer.vertices_begin(), outer.vertices_end(), true);
                
                for (auto hole_it = repaired_poly.holes_begin(); hole_it != repaired_poly.holes_end(); ++hole_it) {
                    repair_cdt.insert_constraint(hole_it->vertices_begin(), hole_it->vertices_end(), true);
                }
                
                // Insert original holes if they're still valid
                for (const auto& hole : poly.holes) {
                    if (hole.size() >= 3) {
                        Polygon_2 hole_polygon = convertToPolygon2(hole);
                        if (hole_polygon.is_simple()) {
                            repair_cdt.insert_constraint(hole_polygon.vertices_begin(), hole_polygon.vertices_end(), true);
                        }
                    }
                }
                
                mark_domains(repair_cdt);
                std::vector<Triangle> triangles = extractTriangles(repair_cdt);
                all_triangles.insert(all_triangles.end(), triangles.begin(), triangles.end());
            }
            
            if (!all_triangles.empty()) {
                std::cout << "Polygon repair successful, generated " << all_triangles.size() << " triangles" << std::endl;
                return std::make_pair(all_triangles, 1); // Repaired method
            }
        }
        
        // Method 2: Try polygon decomposition
        std::cout << "Attempting polygon decomposition..." << std::endl;
        
        std::vector<Polygon_2> simple_polygons = decomposePolygonToSimple(outer_polygon);
        
        if (!simple_polygons.empty()) {
            std::vector<Triangle> all_triangles;
            
            for (const auto& simple_poly : simple_polygons) {
                CDT decomp_cdt;
                
                if (simple_poly.is_simple() && simple_poly.size() >= 3) {
                    decomp_cdt.insert_constraint(simple_poly.vertices_begin(), simple_poly.vertices_end(), true);
                    
                    // Add holes if they fit within this simple polygon
                    for (const auto& hole : poly.holes) {
                        if (hole.size() >= 3) {
                            Polygon_2 hole_polygon = convertToPolygon2(hole);
                            if (hole_polygon.is_simple()) {
                                decomp_cdt.insert_constraint(hole_polygon.vertices_begin(), hole_polygon.vertices_end(), true);
                            }
                        }
                    }
                    
                    mark_domains(decomp_cdt);
                    std::vector<Triangle> triangles = extractTriangles(decomp_cdt);
                    all_triangles.insert(all_triangles.end(), triangles.begin(), triangles.end());
                }
            }
            
            if (!all_triangles.empty()) {
                std::cout << "Polygon decomposition successful, generated " << all_triangles.size() << " triangles" << std::endl;
                return std::make_pair(all_triangles, 2); // Decomposed method
            }
        }
        
        // Method 3: Last resort - try basic point insertion
        std::cout << "Attempting basic triangulation as last resort..." << std::endl;
        
        CDT fallback_cdt;
        std::vector<CDT::Vertex_handle> vertices;
        
        // Insert points individually
        for (const auto& pt : poly.outer) {
            vertices.push_back(fallback_cdt.insert(Point_2(pt.x, pt.y)));
        }
        
        // Try to add constraints one by one, skipping problematic ones
        int successful_constraints = 0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            size_t next = (i + 1) % vertices.size();
            try {
                fallback_cdt.insert_constraint(vertices[i], vertices[next]);
                successful_constraints++;
            } catch (...) {
                // Skip this constraint if it causes problems
                continue;
            }
        }
        
        if (successful_constraints > 0) {
            mark_domains(fallback_cdt);
            std::vector<Triangle> triangles = extractTriangles(fallback_cdt);
            
            if (!triangles.empty()) {
                std::cout << "Basic triangulation successful with " << successful_constraints 
                          << "/" << vertices.size() << " constraints preserved" << std::endl;
                return std::make_pair(triangles, 2); // Decomposed method (partial success)
            }
        }
        
        return std::make_pair(std::vector<Triangle>(), 3); // Failed
        
    } catch (const std::exception& e) {
        std::cerr << "All triangulation methods failed: " << e.what() << std::endl;
        return std::make_pair(std::vector<Triangle>(), 3); // Failed
    }
}

size_t countTriangles(const std::vector<std::vector<Triangle>>& triangulated_polygons)
{
    size_t total = 0;
    for(const auto& triangles : triangulated_polygons)
    {
        total += triangles.size();
    }
    return total;
}