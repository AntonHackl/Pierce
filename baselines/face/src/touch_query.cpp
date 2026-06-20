#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <set>
#include <cmath>
#include <limits>
#include <memory> 

#ifdef _OPENMP
#include <omp.h>
#endif

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "BinaryIO.h"

using namespace RaySpace;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Triangle_3 Triangle_3;
typedef Kernel::Iso_cuboid_3 Iso_cuboid_3; // Axis-aligned bounding box

// Structure to hold a triangle and its parent object ID (for CGAL AABB tree)
struct IndexedTriangle {
    Triangle_3 triangle;
    int objectId;
};

// Custom primitive to use IndexedTriangle with AABB_tree
struct IndexedTrianglePrimitive {
    typedef std::vector<IndexedTriangle>::const_iterator Id;
    typedef Triangle_3 Datum;
    typedef Point_3 Point;

    IndexedTrianglePrimitive() {}
    IndexedTrianglePrimitive(Id id) : m_id(id) {}

    Id id() const { return m_id; }
    const Datum& datum() const { return m_id->triangle; }
    Point reference_point() const { return m_id->triangle.vertex(0); }

private:
    Id m_id;
};

typedef CGAL::AABB_traits<Kernel, IndexedTrianglePrimitive> AABB_Traits;
typedef CGAL::AABB_tree<AABB_Traits> AABB_Tree;

// --- TOUCH Structures ---

struct AABB {
    float min[3];
    float max[3];

    AABB() {
        min[0] = min[1] = min[2] = std::numeric_limits<float>::max();
        max[0] = max[1] = max[2] = -std::numeric_limits<float>::max();
    }

    void extend(const float p[3]) {
        for(int i=0; i<3; ++i) {
            if(p[i] < min[i]) min[i] = p[i];
            if(p[i] > max[i]) max[i] = p[i];
        }
    }

    void extend(const AABB& other) {
        for(int i=0; i<3; ++i) {
            if(other.min[i] < min[i]) min[i] = other.min[i];
            if(other.max[i] > max[i]) max[i] = other.max[i];
        }
    }

    bool overlaps(const AABB& other) const {
        for(int i=0; i<3; ++i) {
            if(max[i] < other.min[i] || min[i] > other.max[i]) return false;
        }
        return true;
    }
};

struct Object {
    int id;
    AABB mbr;
    std::vector<Triangle_3> triangles;
    // Pre-computed AABB tree for this object (for efficient refinement)
    // We use shared_cgal_tris because the AABB tree holds iterators to it
    std::vector<IndexedTriangle> cgal_tris; 
    std::shared_ptr<AABB_Tree> cgal_tree; 
};

struct Node {
    AABB mbr;
    std::vector<Node*> children;
    std::vector<Object*> leaf_objects; // Objects of A in leaf nodes
    std::vector<Object*> assigned_b_objects; // Objects of B assigned to this node
    bool is_leaf = false;
    Node* parent = nullptr;
};

// --- TOUCH Implementation ---

struct ObjectComparator {
    int dim;
    ObjectComparator(int d) : dim(d) {}
    bool operator()(const Object* a, const Object* b) const {
        return (a->mbr.min[dim] + a->mbr.max[dim]) < (b->mbr.min[dim] + b->mbr.max[dim]);
    }
};

struct NodeComparator {
    int dim;
    NodeComparator(int d) : dim(d) {}
    bool operator()(const Node* a, const Node* b) const {
        return (a->mbr.min[dim] + a->mbr.max[dim]) < (b->mbr.min[dim] + b->mbr.max[dim]);
    }
};

Node* build_str_tree(std::vector<Object*>& objects, int fanout) {
    if (objects.empty()) return nullptr;

    size_t P = (objects.size() + fanout - 1) / fanout;
    double p_double = (double)P;
    int Sx = std::ceil(std::pow(p_double, 1.0/3.0));
    int Sy = std::ceil(std::sqrt(p_double / Sx));
    
    std::sort(objects.begin(), objects.end(), ObjectComparator(0));
    
    std::vector<Node*> leaf_nodes;
    size_t current_idx = 0;
    int current_sx = Sx;
    while(current_idx < objects.size()) {
        size_t chunk_size_x = (objects.size() - current_idx + current_sx - 1) / current_sx;
        size_t end_x = std::min(objects.size(), current_idx + chunk_size_x);
        
        std::sort(objects.begin() + current_idx, objects.begin() + end_x, ObjectComparator(1));
        
        size_t current_y = current_idx;
        int current_sy = Sy;
        while(current_y < end_x) {
            size_t chunk_size_y = (end_x - current_y + current_sy - 1) / current_sy;
            size_t end_y = std::min(end_x, current_y + chunk_size_y);
             
            std::sort(objects.begin() + current_y, objects.begin() + end_y, ObjectComparator(2));
             
            size_t current_z = current_y;
            while(current_z < end_y) {
                 Node* node = new Node();
                 node->is_leaf = true;
                 for(int i=0; i<fanout && current_z < end_y; ++i, ++current_z) {
                     node->leaf_objects.push_back(objects[current_z]);
                     node->mbr.extend(objects[current_z]->mbr);
                 }
                 leaf_nodes.push_back(node);
            }
            current_sy--;
            current_y = end_y;
        }
        current_sx--;
        current_idx = end_x;
    }
    
    std::vector<Node*> level_nodes = leaf_nodes;
    while(level_nodes.size() > 1) {
        std::vector<Node*> next_level;
        P = (level_nodes.size() + fanout - 1) / fanout;
        p_double = (double)P;
        Sx = std::ceil(std::pow(p_double, 1.0/3.0));
        Sy = std::ceil(std::sqrt(p_double / Sx));
        
        std::sort(level_nodes.begin(), level_nodes.end(), NodeComparator(0));
        
        current_idx = 0;
        current_sx = Sx;
        while(current_idx < level_nodes.size()) {
            size_t chunk_size_x = (level_nodes.size() - current_idx + current_sx - 1) / current_sx;
            size_t end_x = std::min(level_nodes.size(), current_idx + chunk_size_x);
            
            std::sort(level_nodes.begin() + current_idx, level_nodes.begin() + end_x, NodeComparator(1));
            
            size_t current_y = current_idx;
            int current_sy = Sy;
            while(current_y < end_x) {
                size_t chunk_size_y = (end_x - current_y + current_sy - 1) / current_sy;
                size_t end_y = std::min(end_x, current_y + chunk_size_y);
                
                std::sort(level_nodes.begin() + current_y, level_nodes.begin() + end_y, NodeComparator(2));
                
                size_t current_z = current_y;
                while(current_z < end_y) {
                    Node* parent = new Node();
                    parent->is_leaf = false;
                    for(int i=0; i<fanout && current_z < end_y; ++i, ++current_z) {
                        parent->children.push_back(level_nodes[current_z]);
                        level_nodes[current_z]->parent = parent;
                        parent->mbr.extend(level_nodes[current_z]->mbr);
                    }
                    next_level.push_back(parent);
                }
                current_sy--;
                current_y = end_y;
            }
            current_sx--;
            current_idx = end_x;
        }
        level_nodes = next_level;
    }
    
    return level_nodes.front();
}

void assign_objects(Node* root, std::vector<Object*>& objects_b) {
    if (!root) return;
    
    for(Object* obj : objects_b) {
        Node* p = root;
        bool assigned = false;
        
        if (!obj->mbr.overlaps(p->mbr)) continue;
        
        while(!p->is_leaf) {
            int overlap_count = 0;
            Node* overlap_child = nullptr;
            
            for(Node* ch : p->children) {
                if (obj->mbr.overlaps(ch->mbr)) {
                    overlap_count++;
                    overlap_child = ch;
                }
            }
            
            if (overlap_count == 0) {
                 break; 
            } else if (overlap_count > 1) {
                p->assigned_b_objects.push_back(obj);
                assigned = true;
                break;
            } else {
                p = overlap_child;
            }
        }
        if (!assigned && p->is_leaf) {
             p->assigned_b_objects.push_back(obj);
        }
    }
}

void collect_leaves(Node* n, std::vector<Node*>& leaves) {
    if (n->is_leaf) {
        leaves.push_back(n);
    } else {
        for(Node* ch : n->children) {
            collect_leaves(ch, leaves);
        }
    }
}

void collect_join_work_nodes(Node* n, std::vector<Node*>& work_nodes) {
    if (!n) {
        return;
    }
    if (!n->assigned_b_objects.empty()) {
        work_nodes.push_back(n);
    }
    if (!n->is_leaf) {
        for (Node* ch : n->children) {
            collect_join_work_nodes(ch, work_nodes);
        }
    }
}

std::vector<std::pair<int, int>> build_candidates_for_node(Node* n) {
    std::vector<std::pair<int, int>> local_candidates;
    if (!n || n->assigned_b_objects.empty()) {
        return local_candidates;
    }

    std::vector<Node*> descendant_leaves;
    collect_leaves(n, descendant_leaves);

    for (Node* leaf : descendant_leaves) {
        // Assigned objects may have been attached higher in the tree, so
        // we still need an MBR guard against each descendant leaf.
        for (Object* objB : n->assigned_b_objects) {
            if (!objB->mbr.overlaps(leaf->mbr)) {
                continue;
            }
            for (Object* objA : leaf->leaf_objects) {
                if (objB->mbr.overlaps(objA->mbr)) {
                    local_candidates.emplace_back(objA->id, objB->id);
                }
            }
        }
    }

    return local_candidates;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <datasetA.bin> <datasetB.bin> [threads]" << std::endl;
        return 1;
    }

    std::string fileA = argv[1];
    std::string fileB = argv[2];
    int numThreads = -1;
    if (argc > 3) numThreads = std::stoi(argv[3]);

    #ifdef _OPENMP
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
    std::cout << "OpenMP enabled";
    if (numThreads > 0) {
        std::cout << " (requested threads: " << numThreads << ")";
    } else {
        std::cout << " (using runtime default thread count)";
    }
    std::cout << std::endl;
    #else
    if (numThreads > 0) {
        std::cout << "OpenMP not available; requested threads ignored" << std::endl;
    }
    #endif

    std::cout << "Loading datasets..." << std::endl;
    auto startAll = std::chrono::high_resolution_clock::now();
    GeometryData dataA = RaySpace::IO::readBinaryFile(fileA);
    GeometryData dataB = RaySpace::IO::readBinaryFile(fileB);

    int maxObjA = -1;
    for(int id : dataA.triangleToObject) if(id > maxObjA) maxObjA = id;
    int maxObjB = -1;
    for(int id : dataB.triangleToObject) if(id > maxObjB) maxObjB = id;

    std::vector<Object> objsA(maxObjA + 1);
    for(size_t i=0; i<objsA.size(); ++i) objsA[i].id = i;
    
    for(size_t i=0; i<dataA.indices.size(); ++i) {
        int oid = dataA.triangleToObject[i];
        if(oid < 0) continue;
        uint3 idx = dataA.indices[i];
        Triangle_3 tri(
             Point_3(dataA.vertices[idx.x].x, dataA.vertices[idx.x].y, dataA.vertices[idx.x].z),
             Point_3(dataA.vertices[idx.y].x, dataA.vertices[idx.y].y, dataA.vertices[idx.y].z),
             Point_3(dataA.vertices[idx.z].x, dataA.vertices[idx.z].y, dataA.vertices[idx.z].z)
        );
        objsA[oid].triangles.push_back(tri);
        for(int k=0; k<3; ++k) {
             float p[3] = {(float)tri.vertex(k).x(), (float)tri.vertex(k).y(), (float)tri.vertex(k).z()}; 
             objsA[oid].mbr.extend(p);
        }
    }
    
    std::vector<Object> objsB(maxObjB + 1);
    for(size_t i=0; i<objsB.size(); ++i) objsB[i].id = i;
    for(size_t i=0; i<dataB.indices.size(); ++i) {
        int oid = dataB.triangleToObject[i];
        if(oid < 0) continue;
        uint3 idx = dataB.indices[i];
        Triangle_3 tri(
             Point_3(dataB.vertices[idx.x].x, dataB.vertices[idx.x].y, dataB.vertices[idx.x].z),
             Point_3(dataB.vertices[idx.y].x, dataB.vertices[idx.y].y, dataB.vertices[idx.y].z),
             Point_3(dataB.vertices[idx.z].x, dataB.vertices[idx.z].y, dataB.vertices[idx.z].z)
        );
        objsB[oid].triangles.push_back(tri);
        objsB[oid].cgal_tris.push_back({tri, oid}); 
        for(int k=0; k<3; ++k) {
             float p[3] = {(float)tri.vertex(k).x(), (float)tri.vertex(k).y(), (float)tri.vertex(k).z()};
             objsB[oid].mbr.extend(p);
        }
    }

    // Build per-object AABB trees for B
    #pragma omp parallel for
    for(int i=0; i<(int)objsB.size(); ++i) {
        if(objsB[i].cgal_tris.empty()) continue;
        objsB[i].cgal_tree = std::make_shared<AABB_Tree>(objsB[i].cgal_tris.begin(), objsB[i].cgal_tris.end());
        objsB[i].cgal_tree->build();
    }

    std::cout << "Building TOUCH Tree on Dataset A..." << std::endl;
    std::vector<Object*> ptrsA;
    for(auto& o : objsA) if(!o.triangles.empty()) ptrsA.push_back(&o);
    
    Node* root = nullptr;
    if(!ptrsA.empty()) root = build_str_tree(ptrsA, 2);

    std::cout << "Assigning Dataset B to Tree..." << std::endl;
    std::vector<Object*> ptrsB;
    for(auto& o : objsB) if(!o.triangles.empty()) ptrsB.push_back(&o);
    
    if(root) assign_objects(root, ptrsB);

    std::cout << "Executing Join..." << std::endl;
    auto startQuery = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, int>> candidates;
    if (root) {
        std::vector<Node*> join_work_nodes;
        collect_join_work_nodes(root, join_work_nodes);

        #ifdef _OPENMP
        std::vector<std::vector<std::pair<int, int>>> thread_candidate_buffers;
        thread_candidate_buffers.resize(omp_get_max_threads());

        #pragma omp parallel
        {
            #pragma omp single
            {
                std::cout << "Candidate generation threads: " << omp_get_num_threads() << std::endl;
            }

            std::vector<std::pair<int, int>> local_candidates;
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < static_cast<int>(join_work_nodes.size()); ++i) {
                std::vector<std::pair<int, int>> node_candidates = build_candidates_for_node(join_work_nodes[i]);
                local_candidates.insert(
                    local_candidates.end(),
                    node_candidates.begin(),
                    node_candidates.end());
            }

            thread_candidate_buffers[omp_get_thread_num()] = std::move(local_candidates);
        }

        size_t total_candidates = 0;
        for (const auto& buf : thread_candidate_buffers) {
            total_candidates += buf.size();
        }
        candidates.reserve(total_candidates);
        for (auto& buf : thread_candidate_buffers) {
            candidates.insert(candidates.end(), buf.begin(), buf.end());
        }
        #else
        for (Node* work_node : join_work_nodes) {
            std::vector<std::pair<int, int>> node_candidates = build_candidates_for_node(work_node);
            candidates.insert(candidates.end(), node_candidates.begin(), node_candidates.end());
        }
        #endif
    }

    std::cout << "Candidates found: " << candidates.size() << std::endl;
    
    size_t totalOverlaps = 0;

    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Refinement threads: " << omp_get_num_threads() << std::endl;
        }

        #pragma omp for reduction(+:totalOverlaps)
        for(int i=0; i<(int)candidates.size(); ++i) {
            int idA = candidates[i].first;
            int idB = candidates[i].second;
            
            Object& objA = objsA[idA];
            Object& objB = objsB[idB];
            
            bool intersects = false;
            if(objB.cgal_tree) {
                for(const auto& tri : objA.triangles) {
                    if(objB.cgal_tree->do_intersect(tri)) {
                        intersects = true;
                        break;
                    }
                }
            }
            if(intersects) totalOverlaps++;
        }
    }
    #else
    for(int i=0; i<(int)candidates.size(); ++i) {
        int idA = candidates[i].first;
        int idB = candidates[i].second;
        
        Object& objA = objsA[idA];
        Object& objB = objsB[idB];
        
        bool intersects = false;
        if(objB.cgal_tree) {
            for(const auto& tri : objA.triangles) {
                if(objB.cgal_tree->do_intersect(tri)) {
                    intersects = true;
                    break;
                }
            }
        }
        if(intersects) totalOverlaps++;
    }
    #endif

    auto endAll = std::chrono::high_resolution_clock::now();
    uint64_t queryTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(endAll - startQuery).count();
    
    std::cout << "Total Time: " << std::chrono::duration<double>(endAll - startAll).count() << "s" << std::endl;
    std::cout << "Query Time: " << queryTimeUs << " us (" << queryTimeUs / 1000.0 << " ms)" << std::endl;
    std::cout << "Total Overlaps: " << totalOverlaps << std::endl;

    return 0;
}
