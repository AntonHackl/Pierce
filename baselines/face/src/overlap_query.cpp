#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <set>
#include <atomic>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "BinaryIO.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Triangle_3 Triangle_3;

// A structure to hold a triangle and its parent object ID
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

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <datasetA.bin> <datasetB.bin> [threads] [--output-csv path]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  datasetA.bin   First dataset in custom binary format" << std::endl;
    std::cout << "  datasetB.bin   Second dataset in custom binary format" << std::endl;
    std::cout << "  threads        Number of OpenMP threads (optional, default: auto)" << std::endl;
    std::cout << "  --output-csv   Optional output CSV path for overlapping object pairs" << std::endl;
}

static void exportCsv(const std::string& path, const std::vector<std::set<int>>& overlapByA) {
    std::ofstream csv(path);
    if (!csv.is_open()) {
        std::cerr << "Warning: could not open CSV output path: " << path << std::endl;
        return;
    }

    csv << "a_object_id,b_object_id\n";
    for (size_t a = 0; a < overlapByA.size(); ++a) {
        for (int b : overlapByA[a]) {
            csv << a << "," << b << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (arg1 == "--help" || arg1 == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string fileA = argv[1];
    std::string fileB = argv[2];
    int numThreads = -1;
    std::string outputCsv = "mesh_overlap_results.csv";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-csv" && i + 1 < argc) {
            outputCsv = argv[++i];
            continue;
        }
        if (!arg.empty() && arg[0] != '-' && numThreads < 0) {
            numThreads = std::stoi(arg);
            continue;
        }
        std::cerr << "Unknown argument: " << arg << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    #ifdef _OPENMP
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
    #endif

    std::cout << "Loading datasets..." << std::endl;
    auto startAll = std::chrono::high_resolution_clock::now();
    
    GeometryData dataA = RaySpace::IO::readBinaryFile(fileA);
    GeometryData dataB = RaySpace::IO::readBinaryFile(fileB);

    std::cout << "Building AABB Tree for Dataset B..." << std::endl;
    std::vector<IndexedTriangle> trisB;
    trisB.reserve(dataB.indices.size());
    for (size_t i = 0; i < dataB.indices.size(); ++i) {
        uint3 idx = dataB.indices[i];
        trisB.push_back({
            Triangle_3(
                Point_3(dataB.vertices[idx.x].x, dataB.vertices[idx.x].y, dataB.vertices[idx.x].z),
                Point_3(dataB.vertices[idx.y].x, dataB.vertices[idx.y].y, dataB.vertices[idx.y].z),
                Point_3(dataB.vertices[idx.z].x, dataB.vertices[idx.z].y, dataB.vertices[idx.z].z)
            ),
            dataB.triangleToObject[i]
        });
    }

    AABB_Tree treeB(trisB.begin(), trisB.end());
    treeB.build();

    std::cout << "Grouping Dataset A into objects..." << std::endl;
    int maxObjA = -1;
    for (int id : dataA.triangleToObject) if (id > maxObjA) maxObjA = id;
    
    struct ObjectA {
        std::vector<Triangle_3> triangles;
    };
    std::vector<ObjectA> objsA(maxObjA + 1);
    for (size_t i = 0; i < dataA.indices.size(); ++i) {
        int oid = dataA.triangleToObject[i];
        if (oid < 0) continue;
        uint3 idx = dataA.indices[i];
        objsA[oid].triangles.emplace_back(
            Point_3(dataA.vertices[idx.x].x, dataA.vertices[idx.x].y, dataA.vertices[idx.x].z),
            Point_3(dataA.vertices[idx.y].x, dataA.vertices[idx.y].y, dataA.vertices[idx.y].z),
            Point_3(dataA.vertices[idx.z].x, dataA.vertices[idx.z].y, dataA.vertices[idx.z].z)
        );
    }

    std::cout << "Executing Join Query (Parallel over Dataset A objects)..." << std::endl;
    auto startQuery = std::chrono::high_resolution_clock::now();
    size_t totalOverlapsCount = 0;
    std::vector<std::set<int>> overlapByA(objsA.size());

    #pragma omp parallel for reduction(+:totalOverlapsCount)
    for (int i = 0; i < (int)objsA.size(); ++i) {
        if (objsA[i].triangles.empty()) continue;
        
        std::set<int> seenB;
        for (const auto& triA : objsA[i].triangles) {
            std::vector<IndexedTrianglePrimitive::Id> intersected;
            treeB.all_intersected_primitives(triA, std::back_inserter(intersected));
            for (auto p : intersected) {
                seenB.insert(p->objectId);
            }
        }
        overlapByA[i] = std::move(seenB);
        totalOverlapsCount += overlapByA[i].size();
    }

    auto endAll = std::chrono::high_resolution_clock::now();
    uint64_t queryTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(endAll - startQuery).count();
    std::cout << "Total Time: " << std::chrono::duration<double>(endAll - startAll).count() << "s" << std::endl;
    std::cout << "Query Time: " << queryTimeUs << " us (" << queryTimeUs / 1000.0 << " ms)" << std::endl;
    std::cout << "Total Overlaps: " << totalOverlapsCount << std::endl;
    exportCsv(outputCsv, overlapByA);
    std::cout << "Wrote overlap pairs CSV: " << outputCsv << std::endl;

    return 0;
}
