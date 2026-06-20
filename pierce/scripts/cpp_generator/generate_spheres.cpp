#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sstream>

using namespace std;

struct Vec3 { double x, y, z; };
struct Face { int a, b, c; };

struct Mesh {
    vector<Vec3> vertices;
    vector<Face> faces;
};

Mesh load_template(const string& filename) {
    Mesh mesh;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Could not open template OBJ: " << filename << endl;
        exit(1);
    }

    string line;
    double min_coords[3] = {1e18, 1e18, 1e18};
    double max_coords[3] = {-1e18, -1e18, -1e18};

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string type;
        if (!(ss >> type)) continue;

        if (type == "v") {
            Vec3 v;
            if (ss >> v.x >> v.y >> v.z) {
                mesh.vertices.push_back(v);
                min_coords[0] = min(min_coords[0], v.x);
                min_coords[1] = min(min_coords[1], v.y);
                min_coords[2] = min(min_coords[2], v.z);
                max_coords[0] = max(max_coords[0], v.x);
                max_coords[1] = max(max_coords[1], v.y);
                max_coords[2] = max(max_coords[2], v.z);
            }
        } else if (type == "f") {
            int indices[3];
            string parts[3];
            if (ss >> parts[0] >> parts[1] >> parts[2]) {
                for (int i = 0; i < 3; ++i) {
                    indices[i] = stoi(parts[i]); 
                }
                mesh.faces.push_back({indices[0]-1, indices[1]-1, indices[2]-1});
            }
        }
    }

    double cx = (min_coords[0] + max_coords[0]) / 2.0;
    double cy = (min_coords[1] + max_coords[1]) / 2.0;
    double cz = (min_coords[2] + max_coords[2]) / 2.0;
    
    double size_x = max_coords[0] - min_coords[0];
    double size_y = max_coords[1] - min_coords[1];
    double size_z = max_coords[2] - min_coords[2];
    double max_ext = max({size_x, size_y, size_z});
    if (max_ext == 0.0) max_ext = 1.0;

    for (auto& v : mesh.vertices) {
        v.x = (v.x - cx) / max_ext;
        v.y = (v.y - cy) / max_ext;
        v.z = (v.z - cz) / max_ext;
    }

    if (mesh.vertices.empty() || mesh.faces.empty()) {
        cerr << "Error: Template has no geometry." << endl;
        exit(1);
    }

    return mesh;
}

double compute_universe_for_spheres(double selectivity, double min_size, double max_size) {
    if (selectivity <= 0.0) return 100.0;
    // Expected center-to-center overlap volume for spheres with diameter s1, s2
    // V = pi/6 * E[(s1+s2)^3] = pi/3 * (E[s^3] + 3*E[s^2]*E[s])
    double e_s = (min_size + max_size) / 2.0;
    double e_s2 = (min_size*min_size + min_size*max_size + max_size*max_size) / 3.0;
    double e_s3 = (min_size*min_size*min_size + min_size*min_size*max_size + min_size*max_size*max_size + max_size*max_size*max_size) / 4.0;
    double v_overlap_exp = (M_PI / 3.0) * (e_s3 + 3.0 * e_s2 * e_s);
    return std::cbrt(v_overlap_exp / selectivity);
}

void write_obj_file(const string& filepath, int num_objs, double min_size, double max_size, double extent, int seed, const Mesh& tmpl) {
    auto start_time = chrono::high_resolution_clock::now();
    
    mt19937 gen(seed);
    uniform_real_distribution<double> dist_pos(0.0, extent);
    uniform_real_distribution<double> dist_size(min_size, max_size);

    // Use a large buffer for FILE output (64MB)
    FILE* out = fopen(filepath.c_str(), "wb");
    if (!out) { cerr << "Could not open " << filepath << endl; exit(1); }
    char* big_buf = new char[64 * 1024 * 1024];
    setvbuf(out, big_buf, _IOFBF, 64 * 1024 * 1024);

    fprintf(out, "# Generated mesh\n# Objects: %d\n\n", num_objs);

    long long num_v = tmpl.vertices.size();

    for (int i = 0; i < num_objs; ++i) {
        fprintf(out, "o obj_%d\n", i);
        double cx = dist_pos(gen);
        double cy = dist_pos(gen);
        double cz = dist_pos(gen);
        double s  = dist_size(gen);

        for (const auto& v : tmpl.vertices) {
            fprintf(out, "v %.6f %.6f %.6f\n", v.x * s + cx, v.y * s + cy, v.z * s + cz);
        }
        fprintf(out, "\n");

        long long v_off = (long long)i * num_v + 1;
        for (const auto& f : tmpl.faces) {
            fprintf(out, "f %lld %lld %lld\n", v_off + f.a, v_off + f.b, v_off + f.c);
        }
        fprintf(out, "\n");

        if (i % 5000 == 0 && i > 0) cout << "  ... " << i << " objects written" << endl;
    }

    fclose(out);
    delete[] big_buf;
    
    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    cout << "Generated " << num_objs << " objects in " << duration << "s" << endl;
}

int main(int argc, char* argv[]) {
    string template_obj, output_a, output_b;
    int num_objs_a = 0, num_objs_b = 0;
    double min_size = 0, max_size = 0, selectivity = 0;
    int seed = 42;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--template-obj") template_obj = argv[++i];
        else if (arg == "-na" || arg == "--num-objs-a") num_objs_a = stoi(argv[++i]);
        else if (arg == "-nb" || arg == "--num-objs-b") num_objs_b = stoi(argv[++i]);
        else if (arg == "--min-size") min_size = stod(argv[++i]);
        else if (arg == "--max-size") max_size = stod(argv[++i]);
        else if (arg == "-s" || arg == "--selectivity") selectivity = stod(argv[++i]);
        else if (arg == "-oa" || arg == "--output-a") output_a = argv[++i];
        else if (arg == "-ob" || arg == "--output-b") output_b = argv[++i];
        else if (arg == "--seed") seed = stoi(argv[++i]);
    }

    Mesh tmpl = load_template(template_obj);
    double universe_extent = compute_universe_for_spheres(selectivity, min_size, max_size);

    cout << "Generating datasets with universe extent: " << universe_extent << endl;
    write_obj_file(output_a, num_objs_a, min_size, max_size, universe_extent, seed, tmpl);
    write_obj_file(output_b, num_objs_b, min_size, max_size, universe_extent, seed + 1, tmpl);

    return 0;
}
