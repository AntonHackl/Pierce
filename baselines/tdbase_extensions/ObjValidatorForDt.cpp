#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================================
// Data Structures
// ============================================================================

struct Vertex {
    double x, y, z;
};

struct Face {
    unsigned int v1, v2, v3;
};

struct ValidationReport {
    bool file_exists = false;
    bool parse_success = false;
    bool all_triangles = false;
    bool valid_indices = false;
    bool no_isolated_vertices = false;
    bool min_geometry = false;
    
    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t non_triangular_faces = 0;
    size_t invalid_index_errors = 0;
    size_t isolated_vertices = 0;
    
    std::vector<std::string> error_messages;
    
    bool is_valid() const {
        return file_exists && parse_success && all_triangles && 
               valid_indices && no_isolated_vertices && min_geometry;
    }
};

// ============================================================================
// OBJ Validator Class
// ============================================================================

class ObjValidatorForDt {
private:
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    ValidationReport report;
    std::string input_file;
    
public:
    ObjValidatorForDt(const std::string& obj_file) : input_file(obj_file) {}
    
    // ========================================================================
    // Phase 1: Parse and Validate OBJ
    // ========================================================================
    
    bool parseObjFile() {
        // Check file exists
        report.file_exists = fs::exists(input_file);
        if (!report.file_exists) {
            report.error_messages.push_back("File does not exist: " + input_file);
            return false;
        }
        
        // Open and parse
        std::ifstream file(input_file);
        if (!file.is_open()) {
            report.error_messages.push_back("Could not open file for reading: " + input_file);
            return false;
        }
        
        vertices.clear();
        faces.clear();
        
        std::string line;
        int line_num = 0;
        size_t non_triangular_count = 0;
        
        while (std::getline(file, line)) {
            line_num++;
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            std::string type;
            iss >> type;
            
            if (type == "v") {
                // ============================================================
                // Parse Vertex
                // ============================================================
                double x, y, z;
                if (iss >> x >> y >> z) {
                    vertices.push_back({x, y, z});
                } else {
                    report.error_messages.push_back(
                        "Line " + std::to_string(line_num) + 
                        ": Invalid vertex format. Expected: v x y z");
                }
            }
            else if (type == "f") {
                // ============================================================
                // Parse Face
                // ============================================================
                std::vector<unsigned int> face_indices;
                std::string token;
                
                while (iss >> token) {
                    // Parse vertex index from format: v, v/vt, v/vt/vn, v//vn
                    size_t slash_pos = token.find('/');
                    std::string vertex_idx_str = 
                        (slash_pos != std::string::npos) ? 
                        token.substr(0, slash_pos) : token;
                    
                    try {
                        unsigned int idx = std::stoul(vertex_idx_str);
                        // Convert from 1-based OBJ indices to 0-based
                        face_indices.push_back(idx - 1);
                    } catch (const std::exception& e) {
                        report.invalid_index_errors++;
                        report.error_messages.push_back(
                            "Line " + std::to_string(line_num) + 
                            ": Invalid vertex index: " + vertex_idx_str);
                    }
                }
                
                // Check if face is a triangle (must have exactly 3 vertices)
                if (face_indices.size() == 3) {
                    faces.push_back({face_indices[0], face_indices[1], face_indices[2]});
                } else if (face_indices.size() > 0) {
                    non_triangular_count++;
                    report.error_messages.push_back(
                        "Line " + std::to_string(line_num) + 
                        ": Face has " + std::to_string(face_indices.size()) + 
                        " vertices (expected 3 for triangle)");
                }
            }
        }
        
        file.close();
        
        report.parse_success = true;
        report.vertex_count = vertices.size();
        report.face_count = faces.size();
        report.non_triangular_faces = non_triangular_count;
        
        return true;
    }
    
    // ========================================================================
    // Validate OBJ Structure
    // ========================================================================
    
    void validate() {
        if (!report.parse_success) return;
        
        // Check all triangles
        report.all_triangles = (report.non_triangular_faces == 0);
        
        // Check minimum geometry
        report.min_geometry = (report.vertex_count >= 4 && report.face_count >= 1);
        if (!report.min_geometry) {
            report.error_messages.push_back(
                "Insufficient geometry: need at least 4 vertices and 1 triangle, "
                "got " + std::to_string(report.vertex_count) + " vertices and " + 
                std::to_string(report.face_count) + " triangles");
        }
        
        // Validate all face indices are within vertex range
        report.valid_indices = true;
        for (size_t i = 0; i < faces.size(); ++i) {
            if (faces[i].v1 >= vertices.size() || 
                faces[i].v2 >= vertices.size() || 
                faces[i].v3 >= vertices.size()) {
                report.valid_indices = false;
                report.error_messages.push_back(
                    "Face " + std::to_string(i) + 
                    ": vertex index out of range. Indices: (" + 
                    std::to_string(faces[i].v1) + ", " +
                    std::to_string(faces[i].v2) + ", " +
                    std::to_string(faces[i].v3) + ") but only " +
                    std::to_string(vertices.size()) + " vertices available");
            }
        }
        
        // Check for isolated vertices (vertices not used in any face)
        std::set<unsigned int> used_vertices;
        for (const auto& face : faces) {
            used_vertices.insert(face.v1);
            used_vertices.insert(face.v2);
            used_vertices.insert(face.v3);
        }
        
        report.isolated_vertices = vertices.size() - used_vertices.size();
        report.no_isolated_vertices = (report.isolated_vertices == 0);
        
        if (report.isolated_vertices > 0) {
            report.error_messages.push_back(
                "Found " + std::to_string(report.isolated_vertices) + 
                " isolated vertices (not used in any face)");
        }
    }
    
    // ========================================================================
    // Phase 2: Convert to OFF Format and Write
    // ========================================================================
    
    bool writeOffFile(const std::string& output_file) {
        if (!report.valid_indices) {
            report.error_messages.push_back(
                "Cannot write OFF file: invalid face indices detected");
            return false;
        }
        
        std::ofstream out(output_file);
        if (!out.is_open()) {
            report.error_messages.push_back(
                "Could not open output file for writing: " + output_file);
            return false;
        }
        
        try {
            // Write OFF header
            out << "OFF\n";
            out << vertices.size() << " " << faces.size() << " 0\n";
            
            // Write vertices
            out << std::fixed << std::setprecision(15);
            for (const auto& v : vertices) {
                out << v.x << " " << v.y << " " << v.z << "\n";
            }
            
            // Write faces (OFF format: n v1 v2 v3 ... for n-sided polygon)
            // For triangles, format is: 3 v1 v2 v3
            for (const auto& f : faces) {
                out << "3 " << f.v1 << " " << f.v2 << " " << f.v3 << "\n";
            }
            
            out.close();
            return true;
        } catch (const std::exception& e) {
            report.error_messages.push_back(
                "Error writing OFF file: " + std::string(e.what()));
            return false;
        }
    }
    
    // ========================================================================
    // Reporting
    // ========================================================================
    
    void printReport() const {
        std::cout << "\n";
        std::cout << "=" << std::string(77, '=') << "\n";
        std::cout << "OBJ VALIDATOR FOR DT CONVERSION - REPORT\n";
        std::cout << "=" << std::string(77, '=') << "\n\n";
        
        std::cout << "Input File: " << input_file << "\n\n";
        
        // Overall status
        std::cout << "OVERALL STATUS: ";
        if (report.is_valid()) {
            std::cout << "✓ VALID - Can be converted to DT format\n";
        } else {
            std::cout << "✗ INVALID - Cannot convert to DT format\n";
        }
        
        std::cout << "\n";
        std::cout << "VALIDATION RESULTS:\n";
        std::cout << "-" << std::string(76, '-') << "\n";
        // File checks
        std::cout << "  File Exists:                 ";
        std::cout << (report.file_exists ? "✓ Yes" : "✗ No") << "\n";
        
        // Parsing
        std::cout << "  Parse Successful:            ";
        std::cout << (report.parse_success ? "✓ Yes" : "✗ No") << "\n";
        
        if (report.parse_success) {
            // Geometry statistics
            std::cout << "\n  Geometry Statistics:\n";
            std::cout << "    - Vertices:                " << report.vertex_count << "\n";
            std::cout << "    - Faces:                   " << report.face_count << "\n";
            std::cout << "    - Non-triangular faces:    " << report.non_triangular_faces << "\n";
            std::cout << "    - Invalid index errors:    " << report.invalid_index_errors << "\n";
            std::cout << "    - Isolated vertices:       " << report.isolated_vertices << "\n";
            
            // Validation checks
            std::cout << "\n  Validation Checks:\n";
            std::cout << "    - All Triangles:           ";
            std::cout << (report.all_triangles ? "✓ Yes" : "✗ No") << "\n";
            
            std::cout << "    - Valid Indices:           ";
            std::cout << (report.valid_indices ? "✓ Yes" : "✗ No") << "\n";
            
            std::cout << "    - Minimum Geometry:        ";
            std::cout << (report.min_geometry ? "✓ Yes" : "✗ No") << "\n";
            std::cout << "      (need ≥4 vertices, ≥1 triangle)\n";
            
            std::cout << "    - No Isolated Vertices:    ";
            std::cout << (report.no_isolated_vertices ? "✓ Yes" : "✗ No") << "\n";
        }
        
        // Error messages
        if (!report.error_messages.empty()) {
            std::cout << "\n";
            std::cout << "ERRORS / WARNINGS:\n";
            std::cout << "-" << std::string(76, '-') << "\n";
            for (size_t i = 0; i < report.error_messages.size(); ++i) {
                std::cout << "  [" << (i + 1) << "] " << report.error_messages[i] << "\n";
            }
        }
        
        std::cout << "\n";
        std::cout << "=" << std::string(77, '=') << "\n\n";
    }
    
    const ValidationReport& getReport() const {
        return report;
    }
    
    // ========================================================================
    // Main Workflow
    // ========================================================================
    
    bool validate_and_convert(const std::string& output_file) {
        std::cout << "Processing OBJ file: " << input_file << "\n";
        std::cout << "Output OFF file: " << output_file << "\n\n";
        
        // Phase 1: Parse
        std::cout << "[Phase 1] Parsing OBJ file...\n";
        if (!parseObjFile()) {
            std::cout << "✗ Parsing failed\n";
            printReport();
            return false;
        }
        std::cout << "✓ Parsing complete: " << report.vertex_count << " vertices, " 
                  << report.face_count << " triangles\n\n";
        
        // Validate
        std::cout << "[Phase 1] Validating OBJ structure...\n";
        validate();
        
        if (!report.is_valid()) {
            std::cout << "✗ Validation failed\n";
            printReport();
            return false;
        }
        std::cout << "✓ All validation checks passed\n\n";
        
        // Phase 2: Convert to OFF
        std::cout << "[Phase 2] Converting to OFF format...\n";
        if (!writeOffFile(output_file)) {
            std::cout << "✗ Conversion failed\n";
            printReport();
            return false;
        }
        std::cout << "✓ Successfully wrote OFF file: " << output_file << "\n\n";
        
        // Print report
        printReport();
        return true;
    }
};

// ============================================================================
// Main Program
// ============================================================================

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input.obj> [output.off]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  input.obj   - Path to OBJ file to validate\n";
    std::cout << "  output.off  - Path to output OFF file (optional)\n";
    std::cout << "                If not specified, uses: <input_base>.off\n\n";
    std::cout << "Description:\n";
    std::cout << "  Validates OBJ files for conversion to TDBase DT format.\n";
    std::cout << "  Checks for:\n";
    std::cout << "    - Valid OBJ parsing\n";
    std::cout << "    - All triangular faces\n";
    std::cout << "    - Valid vertex indices\n";
    std::cout << "    - No isolated vertices\n";
    std::cout << "    - Minimum geometry (≥4 vertices, ≥1 triangle)\n\n";
    std::cout << "  If validation passes, converts to OFF format.\n";
    std::cout << "  OFF files can be further validated with Phase 3 (CGAL checks).\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    
    std::string input_file = argv[1];
    
    // Determine output file
    std::string output_file;
    if (argc >= 3) {
        output_file = argv[2];
    } else {
        // Generate from input: input.obj -> input.off
        output_file = input_file;
        size_t dot_pos = output_file.rfind('.');
        if (dot_pos != std::string::npos) {
            output_file = output_file.substr(0, dot_pos);
        }
        output_file += ".off";
    }
    
    // Run validator
    ObjValidatorForDt validator(input_file);
    bool success = validator.validate_and_convert(output_file);
    
    return success ? 0 : 1;
}
