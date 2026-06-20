#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using std::ofstream;
using std::string;
using std::vector;

struct RunConfig {
    string manifest;
    vector<string> obj_files;
    vector<string> off_files;
    string output_name = "converted";
    string output_dir;
    string work_dir;
    string log_dir;
    string validator_path;
    int sample_rate = 30;
    bool use_hausdorff = true;
    bool skip_validation = false;

    string manifest_a;
    string manifest_b;
    string output_a;
    string output_b;
    bool split_mode = false;
};

struct ObjResult {
    string obj_path;
    string obj_id;
    bool success = false;
    string reason;
    string off_path;
};

class Logger {
public:
    explicit Logger(const fs::path& log_path) {
        fs::create_directories(log_path.parent_path());
        stream_.open(log_path, std::ios::out | std::ios::trunc);
        if (!stream_.is_open()) {
            throw std::runtime_error("Failed to open log file: " + log_path.string());
        }
    }

    template <typename... Args>
    void info(const char* tag, Args&&... args) {
        log("INFO", tag, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warn(const char* tag, Args&&... args) {
        log("WARN", tag, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(const char* tag, Args&&... args) {
        log("ERROR", tag, std::forward<Args>(args)...);
    }

private:
    ofstream stream_;

    static string now_str() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    template <typename... Args>
    static string concat(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        return oss.str();
    }

    template <typename... Args>
    void log(const char* level, const char* tag, Args&&... args) {
        const string msg = concat(std::forward<Args>(args)...);
        const string line = "[" + now_str() + "] [" + level + "] [" + tag + "] " + msg;
        std::fprintf(stderr, "%s\n", line.c_str());
        stream_ << line << '\n';
        stream_.flush();
    }
};

static string trim(const string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static string obj_id_from_path(const string& obj_path) {
    return fs::path(obj_path).stem().string();
}

static string shell_quote(const string& s) {
    string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out += "'";
    return out;
}

static string find_validator(const RunConfig& cfg) {
    if (!cfg.validator_path.empty() && fs::exists(cfg.validator_path)) {
        return cfg.validator_path;
    }

    const vector<string> candidates = {
        "./baselines/obj_validator_for_dt",
        "./baselines/tdbase_extensions/build/obj_validator_for_dt",
        "./baselines/tdbase_extensions/build_release/obj_validator_for_dt",
        "./baselines/tdbase_extensions/build_agent/obj_validator_for_dt",
        "./obj_validator_for_dt",
    };

    for (const auto& c : candidates) {
        if (fs::exists(c)) {
            return c;
        }
    }
    return "";
}

static string find_tdbase_exec() {
    const vector<string> candidates = {
        "./baselines/tdbase_extensions/build/tdbase",
        "./baselines/tdbase_extensions/build_release/tdbase",
        "./baselines/tdbase_extensions/build_agent/tdbase",
        "./tdbase",
    };

    for (const auto& c : candidates) {
        if (fs::exists(c)) {
            return c;
        }
    }
    return "";
}

static vector<string> read_manifest(const string& manifest_path, Logger& log) {
    vector<string> paths;
    std::ifstream in(manifest_path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + manifest_path);
    }

    string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }
        paths.push_back(t);
    }

    log.info("MANIFEST", "Loaded ", paths.size(), " entries from ", manifest_path);
    return paths;
}

static string sanitize_stem(const string& p) {
    string s = fs::path(p).stem().string();
    for (char& c : s) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    if (s.empty()) {
        s = "mesh";
    }
    return s;
}

static bool validate_to_off(
    const string& validator_exec,
    const string& obj_path,
    const string& off_path,
    Logger& log) {

    const string cmd = shell_quote(validator_exec) + " " +
                       shell_quote(obj_path) + " " +
                       shell_quote(off_path) +
                       " > /dev/null 2>&1";

    log.info("VALIDATE", "Running validator for ", obj_path);
    const int rc = std::system(cmd.c_str());
    log.info("VALIDATE", "Validator exit code=", rc, " for ", obj_path);

    if (rc != 0) {
        return false;
    }
    if (!fs::exists(off_path)) {
        log.warn("VALIDATE", "Validator returned success but OFF missing: ", off_path);
        return false;
    }
    return true;
}

static string dt_output_path(const string& stem, const string& output_dir) {
    return (fs::path(output_dir) / (stem + ".dt")).string();
}

static int run_tdbase_pack_with_logs(
    const string& tdbase_exec,
    const string& output_dt,
    const vector<string>& off_paths,
    const string& stdout_log,
    const string& stderr_log,
    Logger& log,
    const string& log_tag,
    const string& command_tag);

struct OffRepairStats {
    size_t input_faces = 0;
    size_t kept_faces = 0;
    size_t dropped_small_faces = 0;
    size_t dropped_invalid_indices = 0;
    size_t dropped_degenerate = 0;
    size_t dropped_halfedge_conflict = 0;
};

static bool sanitize_off_for_tdbase(
    const string& input_off,
    const string& output_off,
    OffRepairStats& stats,
    Logger& log) {
    std::ifstream in(input_off);
    if (!in.is_open()) {
        log.warn("REPAIR", "Cannot open OFF for repair: ", input_off);
        return false;
    }

    const string body_tmp = output_off + ".body.tmp";
    std::ofstream body(body_tmp, std::ios::out | std::ios::trunc);
    if (!body.is_open()) {
        log.warn("REPAIR", "Cannot open temporary OFF body file: ", body_tmp);
        return false;
    }

    string magic;
    if (!(in >> magic) || magic != "OFF") {
        log.warn("REPAIR", "Invalid OFF header in ", input_off);
        return false;
    }

    size_t vertex_count = 0, face_count = 0, edge_count = 0;
    if (!(in >> vertex_count >> face_count >> edge_count)) {
        log.warn("REPAIR", "Invalid OFF counts line in ", input_off);
        return false;
    }

    stats.input_faces = face_count;

    for (size_t i = 0; i < vertex_count; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(in >> x >> y >> z)) {
            log.warn("REPAIR", "Failed to parse vertex ", i, " in ", input_off);
            return false;
        }
        body << std::fixed << std::setprecision(15) << x << " " << y << " " << z << "\n";
    }

    std::unordered_set<uint64_t> directed_edges;
    directed_edges.reserve(face_count * 2);

    auto edge_key = [](uint32_t a, uint32_t b) -> uint64_t {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    for (size_t fi = 0; fi < face_count; ++fi) {
        size_t n = 0;
        if (!(in >> n)) {
            log.warn("REPAIR", "Failed to parse face size at face index ", fi, " in ", input_off);
            return false;
        }

        vector<uint32_t> idx;
        idx.reserve(n);
        bool index_parse_ok = true;
        for (size_t k = 0; k < n; ++k) {
            long long raw = -1;
            if (!(in >> raw)) {
                index_parse_ok = false;
                break;
            }
            if (raw < 0 || raw >= static_cast<long long>(vertex_count)) {
                index_parse_ok = false;
            }
            idx.push_back(static_cast<uint32_t>(raw));
        }
        if (!index_parse_ok) {
            ++stats.dropped_invalid_indices;
            continue;
        }
        if (n < 3) {
            ++stats.dropped_small_faces;
            continue;
        }

        const uint32_t v0 = idx[0];
        for (size_t k = 1; k + 1 < idx.size(); ++k) {
            const uint32_t v1 = idx[k];
            const uint32_t v2 = idx[k + 1];

            if (v0 == v1 || v1 == v2 || v0 == v2) {
                ++stats.dropped_degenerate;
                continue;
            }

            const uint64_t e01 = edge_key(v0, v1);
            const uint64_t e12 = edge_key(v1, v2);
            const uint64_t e20 = edge_key(v2, v0);
            if (directed_edges.count(e01) || directed_edges.count(e12) || directed_edges.count(e20)) {
                ++stats.dropped_halfedge_conflict;
                continue;
            }

            directed_edges.insert(e01);
            directed_edges.insert(e12);
            directed_edges.insert(e20);

            body << "3 " << v0 << " " << v1 << " " << v2 << "\n";
            ++stats.kept_faces;
        }
    }

    body.close();

    std::ofstream out(output_off, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        log.warn("REPAIR", "Cannot write repaired OFF: ", output_off);
        return false;
    }
    out << "OFF\n";
    out << vertex_count << " " << stats.kept_faces << " 0\n";

    std::ifstream body_in(body_tmp);
    out << body_in.rdbuf();
    out.close();
    body_in.close();

    std::error_code ec;
    fs::remove(body_tmp, ec);

    log.info(
        "REPAIR",
        "Repaired OFF ", input_off,
        " -> ", output_off,
        " faces_in=", stats.input_faces,
        " faces_kept=", stats.kept_faces,
        " dropped_small=", stats.dropped_small_faces,
        " dropped_invalid_idx=", stats.dropped_invalid_indices,
        " dropped_degenerate=", stats.dropped_degenerate,
        " dropped_halfedge_conflict=", stats.dropped_halfedge_conflict);
    return true;
}

static bool extract_facet_index_from_stderr(const string& stderr_path, size_t& facet_index_out) {
    std::ifstream in(stderr_path);
    if (!in.is_open()) {
        return false;
    }
    string line;
    while (std::getline(in, line)) {
        const string needle = "facet ";
        const size_t p = line.find(needle);
        if (p == string::npos) {
            continue;
        }
        size_t i = p + needle.size();
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) {
            ++i;
        }
        size_t j = i;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) {
            ++j;
        }
        if (j > i) {
            facet_index_out = static_cast<size_t>(std::strtoull(line.substr(i, j - i).c_str(), nullptr, 10));
            return true;
        }
    }
    return false;
}

static bool remove_off_facet_by_index_1based(const string& off_path, size_t facet_1based, Logger& log) {
    std::ifstream in(off_path);
    if (!in.is_open()) {
        log.warn("AUTOFIX", "Cannot open OFF file for face removal: ", off_path);
        return false;
    }

    string magic;
    if (!(in >> magic) || magic != "OFF") {
        log.warn("AUTOFIX", "Invalid OFF header in ", off_path);
        return false;
    }
    size_t vertex_count = 0, face_count = 0, edge_count = 0;
    if (!(in >> vertex_count >> face_count >> edge_count)) {
        log.warn("AUTOFIX", "Invalid OFF counts in ", off_path);
        return false;
    }
    if (facet_1based < 1 || facet_1based > face_count) {
        log.warn("AUTOFIX", "Facet index out of range for removal: ", facet_1based, " face_count=", face_count);
        return false;
    }

    const string tmp_path = off_path + ".autofix.tmp";
    std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        log.warn("AUTOFIX", "Cannot open OFF tmp output: ", tmp_path);
        return false;
    }
    out << "OFF\n";
    out << vertex_count << " " << (face_count - 1) << " 0\n";

    for (size_t i = 0; i < vertex_count; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(in >> x >> y >> z)) {
            log.warn("AUTOFIX", "Failed to parse vertex ", i, " during face removal in ", off_path);
            return false;
        }
        out << std::fixed << std::setprecision(15) << x << " " << y << " " << z << "\n";
    }

    string dummy;
    std::getline(in, dummy); // consume the remainder of the current line

    size_t written_faces = 0;
    size_t seen_faces = 0;
    string line;
    while (seen_faces < face_count && std::getline(in, line)) {
        ++seen_faces;
        if (seen_faces == facet_1based) {
            continue;
        }
        out << line << "\n";
        ++written_faces;
    }

    if (seen_faces < face_count) {
        log.warn("AUTOFIX", "Unexpected EOF while rewriting faces in ", off_path);
        return false;
    }
    if (written_faces != face_count - 1) {
        log.warn("AUTOFIX", "Face rewrite count mismatch in ", off_path);
        return false;
    }

    out.close();
    in.close();

    std::error_code ec;
    fs::rename(tmp_path, off_path, ec);
    if (ec) {
        log.warn("AUTOFIX", "Failed to replace OFF after face removal: ", ec.message());
        return false;
    }
    return true;
}

static bool auto_repair_off_with_probes(
    const string& tdbase_exec,
    const string& original_off,
    const string& work_dir,
    const string& log_dir,
    const string& probe_tag,
    Logger& log,
    string& repaired_off_out) {
    const string repaired_off = (fs::path(work_dir) / (probe_tag + ".autofix.off")).string();
    std::error_code ec;
    fs::copy_file(original_off, repaired_off, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        log.warn("AUTOFIX", "Failed to create autofix OFF copy from ", original_off, ": ", ec.message());
        return false;
    }

    const size_t max_attempts = 24;
    size_t prev_facet_idx = 0;
    size_t descending_by_one_streak = 0;
    for (size_t attempt = 0; attempt < max_attempts; ++attempt) {
        const string probe_out = (fs::path(work_dir) / (probe_tag + ".autofix_probe.dt")).string();
        const string probe_stdout = (fs::path(log_dir) / (probe_tag + ".autofix_probe.stdout.log")).string();
        const string probe_stderr = (fs::path(log_dir) / (probe_tag + ".autofix_probe.stderr.log")).string();
        fs::remove(probe_out, ec);

        const int rc = run_tdbase_pack_with_logs(
            tdbase_exec,
            probe_out,
            vector<string>{repaired_off},
            probe_stdout,
            probe_stderr,
            log,
            "AUTOFIX",
            "autofix_probe_command");
        const bool ok = (rc == 0) && fs::exists(probe_out);
        fs::remove(probe_out, ec);
        if (ok) {
            repaired_off_out = repaired_off;
            log.info("AUTOFIX", "Probe repair succeeded after ", attempt, " face removals for ", original_off);
            return true;
        }

        size_t facet_idx = 0;
        if (!extract_facet_index_from_stderr(probe_stderr, facet_idx)) {
            log.warn("AUTOFIX", "Could not parse failing facet index from ", probe_stderr);
            return false;
        }
        if (attempt > 0) {
            if (prev_facet_idx > 0 && facet_idx + 1 == prev_facet_idx) {
                ++descending_by_one_streak;
            } else {
                descending_by_one_streak = 0;
            }
            if (descending_by_one_streak >= 8) {
                log.warn(
                    "AUTOFIX",
                    "Stopping autofix due to non-converging pattern (failing facet index keeps descending by 1): ",
                    facet_idx);
                return false;
            }
        }
        prev_facet_idx = facet_idx;

        size_t chosen = facet_idx;
        if (chosen < 1) {
            chosen = 1;
        }
        bool removed = remove_off_facet_by_index_1based(repaired_off, chosen, log);
        if (!removed) {
            removed = remove_off_facet_by_index_1based(repaired_off, chosen + 1, log);
            if (removed) {
                ++chosen;
            }
        }
        if (!removed && chosen > 1) {
            removed = remove_off_facet_by_index_1based(repaired_off, chosen - 1, log);
            if (removed) {
                --chosen;
            }
        }
        if (!removed) {
            log.warn("AUTOFIX", "Failed to remove any candidate facet near reported index ", facet_idx);
            return false;
        }
        log.warn("AUTOFIX", "Removed facet index ", chosen, " and retrying repair probe for ", original_off);
    }

    log.warn("AUTOFIX", "Exceeded max attempts while repairing ", original_off);
    return false;
}

static int run_tdbase_pack_with_logs(
    const string& tdbase_exec,
    const string& output_dt,
    const vector<string>& off_paths,
    const string& stdout_log,
    const string& stderr_log,
    Logger& log,
    const string& log_tag,
    const string& command_tag) {
    string cmd = shell_quote(tdbase_exec) + " pack " + shell_quote(output_dt);
    for (const auto& off_path : off_paths) {
        cmd += " " + shell_quote(off_path);
    }
    cmd += " > " + shell_quote(stdout_log) + " 2> " + shell_quote(stderr_log);

    log.info(log_tag.c_str(), "pack_stdout_log=", stdout_log);
    log.info(log_tag.c_str(), "pack_stderr_log=", stderr_log);
    log.info(log_tag.c_str(), command_tag, "=", cmd);
    return std::system(cmd.c_str());
}

static void write_summary_csv(const vector<ObjResult>& results, const fs::path& csv_path, Logger& log) {
    ofstream out(csv_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        log.error("SUMMARY", "Failed to write summary CSV: ", csv_path.string());
        return;
    }

    out << "obj_id,obj_path,success,reason,off_path\n";
    for (const auto& r : results) {
        out << '"' << r.obj_id << "\",";
        out << '"' << r.obj_path << "\",";
        out << (r.success ? "1" : "0") << ",";
        out << '"' << r.reason << "\",";
        out << '"' << r.off_path << "\"\n";
    }

    log.info("SUMMARY", "Wrote summary CSV: ", csv_path.string());
}

static void process_entries(
    const RunConfig& cfg,
    const vector<string>& entries,
    const string& output_dir,
    const string& split_tag,
    const string& output_stem,
    const string& input_label,
    const bool entries_are_off) {
    fs::create_directories(cfg.work_dir);
    fs::create_directories(cfg.log_dir);
    fs::create_directories(output_dir);

    const fs::path split_log_path = fs::path(cfg.log_dir) / (split_tag + ".log");
    Logger log(split_log_path);

    log.info("START", "Processing split=", split_tag);
    log.info("CFG", "input=", input_label);
    log.info("CFG", "entries=", entries.size());
    log.info("CFG", "output_dir=", output_dir);
    log.info("CFG", "output_stem=", output_stem);
    log.info("CFG", "work_dir=", cfg.work_dir);
    log.info("CFG", "log_dir=", cfg.log_dir);
    log.info("CFG", "sample_rate=", cfg.sample_rate);
    log.info("CFG", "use_hausdorff=", (cfg.use_hausdorff ? "true" : "false"));

    string validator_exec;
    auto ensure_validator_exec = [&]() -> const string& {
        if (validator_exec.empty()) {
            validator_exec = find_validator(cfg);
            if (validator_exec.empty()) {
                throw std::runtime_error("Could not locate obj_validator_for_dt binary");
            }
            log.info("CFG", "validator_exec=", validator_exec);
        }
        return validator_exec;
    };
    if (!entries_are_off && !cfg.skip_validation) {
        ensure_validator_exec();
    }
    log.info("CFG", "skip_validation=", (cfg.skip_validation ? "true" : "false"));
    log.info("CFG", "entries_are_off=", (entries_are_off ? "true" : "false"));

    vector<ObjResult> results;
    results.reserve(entries.size());

    vector<string> off_paths;
    off_paths.reserve(entries.size());

    const auto t0 = std::chrono::steady_clock::now();
    auto progress_maybe_log = [&](size_t done, size_t total) {
        if (total == 0) {
            return;
        }
        const bool is_last = (done == total);
        const size_t stride = (total >= 100) ? std::max<size_t>(size_t(1), total / 100) : size_t(1);
        if (!is_last && (done % stride != 0)) {
            return;
        }
        const double pct = 100.0 * static_cast<double>(done) / static_cast<double>(total);
        log.info("PROGRESS", "OFF prep ", done, "/", total, " (", std::fixed, std::setprecision(1), pct, "%)");
    };

    for (size_t i = 0; i < entries.size(); ++i) {
        const string& obj_path = entries[i];
        ObjResult r;
        r.obj_path = obj_path;
        r.obj_id = obj_id_from_path(obj_path);

        log.info("OBJ", "[", (i + 1), "/", entries.size(), "] id=", r.obj_id, " path=", obj_path);

        if (!fs::exists(obj_path)) {
            r.reason = entries_are_off ? "missing_off" : "missing_obj";
            log.warn("OBJ", "Missing input file: ", obj_path);
            results.push_back(r);
            progress_maybe_log(i + 1, entries.size());
            continue;
        }

        if (entries_are_off) {
            r.off_path = obj_path;
            const string repaired_off = (fs::path(cfg.work_dir) / (r.obj_id + ".sanitized.off")).string();
            OffRepairStats repair_stats;
            if (!sanitize_off_for_tdbase(r.off_path, repaired_off, repair_stats, log)) {
                r.reason = "off_repair_failed";
                log.warn("OBJ", "OFF repair failed id=", r.obj_id);
                results.push_back(r);
                progress_maybe_log(i + 1, entries.size());
                continue;
            }
            if (repair_stats.kept_faces == 0) {
                r.reason = "off_repair_empty";
                log.warn("OBJ", "OFF repair produced zero faces id=", r.obj_id);
                results.push_back(r);
                progress_maybe_log(i + 1, entries.size());
                continue;
            }
            r.off_path = repaired_off;
            off_paths.push_back(r.off_path);
            r.success = true;
            r.reason = "ok";
            results.push_back(r);
            log.info("OBJ", "Accepted OFF id=", r.obj_id);
            progress_maybe_log(i + 1, entries.size());
            continue;
        }

        const string off_path = (fs::path(cfg.work_dir) / (r.obj_id + ".off")).string();
        r.off_path = off_path;

        if (cfg.skip_validation) {
            if (fs::exists(off_path)) {
                log.info("OBJ", "Reusing existing OFF (skip-validation): ", off_path);
            } else {
                log.warn("OBJ", "Validation skipped but OFF missing; generating OFF now: ", off_path);
                const bool generated = validate_to_off(ensure_validator_exec(), obj_path, off_path, log);
                if (!generated) {
                    r.reason = "off_generation_failed";
                    log.warn("OBJ", "Failed to generate OFF from OBJ: ", obj_path);
                    results.push_back(r);
                    progress_maybe_log(i + 1, entries.size());
                    continue;
                }
            }
        } else {
            const bool valid = validate_to_off(ensure_validator_exec(), obj_path, off_path, log);
            if (!valid) {
                r.reason = "validator_rejected";
                log.warn("OBJ", "Rejected by validator: ", obj_path);
                results.push_back(r);
                progress_maybe_log(i + 1, entries.size());
                continue;
            }
        }

        const string repaired_off = (fs::path(cfg.work_dir) / (r.obj_id + ".sanitized.off")).string();
        OffRepairStats repair_stats;
        if (!sanitize_off_for_tdbase(off_path, repaired_off, repair_stats, log)) {
            r.reason = "off_repair_failed";
            log.warn("OBJ", "OFF repair failed id=", r.obj_id);
            results.push_back(r);
            progress_maybe_log(i + 1, entries.size());
            continue;
        }
        if (repair_stats.kept_faces == 0) {
            r.reason = "off_repair_empty";
            log.warn("OBJ", "OFF repair produced zero faces id=", r.obj_id);
            results.push_back(r);
            progress_maybe_log(i + 1, entries.size());
            continue;
        }

        r.off_path = repaired_off;
        off_paths.push_back(r.off_path);
        r.success = true;
        r.reason = "ok";
        results.push_back(r);
        log.info("OBJ", "Accepted id=", r.obj_id);
        progress_maybe_log(i + 1, entries.size());
    }

    size_t accepted = 0;
    for (const auto& r : results) {
        if (r.success) {
            ++accepted;
        }
    }
    const size_t skipped = results.size() - accepted;

    log.info("SUMMARY", "Total=", results.size(), " accepted=", accepted, " skipped=", skipped);

    const fs::path summary_csv = fs::path(cfg.log_dir) / (split_tag + "_summary.csv");
    write_summary_csv(results, summary_csv, log);

    if (off_paths.empty()) {
        log.warn("SUMMARY", "No valid meshes accepted. No DT file produced.");
        return;
    }

    const string output_dt = dt_output_path(output_stem, output_dir);
    const string tdbase_exec = find_tdbase_exec();
    if (tdbase_exec.empty()) {
        throw std::runtime_error("Could not locate tdbase executable");
    }

    const string pack_stdout = (fs::path(cfg.log_dir) / (split_tag + "_tdbase_pack.stdout.log")).string();
    const string pack_stderr = (fs::path(cfg.log_dir) / (split_tag + "_tdbase_pack.stderr.log")).string();
    log.info("DT", "Packing ", off_paths.size(), " OFF meshes into ", output_dt);
    log.info("PROGRESS", "Starting DT pack phase");
    int rc = run_tdbase_pack_with_logs(
        tdbase_exec,
        output_dt,
        off_paths,
        pack_stdout,
        pack_stderr,
        log,
        "DT",
        "pack_command");
    if (rc != 0) {
        const int exit_code = WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
        auto read_tail = [](const string& path, size_t max_lines) {
            vector<string> lines;
            std::ifstream in(path);
            string line;
            while (std::getline(in, line)) {
                lines.push_back(line);
            }
            if (lines.size() > max_lines) {
                lines.erase(lines.begin(), lines.end() - static_cast<std::ptrdiff_t>(max_lines));
            }
            return lines;
        };
        const auto stderr_tail = read_tail(pack_stderr, 40);
        const auto stdout_tail = read_tail(pack_stdout, 20);
        log.error("DT", "tdbase pack failed with exit code ", exit_code);
        if (!stderr_tail.empty()) {
            log.error("DT", "Last stderr lines:");
            for (const auto& line : stderr_tail) {
                log.error("DT", line);
            }
        }
        if (!stdout_tail.empty()) {
            log.error("DT", "Last stdout lines:");
            for (const auto& line : stdout_tail) {
                log.error("DT", line);
            }
        }

        log.warn("DT", "Bulk pack failed; attempting per-mesh OFF filtering and retry.");
        vector<string> valid_offs;
        vector<string> rejected_offs;
        valid_offs.reserve(off_paths.size());
        rejected_offs.reserve(off_paths.size());

        for (size_t i = 0; i < off_paths.size(); ++i) {
            const string& candidate = off_paths[i];
            const string probe_out = (fs::path(cfg.work_dir) / (split_tag + "_probe_" + std::to_string(i) + ".dt")).string();
            const string probe_stdout = (fs::path(cfg.log_dir) / (split_tag + "_probe_" + std::to_string(i) + ".stdout.log")).string();
            const string probe_stderr = (fs::path(cfg.log_dir) / (split_tag + "_probe_" + std::to_string(i) + ".stderr.log")).string();
            std::error_code rm_ec;
            fs::remove(probe_out, rm_ec);

            const int probe_rc = run_tdbase_pack_with_logs(
                tdbase_exec,
                probe_out,
                vector<string>{candidate},
                probe_stdout,
                probe_stderr,
                log,
                "PROBE",
                "probe_command");
            const bool ok = (probe_rc == 0) && fs::exists(probe_out);
            if (ok) {
                valid_offs.push_back(candidate);
                log.info("PROBE", "Accepted OFF: ", candidate);
            } else {
                const int probe_exit = WIFEXITED(probe_rc) ? WEXITSTATUS(probe_rc) : probe_rc;
                log.warn("PROBE", "Probe failed (exit=", probe_exit, ") for OFF: ", candidate, " -> attempting autofix");
                string repaired_candidate;
                const string probe_tag = split_tag + "_probe_" + std::to_string(i);
                if (auto_repair_off_with_probes(
                        tdbase_exec,
                        candidate,
                        cfg.work_dir,
                        cfg.log_dir,
                        probe_tag,
                        log,
                        repaired_candidate)) {
                    valid_offs.push_back(repaired_candidate);
                    log.info("PROBE", "Accepted OFF after autofix: ", repaired_candidate);
                } else {
                    rejected_offs.push_back(candidate);
                    log.warn("PROBE", "Rejected OFF after autofix attempts: ", candidate);
                }
            }
            fs::remove(probe_out, rm_ec);
        }

        log.info("PROBE", "Filtering result: valid=", valid_offs.size(), " rejected=", rejected_offs.size());
        for (const auto& rej : rejected_offs) {
            log.warn("PROBE", "Skipped OFF: ", rej);
        }

        if (valid_offs.empty()) {
            throw std::runtime_error(
                string("tdbase pack failed and all OFF meshes were rejected in fallback probing ") +
                "(see logs: " + pack_stderr + " and " + pack_stdout + ")");
        }

        log.info("DT", "Retrying pack with valid OFF subset (", valid_offs.size(), " meshes)");
        rc = run_tdbase_pack_with_logs(
            tdbase_exec,
            output_dt,
            valid_offs,
            pack_stdout,
            pack_stderr,
            log,
            "DT",
            "retry_pack_command");
        if (rc != 0) {
            const int retry_exit = WIFEXITED(rc) ? WEXITSTATUS(rc) : rc;
            throw std::runtime_error(
                "tdbase retry pack failed with exit code " + std::to_string(retry_exit) +
                " (see logs: " + pack_stderr + " and " + pack_stdout + ")");
        }
        log.info("DT", "Retry pack succeeded after skipping ", rejected_offs.size(), " OFF meshes");
    }
    if (!fs::exists(output_dt)) {
        throw std::runtime_error("tdbase pack returned success but output DT was not created: " + output_dt);
    }
    log.info("DT", "DT written: ", output_dt);
    log.info("PROGRESS", "DT pack phase completed");

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log.info("DONE", "Completed split=", split_tag, " elapsed_ms=", ms);
}

static void process_manifest(const RunConfig& cfg, const string& manifest_path, const string& output_dir, const string& split_tag) {
    Logger manifest_log(fs::path(cfg.log_dir) / (split_tag + "_manifest.log"));
    const vector<string> entries = read_manifest(manifest_path, manifest_log);
    const string stem = fs::path(manifest_path).stem().string();
    process_entries(cfg, entries, output_dir, split_tag, stem, string("manifest=") + manifest_path, false);
}

static void process_obj_files(const RunConfig& cfg) {
    if (cfg.obj_files.empty()) {
        throw std::runtime_error("No --obj files provided");
    }
    vector<string> entries = cfg.obj_files;
    string stem = cfg.output_name;
    if (stem.empty()) {
        stem = sanitize_stem(entries.front());
    }
    process_entries(cfg, entries, cfg.output_dir, "single", stem, "--obj inputs", false);
}

static void process_off_files(const RunConfig& cfg) {
    if (cfg.off_files.empty()) {
        throw std::runtime_error("No --off files provided");
    }
    vector<string> entries = cfg.off_files;
    string stem = cfg.output_name;
    if (stem.empty()) {
        stem = sanitize_stem(entries.front());
    }
    process_entries(cfg, entries, cfg.output_dir, "single", stem, "--off inputs", true);
}

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage:\n");
    std::fprintf(stderr, "  %s --obj <file.obj> [--obj <file2.obj> ...] --output <dir> [--output-name <stem>] [--work <dir>] [--log <dir>]\n", prog);
    std::fprintf(stderr, "      [--validator <path>] [--skip-validation] [--sample-rate <int>] [--no-hausdorff]\n");
    std::fprintf(stderr, "  %s --off <file.off> [--off <file2.off> ...] --output <dir> [--output-name <stem>] [--work <dir>] [--log <dir>]\n", prog);
    std::fprintf(stderr, "      [--sample-rate <int>] [--no-hausdorff]\n");
    std::fprintf(stderr, "  %s --manifest <path> --output <dir> [--work <dir>] [--log <dir>]\n", prog);
    std::fprintf(stderr, "      [--validator <path>] [--skip-validation] [--sample-rate <int>] [--no-hausdorff]\n");
    std::fprintf(stderr, "  %s --manifest-a <path> --manifest-b <path> --output-a <dir> --output-b <dir>\n", prog);
    std::fprintf(stderr, "      [--work <dir>] [--log <dir>] [--validator <path>] [--skip-validation] [--sample-rate <int>] [--no-hausdorff]\n");
}

int main(int argc, char** argv) {
    RunConfig cfg;
    cfg.work_dir = "./tmp/obj_to_dt_work";
    cfg.log_dir = "./tmp/obj_to_dt_logs";

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];

        if (a == "--manifest" && i + 1 < argc) {
            cfg.manifest = argv[++i];
        } else if (a == "--obj" && i + 1 < argc) {
            cfg.obj_files.push_back(argv[++i]);
        } else if (a == "--off" && i + 1 < argc) {
            cfg.off_files.push_back(argv[++i]);
        } else if (a == "--output" && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        } else if (a == "--output-name" && i + 1 < argc) {
            cfg.output_name = argv[++i];
        } else if (a == "--manifest-a" && i + 1 < argc) {
            cfg.manifest_a = argv[++i];
            cfg.split_mode = true;
        } else if (a == "--manifest-b" && i + 1 < argc) {
            cfg.manifest_b = argv[++i];
            cfg.split_mode = true;
        } else if (a == "--output-a" && i + 1 < argc) {
            cfg.output_a = argv[++i];
            cfg.split_mode = true;
        } else if (a == "--output-b" && i + 1 < argc) {
            cfg.output_b = argv[++i];
            cfg.split_mode = true;
        } else if (a == "--work" && i + 1 < argc) {
            cfg.work_dir = argv[++i];
        } else if (a == "--log" && i + 1 < argc) {
            cfg.log_dir = argv[++i];
        } else if (a == "--validator" && i + 1 < argc) {
            cfg.validator_path = argv[++i];
        } else if (a == "--skip-validation") {
            cfg.skip_validation = true;
        } else if (a == "--sample-rate" && i + 1 < argc) {
            cfg.sample_rate = std::atoi(argv[++i]);
        } else if (a == "--no-hausdorff") {
            cfg.use_hausdorff = false;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        if (cfg.split_mode) {
            if (cfg.manifest_a.empty() || cfg.manifest_b.empty() || cfg.output_a.empty() || cfg.output_b.empty()) {
                print_usage(argv[0]);
                return 1;
            }

            RunConfig a_cfg = cfg;
            a_cfg.work_dir = (fs::path(cfg.work_dir) / "split_a").string();
            a_cfg.log_dir = (fs::path(cfg.log_dir) / "split_a").string();
            process_manifest(a_cfg, cfg.manifest_a, cfg.output_a, "split_a");

            RunConfig b_cfg = cfg;
            b_cfg.work_dir = (fs::path(cfg.work_dir) / "split_b").string();
            b_cfg.log_dir = (fs::path(cfg.log_dir) / "split_b").string();
            process_manifest(b_cfg, cfg.manifest_b, cfg.output_b, "split_b");
        } else {
            if (cfg.output_dir.empty()) {
                print_usage(argv[0]);
                return 1;
            }
            if (!cfg.obj_files.empty() && !cfg.off_files.empty()) {
                throw std::runtime_error("Use either --obj or --off inputs, not both");
            } else if (!cfg.off_files.empty()) {
                process_off_files(cfg);
            } else if (!cfg.obj_files.empty()) {
                process_obj_files(cfg);
            } else if (!cfg.manifest.empty()) {
                process_manifest(cfg, cfg.manifest, cfg.output_dir, "single");
            } else {
                print_usage(argv[0]);
                return 1;
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal error: %s\n", e.what());
        return 2;
    }
}
