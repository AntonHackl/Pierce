#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>

class CliOptionsBase {
public:
    virtual ~CliOptionsBase() = default;

    using HelpEntry = std::pair<std::string, std::string>;

    bool helpRequested = false;

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                helpRequested = true;
                continue;
            }
            if (parseCommonOption(arg, i, argc, argv)) {
                continue;
            }
            parseApplicationOption(arg, i, argc, argv);
        }
    }

protected:
    static void printHelpMessage(
        const char* exeName,
        const std::string& usage,
        const std::string& description,
        const std::vector<HelpEntry>& options
    ) {
        std::cout << "Usage: " << exeName << " " << usage << std::endl;
        if (!description.empty()) {
            std::cout << description << std::endl;
        }
        std::cout << "Options:" << std::endl;
        for (const auto& [flag, text] : options) {
            std::cout << "  " << std::left << std::setw(34) << flag << text << std::endl;
        }
    }

    static void appendHelpFlag(std::vector<HelpEntry>& options) {
        options.emplace_back("--help, -h", "Show this help message");
    }

    virtual bool parseCommonOption(const std::string& arg, int& i, int argc, char* argv[]) = 0;
    virtual bool parseApplicationOption(const std::string& arg, int& i, int argc, char* argv[]) {
        (void)arg;
        (void)i;
        (void)argc;
        (void)argv;
        return false;
    }
};

class MeshPairCliOptions : public CliOptionsBase {
public:
    explicit MeshPairCliOptions(const std::string& defaultOutputPath)
        : outputJsonPath(defaultOutputPath) {}

    std::string mesh1Path;
    std::string mesh2Path;
    std::string outputJsonPath;
    std::string ptxPath;

    bool hasRequiredMeshInputs() const {
        return !mesh1Path.empty() && !mesh2Path.empty();
    }

protected:
    static void appendMeshPairHelp(
        std::vector<HelpEntry>& options,
        const std::string& mesh1Description = "Path to first mesh dataset (geometry file)",
        const std::string& mesh2Description = "Path to second mesh dataset (geometry file)",
        bool includeOutput = true,
        bool includePtx = true
    ) {
        options.emplace_back("--mesh1 <path>", mesh1Description);
        options.emplace_back("--mesh2 <path>", mesh2Description);
        if (includeOutput) {
            options.emplace_back("--output <path>", "Path to JSON file for performance timing output");
        }
        if (includePtx) {
            options.emplace_back("--ptx <path>", "Path to PTX file or directory (default: auto-detect)");
        }
    }

    bool parseCommonOption(const std::string& arg, int& i, int argc, char* argv[]) override {
        if (arg == "--mesh1" && i + 1 < argc) {
            mesh1Path = argv[++i];
            return true;
        }
        if (arg == "--mesh2" && i + 1 < argc) {
            mesh2Path = argv[++i];
            return true;
        }
        if (arg == "--output" && i + 1 < argc) {
            outputJsonPath = argv[++i];
            return true;
        }
        if (arg == "--ptx" && i + 1 < argc) {
            ptxPath = argv[++i];
            return true;
        }
        return false;
    }
};

class BenchmarkMeshPairCliOptions : public MeshPairCliOptions {
public:
    explicit BenchmarkMeshPairCliOptions(const std::string& defaultOutputPath)
        : MeshPairCliOptions(defaultOutputPath) {}

    int numberOfRuns = 1;
    int warmupRuns = 2;
    bool exportResults = true;
    bool allowNoExportFlag = false;

    void sanitizeRunCounts() {
        if (numberOfRuns < 1) {
            numberOfRuns = 1;
        }
        if (warmupRuns < 0) {
            warmupRuns = 0;
        }
    }

protected:
    static void appendBenchmarkRunHelp(std::vector<HelpEntry>& options) {
        options.emplace_back("--runs <number>", "Number of measured query runs (default: 1)");
        options.emplace_back("--warmup-runs <number>", "Number of warmup iterations (default: 2)");
    }

    static void appendNoExportHelp(std::vector<HelpEntry>& options) {
        options.emplace_back("--no-export", "Do not export results to CSV");
    }

    bool parseCommonOption(const std::string& arg, int& i, int argc, char* argv[]) override {
        if (MeshPairCliOptions::parseCommonOption(arg, i, argc, argv)) {
            return true;
        }
        if (arg == "--runs" && i + 1 < argc) {
            numberOfRuns = std::atoi(argv[++i]);
            return true;
        }
        if (arg == "--warmup-runs" && i + 1 < argc) {
            warmupRuns = std::atoi(argv[++i]);
            return true;
        }
        if (allowNoExportFlag && arg == "--no-export") {
            exportResults = false;
            return true;
        }
        return false;
    }
};
