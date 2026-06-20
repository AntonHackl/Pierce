#pragma once

#include <string>
#include <fstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

inline bool pathExists(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

inline bool isDirectoryPath(const std::string& path)
{
    struct stat info;
#ifdef _WIN32
    return stat(path.c_str(), &info) == 0 && (info.st_mode & _S_IFDIR);
#else
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
#endif
}

inline std::string joinPath(const std::string& dir, const std::string& file)
{
    if (dir.empty()) return file;
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') return dir + file;
#ifdef _WIN32
    return dir + "\\" + file;
#else
    return dir + "/" + file;
#endif
}

inline std::string detectPTXPath(const std::string& moduleName, const std::string& hint = "")
{
    if (!hint.empty()) {
        if (isDirectoryPath(hint)) {
            return joinPath(hint, moduleName);
        }
        if (pathExists(hint)) {
            size_t pos = hint.find_last_of("\\/");
            std::string file = (pos == std::string::npos) ? hint : hint.substr(pos + 1);
            if (file == moduleName) {
                return hint;
            }
            return joinPath(hint.substr(0, pos + 1), moduleName);
        }
        size_t pos = hint.find_last_of("\\/");
        if (pos != std::string::npos) {
            return joinPath(hint.substr(0, pos + 1), moduleName);
        }
    }

    std::string dir;

#ifdef _WIN32
    char exePath[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::string path(exePath, len);
        size_t pos = path.find_last_of("\\/");
        if (pos != std::string::npos) {
            dir = path.substr(0, pos + 1);
        }
    }
#else
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1) {
        exePath[len] = '\0';
        std::string path(exePath);
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            dir = path.substr(0, pos + 1);
        }
    }
#endif

    if (!dir.empty()) {
        {
            std::string candidate = dir + moduleName;
            if (pathExists(candidate)) return candidate;
        }

#ifdef _WIN32
        {
            std::string candidate = dir + "..\\" + moduleName;
            if (pathExists(candidate)) return candidate;
        }
        {
            std::string candidate = dir + "..\\..\\" + moduleName;
            if (pathExists(candidate)) return candidate;
        }
#else
        {
            std::string candidate = dir + "../" + moduleName;
            if (pathExists(candidate)) return candidate;
        }
        {
            std::string candidate = dir + "../../" + moduleName;
            if (pathExists(candidate)) return candidate;
        }
#endif
    }

    {
        std::string candidate = moduleName;
        if (pathExists(candidate)) return candidate;
    }

    return moduleName;
}
