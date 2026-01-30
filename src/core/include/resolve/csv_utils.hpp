#pragma once

#include <string>

namespace resolve {

// Parse float safely with default value
inline float safe_stof(const std::string& s, float default_val = 0.0f) {
    if (s.empty()) return default_val;
    try {
        return std::stof(s);
    } catch (...) {
        return default_val;
    }
}

// Parse int safely with default value
inline int safe_stoi(const std::string& s, int default_val = 0) {
    if (s.empty()) return default_val;
    try {
        return std::stoi(s);
    } catch (...) {
        return default_val;
    }
}

} // namespace resolve
