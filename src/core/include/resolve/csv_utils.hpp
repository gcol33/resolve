#pragma once

#include <string>
#include <optional>
#include <sstream>
#include <locale>

namespace resolve {

// Locale-independent strict float parse. Returns nullopt on empty input,
// unparseable text, OR trailing non-whitespace characters, so "12abc" and
// "1,234" are rejected rather than silently truncated to 12 / 1 (the
// leading-prefix behavior of std::stof). The stream is imbued with the
// classic ("C") locale, so a process LC_NUMERIC using a comma decimal
// separator (common on European Windows) cannot change how coordinates,
// abundances, covariates, or regression targets parse: the decimal point is
// always '.', and a comma is treated as trailing garbage.
//
// std::from_chars for floating point would be faster, but libc++ (the macOS
// CI toolchain) only gained floating-point from_chars very recently; the
// classic-locale stream is available on every supported compiler and yields
// identical results across platforms, so there is a single parse path rather
// than a Linux/macOS split.
inline std::optional<float> parse_float_strict(const std::string& s) {
    if (s.empty()) return std::nullopt;
    std::istringstream iss(s);
    iss.imbue(std::locale::classic());
    float v;
    iss >> v;
    if (iss.fail()) return std::nullopt;
    char extra;
    if (iss >> extra) return std::nullopt;  // trailing non-whitespace
    return v;
}

// Parse float safely with default value. Uses parse_float_strict, so failures
// (empty, unparseable, trailing garbage) map to default_val instead of a
// silently-truncated leading prefix.
inline float safe_stof(const std::string& s, float default_val = 0.0f) {
    auto v = parse_float_strict(s);
    return v ? *v : default_val;
}

// Parse int safely with default value. Integer parsing is unaffected by
// LC_NUMERIC (no decimal separator), but trailing garbage is rejected so a
// value like "12abc" maps to default_val rather than 12.
inline int safe_stoi(const std::string& s, int default_val = 0) {
    if (s.empty()) return default_val;
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return default_val;  // trailing garbage
        return v;
    } catch (...) {
        return default_val;
    }
}

} // namespace resolve
