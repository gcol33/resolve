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

} // namespace resolve
