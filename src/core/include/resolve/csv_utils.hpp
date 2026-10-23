#pragma once

#include "resolve/number_text.hpp"

#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace resolve {

namespace detail {

inline bool is_c_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

inline bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

inline std::string_view trim_c_space(std::string_view s) noexcept {
    while (!s.empty() && is_c_space(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_c_space(s.back())) s.remove_suffix(1);
    return s;
}

}  // namespace detail

// Strict, locale-independent float parse of a whole cell. Surrounding
// whitespace is ignored and a leading '+' accepted; anything else that is not
// a decimal number (digits with an optional '.', an optional exponent, an
// optional sign) is rejected rather than read up to the first bad character:
// "12abc" and "1,234" give nullopt, not 12 and 1. The decimal point is always
// '.', so a process LC_NUMERIC using a comma decimal separator (common on
// European Windows) cannot change how coordinates, abundances, covariates or
// regression targets parse. Spelled-out infinities and NaNs, hexadecimal
// numbers and values outside the float range are rejected as well; the empty
// cell is nullopt.
inline std::optional<float> parse_float_strict(std::string_view s) {
    s = detail::trim_c_space(s);
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
        if (!s.empty() && (s.front() == '-' || s.front() == '+')) return std::nullopt;
    }
    const size_t magnitude = !s.empty() && s.front() == '-' ? 1 : 0;
    if (magnitude >= s.size()) return std::nullopt;
    if (!detail::is_digit(s[magnitude]) && s[magnitude] != '.') return std::nullopt;
    for (size_t i = magnitude; i < s.size(); ++i) {
        const char c = s[i];
        if (!detail::is_digit(c) && c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-') {
            return std::nullopt;
        }
    }
    float value = 0.0f;
    if (!decimal_to_number(s, value)) return std::nullopt;
    return value;
}

// Strict base-10 int64 parse of a whole cell: optional leading whitespace, an
// optional sign, digits, and nothing after them (trailing whitespace
// included). nullopt for anything else and on overflow.
inline std::optional<int64_t> parse_int64_strict(std::string_view s) {
    while (!s.empty() && detail::is_c_space(s.front())) s.remove_prefix(1);
    if (!s.empty() && s.front() == '+') {
        s.remove_prefix(1);
        if (s.empty() || !detail::is_digit(s.front())) return std::nullopt;
    }
    if (s.empty()) return std::nullopt;
    int64_t value = 0;
    const char* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, value);
    if (ec != std::errc() || ptr != end) return std::nullopt;
    return value;
}

} // namespace resolve
