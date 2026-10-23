#pragma once

// Locale-free conversion between floating-point numbers and decimal text.
//
// Every number the engine reads from text (CSV cells, JSON documents) or writes
// as text (JSON documents, prediction CSVs, CLI help defaults) goes through
// these two functions, so a process LC_NUMERIC with a comma decimal separator
// can change neither. A number is written in the shortest form that reads back
// to the same value: a float32 prediction prints as "47.12346", where six
// significant digits would print "47.1235" and the digits of its widening to
// double "47.123458862304688".
//
// The conversion is std::from_chars / std::to_chars where the standard library
// has them for floating point; otherwise (libc++ before LLVM 20) the C
// library's strtod / snprintf under a "C" locale object, which are the same
// correctly rounded conversions and are not affected by the process locale.

#include <charconv>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#if !(defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L)
#include <cerrno>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#if defined(__APPLE__)
#include <xlocale.h>
#else
#include <locale.h>
#endif
#define RESOLVE_NUMBER_TEXT_C_LOCALE 1
#endif

namespace resolve {

namespace number_text_detail {

#if defined(RESOLVE_NUMBER_TEXT_C_LOCALE)
inline locale_t c_locale() {
    static const locale_t locale = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(0));
    return locale;
}

inline int print_c(char* buffer, std::size_t size, int precision, double value) {
#if defined(__APPLE__)
    return snprintf_l(buffer, size, c_locale(), "%.*g", precision, value);
#else
    const locale_t previous = uselocale(c_locale());
    const int written = std::snprintf(buffer, size, "%.*g", precision, value);
    uselocale(previous);
    return written;
#endif
}
#endif

}  // namespace number_text_detail

// The correctly rounded float or double nearest to the decimal number `text`.
// False when the text does not convert as a whole (no surrounding whitespace,
// no leading '+'), or when the value lies outside the type's range.
template <typename T>
bool decimal_to_number(std::string_view text, T& value) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "decimal_to_number converts to float or double");
    if (text.empty()) return false;
#if !defined(RESOLVE_NUMBER_TEXT_C_LOCALE)
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    return ec == std::errc() && ptr == end;
#else
    if (text.front() == '+' || text.front() == ' ' || text.front() == '\t') return false;
    const std::string buffer(text);
    char* end = nullptr;
    errno = 0;
    if constexpr (std::is_same_v<T, float>) {
        value = strtof_l(buffer.c_str(), &end, number_text_detail::c_locale());
    } else {
        value = strtod_l(buffer.c_str(), &end, number_text_detail::c_locale());
    }
    return errno != ERANGE && end == buffer.c_str() + buffer.size();
#endif
}

// The shortest decimal text that reads back to `value` exactly. A finite value
// only: a caller decides what a NaN or an infinity is spelled as in its format.
template <typename T>
std::string number_to_decimal(T value) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "number_to_decimal formats a float or a double");
    char buffer[64];
#if !defined(RESOLVE_NUMBER_TEXT_C_LOCALE)
    const auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    (void)ec;
    return std::string(buffer, ptr);
#else
    for (int precision = 1; precision <= std::numeric_limits<T>::max_digits10; ++precision) {
        const int n = number_text_detail::print_c(buffer, sizeof(buffer), precision,
                                                  static_cast<double>(value));
        T back{};
        const std::string_view text(buffer, static_cast<std::size_t>(n));
        if (decimal_to_number(text, back) && back == value) return std::string(text);
    }
    return std::string(buffer);
#endif
}

}  // namespace resolve
