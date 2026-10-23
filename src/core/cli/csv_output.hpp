#pragma once

// Writing prediction CSVs: shared by `resolve predict --model` and
// `resolve predict --suite`, so a label, a number and a missing value are
// spelled the same in both.

#include <cmath>
#include <string>
#include <type_traits>

#include "resolve/number_text.hpp"

namespace resolve_cli {

// Quote a CSV field when it carries a comma, quote, or newline. Class labels
// come from a user's CSV column and may contain any of them; emitting them raw
// would shift every following column.
inline std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

// A number as a CSV cell: the shortest decimal that reads back to the same
// float or double, so a coordinate keeps every digit it has; NaN is written NA,
// the spelling R and pandas read as missing.
template <typename T>
std::string csv_number(T v) {
    static_assert(std::is_floating_point_v<T>, "csv_number writes a float or a double");
    if (std::isnan(v)) return "NA";
    if (std::isinf(v)) return v > 0 ? "Inf" : "-Inf";
    return resolve::number_to_decimal(v);
}

}  // namespace resolve_cli
