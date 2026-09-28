#pragma once

// Writing prediction CSVs: shared by `resolve predict --model` and
// `resolve predict --suite`, so a label, a number and a missing value are
// spelled the same in both.

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace resolve_cli {

// Quote a CSV field when it carries a comma, quote, or newline. Class labels
// come from a user's CSV column and may contain any of them; emitting them raw
// would shift every following column (issue #110 item 5).
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

// A number as a CSV cell; NaN is written NA, the spelling R and pandas read as
// missing.
inline std::string csv_number(double v) {
    if (std::isnan(v)) return "NA";
    std::ostringstream out;
    out << v;
    return out.str();
}

// The label a class code prints as: the original CSV label when the checkpoint
// carries the class vocabulary (persisted since #76), otherwise the code
// itself (a pre-#76 checkpoint, or a column that was already integer-coded).
inline std::string class_label(const std::vector<std::string>& class_names, int64_t code) {
    const bool in_vocab = code >= 0 && code < static_cast<int64_t>(class_names.size());
    return in_vocab ? class_names[static_cast<size_t>(code)] : std::to_string(code);
}

}  // namespace resolve_cli
