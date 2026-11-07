#pragma once

// Reading named columns and numeric cells from the rows of a table
// (row_source.hpp), the way every loader reads them: a named column the table
// lacks is an error, and a numeric cell that is empty, an NA marker,
// unparseable or not finite is missing, never a zero.

#include "resolve/categorical.hpp"
#include "resolve/csv_utils.hpp"
#include "resolve/row_source.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace resolve {

// A numeric cell, or nullopt for an empty / NA marker / unparseable /
// non-finite value.
inline std::optional<float> parse_numeric_cell(std::string_view s) {
    if (is_na_string(s)) return std::nullopt;
    auto v = parse_float_strict(s);  // locale-free, rejects trailing garbage
    if (!v || !std::isfinite(*v)) return std::nullopt;
    return v;
}

// The index of a named role column in `source`. A named column the source
// lacks is a configuration error (typically a typo'd role name), never a
// silent "no such feature": it would otherwise train a model without the input
// the caller asked for and bake the wrong feature count into the checkpoint.
inline int require_column(const RowSource& source, const std::string& name, const char* role,
                          const char* table) {
    const int col = source.column_index(name);
    if (col < 0) {
        throw std::runtime_error(std::string(role) + " column '" + name + "' not found in the " +
                                 table + " table");
    }
    return col;
}

// Index of an optional role, -1 when the role is unset.
inline int optional_column(const RowSource& source, const std::optional<std::string>& name,
                           const char* role, const char* table) {
    return name ? require_column(source, *name, role, table) : -1;
}

// The cell of `row` at `col`, or nullopt when the row is too short to hold it.
inline std::optional<std::string_view> cell_at(RowFields row, int col) {
    if (col < 0 || row.size() <= static_cast<size_t>(col)) return std::nullopt;
    return row[static_cast<size_t>(col)];
}

inline std::optional<float> parse_cell_at(RowFields row, int col) {
    const auto cell = cell_at(row, col);
    return cell ? parse_numeric_cell(*cell) : std::nullopt;
}

}  // namespace resolve
