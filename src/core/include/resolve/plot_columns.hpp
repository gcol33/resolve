#pragma once

// Numeric plot-level columns of a dataset's own tables, for a given list of
// plots: what a step after prediction reads about the plots it was given
// (gap fitting's positions, years and recorded values; terrain correction's
// coordinates, elevation-model descriptors and location uncertainty).

#include "resolve/row_source.hpp"

#include <optional>
#include <string>
#include <vector>

namespace resolve {

// The values of `columns` for the plots `plot_ids`, in that order: one vector
// per column, NaN where a cell is empty, an NA marker or not a number, parsed
// in double precision. Read
// from the header table (one row per plot), or without one from each plot's
// first row of the species table. Rows of plots not listed are skipped.
// Throws std::runtime_error for a named column the table lacks, a plot the
// table does not hold, or a plot the header table lists twice. `role` names
// the caller in those messages ("gap fitting", "terrain correction").
[[nodiscard]] std::vector<std::vector<double>> read_plot_columns(const std::optional<TableSource>& header,
                                                                 const TableSource& species,
                                                                 const std::string& plot_id_column,
                                                                 const std::vector<std::string>& columns,
                                                                 const std::vector<std::string>& plot_ids,
                                                                 const char* role);

// Every row of one table, in file order: the row's plot id and the values of
// `columns` (NaN where a cell is empty, an NA marker or not a number). The
// whole-table reading of a step that is handed a table of its own (the
// terrain CLI's coordinates and fitting tables). Throws std::runtime_error
// for a named column the table lacks.
struct TableColumns {
    std::vector<std::string> plot_ids;
    std::vector<std::vector<double>> values;  // one vector per column
};
[[nodiscard]] TableColumns read_table_columns(const TableSource& table, const std::string& plot_id_column,
                                              const std::vector<std::string>& columns, const char* role);

}  // namespace resolve
