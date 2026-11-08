#include "resolve/plot_columns.hpp"

#include "resolve/row_cells.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace resolve {

std::vector<std::vector<double>> read_plot_columns(const std::optional<TableSource>& header,
                                                   const TableSource& species, const std::string& plot_id_column,
                                                   const std::vector<std::string>& columns,
                                                   const std::vector<std::string>& plot_ids, const char* role) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, int32_t> plot_of;
    plot_of.reserve(plot_ids.size());
    for (std::size_t i = 0; i < plot_ids.size(); ++i) plot_of.emplace(plot_ids[i], static_cast<int32_t>(i));

    std::vector<std::vector<double>> values(columns.size(), std::vector<double>(plot_ids.size(), kNaN));
    std::vector<char> seen(plot_ids.size(), 0);
    const char* table = header ? "header" : "species";
    const bool first_row_only = !header;
    const auto source = header ? header->open() : species.open();
    const int plot_col = require_column(*source, plot_id_column, "plot_id", table);
    std::vector<int> cols;
    cols.reserve(columns.size());
    for (const auto& name : columns) cols.push_back(require_column(*source, name, role, table));
    source->read_rows([&](size_t, RowFields row) {
        const auto id = cell_at(row, plot_col);
        if (!id) return;
        const auto it = plot_of.find(std::string(*id));
        if (it == plot_of.end()) return;
        const auto p = static_cast<std::size_t>(it->second);
        if (seen[p]) {
            if (first_row_only) return;
            throw std::runtime_error("Duplicate plot_id '" + std::string(*id) + "' in header data.");
        }
        seen[p] = 1;
        for (std::size_t c = 0; c < cols.size(); ++c) {
            if (const auto v = parse_cell_at<double>(row, cols[c])) values[c][p] = *v;
        }
    });
    for (std::size_t p = 0; p < plot_ids.size(); ++p) {
        if (!seen[p]) {
            throw std::runtime_error(std::string(role) + ": plot '" + plot_ids[p] + "' is not in the " + table +
                                     " table");
        }
    }
    return values;
}

TableColumns read_table_columns(const TableSource& table, const std::string& plot_id_column,
                                const std::vector<std::string>& columns, const char* role) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    TableColumns out;
    out.values.resize(columns.size());
    const auto source = table.open();
    const int plot_col = require_column(*source, plot_id_column, "plot_id", "input");
    std::vector<int> cols;
    cols.reserve(columns.size());
    for (const auto& name : columns) cols.push_back(require_column(*source, name, role, "input"));
    source->read_rows([&](size_t, RowFields row) {
        const auto id = cell_at(row, plot_col);
        out.plot_ids.emplace_back(id ? std::string(*id) : std::string());
        for (std::size_t c = 0; c < cols.size(); ++c) {
            const auto v = parse_cell_at<double>(row, cols[c]);
            out.values[c].push_back(v ? *v : kNaN);
        }
    });
    return out;
}

}  // namespace resolve
