#pragma once

#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace resolve {

// The fields of one data row, one per column in header order (a short row
// holds fewer). The views are valid only for the duration of the row callback:
// a source reuses the storage behind them for the next row.
using RowFields = std::span<const std::string_view>;

// Abstract tabular row source consumed by the dataset loaders. Both CSVReader
// (streaming a file from disk) and InMemoryRowSource (a column-oriented frame
// held in RAM, e.g. a pandas DataFrame or an R data.frame) implement it, so one
// loader body serves the on-disk and in-memory paths and a dataset built from a
// frame is identical to one built from the equivalent CSV.
class RowSource {
public:
    virtual ~RowSource() = default;

    // Header column names, in order.
    virtual const std::vector<std::string>& columns() const = 0;

    // Index of a named column, or -1 if absent.
    virtual int column_index(const std::string& name) const = 0;

    // Invoke cb(row_idx, fields) for each data row.
    virtual void read_rows(const std::function<void(size_t, RowFields)>& cb) = 0;
};

// Column-oriented in-memory table: the cross-binding carrier for an in-memory
// DataFrame. Every cell is a string so the semantics match a CSV cell exactly
// (the empty string denotes a missing value, parsed/factorized by the same code
// paths as a CSV), which makes from_dataframe(df) == from_csv(df.to_csv()) by
// construction. Columns must be equal length; column names must be unique.
class ColumnTable {
public:
    ColumnTable() = default;

    ColumnTable(std::vector<std::string> names,
                std::vector<std::vector<std::string>> columns)
        : names_(std::move(names)), columns_(std::move(columns)) {
        if (names_.size() != columns_.size()) {
            throw std::runtime_error(
                "ColumnTable: number of names (" + std::to_string(names_.size()) +
                ") != number of columns (" + std::to_string(columns_.size()) + ")");
        }
        num_rows_ = columns_.empty() ? 0 : columns_.front().size();
        for (size_t c = 0; c < columns_.size(); ++c) {
            if (columns_[c].size() != num_rows_) {
                throw std::runtime_error(
                    "ColumnTable: column '" + names_[c] + "' has " +
                    std::to_string(columns_[c].size()) + " rows; expected " +
                    std::to_string(num_rows_));
            }
            auto [it, inserted] = index_.emplace(names_[c], static_cast<int>(c));
            if (!inserted) {
                throw std::runtime_error(
                    "ColumnTable: duplicate column name '" + names_[c] + "' (columns " +
                    std::to_string(it->second + 1) + " and " + std::to_string(c + 1) + ")");
            }
        }
    }

    const std::vector<std::string>& names() const { return names_; }
    const std::vector<std::vector<std::string>>& columns() const { return columns_; }
    size_t num_rows() const { return num_rows_; }
    size_t num_cols() const { return names_.size(); }

    int column_index(const std::string& name) const {
        auto it = index_.find(name);
        return it != index_.end() ? it->second : -1;
    }

private:
    std::vector<std::string> names_;
    std::vector<std::vector<std::string>> columns_;  // column-major
    size_t num_rows_ = 0;
    std::unordered_map<std::string, int> index_;
};

// Adapts a ColumnTable to the RowSource interface: each row is a set of views
// onto the table's cells (column-major storage -> row-major view), no cell is
// copied. Holds the table by reference; the table must outlive the source.
class InMemoryRowSource : public RowSource {
public:
    explicit InMemoryRowSource(const ColumnTable& table) : table_(table) {}

    const std::vector<std::string>& columns() const override { return table_.names(); }

    int column_index(const std::string& name) const override {
        return table_.column_index(name);
    }

    void read_rows(const std::function<void(size_t, RowFields)>& cb) override {
        const size_t n_cols = table_.num_cols();
        const size_t n_rows = table_.num_rows();
        const auto& cols = table_.columns();
        std::vector<std::string_view> row(n_cols);
        for (size_t r = 0; r < n_rows; ++r) {
            for (size_t c = 0; c < n_cols; ++c) {
                row[c] = cols[c][r];
            }
            cb(r, row);
        }
    }

private:
    const ColumnTable& table_;
};

} // namespace resolve
