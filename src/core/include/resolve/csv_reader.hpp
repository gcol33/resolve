#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "resolve/io_retry.hpp"  // io::IOError for transient-read classification
#include "resolve/row_source.hpp"  // RowSource interface (shared loader seam)

namespace resolve {

// Simple CSV reader that doesn't depend on external libraries
// For production use, consider fast-cpp-csv-parser
class CSVReader : public RowSource {
public:
    explicit CSVReader(const std::string& filename, char delimiter = ',');

    // Get column names (header row)
    const std::vector<std::string>& columns() const override { return columns_; }

    // Get column index by name (-1 if not found)
    int column_index(const std::string& name) const override;

    // Read all rows, calling callback for each row
    // Callback receives row index and vector of string values
    void read_rows(
        const std::function<void(size_t, const std::vector<std::string>&)>& callback) override;

    // Number of data rows (RowSource interface; reads the entire file).
    size_t num_rows() const override { return count_rows(); }

    // Read all rows into memory
    std::vector<std::vector<std::string>> read_all();

    // Get number of rows (requires reading entire file)
    size_t count_rows() const;

private:
    void parse_header();
    std::vector<std::string> parse_line(const std::string& line);

    std::string filename_;
    char delimiter_;
    std::vector<std::string> columns_;
    std::unordered_map<std::string, int> column_indices_;
};

// Inline implementations

inline CSVReader::CSVReader(const std::string& filename, char delimiter)
    : filename_(filename), delimiter_(delimiter) {
    parse_header();
}

inline void CSVReader::parse_header() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw io::IOError("Cannot open file: " + filename_);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        // bad() is a transient read fault (retryable); a clean EOF on the first
        // line is a genuinely empty file (a data error, not retried).
        if (file.bad()) throw io::IOError("Read error in header of: " + filename_);
        throw std::runtime_error("Empty CSV file: " + filename_);
    }

    // Strip a leading UTF-8 BOM (EF BB BF). Files exported from Excel and many
    // Windows tools prepend one; without this the first column name binds to
    // "﻿name" and every role/target lookup against it silently fails.
    if (header_line.size() >= 3 &&
        static_cast<unsigned char>(header_line[0]) == 0xEF &&
        static_cast<unsigned char>(header_line[1]) == 0xBB &&
        static_cast<unsigned char>(header_line[2]) == 0xBF) {
        header_line.erase(0, 3);
    }

    columns_ = parse_line(header_line);
    for (size_t i = 0; i < columns_.size(); ++i) {
        auto [it, inserted] = column_indices_.emplace(columns_[i], static_cast<int>(i));
        if (!inserted) {
            // Duplicate header names would otherwise bind every reader of this
            // name to the last occurrence, silently mis-reading roles/targets.
            throw std::runtime_error(
                "Duplicate column name in CSV header: '" + columns_[i] +
                "' (columns " + std::to_string(it->second + 1) + " and " +
                std::to_string(i + 1) + ") in file: " + filename_);
        }
    }
}

inline int CSVReader::column_index(const std::string& name) const {
    auto it = column_indices_.find(name);
    return it != column_indices_.end() ? it->second : -1;
}

// True if `s` ends with an open (unterminated) quoted field, i.e. a quoted
// field runs past the end of this physical line into the next (an RFC-4180
// embedded newline). Quote handling mirrors parse_line exactly (a doubled ""
// inside a quoted field is an escaped quote, not a terminator).
inline bool csv_has_open_quote(const std::string& s) {
    bool in_quotes = false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"') {
            if (in_quotes && i + 1 < s.size() && s[i + 1] == '"') {
                ++i;  // escaped quote — consume the pair
            } else {
                in_quotes = !in_quotes;
            }
        }
    }
    return in_quotes;
}

inline std::vector<std::string> CSVReader::parse_line(const std::string& line) {
    std::vector<std::string> result;
    std::string field;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                // Escaped quote
                field += '"';
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == delimiter_ && !in_quotes) {
            result.push_back(field);
            field.clear();
        } else if (c == '\r') {
            // Skip carriage return
        } else {
            field += c;
        }
    }
    result.push_back(field);  // Last field

    return result;
}

inline void CSVReader::read_rows(
    const std::function<void(size_t, const std::vector<std::string>&)>& callback) {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw io::IOError("Cannot open file: " + filename_);
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    size_t row_idx = 0;
    std::string record;  // accumulates physical lines into one logical record
    while (std::getline(file, line)) {
        if (record.empty()) {
            record = line;
        } else {
            record += '\n';  // restore the embedded newline inside the quoted field
            record += line;
        }
        // A quoted field may span multiple physical lines; keep accumulating
        // until the quotes balance so the field's embedded newline is preserved
        // rather than splitting the record (which would corrupt the row and
        // desync count_rows from read_rows).
        if (csv_has_open_quote(record)) continue;

        if (!record.empty() && !(record.size() == 1 && record[0] == '\r')) {
            callback(row_idx++, parse_line(record));
        }
        record.clear();
    }
    // A leftover record with an open quote means the file ended inside a quoted
    // field — a malformed CSV. Fail loudly rather than emit a corrupt row.
    if (!record.empty()) {
        if (csv_has_open_quote(record)) {
            throw std::runtime_error(
                "Unbalanced quote in CSV (file ended inside a quoted field): " + filename_);
        }
        if (!(record.size() == 1 && record[0] == '\r')) {
            callback(row_idx++, parse_line(record));
        }
    }
    // badbit (as opposed to a clean EOF) means the stream errored mid-read --
    // a transient storage fault the caller can retry (issue #20).
    if (file.bad()) {
        throw io::IOError("Read error while streaming rows of: " + filename_);
    }
}

inline std::vector<std::vector<std::string>> CSVReader::read_all() {
    std::vector<std::vector<std::string>> result;
    read_rows([&result](size_t, const std::vector<std::string>& row) {
        result.push_back(row);
    });
    return result;
}

inline size_t CSVReader::count_rows() const {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw io::IOError("Cannot open file: " + filename_);
    }

    size_t count = 0;
    std::string line;
    // Skip header
    std::getline(file, line);

    std::string record;  // must match read_rows' multi-line record accumulation
    while (std::getline(file, line)) {
        if (record.empty()) {
            record = line;
        } else {
            record += '\n';
            record += line;
        }
        if (csv_has_open_quote(record)) continue;  // quoted field spans lines
        if (!record.empty() && !(record.size() == 1 && record[0] == '\r')) {
            ++count;
        }
        record.clear();
    }
    // A trailing record (file did not end in a newline) still counts if present.
    if (!record.empty() && !(record.size() == 1 && record[0] == '\r')) {
        ++count;
    }
    if (file.bad()) {
        throw io::IOError("Read error while counting rows of: " + filename_);
    }
    return count;
}

} // namespace resolve
