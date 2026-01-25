#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace resolve {

// Simple CSV reader that doesn't depend on external libraries
// For production use, consider fast-cpp-csv-parser
class CSVReader {
public:
    explicit CSVReader(const std::string& filename, char delimiter = ',');

    // Get column names (header row)
    const std::vector<std::string>& columns() const { return columns_; }

    // Get column index by name (-1 if not found)
    int column_index(const std::string& name) const;

    // Read all rows, calling callback for each row
    // Callback receives row index and vector of string values
    void read_rows(std::function<void(size_t, const std::vector<std::string>&)> callback);

    // Read all rows into memory
    std::vector<std::vector<std::string>> read_all();

    // Get number of rows (requires reading entire file)
    size_t count_rows();

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
        throw std::runtime_error("Cannot open file: " + filename_);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Empty CSV file: " + filename_);
    }

    columns_ = parse_line(header_line);
    for (size_t i = 0; i < columns_.size(); ++i) {
        column_indices_[columns_[i]] = static_cast<int>(i);
    }
}

inline int CSVReader::column_index(const std::string& name) const {
    auto it = column_indices_.find(name);
    return it != column_indices_.end() ? it->second : -1;
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

inline void CSVReader::read_rows(std::function<void(size_t, const std::vector<std::string>&)> callback) {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename_);
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    size_t row_idx = 0;
    while (std::getline(file, line)) {
        if (line.empty() || (line.size() == 1 && line[0] == '\r')) {
            continue;
        }
        auto fields = parse_line(line);
        callback(row_idx++, fields);
    }
}

inline std::vector<std::vector<std::string>> CSVReader::read_all() {
    std::vector<std::vector<std::string>> result;
    read_rows([&result](size_t, const std::vector<std::string>& row) {
        result.push_back(row);
    });
    return result;
}

inline size_t CSVReader::count_rows() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename_);
    }

    size_t count = 0;
    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (!line.empty() && !(line.size() == 1 && line[0] == '\r')) {
            ++count;
        }
    }
    return count;
}

} // namespace resolve
