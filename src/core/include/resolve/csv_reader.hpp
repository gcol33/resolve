#pragma once

#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "resolve/io_retry.hpp"  // io::IOError for transient-read classification
#include "resolve/row_source.hpp"  // RowSource interface (shared loader seam)

namespace resolve {

// A CSV file read one logical record at a time, through one reused buffer.
// Handles the RFC-4180 forms the loaders meet: a configurable delimiter,
// quoted fields, doubled quotes as an escaped quote, newlines embedded in a
// quoted field, CRLF line endings (every carriage return is dropped), and a
// leading UTF-8 BOM. Blank lines are skipped. A field without quotes is a view
// straight into the read buffer; a quoted one is unescaped into a scratch
// buffer kept for the record. Both the header and the data rows are read
// through it, so they follow one set of rules.
class CsvRecordStream {
public:
    CsvRecordStream(const std::string& filename, char delimiter)
        : filename_(filename), delimiter_(delimiter), file_(filename, std::ios::binary),
          buffer_(kInitialBuffer) {
        if (!file_.is_open()) {
            throw io::IOError("Cannot open file: " + filename_);
        }
    }

    // Advance to the next non-blank record and split it into fields. False at
    // the end of the file.
    bool next() {
        std::string_view record;
        while (find_record(record)) {
            if (record.empty() || record == "\r") continue;
            split(record);
            return true;
        }
        return false;
    }

    // The fields of the current record, valid until the next call to next().
    [[nodiscard]] RowFields fields() const noexcept { return fields_; }

private:
    static constexpr size_t kInitialBuffer = size_t{1} << 20;

    struct FieldSpan {
        bool in_scratch;
        size_t offset;
        size_t length;
    };

    // The next record of the file, its terminating newline excluded. A newline
    // inside a quoted field belongs to the record.
    bool find_record(std::string_view& record) {
        size_t scan = begin_;
        bool in_quotes = false;
        for (;;) {
            for (; scan < end_; ++scan) {
                const char c = buffer_[scan];
                if (c == '"') {
                    in_quotes = !in_quotes;
                } else if (c == '\n' && !in_quotes) {
                    record = std::string_view(buffer_.data() + begin_, scan - begin_);
                    begin_ = scan + 1;
                    return true;
                }
            }
            if (at_eof_) {
                if (begin_ == end_) return false;
                if (in_quotes) {
                    throw std::runtime_error(
                        "Unbalanced quote in CSV (file ended inside a quoted field): " +
                        filename_);
                }
                record = std::string_view(buffer_.data() + begin_, end_ - begin_);
                begin_ = end_;
                return true;
            }
            scan -= begin_;
            fill();
        }
    }

    // Move the unread bytes to the front of the buffer (growing it when a
    // record fills it) and read more of the file behind them.
    void fill() {
        const size_t pending = end_ - begin_;
        if (begin_ > 0 && pending > 0) {
            std::memmove(buffer_.data(), buffer_.data() + begin_, pending);
        }
        begin_ = 0;
        end_ = pending;
        if (end_ == buffer_.size()) buffer_.resize(buffer_.size() * 2);
        file_.read(buffer_.data() + end_, static_cast<std::streamsize>(buffer_.size() - end_));
        // badbit (as opposed to a clean EOF) is a transient storage fault the
        // caller can retry.
        if (file_.bad()) {
            throw io::IOError("Read error while streaming rows of: " + filename_);
        }
        const auto got = static_cast<size_t>(file_.gcount());
        if (first_fill_) {
            first_fill_ = false;
            if (got >= 3 && static_cast<unsigned char>(buffer_[0]) == 0xEF &&
                static_cast<unsigned char>(buffer_[1]) == 0xBB &&
                static_cast<unsigned char>(buffer_[2]) == 0xBF) {
                begin_ = 3;
            }
        }
        end_ += got;
        if (got == 0 || file_.eof()) at_eof_ = true;
    }

    void split(std::string_view record) {
        spans_.clear();
        scratch_.clear();
        const size_t n = record.size();
        size_t i = 0;
        for (;;) {
            size_t j = i;
            while (j < n && record[j] != delimiter_ && record[j] != '"' && record[j] != '\r') ++j;
            if (j == n || record[j] == delimiter_) {
                spans_.push_back({false, i, j - i});
            } else {
                // Quotes or carriage returns: unescape the field.
                const size_t out = scratch_.size();
                scratch_.append(record.data() + i, j - i);
                bool in_quotes = false;
                for (; j < n; ++j) {
                    const char c = record[j];
                    if (c == '"') {
                        if (in_quotes && j + 1 < n && record[j + 1] == '"') {
                            scratch_ += '"';
                            ++j;
                        } else {
                            in_quotes = !in_quotes;
                        }
                    } else if (c == delimiter_ && !in_quotes) {
                        break;
                    } else if (c != '\r') {
                        scratch_ += c;
                    }
                }
                spans_.push_back({true, out, scratch_.size() - out});
            }
            if (j >= n) break;
            i = j + 1;
        }
        fields_.clear();
        for (const auto& span : spans_) {
            const char* base = span.in_scratch ? scratch_.data() : record.data();
            fields_.emplace_back(base + span.offset, span.length);
        }
    }

    std::string filename_;
    char delimiter_;
    std::ifstream file_;
    std::vector<char> buffer_;
    size_t begin_ = 0;
    size_t end_ = 0;
    bool at_eof_ = false;
    bool first_fill_ = true;
    std::vector<FieldSpan> spans_;
    std::string scratch_;
    std::vector<std::string_view> fields_;
};

// The engine's CSV reader, hand-rolled against no external dependency: a
// RowSource over a CsvRecordStream. The first record is the header; a
// duplicate header name is rejected rather than silently binding every lookup
// to the last occurrence.
class CSVReader : public RowSource {
public:
    explicit CSVReader(const std::string& filename, char delimiter = ',')
        : filename_(filename), delimiter_(delimiter) {
        CsvRecordStream stream(filename_, delimiter_);
        if (!stream.next()) {
            throw std::runtime_error("Empty CSV file: " + filename_);
        }
        for (const auto field : stream.fields()) columns_.emplace_back(field);
        for (size_t i = 0; i < columns_.size(); ++i) {
            auto [it, inserted] = column_indices_.emplace(columns_[i], static_cast<int>(i));
            if (!inserted) {
                throw std::runtime_error(
                    "Duplicate column name in CSV header: '" + columns_[i] +
                    "' (columns " + std::to_string(it->second + 1) + " and " +
                    std::to_string(i + 1) + ") in file: " + filename_);
            }
        }
    }

    const std::vector<std::string>& columns() const override { return columns_; }

    int column_index(const std::string& name) const override {
        auto it = column_indices_.find(name);
        return it != column_indices_.end() ? it->second : -1;
    }

    // Streams the data rows from the start of the file on every call.
    void read_rows(const std::function<void(size_t, RowFields)>& callback) override {
        CsvRecordStream stream(filename_, delimiter_);
        if (!stream.next()) return;  // the header
        size_t row = 0;
        while (stream.next()) callback(row++, stream.fields());
    }

private:
    std::string filename_;
    char delimiter_;
    std::vector<std::string> columns_;
    std::unordered_map<std::string, int> column_indices_;
};

} // namespace resolve
