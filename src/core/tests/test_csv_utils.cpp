#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "resolve/csv_utils.hpp"
#include "resolve/csv_reader.hpp"
#include "test_support.hpp"
#include <cstdint>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;
using namespace Catch::Matchers;

namespace {

// The reader sees the bytes exactly as written, so a leading BOM reaches it
// verbatim.
TempFile binary_csv(const std::string& content) {
    return TempFile(content, ".csv", WriteMode::Binary);
}

}  // namespace

// ============================================================================
// parse_float_strict — locale-free, trailing-garbage-rejecting
// ============================================================================

TEST_CASE("parse_float_strict parses plain decimals", "[csv_utils][parse]") {
    REQUIRE_THAT(*parse_float_strict("1.5"), WithinAbs(1.5f, 1e-6));
    REQUIRE_THAT(*parse_float_strict("-2.5"), WithinAbs(-2.5f, 1e-6));
    REQUIRE_THAT(*parse_float_strict("0"), WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(*parse_float_strict("1e3"), WithinAbs(1000.0f, 1e-6));
    // Leading / trailing whitespace is tolerated.
    REQUIRE_THAT(*parse_float_strict("  3.25 "), WithinAbs(3.25f, 1e-6));
}

TEST_CASE("parse_float_strict rejects trailing garbage", "[csv_utils][parse]") {
    // std::stof would have returned 12 / 1 for these (leading-prefix parse).
    REQUIRE(parse_float_strict("12abc") == std::nullopt);
    REQUIRE(parse_float_strict("1.5x") == std::nullopt);
    REQUIRE(parse_float_strict("") == std::nullopt);
    REQUIRE(parse_float_strict("abc") == std::nullopt);
}

TEST_CASE("parse_float_strict is locale-free: comma is not a decimal point",
          "[csv_utils][parse][locale]") {
    // On a process LC_NUMERIC using a comma decimal separator, std::stof would
    // read "1,234" as 1.234 (or 1 with grouping). The classic-locale stream
    // always treats '.' as the decimal point, so the comma is trailing garbage
    // and the value is rejected rather than silently mis-parsed.
    REQUIRE(parse_float_strict("1,234") == std::nullopt);
    REQUIRE(parse_float_strict("3,14") == std::nullopt);
}

// ============================================================================
// CSVReader — BOM stripping + duplicate-header rejection
// ============================================================================

TEST_CASE("CSVReader strips a leading UTF-8 BOM from the header",
          "[csv_utils][csv_reader][bom]") {
    const std::string bom = "\xEF\xBB\xBF";
    const TempFile f = binary_csv(bom + "plot_id,area\np1,100\n");

    CSVReader reader(f.path());
    // Without BOM stripping the first column would be "﻿plot_id" and the
    // lookup would fail (-1).
    REQUIRE(reader.column_index("plot_id") == 0);
    REQUIRE(reader.column_index("area") == 1);
    REQUIRE(reader.columns().at(0) == "plot_id");
}

TEST_CASE("CSVReader reads a normal header without a BOM",
          "[csv_utils][csv_reader]") {
    const TempFile f = binary_csv("plot_id,area,habitat\np1,100,M\n");
    CSVReader reader(f.path());
    REQUIRE(reader.column_index("plot_id") == 0);
    REQUIRE(reader.column_index("area") == 1);
    REQUIRE(reader.column_index("habitat") == 2);
    REQUIRE(reader.column_index("missing") == -1);
}

TEST_CASE("CSVReader throws on duplicate header column names",
          "[csv_utils][csv_reader][duplicate]") {
    // "id" appears twice: without the guard every reader of "id" would silently
    // bind to the last occurrence.
    const TempFile f = binary_csv("id,area,id\np1,100,p1\n");
    REQUIRE_THROWS_AS(CSVReader(f.path()), std::runtime_error);
}

// ============================================================================
// Number parsing: whole-cell, locale-free
// ============================================================================

TEST_CASE("parse_float_strict takes signs, exponents and a bare fraction",
          "[csv_utils][parse]") {
    REQUIRE(*parse_float_strict("+1.5") == 1.5f);
    REQUIRE(*parse_float_strict(".5") == 0.5f);
    REQUIRE(*parse_float_strict("5.") == 5.0f);
    REQUIRE(*parse_float_strict("-1e-3") == -1e-3f);
    REQUIRE(*parse_float_strict("2.5E+2") == 250.0f);
    REQUIRE(*parse_float_strict("\t7\r") == 7.0f);
    // The correctly rounded float, as the C library reads the same text.
    REQUIRE(*parse_float_strict("0.1") == 0.1f);
    REQUIRE(*parse_float_strict("3.4028234e38") == 3.4028234e38f);
}

TEST_CASE("parse_float_strict rejects what is not a plain decimal number",
          "[csv_utils][parse]") {
    for (const char* text : {"+-1", "--1", "+", "-", ".", "1e", "1.2.3", "0x10", "inf",
                             "-inf", "nan", "NaN", "infinity", "1 2", "1e999", "1e-999"}) {
        INFO(text);
        REQUIRE(parse_float_strict(text) == std::nullopt);
    }
}

TEST_CASE("parse_int64_strict reads whole base-10 integers", "[csv_utils][parse]") {
    REQUIRE(*parse_int64_strict("42") == 42);
    REQUIRE(*parse_int64_strict("-7") == -7);
    REQUIRE(*parse_int64_strict("+3") == 3);
    REQUIRE(*parse_int64_strict("  12") == 12);
    REQUIRE(*parse_int64_strict("9223372036854775807") == INT64_MAX);
    for (const char* text : {"", " ", "12 ", "1.0", "1e3", "+-1", "abc", "0x1F",
                             "9223372036854775808"}) {
        INFO(text);
        REQUIRE(parse_int64_strict(text) == std::nullopt);
    }
}

// ============================================================================
// CSVReader: one record stream for the header and the rows
// ============================================================================

namespace {

std::vector<std::vector<std::string>> read_all(CSVReader& reader) {
    std::vector<std::vector<std::string>> rows;
    reader.read_rows([&](size_t i, RowFields row) {
        REQUIRE(i == rows.size());
        rows.emplace_back(row.begin(), row.end());
    });
    return rows;
}

using Rows = std::vector<std::vector<std::string>>;

}  // namespace

TEST_CASE("CSVReader keeps a newline inside a quoted field and drops carriage returns",
          "[csv_utils][csv_reader]") {
    const TempFile f = binary_csv(
        "id,\"note\r\nline\"\r\n"
        "a,\"two\r\nlines\"\r\n"
        "\r\n"
        "b,\"\"\"quoted\"\" and, comma\"\r\n"
        "c,trailing,\r\n"
        "d,last");
    CSVReader reader(f.path());
    REQUIRE(reader.columns() == std::vector<std::string>{"id", "note\nline"});
    const Rows rows = read_all(reader);
    REQUIRE(rows == Rows{{"a", "two\nlines"},
                         {"b", "\"quoted\" and, comma"},
                         {"c", "trailing", ""},
                         {"d", "last"}});
    // Every call streams the rows from the start.
    REQUIRE(read_all(reader) == rows);
}

TEST_CASE("CSVReader reads records across its buffer boundary", "[csv_utils][csv_reader]") {
    // 1 MiB is the initial buffer: rows run well past it, and one field alone
    // is larger than it.
    std::string content = "id,payload\n";
    const std::string big(3 << 20, 'x');
    const int n_rows = 60000;
    for (int i = 0; i < n_rows; ++i) {
        content += "r" + std::to_string(i) + ",\"v," + std::to_string(i) + "\"\n";
        if (i == 30000) content += "big," + big + "\n";
    }
    const TempFile f = binary_csv(content);
    CSVReader reader(f.path());
    size_t n = 0;
    size_t mismatched = 0;
    bool saw_big = false;
    reader.read_rows([&](size_t, RowFields row) {
        if (row.size() == 2 && row[0] == "big") {
            saw_big = row[1].size() == big.size();
            return;
        }
        if (row.size() != 2 || row[0] != "r" + std::to_string(n) ||
            row[1] != "v," + std::to_string(n)) {
            ++mismatched;
        }
        ++n;
    });
    REQUIRE(n == static_cast<size_t>(n_rows));
    REQUIRE(mismatched == 0);
    REQUIRE(saw_big);
}

TEST_CASE("CSVReader refuses a file that ends inside a quoted field",
          "[csv_utils][csv_reader]") {
    const TempFile f = binary_csv("id,note\na,\"open\nb,c\n");
    CSVReader reader(f.path());
    REQUIRE_THROWS_WITH(read_all(reader), ContainsSubstring("Unbalanced quote"));
}

TEST_CASE("CSVReader reports an empty file and a missing one", "[csv_utils][csv_reader]") {
    const TempFile empty = binary_csv("");
    REQUIRE_THROWS_WITH(CSVReader(empty.path()), ContainsSubstring("Empty CSV file"));
    REQUIRE_THROWS_AS(CSVReader("no/such/file.csv"), io::IOError);
}

TEST_CASE("CSVReader takes another delimiter", "[csv_utils][csv_reader]") {
    const TempFile f = binary_csv("a;b\n1,5;\"x;y\"\n");
    CSVReader reader(f.path(), ';');
    REQUIRE(read_all(reader) == Rows{{"1,5", "x;y"}});
}
