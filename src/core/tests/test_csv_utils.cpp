#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/csv_utils.hpp"
#include "resolve/csv_reader.hpp"
#include <fstream>
#include <filesystem>
#include <string>

using namespace resolve;
using namespace Catch::Matchers;

namespace {

// Local temp-file helper. Named distinctly from test_dataset.cpp's TempFile to
// avoid a duplicate-symbol/ODR clash when both link into resolve_tests. Writes
// in binary mode so a leading BOM is emitted verbatim (text mode is fine too,
// but binary keeps the bytes exact for the BOM case).
class TempCSV {
public:
    explicit TempCSV(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_csvutils_" + std::to_string(counter_++) + ".csv");
        std::ofstream file(path_, std::ios::binary);
        file << content;
    }
    ~TempCSV() { std::filesystem::remove(path_); }
    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
    static int counter_;
};

int TempCSV::counter_ = 0;

}  // namespace

// ============================================================================
// parse_float_strict / safe_stof — locale-free, trailing-garbage-rejecting
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

TEST_CASE("safe_stof returns default on failure, value on success",
          "[csv_utils][parse]") {
    REQUIRE_THAT(safe_stof("2.75"), WithinAbs(2.75f, 1e-6));
    REQUIRE_THAT(safe_stof("12abc", -1.0f), WithinAbs(-1.0f, 1e-6));
    REQUIRE_THAT(safe_stof("", 7.0f), WithinAbs(7.0f, 1e-6));
    REQUIRE_THAT(safe_stof("1,5", 9.0f), WithinAbs(9.0f, 1e-6));
}

TEST_CASE("safe_stoi rejects trailing garbage", "[csv_utils][parse]") {
    REQUIRE(safe_stoi("42") == 42);
    REQUIRE(safe_stoi("42abc", -1) == -1);
    REQUIRE(safe_stoi("", 5) == 5);
}

// ============================================================================
// CSVReader — BOM stripping + duplicate-header rejection
// ============================================================================

TEST_CASE("CSVReader strips a leading UTF-8 BOM from the header",
          "[csv_utils][csv_reader][bom]") {
    const std::string bom = "\xEF\xBB\xBF";
    TempCSV f(bom + "plot_id,area\np1,100\n");

    CSVReader reader(f.path());
    // Without BOM stripping the first column would be "﻿plot_id" and the
    // lookup would fail (-1).
    REQUIRE(reader.column_index("plot_id") == 0);
    REQUIRE(reader.column_index("area") == 1);
    REQUIRE(reader.columns().at(0) == "plot_id");
}

TEST_CASE("CSVReader reads a normal header without a BOM",
          "[csv_utils][csv_reader]") {
    TempCSV f("plot_id,area,habitat\np1,100,M\n");
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
    TempCSV f("id,area,id\np1,100,p1\n");
    REQUIRE_THROWS_AS(CSVReader(f.path()), std::runtime_error);
}
