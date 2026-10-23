#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/csv_utils.hpp"
#include "resolve/csv_reader.hpp"
#include "test_support.hpp"
#include <string>

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
