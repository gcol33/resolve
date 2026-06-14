#include <catch2/catch_test_macros.hpp>

#include "resolve/io_retry.hpp"

#include <stdexcept>
#include <string>

using namespace resolve;

namespace {

// A parse/logic error that is NOT an io::IOError -- must never be retried.
struct ParseError : std::runtime_error {
    ParseError() : std::runtime_error("parse error") {}
};

}  // namespace

TEST_CASE("with_retry returns the value on first success", "[io_retry]") {
    int calls = 0;
    int result = io::with_retry([&]() {
        ++calls;
        return 42;
    }, "ok", /*attempts=*/3, /*backoff_ms=*/0);
    REQUIRE(result == 42);
    REQUIRE(calls == 1);  // no retry when the first try succeeds
}

TEST_CASE("with_retry retries a transient IOError then succeeds", "[io_retry]") {
    int calls = 0;
    int result = io::with_retry<io::IOError>([&]() {
        ++calls;
        if (calls < 3) throw io::IOError("transient");
        return 7;
    }, "transient", /*attempts=*/5, /*backoff_ms=*/0);
    REQUIRE(result == 7);
    REQUIRE(calls == 3);  // failed twice, succeeded on the third try
}

TEST_CASE("with_retry rethrows after exhausting attempts", "[io_retry]") {
    int calls = 0;
    REQUIRE_THROWS_AS(
        io::with_retry<io::IOError>([&]() -> int {
            ++calls;
            throw io::IOError("always");
        }, "always", /*attempts=*/3, /*backoff_ms=*/0),
        io::IOError);
    REQUIRE(calls == 3);  // exactly `attempts` tries, then rethrow
}

TEST_CASE("with_retry does not retry an exception outside Exc", "[io_retry]") {
    // Gated on io::IOError: a ParseError (permanent) must propagate on the first
    // try, so a permanent fault never triggers a re-read of a multi-GB file.
    int calls = 0;
    REQUIRE_THROWS_AS(
        io::with_retry<io::IOError>([&]() -> int {
            ++calls;
            throw ParseError();
        }, "parse", /*attempts=*/5, /*backoff_ms=*/0),
        ParseError);
    REQUIRE(calls == 1);  // no retry for a non-IOError
}

TEST_CASE("with_retry default Exc catches any std::exception", "[io_retry]") {
    int calls = 0;
    int result = io::with_retry([&]() {
        ++calls;
        if (calls < 2) throw std::runtime_error("generic");
        return 1;
    }, "generic", /*attempts=*/3, /*backoff_ms=*/0);
    REQUIRE(result == 1);
    REQUIRE(calls == 2);
}

TEST_CASE("with_retry supports a void-returning operation", "[io_retry]") {
    int calls = 0;
    bool done = false;
    io::with_retry([&]() {
        ++calls;
        if (calls < 2) throw io::IOError("retry me");
        done = true;
    }, "void-op", /*attempts=*/3, /*backoff_ms=*/0);
    REQUIRE(done);
    REQUIRE(calls == 2);
}

TEST_CASE("with_retry attempts floor of 1 still runs once", "[io_retry]") {
    int calls = 0;
    REQUIRE_THROWS_AS(
        io::with_retry<io::IOError>([&]() -> int {
            ++calls;
            throw io::IOError("x");
        }, "single", /*attempts=*/0, /*backoff_ms=*/0),  // clamped to 1
        io::IOError);
    REQUIRE(calls == 1);
}
