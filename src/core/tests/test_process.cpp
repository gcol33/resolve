#include <catch2/catch_test_macros.hpp>

#include "resolve/process.hpp"

using namespace resolve;

// The crash handler itself cannot be unit-tested without faulting the test
// process, and installing it (or pinning the thread pools) would mutate
// process-global state shared with every other test in this binary -- which
// could mask resolve_tests' own exit code. So the unit tests cover only the
// pure exit-code policy and the side-effect-free guard paths; the handler's
// runtime behavior is verified end to end by the issue #18 Rscript repro and
// the issue #19 fail-fast smoke (see src/core/dev_notes/).

TEST_CASE("crash_exit_code: work complete returns the shutdown code", "[process]") {
    // Teardown-after-success path (issue #18): the fault is benign, so the
    // process exits with the caller's chosen code regardless of the NTSTATUS.
    REQUIRE(crash_exit_code(true, 0, 0xC0000005UL) == 0U);
    REQUIRE(crash_exit_code(true, 7, 0xC0000006UL) == 7U);
    REQUIRE(crash_exit_code(true, 0, 0UL) == 0U);
}

TEST_CASE("crash_exit_code: mid-run fault never reports success", "[process]") {
    // Mid-run path (issue #19): the orchestrator must record a failure. The
    // real NTSTATUS is preserved when present so the recorded code matches the
    // fault (e.g. 0xC0000005 access violation, 0xC0000006 in-page error).
    REQUIRE(crash_exit_code(false, 0, 0xC0000005UL) == 0xC0000005U);
    REQUIRE(crash_exit_code(false, 0, 0xC0000006UL) == 0xC0000006U);

    SECTION("no usable NTSTATUS still yields a non-zero code") {
        REQUIRE(crash_exit_code(false, 0, 0UL) != 0U);
    }

    SECTION("a zero shutdown code does not leak into the failure path") {
        // work_complete is false, so the result must be the fault code, not 0.
        REQUIRE(crash_exit_code(false, 0, 0xC0000005UL) != 0U);
    }
}

TEST_CASE("set_thread_pools no-op guards do not touch the pools", "[process]") {
    // Non-positive values are a no-op for the corresponding pool (so they
    // neither throw nor mutate libtorch's global thread counts shared with the
    // rest of this test binary). Positive-value pinning is exercised by the
    // binding load paths, not here, to keep this binary's state clean.
    REQUIRE_NOTHROW(set_thread_pools(0, 0));
    REQUIRE_NOTHROW(set_thread_pools(-1, -1));
}
