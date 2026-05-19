#include <catch2/catch_test_macros.hpp>

#include "resolve/gpu.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace resolve;

namespace {

// Logger that records every line for assertions instead of printing.
struct RecordingLogger {
    std::vector<std::string> messages;

    void operator()(const std::string& msg) {
        messages.push_back(msg);
    }
};

} // namespace

TEST_CASE("set_vram_fraction rejects out-of-range values", "[gpu]") {
    SECTION("fraction <= 0 throws") {
        REQUIRE_THROWS_AS(set_vram_fraction(0.0, -1, null_log),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(set_vram_fraction(-0.5, -1, null_log),
                          std::invalid_argument);
    }

    SECTION("fraction > 1 throws") {
        REQUIRE_THROWS_AS(set_vram_fraction(1.5, -1, null_log),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(set_vram_fraction(2.0, -1, null_log),
                          std::invalid_argument);
    }
}

TEST_CASE("set_vram_fraction with 1.0 is uncapped no-op", "[gpu]") {
    RecordingLogger log;
    LogCallback cb = [&](const std::string& m) { log(m); };

    REQUIRE_NOTHROW(set_vram_fraction(1.0, -1, cb));

    REQUIRE(log.messages.size() == 1);
    REQUIRE(log.messages[0].find("disabled") != std::string::npos);
}

TEST_CASE("set_vram_fraction no-ops cleanly with no usable device", "[gpu]") {
    // Either we're built without CUDA (RESOLVE_HAS_CUDA undefined → silent
    // skip with an informational log) or CUDA is built in but no device is
    // present at test time (still a silent skip).
    //
    // Either way the call must not throw and must produce at least one log
    // line so the user knows what happened.
    RecordingLogger log;
    LogCallback cb = [&](const std::string& m) { log(m); };

#ifdef RESOLVE_HAS_CUDA
    // With CUDA built in, the call should succeed regardless of whether a
    // device is actually attached — c10::cuda::device_count() returns 0 on a
    // CPU-only host and we short-circuit before calling the allocator.
    REQUIRE_NOTHROW(set_vram_fraction(0.5, -1, cb));
    REQUIRE_FALSE(log.messages.empty());
#else
    REQUIRE_NOTHROW(set_vram_fraction(0.5, -1, cb));
    REQUIRE(log.messages.size() == 1);
    REQUIRE(log.messages[0].find("CUDA") != std::string::npos);
#endif
}

#ifdef RESOLVE_HAS_CUDA
#include <c10/cuda/CUDAFunctions.h>

TEST_CASE("set_vram_fraction applies cap on a CUDA device", "[gpu][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SUCCEED("No CUDA device available; skipping live cap check");
        return;
    }

    RecordingLogger log;
    LogCallback cb = [&](const std::string& m) { log(m); };

    // Should not throw and should log a "VRAM cap:" line.
    REQUIRE_NOTHROW(set_vram_fraction(0.5, -1, cb));
    bool saw_cap_msg = false;
    for (const auto& m : log.messages) {
        if (m.find("VRAM cap") != std::string::npos) {
            saw_cap_msg = true;
            break;
        }
    }
    REQUIRE(saw_cap_msg);

    // Reset to uncapped so subsequent tests in this binary aren't affected.
    REQUIRE_NOTHROW(set_vram_fraction(1.0, -1, null_log));
}
#endif
