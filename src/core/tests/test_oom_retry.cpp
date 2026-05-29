// Tests for the auto-halve-on-OOM retry policy in Trainer::fit.
//
// The actual catch-and-retry happens inside Trainer::fit, but the
// retry-vs-rethrow decision is extracted into resolve::decide_oom_retry()
// in gpu.hpp so it can be unit-tested without exhausting CUDA VRAM.
// A real OOM smoke test lives in dev_notes/oom_retry_smoke.py and runs
// against an oversized batch_size on a CUDA device.

#include <catch2/catch_test_macros.hpp>

#include "resolve/gpu.hpp"
#include "resolve/types.hpp"

#include <string>

using namespace resolve;

TEST_CASE("decide_oom_retry halves and signals retry above the floor", "[oom_retry]") {
    int new_bs = -1;
    std::string log_msg;
    std::string err_msg;

    const bool retry = decide_oom_retry(
        /*prev_bs=*/16384,
        /*floor=*/1024,
        /*original_bs=*/16384,
        /*oom_what=*/"CUDA out of memory. Tried to allocate 618.00 MiB.",
        new_bs,
        log_msg,
        err_msg
    );

    REQUIRE(retry);
    REQUIRE(new_bs == 8192);
    REQUIRE(err_msg.empty());
    REQUIRE(log_msg.find("OOM at batch_size=16384") != std::string::npos);
    REQUIRE(log_msg.find("retrying at batch_size=8192") != std::string::npos);
    REQUIRE(log_msg.find("floor=1024") != std::string::npos);
}

TEST_CASE("decide_oom_retry refuses to drop below the floor", "[oom_retry]") {
    int new_bs = -1;
    std::string log_msg;
    std::string err_msg;

    // prev_bs/2 = 1024 which still meets the floor; one more halve from
    // there (the next attempt's prev_bs == 1024 -> new_bs = 512) would
    // breach floor=1024. Exercise both.
    SECTION("halving exactly to floor is allowed") {
        const bool retry = decide_oom_retry(
            /*prev_bs=*/2048,
            /*floor=*/1024,
            /*original_bs=*/16384,
            /*oom_what=*/"CUDA out of memory.",
            new_bs,
            log_msg,
            err_msg
        );
        REQUIRE(retry);
        REQUIRE(new_bs == 1024);
        REQUIRE(err_msg.empty());
    }

    SECTION("halving below floor rethrows with a diagnostic message") {
        const bool retry = decide_oom_retry(
            /*prev_bs=*/1024,
            /*floor=*/1024,
            /*original_bs=*/16384,
            /*oom_what=*/"CUDA out of memory. Tried to allocate 200.00 MiB.",
            new_bs,
            log_msg,
            err_msg
        );
        REQUIRE_FALSE(retry);
        REQUIRE(log_msg.empty());
        // Error message must contain the relevant context: original bs,
        // current bs, floor, and the underlying OOM string.
        REQUIRE(err_msg.find("Original batch_size=16384") != std::string::npos);
        REQUIRE(err_msg.find("current batch_size=1024") != std::string::npos);
        REQUIRE(err_msg.find("batch_size_floor=1024") != std::string::npos);
        REQUIRE(err_msg.find("CUDA out of memory") != std::string::npos);
    }
}

TEST_CASE("decide_oom_retry tolerates a null oom_what pointer", "[oom_retry]") {
    int new_bs = -1;
    std::string log_msg;
    std::string err_msg;

    const bool retry = decide_oom_retry(
        /*prev_bs=*/1024,
        /*floor=*/1024,
        /*original_bs=*/16384,
        /*oom_what=*/nullptr,
        new_bs,
        log_msg,
        err_msg
    );

    REQUIRE_FALSE(retry);
    REQUIRE(err_msg.find("Original OOM: <null>") != std::string::npos);
}

TEST_CASE("TrainConfig exposes batch_size_floor with the documented default", "[oom_retry]") {
    TrainConfig cfg;
    REQUIRE(cfg.batch_size_floor == 1024);
}
