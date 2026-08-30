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

// ===========================================================================
// use_hash_prefetch (issue #114)
// ===========================================================================
//
// Trainer::train_epoch acquired its two CUDA streams before checking whether
// the run was on CUDA at all. RESOLVE_HAS_CUDA is a COMPILE-time guard, so a
// CUDA-enabled build ran those lines on a device="cpu" run too, initializing
// the CUDA runtime and killing the first epoch on a host with no usable driver
// (a CPU queue node, a CI runner, a laptop). The acquisition is now behind this
// predicate, which is the same condition every USE of the streams was already
// gated on -- so the GPU prefetch path is unchanged and the CPU path never
// touches the CUDA runtime.
//
// The failure it prevents needs a driverless host to reproduce, and a CPU-only
// build compiles the whole block out, so the decision is pinned here as pure
// logic instead.

TEST_CASE("use_hash_prefetch requires a CUDA device", "[gpu][prefetch][issue114]") {
    SECTION("a CPU run never prefetches, whatever else is set") {
        REQUIRE_FALSE(use_hash_prefetch(/*use_cuda_hash=*/true, /*device_is_cuda=*/false,
                                        /*n_train=*/1024, /*batch_size=*/32));
        REQUIRE_FALSE(use_hash_prefetch(false, false, 1024, 32));
    }

    SECTION("a CUDA run without the CUDA hash path never prefetches") {
        REQUIRE_FALSE(use_hash_prefetch(/*use_cuda_hash=*/false, /*device_is_cuda=*/true,
                                        1024, 32));
    }

    SECTION("a single batch has nothing to overlap with") {
        REQUIRE_FALSE(use_hash_prefetch(true, true, /*n_train=*/32, /*batch_size=*/32));
        REQUIRE_FALSE(use_hash_prefetch(true, true, /*n_train=*/16, /*batch_size=*/32));
    }

    SECTION("the GPU path is unchanged: it prefetches when it always did") {
        REQUIRE(use_hash_prefetch(true, true, /*n_train=*/33, /*batch_size=*/32));
        REQUIRE(use_hash_prefetch(true, true, 1024, 32));
    }
}
