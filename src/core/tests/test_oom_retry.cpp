// The out-of-memory policies of the trainer: the auto-halve retry of the
// training batch (decide_oom_retry), the shrink of the held-out forward's chunk
// (decide_eval_chunk_retry) and the classification of an error as an OOM.
//
// The catch-and-retry happens inside Trainer::fit, but the retry-vs-rethrow
// decisions are free functions in gpu.hpp, so they are tested here without
// exhausting CUDA VRAM.

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
// use_hash_prefetch
// ===========================================================================
//
// Trainer::train_epoch acquires its two CUDA streams only behind this
// predicate. RESOLVE_HAS_CUDA is a COMPILE-time guard, so a CUDA-enabled build
// reaches the prefetch setup on a device="cpu" run too, and acquiring the
// streams there initializes the CUDA runtime, which fails on a host with no
// usable driver (a CPU queue node, a CI runner, a laptop). That needs a
// driverless host to reproduce, and a CPU-only build compiles the whole block
// out, so the decision is pinned here as pure logic.

TEST_CASE("use_hash_prefetch requires a CUDA device", "[gpu][prefetch]") {
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

    SECTION("a CUDA-hash run with more than one batch prefetches") {
        REQUIRE(use_hash_prefetch(true, true, /*n_train=*/33, /*batch_size=*/32));
        REQUIRE(use_hash_prefetch(true, true, 1024, 32));
    }
}

// ===========================================================================
// decide_eval_chunk_retry
// ===========================================================================

TEST_CASE("decide_eval_chunk_retry halves the evaluation chunk", "[oom_retry][eval]") {
    int64_t new_chunk = -1;
    std::string log_msg;
    std::string err_msg;

    const bool retry = decide_eval_chunk_retry(
        /*prev_chunk=*/4096, /*full_batch_required=*/false,
        "CUDA out of memory. Tried to allocate 2.00 GiB.", new_chunk, log_msg, err_msg);

    REQUIRE(retry);
    REQUIRE(new_chunk == 2048);
    REQUIRE(err_msg.empty());
    REQUIRE(log_msg.find("held-out forward at 4096") != std::string::npos);
    REQUIRE(log_msg.find("retrying at 2048") != std::string::npos);
}

TEST_CASE("decide_eval_chunk_retry walks down to a single plot", "[oom_retry][eval]") {
    int64_t chunk = 1000;
    int steps = 0;
    std::string log_msg;
    std::string err_msg;
    while (decide_eval_chunk_retry(chunk, false, "out of memory", chunk, log_msg, err_msg)) {
        ++steps;
        REQUIRE(chunk >= 1);
    }
    // 1000 -> 500 -> 250 -> 125 -> 62 -> 31 -> 15 -> 7 -> 3 -> 1, then refused.
    REQUIRE(steps == 9);
    REQUIRE(chunk == 1);
    REQUIRE(err_msg.find("already one plot per chunk") != std::string::npos);
    REQUIRE(err_msg.find("Original OOM: out of memory") != std::string::npos);
}

TEST_CASE("decide_eval_chunk_retry refuses a full-batch architecture",
          "[oom_retry][eval]") {
    int64_t new_chunk = -1;
    std::string log_msg;
    std::string err_msg;

    const bool retry = decide_eval_chunk_retry(
        /*prev_chunk=*/50000, /*full_batch_required=*/true, nullptr,
        new_chunk, log_msg, err_msg);

    REQUIRE_FALSE(retry);
    REQUIRE(new_chunk == 50000);
    REQUIRE(log_msg.empty());
    REQUIRE(err_msg.find("whole held-out fold at once") != std::string::npos);
    REQUIRE(err_msg.find("Original OOM: <null>") != std::string::npos);
}

TEST_CASE("decide_eval_chunk_retry never touches the training batch size",
          "[oom_retry][eval]") {
    // The evaluation decision takes no batch size and the training decision
    // takes no chunk: an OOM in the held-out forward cannot reach the
    // training batch through the decision logic.
    TrainConfig cfg;
    const int batch_before = cfg.batch_size;
    int64_t new_chunk = 0;
    std::string log_msg;
    std::string err_msg;
    REQUIRE(decide_eval_chunk_retry(8192, false, "out of memory", new_chunk, log_msg, err_msg));
    REQUIRE(new_chunk == 4096);
    REQUIRE(cfg.batch_size == batch_before);
}

TEST_CASE("is_out_of_memory recognises the CUDA OOM failures", "[oom_retry]") {
    SECTION("the allocator's OutOfMemoryError") {
        try {
            TORCH_CHECK_WITH(OutOfMemoryError, false, "CUDA out of memory.");
        } catch (const c10::Error& error) {
            REQUIRE(is_out_of_memory(error));
        }
    }
    SECTION("a cudaMalloc failure reported as a plain c10 error") {
        try {
            TORCH_CHECK(false, "CUDA error: cudaErrorMemoryAllocation");
        } catch (const c10::Error& error) {
            REQUIRE(is_out_of_memory(error));
        }
    }
    SECTION("any other c10 error is not an OOM") {
        try {
            TORCH_CHECK(false, "shape mismatch: [4, 3] vs [4, 2]");
        } catch (const c10::Error& error) {
            REQUIRE_FALSE(is_out_of_memory(error));
        }
    }
}
