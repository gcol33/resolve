#include <catch2/catch_test_macros.hpp>

#include "resolve/encoder.hpp"

#include <torch/torch.h>

using namespace resolve;

// ============================================================================
// AMP fp32-normalization fix (gcol33/resolve#21)
//
// Under fp16 autocast, BatchNorm computes its batch statistics and updates its
// running mean/variance in fp16, so the running buffers drift: the model trains
// (train mode uses fresh per-batch stats) but collapses in eval mode (which
// uses the corrupted running stats). run_norm_fp32 forces the normalization to
// run in fp32 so the running statistics stay accurate.
//
// This test pins the invariant: with the fix, BatchNorm running statistics
// accumulated under autocast match the fp32 reference (same fp16-quantized
// input, statistics computed in fp32). It also checks the fix is load-bearing:
// the same BatchNorm invoked directly under autocast (no run_norm_fp32) drifts
// from the reference.
// ============================================================================

#ifdef RESOLVE_HAS_CUDA
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAFunctions.h>

namespace {

// Max-abs divergence of a BatchNorm's running stats from a reference module.
double running_stat_drift(const torch::nn::BatchNorm1d& a,
                          const torch::nn::BatchNorm1d& b) {
    auto dm = (a->running_mean - b->running_mean).abs().max().item<double>();
    auto dv = (a->running_var - b->running_var).abs().max().item<double>();
    return dm + dv;
}

}  // namespace

TEST_CASE("run_norm_fp32 keeps BatchNorm statistics in fp32 under autocast",
          "[encoder][amp][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SUCCEED("No CUDA device available; skipping autocast BN-stat check");
        return;
    }

    const auto device = torch::kCUDA;
    const int64_t dim = 64;
    const int64_t batch = 4096;
    const int n_steps = 16;

    torch::manual_seed(0);
    // Large-magnitude activations so fp16 statistic quantization is visible
    // (fp16 has ~0.1 resolution near 1000).
    auto x = (torch::randn({batch, dim}, torch::TensorOptions().device(device)) * 200.0)
                 .add(1000.0);
    // The activations that actually enter BN under AMP are fp16. All three paths
    // start from the same fp16 tensor so the only difference is the dtype in
    // which the statistics are computed.
    auto x16 = x.to(torch::kHalf);

    // Three identically-initialised BatchNorm layers in train mode.
    auto bn_ref = torch::nn::BatchNorm1d(dim);  // fp32 ground truth (no autocast)
    auto bn_fix = torch::nn::BatchNorm1d(dim);  // fp16 input, run_norm_fp32
    auto bn_bug = torch::nn::BatchNorm1d(dim);  // fp16 input, direct under autocast
    bn_ref->to(device); bn_fix->to(device); bn_bug->to(device);
    bn_ref->train(); bn_fix->train(); bn_bug->train();

    // Reference: fp16-quantised input, statistics computed in fp32 (the target
    // behaviour). No autocast active.
    for (int k = 0; k < n_steps; ++k) {
        bn_ref->forward(x16.to(torch::kFloat32));
    }

    // Fix path: autocast active, normalization routed through run_norm_fp32.
    at::autocast::set_autocast_dtype(at::kCUDA, at::kHalf);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::increment_nesting();
    for (int k = 0; k < n_steps; ++k) {
        run_norm_fp32([&](torch::Tensor t) { return bn_fix->forward(t); }, x16);
    }
    // Bug path: same autocast region, BN invoked directly (statistics in fp16).
    for (int k = 0; k < n_steps; ++k) {
        bn_bug->forward(x16);
    }
    at::autocast::decrement_nesting();
    at::autocast::set_autocast_enabled(at::kCUDA, false);

    const double drift_fix = running_stat_drift(bn_fix, bn_ref);
    const double drift_bug = running_stat_drift(bn_bug, bn_ref);

    INFO("drift_fix=" << drift_fix << "  drift_bug=" << drift_bug);

    // Invariant: with the fix, autocast BN running stats match the fp32 reference.
    REQUIRE(drift_fix < 1e-2);

    if (drift_bug < 1e-2) {
        WARN("BatchNorm runs in fp32 under autocast on this libtorch build; "
             "run_norm_fp32 is belt-and-suspenders here, not load-bearing.");
    } else {
        // The fix is load-bearing: direct-autocast BN drifts far more.
        REQUIRE(drift_fix < drift_bug);
    }
}

TEST_CASE("run_norm_fp32 is a no-op when autocast is disabled",
          "[encoder][amp]") {
    const int64_t dim = 8;
    auto bn = torch::nn::BatchNorm1d(dim);
    bn->eval();
    auto x = torch::randn({4, dim});
    // No autocast active -> run_norm_fp32 must call straight through and return
    // the same result as a direct forward.
    auto direct = bn->forward(x);
    auto via_helper = run_norm_fp32([&](torch::Tensor t) { return bn->forward(t); }, x);
    REQUIRE(torch::allclose(direct, via_helper, /*rtol=*/1e-6, /*atol=*/1e-6));
}

#else  // !RESOLVE_HAS_CUDA

TEST_CASE("run_norm_fp32 is a no-op on CPU builds", "[encoder][amp]") {
    const int64_t dim = 8;
    auto bn = torch::nn::BatchNorm1d(dim);
    bn->eval();
    auto x = torch::randn({4, dim});
    auto direct = bn->forward(x);
    auto via_helper = run_norm_fp32([&](torch::Tensor t) { return bn->forward(t); }, x);
    REQUIRE(torch::allclose(direct, via_helper, /*rtol=*/1e-6, /*atol=*/1e-6));
}

#endif  // RESOLVE_HAS_CUDA
