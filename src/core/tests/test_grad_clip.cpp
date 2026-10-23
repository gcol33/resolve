// clip_grad_norm: the engine's gradient clipping, which keeps the scale on the
// device. It must clip exactly as torch::nn::utils::clip_grad_norm_ does.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/grad_clip.hpp"

#include <torch/torch.h>

#include <cmath>
#include <vector>

using namespace resolve;

namespace {

// Three parameters on `device`: two whose gradients are standard normal draws
// times `magnitude` (seeded, so two calls agree), and one with no gradient.
std::vector<torch::Tensor> parameters_with_gradients(float magnitude, torch::Device device) {
    torch::manual_seed(17);
    std::vector<torch::Tensor> params = {
        torch::zeros({7, 5}, device).requires_grad_(),
        torch::zeros({5}, device).requires_grad_(),
        torch::zeros({3}, device).requires_grad_(),
    };
    params[0].mutable_grad() = torch::randn({7, 5}).to(device) * magnitude;
    params[1].mutable_grad() = torch::randn({5}).to(device) * magnitude;
    return params;
}

float global_norm(const std::vector<torch::Tensor>& params) {
    double sum = 0.0;
    for (const auto& p : params) {
        if (p.grad().defined()) sum += p.grad().pow(2).sum().item<double>();
    }
    return static_cast<float>(std::sqrt(sum));
}

}  // namespace

TEST_CASE("clip_grad_norm clips bit for bit as torch's clip_grad_norm_", "[grad_clip]") {
    // Below, near and well above the bound.
    for (float magnitude : {0.01f, 0.2f, 1.0f, 50.0f}) {
        INFO("magnitude " << magnitude);
        auto ours = parameters_with_gradients(magnitude, torch::kCPU);
        auto reference = parameters_with_gradients(magnitude, torch::kCPU);
        clip_grad_norm(ours, 1.0);
        torch::nn::utils::clip_grad_norm_(reference, 1.0);
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(torch::equal(ours[i].grad(), reference[i].grad()));
        }
        REQUIRE_FALSE(ours[2].grad().defined());
    }
}

TEST_CASE("clip_grad_norm bounds the global norm and leaves small gradients alone", "[grad_clip]") {
    auto large = parameters_with_gradients(50.0f, torch::kCPU);
    clip_grad_norm(large, 1.0);
    REQUIRE(global_norm(large) == Catch::Approx(1.0f).epsilon(1e-5));

    auto small = parameters_with_gradients(0.01f, torch::kCPU);
    const auto before = small[0].grad().clone();
    clip_grad_norm(small, 1.0);
    REQUIRE(torch::equal(small[0].grad(), before));

    std::vector<torch::Tensor> none = {torch::zeros({2}).requires_grad_()};
    clip_grad_norm(none, 1.0);
    REQUIRE_FALSE(none[0].grad().defined());
}

TEST_CASE("clip_grad_norm agrees with torch's clip_grad_norm_ on CUDA", "[grad_clip][cuda]") {
    if (!torch::cuda::is_available()) SKIP("no CUDA device");
    for (float magnitude : {0.01f, 50.0f}) {
        INFO("magnitude " << magnitude);
        auto ours = parameters_with_gradients(magnitude, torch::kCUDA);
        auto reference = parameters_with_gradients(magnitude, torch::kCUDA);
        clip_grad_norm(ours, 1.0);
        torch::nn::utils::clip_grad_norm_(reference, 1.0);
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(torch::allclose(ours[i].grad(), reference[i].grad(), 1e-6, 1e-7));
        }
    }
}
