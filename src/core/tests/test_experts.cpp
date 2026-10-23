// MixtureOfExperts as a standalone module: construction limits, the gate
// distribution and the load-balancing auxiliary loss.

#include <catch2/catch_test_macros.hpp>

#include "resolve/experts.hpp"

#include <torch/torch.h>

#include <cmath>
#include <vector>

using namespace resolve;

TEST_CASE("MoE rejects a single expert with routing", "[moe]") {
    // n_experts == 1 makes the Soft-routing CV^2 aux loss var() NaN.
    REQUIRE_THROWS(MixtureOfExperts(
        /*input_dim=*/8, /*expert_hidden_dims=*/std::vector<int64_t>{16},
        /*output_dim=*/8, /*n_experts=*/1, MoERoutingType::Soft,
        /*top_k=*/1, /*noise_std=*/0.0f, /*dropout=*/0.0f));
    REQUIRE_THROWS(MixtureOfExperts(
        8, std::vector<int64_t>{16}, 8, 1, MoERoutingType::TopK, 1, 0.0f, 0.0f));
}

TEST_CASE("MoE gate probabilities are a valid distribution", "[moe]") {
    torch::manual_seed(0);
    MixtureOfExperts moe(
        /*input_dim=*/8, std::vector<int64_t>{16},
        /*output_dim=*/8, /*n_experts=*/4, MoERoutingType::Soft,
        /*top_k=*/2, /*noise_std=*/0.0f, /*dropout=*/0.0f);
    moe->eval();  // no gate noise / dropout

    auto x = torch::randn({6, 8});
    auto out = moe->forward(x);

    REQUIRE(out.output.sizes() == std::vector<int64_t>{6, 8});
    REQUIRE(out.gate_probs.sizes() == std::vector<int64_t>{6, 4});
    REQUIRE(out.gate_probs.min().item<float>() >= 0.0f);
    auto row_sums = out.gate_probs.sum(/*dim=*/1);
    REQUIRE(torch::allclose(row_sums, torch::ones_like(row_sums), 1e-4, 1e-4));
    REQUIRE(std::isfinite(out.aux_loss.item<float>()));  // not NaN (n_experts>=2)
}

TEST_CASE("MoE Soft aux loss is finite for two experts", "[moe]") {
    torch::manual_seed(1);
    MixtureOfExperts moe(4, std::vector<int64_t>{8}, 4, 2, MoERoutingType::Soft,
                         1, 0.0f, 0.0f);
    moe->train();
    auto out = moe->forward(torch::randn({16, 4}));
    REQUIRE(std::isfinite(out.aux_loss.item<float>()));
}
