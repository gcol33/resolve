// TabM: BatchEnsembleLinear and the TabMEncoder built from it.

#include <catch2/catch_test_macros.hpp>

#include "resolve/tabm.hpp"

#include <torch/torch.h>

#include <string>
#include <vector>

using namespace resolve;

// ============================================================================
// BatchEnsembleLinear
// ============================================================================

TEST_CASE("BatchEnsembleLinear forward shape", "[tabm]") {
    BatchEnsembleLinear layer(32, 64, 8);
    auto x = torch::randn({16, 32});
    auto out = layer->forward(x);

    REQUIRE(out.size(0) == 16);      // batch
    REQUIRE(out.size(1) == 8);       // n_ensembles
    REQUIRE(out.size(2) == 64);      // out_features
}

TEST_CASE("BatchEnsembleLinear accessors", "[tabm]") {
    BatchEnsembleLinear layer(10, 20, 4);

    REQUIRE(layer->in_features() == 10);
    REQUIRE(layer->out_features() == 20);
    REQUIRE(layer->n_ensembles() == 4);
}

TEST_CASE("BatchEnsembleLinear gradient flow", "[tabm]") {
    BatchEnsembleLinear layer(16, 32, 4);
    auto x = torch::randn({8, 16}, torch::requires_grad(true));
    auto out = layer->forward(x);
    auto loss = out.sum();
    loss.backward();

    REQUIRE(x.grad().defined());
    REQUIRE(x.grad().numel() > 0);
}

TEST_CASE("BatchEnsembleLinear ensemble diversity", "[tabm]") {
    BatchEnsembleLinear layer(16, 32, 4);
    auto x = torch::randn({8, 16});
    auto out = layer->forward(x);

    // Each ensemble member should produce different outputs (due to r/s perturbations)
    auto member0 = out.select(1, 0);  // (8, 32)
    auto member1 = out.select(1, 1);  // (8, 32)
    auto diff = (member0 - member1).abs().mean().item<float>();
    REQUIRE(diff > 1e-6f);  // Members should differ
}

// ============================================================================
// TabMEncoder
// ============================================================================

TEST_CASE("TabMEncoder forward shape", "[tabm]") {
    std::vector<int64_t> dims = {64, 32};
    TabMEncoder encoder(32, dims, 8, 0.1f, TabMAggregation::Mean);
    auto x = torch::randn({16, 32});
    auto out = encoder->forward(x);

    REQUIRE(out.size(0) == 16);      // batch
    REQUIRE(out.size(1) == 32);      // last hidden dim
}

TEST_CASE("TabMEncoder forward_all returns per-ensemble", "[tabm]") {
    std::vector<int64_t> dims = {64, 32};
    TabMEncoder encoder(32, dims, 8, 0.0f, TabMAggregation::Mean);
    auto x = torch::randn({16, 32});
    auto all = encoder->forward_all(x);

    REQUIRE(all.size(0) == 16);      // batch
    REQUIRE(all.size(1) == 8);       // n_ensembles
    REQUIRE(all.size(2) == 32);      // output_dim
}

TEST_CASE("TabMEncoder aggregation matches forward_all mean", "[tabm]") {
    std::vector<int64_t> dims = {32, 16};
    TabMEncoder encoder(16, dims, 4, 0.0f, TabMAggregation::Mean);
    encoder->eval();  // Disable dropout
    auto x = torch::randn({8, 16});

    auto aggregated = encoder->forward(x);
    auto all = encoder->forward_all(x);
    auto manual_mean = all.mean(1);  // Mean across ensemble dim

    auto diff = (aggregated - manual_mean).abs().max().item<float>();
    REQUIRE(diff < 1e-5f);
}

TEST_CASE("TabMEncoder accessors", "[tabm]") {
    std::vector<int64_t> dims = {64, 16};
    TabMEncoder encoder(32, dims, 12);

    REQUIRE(encoder->output_dim() == 16);
    REQUIRE(encoder->n_ensembles() == 12);
}
