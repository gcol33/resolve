// TabM: BatchEnsembleLinear and the TabMEncoder built from it.

#include <catch2/catch_test_macros.hpp>

#include "resolve/tabm.hpp"

#include <torch/torch.h>

#include <set>
#include <string>
#include <vector>

using namespace resolve;

namespace {
MLPBlockConfig block_with_dropout(float dropout) {
    MLPBlockConfig block;
    block.dropout = dropout;
    return block;
}
}  // namespace

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
    TabMEncoder encoder(32, dims, 8, block_with_dropout(0.1f), TabMAggregation::Mean);
    auto x = torch::randn({16, 32});
    auto out = encoder->forward(x);

    REQUIRE(out.size(0) == 16);      // batch
    REQUIRE(out.size(1) == 32);      // last hidden dim
}

TEST_CASE("TabMEncoder forward_all returns per-ensemble", "[tabm]") {
    std::vector<int64_t> dims = {64, 32};
    TabMEncoder encoder(32, dims, 8, block_with_dropout(0.0f), TabMAggregation::Mean);
    auto x = torch::randn({16, 32});
    auto all = encoder->forward_all(x);

    REQUIRE(all.size(0) == 16);      // batch
    REQUIRE(all.size(1) == 8);       // n_ensembles
    REQUIRE(all.size(2) == 32);      // output_dim
}

TEST_CASE("TabMEncoder aggregation matches forward_all mean", "[tabm]") {
    std::vector<int64_t> dims = {32, 16};
    TabMEncoder encoder(16, dims, 4, block_with_dropout(0.0f), TabMAggregation::Mean);
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
    TabMEncoder encoder(32, dims, 12, MLPBlockConfig{});

    REQUIRE(encoder->output_dim() == 16);
    REQUIRE(encoder->n_ensembles() == 12);
}

// ============================================================================
// TabMEncoder honours MLPBlockConfig
// ============================================================================

namespace {
std::set<std::string> names_of(const torch::nn::Module& module) {
    std::set<std::string> names;
    for (const auto& item : module.named_parameters()) names.insert(item.key());
    for (const auto& item : module.named_buffers()) names.insert(item.key());
    return names;
}
}  // namespace

TEST_CASE("TabMEncoder builds the normalization MLPBlockConfig names", "[tabm]") {
    const std::vector<int64_t> dims = {12, 8};
    MLPBlockConfig block;

    block.normalization = NormLayerType::LayerNorm;
    auto layer_norm = names_of(*TabMEncoder(6, dims, 3, block));
    CHECK(layer_norm.count("norms.0.weight") == 1);
    CHECK(layer_norm.count("norms.0.running_mean") == 0);

    block.normalization = NormLayerType::BatchNorm;
    auto batch_norm = names_of(*TabMEncoder(6, dims, 3, block));
    CHECK(batch_norm.count("norms.1.running_mean") == 1);

    block.normalization = NormLayerType::None;
    for (const auto& name : names_of(*TabMEncoder(6, dims, 3, block))) {
        CHECK(name.rfind("norms.", 0) != 0);
    }
}

TEST_CASE("TabMEncoder applies the configured activation per member", "[tabm]") {
    MLPBlockConfig block;
    block.activation = ActivationType::ReLU;
    block.dropout = 0.0f;
    TabMEncoder encoder(6, std::vector<int64_t>{12, 8}, 3, block);
    encoder->eval();
    std::vector<torch::Tensor> activations;
    encoder->forward(torch::randn({5, 6}), &activations);
    REQUIRE(activations.size() == 2);
    CHECK(activations[0].sizes() == torch::IntArrayRef({15, 12}));
    CHECK(activations[1].sizes() == torch::IntArrayRef({15, 8}));
    for (const auto& a : activations) CHECK(a.min().item<float>() >= 0.0f);

    block.activation = ActivationType::PReLU;
    auto prelu = names_of(*TabMEncoder(6, std::vector<int64_t>{12, 8}, 3, block));
    CHECK(prelu.count("activations.0.weight") == 1);
    CHECK(prelu.count("activations.1.weight") == 1);
}

TEST_CASE("TabMEncoder adds a skip connection under use_residual", "[tabm]") {
    MLPBlockConfig block;
    block.use_residual = true;
    block.dropout = 0.0f;
    block.normalization = NormLayerType::LayerNorm;
    // A layer that changes width projects its input; one that keeps it adds
    // its input unchanged.
    auto names = names_of(*TabMEncoder(6, std::vector<int64_t>{12, 12}, 3, block));
    CHECK(names.count("projections.0.weight") == 1);
    CHECK(names.count("projections.1.weight") == 0);

    MLPBlockConfig plain = block;
    plain.use_residual = false;
    TabMEncoder residual(6, std::vector<int64_t>{6}, 3, block);
    TabMEncoder without(6, std::vector<int64_t>{6}, 3, plain);
    {
        torch::NoGradGuard no_grad;
        auto source = residual->named_parameters();
        for (auto& p : without->named_parameters()) p.value().copy_(source[p.key()]);
    }
    residual->eval();
    without->eval();
    const auto x = torch::randn({4, 6});
    CHECK(torch::allclose(residual->forward_all(x), without->forward_all(x) + x.unsqueeze(1),
                          1e-5, 1e-6));
}
