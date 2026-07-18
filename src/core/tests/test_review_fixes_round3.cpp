// Regression tests for the review-fix batch issues #79-#90 that live outside the
// dataset loader (those are in test_dataset.cpp / test_spatial_cv.cpp).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/experts.hpp"
#include "resolve/vae.hpp"
#include "resolve/loss.hpp"
#include "resolve/attention.hpp"
#include "resolve/fuzzy.hpp"

using namespace resolve;

// ---------------------------------------------------------------------------
// #83 / #89 -- Mixture of Experts
// ---------------------------------------------------------------------------

TEST_CASE("MoE rejects a single expert with routing", "[moe][review]") {
    // n_experts == 1 makes the Soft-routing CV^2 aux loss var() NaN (issue #83).
    REQUIRE_THROWS(MixtureOfExperts(
        /*input_dim=*/8, /*expert_hidden_dims=*/std::vector<int64_t>{16},
        /*output_dim=*/8, /*n_experts=*/1, MoERoutingType::Soft,
        /*top_k=*/1, /*noise_std=*/0.0f, /*dropout=*/0.0f));
    REQUIRE_THROWS(MixtureOfExperts(
        8, std::vector<int64_t>{16}, 8, 1, MoERoutingType::TopK, 1, 0.0f, 0.0f));
}

TEST_CASE("MoE gate probabilities are a valid distribution", "[moe][review]") {
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

TEST_CASE("MoE Soft aux loss is finite for two experts", "[moe][review]") {
    torch::manual_seed(1);
    MixtureOfExperts moe(4, std::vector<int64_t>{8}, 4, 2, MoERoutingType::Soft,
                         1, 0.0f, 0.0f);
    moe->train();
    auto out = moe->forward(torch::randn({16, 4}));
    REQUIRE(std::isfinite(out.aux_loss.item<float>()));
}

// ---------------------------------------------------------------------------
// #85 -- VAE projection weight shape matches the (now-correct) header contract
// ---------------------------------------------------------------------------

TEST_CASE("SpeciesVAE projection weights are (encoder_dims[0], input_dim)", "[vae][review]") {
    const int64_t input_dim = 50;
    VAEConfig cfg;  // encoder_dims default {512, 256, 128}, latent_dim 64
    SpeciesVAE vae(input_dim, cfg);
    auto w = vae->get_projection_weights();
    REQUIRE(w.dim() == 2);
    REQUIRE(w.size(0) == cfg.encoder_dims.front());  // 512, NOT latent_dim
    REQUIRE(w.size(1) == input_dim);                 // 50
}

// ---------------------------------------------------------------------------
// #90 -- NCA loss reduction over contributing samples only
// ---------------------------------------------------------------------------

TEST_CASE("NCA loss averages over contributing samples only", "[loss][nca][review]") {
    // Four base samples (classes 0,0,1,1) all have a same-class neighbor. Adding a
    // fifth sample of a unique class with a near-opposite direction (cosine ~ -1,
    // negligible influence on the others' softmax at temperature 0.1) contributes
    // 0 to the loss. The fixed reduction divides by the number of contributing
    // samples, so the loss stays ~unchanged; the old batch-size divisor would have
    // shrunk it by a factor 4/5 (issue #90).
    NCALoss nca(/*latent_dim=*/4, /*n_classes=*/3, /*temperature=*/0.1f, /*n_neighbors=*/0);
    nca->eval();

    auto base = torch::tensor({
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.98f, 0.20f, 0.0f, 0.0f},   // ~ class-0 neighbor of row 0
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.98f, 0.20f},   // ~ class-1 neighbor of row 2
    });
    auto base_tgt = torch::tensor({0, 0, 1, 1}, torch::kLong);

    auto extra = torch::tensor({{-1.0f, -1.0f, -1.0f, -1.0f}});
    auto with_extra = torch::cat({base, extra}, 0);
    auto with_extra_tgt = torch::cat({base_tgt, torch::tensor({2}, torch::kLong)}, 0);

    float loss_base = nca->forward(base, base_tgt).item<float>();
    float loss_extra = nca->forward(with_extra, with_extra_tgt).item<float>();

    REQUIRE(std::isfinite(loss_base));
    REQUIRE(std::isfinite(loss_extra));
    REQUIRE(loss_base > 0.0f);
    // Fixed reduction: within 10% (the extra sample's tiny softmax leakage). The
    // old sum/batch_size reduction would differ by ~20% (factor 4/5).
    REQUIRE(std::abs(loss_extra - loss_base) < 0.10f * loss_base);
}

// ---------------------------------------------------------------------------
// #90 -- typed message passing rejects out-of-range edge types
// ---------------------------------------------------------------------------

TEST_CASE("TypedMessagePassing rejects out-of-range edge types", "[gnn][review]") {
    TypedMessagePassingLayer layer(/*in=*/4, /*out=*/4, /*n_edge_types=*/2,
                                   /*n_heads=*/1, /*dropout=*/0.0f);
    auto nodes = torch::randn({3, 4});
    auto edge_index = torch::tensor({{0, 1}, {1, 2}}, torch::kLong);  // (2, 2)

    SECTION("valid edge types forward") {
        auto edge_type = torch::tensor({0, 1}, torch::kLong);
        REQUIRE_NOTHROW(layer->forward(nodes, edge_index, edge_type));
    }
    SECTION("out-of-range edge type throws") {
        auto edge_type = torch::tensor({0, 5}, torch::kLong);  // 5 >= n_edge_types
        REQUIRE_THROWS(layer->forward(nodes, edge_index, edge_type));
    }
    SECTION("negative edge type throws") {
        auto edge_type = torch::tensor({0, -1}, torch::kLong);
        REQUIRE_THROWS(layer->forward(nodes, edge_index, edge_type));
    }
}

// ---------------------------------------------------------------------------
// #90 -- fuzzy case-folding covers Latin Extended-A
// ---------------------------------------------------------------------------

TEST_CASE("Fuzzy case-insensitive match folds Latin Extended-A", "[fuzzy][review]") {
    // "\xC4\x8D" = U+010D (c-caron, lowercase), "\xC4\x8C" = U+010C (uppercase).
    // "\xC5\xA1" = U+0161 (s-caron), "\xC5\xA0" = U+0160 (uppercase).
    std::vector<std::string> entries = {
        "\xC4\x8D" "ern" "\xC5\xA1",  // "cerns" with carons, lowercase
        "quercus",
    };
    fuzzy::BuildOptions bopts;
    bopts.case_insensitive = true;
    auto index = fuzzy::FuzzyIndex::build(entries, bopts);

    fuzzy::QueryOptions qopts;
    qopts.max_edit_distance = 0;  // require an exact case-folded match
    qopts.top_n = 1;

    // Uppercase query with the same diacritics must fold to distance 0.
    auto matches = index.query("\xC4\x8C" "ERN" "\xC5\xA0", qopts);
    REQUIRE(!matches.empty());
    REQUIRE(matches[0].distance == 0);
    REQUIRE(matches[0].entry == entries[0]);
}
