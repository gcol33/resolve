// Attention-based tabular encoders as standalone modules: FT-Transformer,
// TabNet, SAINT and ExcelFormer.

#include <catch2/catch_test_macros.hpp>

#include "resolve/attention.hpp"

#include <torch/torch.h>

#include <set>
#include <string>
#include <vector>

using namespace resolve;

// ============================================================================
// FT-Transformer, TabNet, SAINT
// ============================================================================

TEST_CASE("FTTransformerEncoder forward shape", "[attention]") {
    std::vector<int64_t> no_cats;
    FTTransformerEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
        /*n_layers=*/2, /*use_cls_token=*/true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("TabNetEncoder forward shape", "[attention]") {
    TabNetEncoder encoder(32, 3, 16, 16, 1.5f, 1e-3f);
    auto x = torch::randn({8, 32});
    auto [out, importance] = encoder->forward(x);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 16);  // n_d
    REQUIRE(importance.size(0) == 8);
    REQUIRE(importance.size(1) == 32);  // input_dim
}

TEST_CASE("SAINTEncoder forward shape", "[attention]") {
    std::vector<int64_t> no_cats;
    SAINTEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
        /*n_layers=*/2, /*use_row_attention=*/true, /*use_cls_token=*/true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

// ============================================================================
// ExcelFormer
// ============================================================================

TEST_CASE("ExcelFormerEncoder forward shape", "[excelformer]") {
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
        /*n_layers=*/2, 0.5f, true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("ExcelFormerEncoder with categoricals", "[excelformer]") {
    std::vector<int64_t> cat_cards = {5, 10};
    ExcelFormerEncoder encoder(
        8, cat_cards, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
        /*n_layers=*/2, 0.5f, true);

    auto numerical = torch::randn({8, 8});
    std::vector<torch::Tensor> categoricals = {
        torch::randint(0, 5, {8}),
        torch::randint(0, 10, {8})
    };
    auto out = encoder->forward(numerical, categoricals);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);
}

TEST_CASE("ExcelFormerEncoder feature_importance", "[excelformer]") {
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(32, 4, 0, 0.1f), /*n_layers=*/2);
    auto importance = encoder->feature_importance();

    // Should be sigmoid of learnable logits, so in [0, 1]
    REQUIRE(importance.min().item<float>() >= 0.0f);
    REQUIRE(importance.max().item<float>() <= 1.0f);
}

TEST_CASE("ExcelFormerEncoder importance_logits receive gradient", "[excelformer]") {
    // The semi-permeable mask must be differentiable so the learnable
    // importance parameter actually trains. A hard boolean mask leaves it
    // gradient-dead (importance stuck at its init, mask permanently open).
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(32, 4, 0, 0.1f),
        /*n_layers=*/2, 0.5f, true);

    torch::Tensor importance_logits;
    for (const auto& named : encoder->named_parameters()) {
        if (named.key().find("importance_logits") != std::string::npos) {
            importance_logits = named.value();
        }
    }
    REQUIRE(importance_logits.defined());

    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);
    auto loss = out.pow(2).mean();
    loss.backward();

    REQUIRE(importance_logits.grad().defined());
    REQUIRE(importance_logits.grad().abs().sum().item<float>() > 0.0f);
}

TEST_CASE("ExcelFormerEncoder does not double-register block parameters",
          "[excelformer][params]") {
    // Each transformer block must be a child of the layers ModuleList only.
    // Registering it a second time by name would alias the same storage under
    // two paths in named_parameters(), doubling optimizer updates and the
    // checkpoint. libtorch C++ named_parameters() does not dedupe shared
    // submodules (unlike Python's remove_duplicate default), so any aliasing
    // surfaces here.
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(
        10, no_cats, TransformerBlockConfig::uniform(64, 4, 0, 0.1f),
        /*n_layers=*/2, 0.5f, true);

    std::set<void*> storages;
    for (const auto& named : encoder->named_parameters()) {
        void* ptr = named.value().data_ptr();
        INFO("duplicate parameter storage under: " << named.key());
        REQUIRE(storages.insert(ptr).second);
    }
}
