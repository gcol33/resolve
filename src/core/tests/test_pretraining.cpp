// Self-supervised pretraining building blocks: the T-JEPA feature masker and
// predictor, the SCARF corruptor and projection head, the SCARF pretrainer on a
// non-hash encoder, and PretrainConfig validation.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/model.hpp"
#include "resolve/pretraining.hpp"

#include <torch/torch.h>

#include <cmath>
#include <stdexcept>

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// T-JEPA: FeatureMasker
// ============================================================================

TEST_CASE("FeatureMasker create_mask shape", "[pretraining]") {
    FeatureMasker masker(100, 0.3f);
    auto mask = masker->create_mask(16);

    REQUIRE(mask.size(0) == 16);
    REQUIRE(mask.size(1) == 100);
}

TEST_CASE("FeatureMasker mask ratio is approximate", "[pretraining]") {
    FeatureMasker masker(200, 0.3f, MaskStrategy::Random);
    auto mask = masker->create_mask(64);

    // Mean of mask should be ~0.7 (70% kept)
    float keep_ratio = mask.to(torch::kFloat).mean().item<float>();
    REQUIRE_THAT(keep_ratio, WithinAbs(0.7f, 0.1f));
}

TEST_CASE("FeatureMasker apply_mask preserves shape", "[pretraining]") {
    FeatureMasker masker(50, 0.3f);
    auto features = torch::randn({8, 50});
    auto mask = masker->create_mask(8);
    auto masked = masker->apply_mask(features, mask);

    REQUIRE(masked.size(0) == 8);
    REQUIRE(masked.size(1) == 50);
}

TEST_CASE("FeatureMasker accessors", "[pretraining]") {
    FeatureMasker masker(64, 0.4f);
    REQUIRE(masker->n_features() == 64);
    REQUIRE_THAT(masker->mask_ratio(), WithinAbs(0.4f, 1e-6f));
}

// ============================================================================
// T-JEPA: JEPAPredictor
// ============================================================================

TEST_CASE("JEPAPredictor forward shape", "[pretraining]") {
    JEPAPredictor predictor(64, 128, 2, 0.1f);
    auto context = torch::randn({16, 64});
    auto predicted = predictor->forward(context);

    REQUIRE(predicted.size(0) == 16);
    REQUIRE(predicted.size(1) == 64);  // Same as latent_dim
}

TEST_CASE("JEPAPredictor gradient flow", "[pretraining]") {
    JEPAPredictor predictor(32, 64, 2, 0.0f);
    auto x = torch::randn({8, 32}, torch::requires_grad(true));
    auto out = predictor->forward(x);
    out.sum().backward();

    REQUIRE(x.grad().defined());
}

// ============================================================================
// SCARF: SCARFCorruptor
// ============================================================================

TEST_CASE("SCARFCorruptor corrupt preserves shape", "[pretraining]") {
    SCARFCorruptor corruptor(50, 0.6f);
    auto features = torch::randn({16, 50});
    auto corrupted = corruptor->corrupt(features);

    REQUIRE(corrupted.size(0) == 16);
    REQUIRE(corrupted.size(1) == 50);
}

TEST_CASE("SCARFCorruptor actually modifies features", "[pretraining]") {
    SCARFCorruptor corruptor(50, 0.6f);
    auto features = torch::randn({16, 50});
    auto corrupted = corruptor->corrupt(features);

    // With 60% corruption, the outputs should differ
    auto diff = (features - corrupted).abs().sum().item<float>();
    REQUIRE(diff > 0.0f);
}

TEST_CASE("SCARFCorruptor accessors", "[pretraining]") {
    SCARFCorruptor corruptor(100, 0.5f);
    REQUIRE(corruptor->n_features() == 100);
    REQUIRE_THAT(corruptor->corruption_rate(), WithinAbs(0.5f, 1e-6f));
}

// ============================================================================
// SCARF: ProjectionHead and the pretrainer
// ============================================================================

TEST_CASE("ProjectionHead forward shape", "[pretraining]") {
    ProjectionHead head(64, 128);
    auto x = torch::randn({16, 64});
    auto out = head->forward(x);

    REQUIRE(out.size(0) == 16);
    REQUIRE(out.size(1) == 128);
}

TEST_CASE("SCARF pretrain runs on embed encoder with masked species views",
          "[pretraining][scarf]") {
    // For non-hash encoders the species composition lives in the ID /
    // explicit-vector tensors, not in `continuous`. SCARF masks the species side
    // of view 2, so the two contrastive views differ in the species channel and
    // the pretext task is not trivially solvable by species identity. This
    // exercises the masked-species path end-to-end and guards its wiring
    // (shape / finite loss).
    ResolveSchema schema;
    schema.n_plots = 64;
    schema.n_species = 50;
    schema.n_species_vocab = 100;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.n_genera_vocab = 25;
    schema.n_families_vocab = 15;
    schema.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Embed;
    mcfg.species_embed_dim = 16;
    mcfg.top_k_species = 5;
    mcfg.n_taxonomy_slots = 3;
    mcfg.hidden_dims = {32, 16};

    ResolveModel model(schema, mcfg);

    PretrainConfig pcfg;
    pcfg.pretrain_epochs = 2;
    pcfg.batch_size = 16;
    pcfg.corruption_rate = 0.6f;
    SCARFPretrainer pretrainer(model, pcfg);

    // n_continuous = 2 (coords) + 1 (unknown_fraction) = 3
    auto continuous = torch::randn({64, 3});
    auto species_ids = torch::randint(1, 100, {64, 5});
    auto genus_ids = torch::randint(1, 25, {64, 3});
    auto family_ids = torch::randint(1, 15, {64, 3});

    auto result = pretrainer.pretrain(continuous, genus_ids, family_ids, species_ids, {});

    REQUIRE(result.epochs_completed == 2);
    REQUIRE(result.loss_history.size() == 2);
    for (float l : result.loss_history) {
        REQUIRE(std::isfinite(l));
    }
}

// ============================================================================
// PretrainConfig validation
// ============================================================================

TEST_CASE("PretrainConfig::validate rejects bad values", "[pretraining][validation]") {
    // batch_size == 0 divides by zero when computing steps; mask_ratio >= 1
    // makes the Block strategy's randint bound <= 0 (throws). The pretrainer
    // constructors call validate(), so a bad config fails fast at construction
    // rather than deep inside pretrain().
    ResolveSchema schema;
    schema.n_plots = 8;
    schema.n_species = 10;
    schema.n_species_vocab = 20;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});
    ModelConfig mcfg;
    mcfg.hidden_dims = {8};
    ResolveModel model(schema, mcfg);

    SECTION("valid config passes") {
        PretrainConfig ok;
        REQUIRE_NOTHROW(ok.validate());
    }
    SECTION("batch_size == 0 throws") {
        PretrainConfig bad;
        bad.batch_size = 0;
        REQUIRE_THROWS_AS(bad.validate(), std::invalid_argument);
        REQUIRE_THROWS_AS(JEPAPretrainer(model, bad), std::invalid_argument);
        REQUIRE_THROWS_AS(SCARFPretrainer(model, bad), std::invalid_argument);
    }
    SECTION("mask_ratio >= 1 throws") {
        PretrainConfig bad;
        bad.mask_ratio = 1.0f;
        REQUIRE_THROWS_AS(bad.validate(), std::invalid_argument);
    }
    SECTION("mask_ratio <= 0 throws") {
        PretrainConfig bad;
        bad.mask_ratio = 0.0f;
        REQUIRE_THROWS_AS(bad.validate(), std::invalid_argument);
    }
    SECTION("corruption_rate out of [0,1] throws") {
        PretrainConfig bad;
        bad.corruption_rate = 1.5f;
        REQUIRE_THROWS_AS(bad.validate(), std::invalid_argument);
    }
}
