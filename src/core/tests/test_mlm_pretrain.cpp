#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/pretraining.hpp"
#include "resolve/encoder.hpp"

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// mask_species_batch Tests
// ============================================================================

TEST_CASE("mask_species_batch respects padding", "[mlm]") {
    auto species_ids = torch::zeros({4, 10}, torch::kInt64);
    // Only first 5 positions are valid
    species_ids.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 5)},
        torch::randint(1, 50, {4, 5}));

    auto valid_mask = (species_ids != 0);

    auto [masked_ids, mlm_mask, targets, mask_token_positions] =
        mask_species_batch(species_ids, valid_mask, 50, 0.5f);

    // No padding positions should be masked
    auto padding_positions = ~valid_mask;
    REQUIRE((mlm_mask & padding_positions).sum().item<int64_t>() == 0);
    REQUIRE((mask_token_positions & padding_positions).sum().item<int64_t>() == 0);
    // The mask-token subset is a subset of all masked positions.
    REQUIRE((mask_token_positions & ~mlm_mask).sum().item<int64_t>() == 0);
}

TEST_CASE("mask_species_batch approximate ratios", "[mlm]") {
    torch::manual_seed(42);
    auto species_ids = torch::randint(1, 100, {100, 50}, torch::kInt64);
    auto valid_mask = torch::ones({100, 50}, torch::kBool);

    auto [masked_ids, mlm_mask, targets, mask_token_positions] =
        mask_species_batch(species_ids, valid_mask, 100, 0.15f);

    // ~15% of 5000 valid positions should be masked
    float mask_ratio = static_cast<float>(mlm_mask.sum().item<int64_t>()) / 5000.0f;
    REQUIRE_THAT(mask_ratio, WithinAbs(0.15, 0.05));

    // Targets should have same count as masked positions
    REQUIRE(targets.numel() == mlm_mask.sum().item<int64_t>());

    // Only ~80% of masked positions carry the mask token, so the subset is a
    // strict minority of masked positions (never the whole set).
    REQUIRE(mask_token_positions.sum().item<int64_t>() < mlm_mask.sum().item<int64_t>());
}

TEST_CASE("mask_species_batch BERT 80/10/10 split", "[mlm]") {
    torch::manual_seed(123);
    auto species_ids = torch::randint(1, 100, {200, 50}, torch::kInt64);
    auto valid_mask = torch::ones({200, 50}, torch::kBool);

    auto [masked_ids, mlm_mask, targets, mask_token_positions] =
        mask_species_batch(species_ids, valid_mask, 100, 0.15f);

    int64_t n_masked = mlm_mask.sum().item<int64_t>();
    if (n_masked > 100) {  // Need enough samples for statistics
        // Count how many masked positions became 0 (mask token)
        auto original = species_ids.index({mlm_mask});
        auto modified = masked_ids.index({mlm_mask});
        int64_t n_zeroed = (modified == 0).sum().item<int64_t>();
        int64_t n_changed_nonzero = ((modified != original) & (modified != 0)).sum().item<int64_t>();
        int64_t n_kept = (modified == original).sum().item<int64_t>();

        float pct_zeroed = static_cast<float>(n_zeroed) / n_masked;
        float pct_random = static_cast<float>(n_changed_nonzero) / n_masked;
        float pct_kept = static_cast<float>(n_kept) / n_masked;

        REQUIRE_THAT(pct_zeroed, WithinAbs(0.8, 0.1));
        REQUIRE_THAT(pct_random, WithinAbs(0.1, 0.08));
        REQUIRE_THAT(pct_kept, WithinAbs(0.1, 0.08));

        // The returned mask-token subset must be exactly the zeroed positions —
        // this is what the encoder replaces with the mask embedding, so the
        // random/keep branches actually reach the encoder.
        auto zeroed_positions = (masked_ids == 0) & mlm_mask;
        REQUIRE((mask_token_positions != zeroed_positions).sum().item<int64_t>() == 0);
        REQUIRE(mask_token_positions.sum().item<int64_t>() == n_zeroed);
    }
}

// ============================================================================
// MaskedSpeciesHead Tests
// ============================================================================

TEST_CASE("MaskedSpeciesHead output shape", "[mlm]") {
    MaskedSpeciesHead head(64, 500);

    auto tokens = torch::randn({100, 64});
    auto logits = head->forward(tokens);

    REQUIRE(logits.size(0) == 100);
    REQUIRE(logits.size(1) == 500);
}

// ============================================================================
// MaskedSpeciesPretrainer Tests
// ============================================================================

TEST_CASE("MaskedSpeciesPretrainer single epoch runs", "[mlm]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/1
    );

    MLMPretrainConfig config;
    config.pretrain_epochs = 1;
    config.batch_size = 16;
    config.mask_prob = 0.3f;

    MaskedSpeciesPretrainer pretrainer(encoder, 50, config);

    auto species_ids = torch::randint(1, 50, {32, 15}, torch::kInt64);
    auto valid_mask = (species_ids != 0);

    auto losses = pretrainer.pretrain(species_ids, {}, {}, {}, valid_mask);

    REQUIRE(losses.size() == 1);
    REQUIRE(std::isfinite(losses[0]));
    REQUIRE(losses[0] > 0);
}

TEST_CASE("MaskedSpeciesPretrainer encoder weights change after training", "[mlm]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/1
    );

    // Snapshot initial weights
    auto initial_weights = encoder->get_species_weights().clone();

    MLMPretrainConfig config;
    config.pretrain_epochs = 3;
    config.batch_size = 16;
    config.pretrain_lr = 1e-3f;

    MaskedSpeciesPretrainer pretrainer(encoder, 50, config);

    auto species_ids = torch::randint(1, 50, {64, 15}, torch::kInt64);
    auto valid_mask = (species_ids != 0);

    pretrainer.pretrain(species_ids, {}, {}, {}, valid_mask);

    // Weights should have changed
    auto final_weights = pretrainer.encoder()->get_species_weights();
    REQUIRE(!torch::allclose(initial_weights, final_weights));
}
