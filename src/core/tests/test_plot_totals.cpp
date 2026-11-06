#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "resolve/plot_encoder_rank_pool.hpp"
#include "resolve/plot_encoder_transformer.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/species_pool.hpp"
#include "test_support.hpp"

#include <cmath>
#include <functional>
#include <memory>

using namespace resolve;
using namespace resolve::test;
using Catch::Approx;

// ============================================================================
// Plot totals: combined cover and species count as model inputs
// ============================================================================

namespace {

constexpr int64_t kBatch = 4;
constexpr int64_t kSlots = 6;

using FeatureFactory =
    std::function<std::unique_ptr<SpeciesFeatures>(torch::nn::Module&, const SpeciesEncoderSpec&)>;

SpeciesEncoderSpec spec_for(SpeciesEncodingMode mode, bool totals, float dropout = 0.0f) {
    auto spec = encoder_spec(mode, /*n_continuous=*/0, /*n_species=*/50);
    spec.config.plot_totals = totals;
    spec.config.plot_totals_dropout = dropout;
    return spec;
}

PlotInputs pool_batch(bool with_totals) {
    torch::manual_seed(3);
    PlotInputs in;
    in.species_ids = torch::randint(1, 50, {kBatch, kSlots}, torch::kInt64);
    in.pool_weights = torch::rand({kBatch, kSlots});
    in.pool_mask = in.species_ids != 0;
    in.pool_has_cover = torch::ones({kBatch});
    if (with_totals) {
        in.pool_totals = torch::tensor({0.75f, 1.1f, 0.2f, 2.3f, 0.9f, 1.6f, 0.0f, 0.7f})
                             .reshape({kBatch, 2});
    }
    return in;
}

struct Encoding {
    SpeciesEncodingMode mode;
    FeatureFactory make;
};

const Encoding kEncodings[] = {
    {SpeciesEncodingMode::RankPool, make_rank_pool_features},
    {SpeciesEncodingMode::Transformer, make_transformer_features},
};

// The last kPlotTotalsWidth columns of `features`.
torch::Tensor tail(const torch::Tensor& features) {
    return features.narrow(1, features.size(1) - kPlotTotalsWidth, kPlotTotalsWidth);
}

}  // namespace

TEST_CASE("Plot totals are the random-overlap cover and log(1 + species count)",
          "[plot_totals]") {
    const std::vector<SpeciesRecord> records = {
        {"a", "", "", 50.0f, "P1"}, {"b", "", "", 50.0f, "P1"},
        {"a", "", "", 250.0f, "P2"}, {"c", "", "", 10.0f, "P2"}, {"d", "", "", 5.0f, "P2"},
        {"b", "", "", 0.5f, "P3"},
    };
    const std::vector<std::string> plots = {"P1", "P2", "P3"};
    RankPoolEncoder encoder;
    encoder.fit(records);

    const auto with_cover = encoder.transform(records, plots, 0, /*has_abundance_column=*/true).totals;
    REQUIRE(with_cover.sizes() == torch::IntArrayRef({3, 2}));
    CHECK(with_cover[0][0].item<float>() == Approx(0.75));   // 1 - 0.5 * 0.5
    CHECK(with_cover[1][0].item<float>() == Approx(1.0));    // a cover above 100% is capped
    CHECK(with_cover[2][0].item<float>() == Approx(0.005));
    CHECK(with_cover[0][1].item<float>() == Approx(std::log1p(2.0)));
    CHECK(with_cover[1][1].item<float>() == Approx(std::log1p(3.0)));
    CHECK(with_cover[2][1].item<float>() == Approx(std::log1p(1.0)));

    // Without an abundance column the cover is unknown: 0, the count stays.
    const auto presence = encoder.transform(records, plots, 0, /*has_abundance_column=*/false).totals;
    CHECK(torch::equal(presence.select(1, 0), torch::zeros({3})));
    CHECK(torch::allclose(presence.select(1, 1), with_cover.select(1, 1)));
}

TEST_CASE("Plot totals are counted before the species cap", "[plot_totals]") {
    std::vector<SpeciesRecord> records;
    for (int s = 0; s < 10; ++s) records.push_back({"sp" + std::to_string(s), "", "", 1.0f, "P"});
    RankPoolEncoder encoder;
    encoder.fit(records);
    const auto out = encoder.transform(records, {"P"}, /*species_cap=*/3, true);
    REQUIRE(out.species_ids.size(1) == 3);
    CHECK(out.totals[0][1].item<float>() == Approx(std::log1p(10.0)));
}

TEST_CASE("Without plot_totals the features ignore the totals", "[plot_totals]") {
    for (const auto& enc : kEncodings) {
        torch::nn::Module owner;
        auto features = enc.make(owner, spec_for(enc.mode, /*totals=*/false));
        owner.eval();
        auto with_totals = pool_batch(true);
        auto without = pool_batch(false);
        const auto a = features->forward(with_totals, /*training=*/false);
        const auto b = features->forward(without, /*training=*/false);
        CHECK(a.size(1) == features->width());
        CHECK(torch::equal(a, b));
    }
}

TEST_CASE("plot_totals appends the totals and a presence flag", "[plot_totals]") {
    for (const auto& enc : kEncodings) {
        torch::nn::Module plain_owner;
        torch::nn::Module owner;
        const auto plain = enc.make(plain_owner, spec_for(enc.mode, false));
        auto features = enc.make(owner, spec_for(enc.mode, true));
        owner.eval();
        REQUIRE(features->width() == plain->width() + kPlotTotalsWidth);

        auto in = pool_batch(true);
        const auto given = tail(features->forward(in, /*training=*/false));
        CHECK(torch::allclose(given.narrow(1, 0, 2), in.pool_totals));
        CHECK(torch::equal(given.select(1, 2), torch::ones({kBatch})));

        // Withheld: undefined pool_totals reads as zeros with the flag cleared.
        const auto withheld = tail(features->forward(pool_batch(false), /*training=*/false));
        CHECK(torch::equal(withheld, torch::zeros({kBatch, kPlotTotalsWidth})));
    }
}

TEST_CASE("Totals dropout withholds the totals in training only", "[plot_totals]") {
    for (const auto& enc : kEncodings) {
        torch::nn::Module owner;
        auto features = enc.make(owner, spec_for(enc.mode, true, /*dropout=*/1.0f));
        auto in = pool_batch(true);
        CHECK(torch::equal(tail(features->forward(in, /*training=*/true)),
                           torch::zeros({kBatch, kPlotTotalsWidth})));
        const auto eval = tail(features->forward(in, /*training=*/false));
        CHECK(torch::equal(eval.select(1, 2), torch::ones({kBatch})));
    }
}

TEST_CASE("Cover dropout clears the combined cover, not the species count", "[plot_totals]") {
    PoolInputs pool = PoolInputs::from(pool_batch(true));
    const auto count = pool.totals.select(1, 1).clone();
    pool.apply_cover_dropout(/*training=*/true, /*cover_dropout=*/1.0f);
    CHECK(torch::equal(pool.totals.select(1, 0), torch::zeros({kBatch})));
    CHECK(torch::equal(pool.totals.select(1, 1), count));
    CHECK(torch::equal(pool.has_totals, torch::ones({kBatch})));
}
