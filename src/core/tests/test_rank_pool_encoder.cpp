#include <catch2/catch_test_macros.hpp>
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"

using namespace resolve;

// ============================================================================
// PlotEncoderRankPool Tests
// ============================================================================

TEST_CASE("PlotEncoderRankPool forward without taxonomy", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*species_embed_dim=*/32
    );

    auto continuous = torch::randn({8, 10});
    auto species_ids = torch::randint(0, 100, {8, 20}, torch::kInt64);
    species_ids.index_put_({torch::indexing::Slice(), torch::indexing::Slice(15, 20)}, 0);  // pad last 5

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(!encoder->has_taxonomy());
}

TEST_CASE("PlotEncoderRankPool forward with taxonomy", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/30,
        /*n_families=*/10,
        /*species_embed_dim=*/32,
        /*genus_embed_dim=*/16,
        /*family_embed_dim=*/8
    );

    auto continuous = torch::randn({4, 10});
    auto species_ids = torch::randint(1, 100, {4, 15}, torch::kInt64);
    auto genus_ids = torch::randint(1, 30, {4, 15}, torch::kInt64);
    auto family_ids = torch::randint(1, 10, {4, 15}, torch::kInt64);
    auto weights = torch::rand({4, 15});

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids, genus_ids, family_ids, weights);

    REQUIRE(out.size(0) == 4);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(encoder->has_taxonomy());
}

TEST_CASE("PlotEncoderRankPool enables taxonomy for family-only data", "[rank_pool][taxonomy]") {
    // Family mapped, no genus: genus vocab is 1 (UNK only), family vocab > 1.
    // The gate must be (n_genera > 1 || n_families > 1), so the family
    // embeddings survive. The old (n_genera > 0 && n_families > 0) gate dropped
    // them whenever genus was absent.
    PlotEncoderRankPool encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/1,      // UNK-only
        /*n_families=*/10,
        /*species_embed_dim=*/32,
        /*genus_embed_dim=*/16,
        /*family_embed_dim=*/8
    );
    REQUIRE(encoder->has_taxonomy());

    auto continuous = torch::randn({4, 10});
    auto species_ids = torch::randint(1, 100, {4, 15}, torch::kInt64);
    auto genus_ids = torch::zeros({4, 15}, torch::kInt64);            // all UNK
    auto family_ids = torch::randint(1, 10, {4, 15}, torch::kInt64);
    auto weights = torch::rand({4, 15});

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids, genus_ids, family_ids, weights);
    REQUIRE(out.size(0) == 4);
    REQUIRE(out.size(1) == encoder->latent_dim());
}

TEST_CASE("PlotEncoderRankPool enables taxonomy for genus-only data", "[rank_pool][taxonomy]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/30,
        /*n_families=*/1,    // UNK-only
        /*species_embed_dim=*/32,
        /*genus_embed_dim=*/16,
        /*family_embed_dim=*/8
    );
    REQUIRE(encoder->has_taxonomy());
}

TEST_CASE("PlotEncoderRankPool mask handling excludes padding", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*species_embed_dim=*/16
    );

    auto continuous = torch::randn({2, 5});
    auto species_ids = torch::zeros({2, 10}, torch::kInt64);
    species_ids[0][0] = 1;
    species_ids[0][1] = 2;
    species_ids[1][0] = 3;

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids);

    REQUIRE(out.size(0) == 2);
    REQUIRE(out.size(1) == encoder->latent_dim());
    // Outputs should be finite (no NaN from division by zero)
    REQUIRE(out.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderRankPool different weights produce different outputs", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*species_embed_dim=*/16
    );

    auto continuous = torch::randn({1, 5});
    auto species_ids = torch::tensor({{1, 2, 3, 4, 5}}, torch::kInt64);
    auto weights_a = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
    auto weights_b = torch::tensor({{0.0f, 0.0f, 0.0f, 0.0f, 1.0f}});

    encoder->eval();
    auto out_a = encoder->forward(continuous, species_ids, {}, {}, weights_a);
    auto out_b = encoder->forward(continuous, species_ids, {}, {}, weights_b);

    REQUIRE(!torch::allclose(out_a, out_b));
}

TEST_CASE("PlotEncoderRankPool embedding extraction shapes", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/5,
        /*n_species=*/100,
        /*n_genera=*/30,
        /*n_families=*/10,
        /*species_embed_dim=*/32,
        /*genus_embed_dim=*/16,
        /*family_embed_dim=*/8
    );

    auto sp_w = encoder->get_species_weights();
    auto g_w = encoder->get_genus_weights();
    auto f_w = encoder->get_family_weights();

    REQUIRE(sp_w.size(0) == 100);
    REQUIRE(sp_w.size(1) == 32);
    REQUIRE(g_w.size(0) == 30);
    REQUIRE(g_w.size(1) == 16);
    REQUIRE(f_w.size(0) == 10);
    REQUIRE(f_w.size(1) == 8);
}

TEST_CASE("PlotEncoderRankPool gradient flow", "[rank_pool]") {
    PlotEncoderRankPool encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*species_embed_dim=*/16
    );

    auto continuous = torch::randn({4, 5}, torch::requires_grad());
    auto species_ids = torch::randint(1, 50, {4, 10}, torch::kInt64);

    encoder->train();
    auto out = encoder->forward(continuous, species_ids);
    auto loss = out.sum();
    loss.backward();

    REQUIRE(continuous.grad().defined());
    REQUIRE(continuous.grad().abs().sum().item<float>() > 0);
}

TEST_CASE("PlotEncoderRankPool fused pool equals explicit weighted mean", "[rank_pool]") {
    // The rank-pool forward replaced an explicit gather-multiply-sum over a
    // (batch, max_sp, embed_dim) intermediate with a fused embedding_bag
    // weighted-sum (issue #6). This guards the invariant that the fused path is
    // numerically identical to the explicit one over the same embedding tables.
    namespace F = torch::nn::functional;
    torch::manual_seed(0);

    const int64_t batch = 6, max_sp = 12;
    const int64_t n_species = 40, n_genera = 15, n_families = 7;
    const int64_t sp_dim = 8, g_dim = 4, f_dim = 4;

    // Embedding tables with row 0 zeroed (padding_idx convention).
    auto make_table = [](int64_t rows, int64_t dim) {
        auto w = torch::randn({rows, dim});
        w.index_put_({0}, torch::zeros({dim}));
        return w;
    };
    auto sp_w = make_table(n_species, sp_dim);
    auto g_w = make_table(n_genera, g_dim);
    auto f_w = make_table(n_families, f_dim);

    // IDs with some padding/UNK (id 0) entries; weights with a per-plot mask.
    auto sp_ids = torch::randint(0, n_species, {batch, max_sp}, torch::kInt64);
    auto g_ids = torch::randint(0, n_genera, {batch, max_sp}, torch::kInt64);
    auto f_ids = torch::randint(0, n_families, {batch, max_sp}, torch::kInt64);
    sp_ids.index_put_({torch::indexing::Slice(), torch::indexing::Slice(9, max_sp)}, 0);
    auto mask = (sp_ids != 0).to(torch::kFloat32);
    auto weights = torch::rand({batch, max_sp}) * mask;
    auto w_sum = weights.sum(1, true).clamp_min(1e-8);
    auto w_normed = weights / w_sum;

    // Explicit path: materialize, multiply, sum.
    auto combined = torch::cat({
        F::embedding(sp_ids, sp_w, F::EmbeddingFuncOptions().padding_idx(0)),
        F::embedding(g_ids, g_w, F::EmbeddingFuncOptions().padding_idx(0)),
        F::embedding(f_ids, f_w, F::EmbeddingFuncOptions().padding_idx(0)),
    }, -1);
    auto explicit_pooled = (combined * w_normed.unsqueeze(-1)).sum(1);

    // Fused path: three embedding_bags, concatenated.
    auto bag_opts = F::EmbeddingBagFuncOptions()
        .mode(torch::kSum).per_sample_weights(w_normed).padding_idx(0);
    auto fused_pooled = torch::cat({
        F::embedding_bag(sp_ids, sp_w, bag_opts),
        F::embedding_bag(g_ids, g_w, bag_opts),
        F::embedding_bag(f_ids, f_w, bag_opts),
    }, -1);

    REQUIRE(fused_pooled.sizes() == explicit_pooled.sizes());
    REQUIRE(torch::allclose(fused_pooled, explicit_pooled, /*rtol=*/1e-5, /*atol=*/1e-6));
}

TEST_CASE("ResolveModel with RankPool mode constructs and forwards", "[rank_pool][model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 500;
    schema.n_species_vocab = 500;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 50;
    schema.n_genera_vocab = 50;
    schema.n_families = 20;
    schema.n_families_vocab = 20;
    schema.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::RankPool;
    config.species_embed_dim = 32;
    config.genus_emb_dim = 16;
    config.family_emb_dim = 8;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);
    model->eval();

    auto continuous = torch::randn({4, 2});
    auto species_ids = torch::randint(1, 500, {4, 20}, torch::kInt64);
    auto pool_genus_ids = torch::randint(1, 50, {4, 20}, torch::kInt64);
    auto pool_family_ids = torch::randint(1, 20, {4, 20}, torch::kInt64);
    auto pool_weights = torch::rand({4, 20});
    auto pool_mask = (species_ids != 0);
    auto pool_has_cover = torch::ones({4});

    auto outputs = model->forward(
        continuous, {}, {}, species_ids, {},
        pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover
    );

    REQUIRE(outputs.count("area") == 1);
    REQUIRE(outputs["area"].size(0) == 4);
}
