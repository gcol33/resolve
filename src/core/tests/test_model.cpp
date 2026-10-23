#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/model.hpp"
#include "resolve/plot_encoder_embed.hpp"
#include "resolve/plot_encoder_hash.hpp"
#include "resolve/plot_encoder_sparse.hpp"
#include "test_support.hpp"

using namespace resolve;
using namespace resolve::test;

TEST_CASE("PlotEncoder forward without taxonomy", "[encoder]") {
    auto spec = encoder_spec(SpeciesEncodingMode::Hash, 10, 0);
    spec.config.hash_dim = 0;
    PlotEncoder encoder(spec);

    auto output = encoder->forward({.continuous = torch::randn({4, 10})});

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 16);  // Last hidden dim
}

TEST_CASE("PlotEncoder forward with taxonomy", "[encoder]") {
    auto spec = encoder_spec(SpeciesEncodingMode::Hash, 10, 0, 100, 50);
    spec.config.hash_dim = 0;
    spec.config.n_taxonomy_slots = 3;
    PlotEncoder encoder(spec);

    auto output = encoder->forward({.continuous = torch::randn({4, 10}),
                                    .genus_ids = torch::randint(0, 100, {4, 3}),
                                    .family_ids = torch::randint(0, 50, {4, 3})});

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 16);
}

// A taxonomy-enabled concat encoder reserves fixed input width for BOTH genus
// and family, so forward() must be given both; check_concat_taxonomy names the
// missing one instead of an opaque torch::cat/Linear shape error.
TEST_CASE("PlotEncoder with taxonomy throws on missing taxonomy input", "[encoder]") {
    auto spec = encoder_spec(SpeciesEncodingMode::Hash, 10, 0, 100, 50);
    spec.config.hash_dim = 0;
    spec.config.n_taxonomy_slots = 3;
    PlotEncoder encoder(spec);
    auto continuous = torch::randn({4, 10});
    auto genus_ids = torch::randint(0, 100, {4, 3});

    SECTION("no taxonomy tensors at all") {
        REQUIRE_THROWS(encoder->forward({.continuous = continuous}));
    }
    SECTION("only genus provided, family missing") {
        REQUIRE_THROWS(encoder->forward({.continuous = continuous, .genus_ids = genus_ids}));
    }
    SECTION("both provided still works") {
        REQUIRE_NOTHROW(encoder->forward({.continuous = continuous,
                                          .genus_ids = genus_ids,
                                          .family_ids = torch::randint(0, 50, {4, 3})}));
    }
}

// check_concat_taxonomy is the shared guard. Direct unit coverage.
TEST_CASE("check_concat_taxonomy gates on both taxonomy tensors", "[encoder]") {
    auto g = torch::randint(0, 10, {4, 3});
    auto f = torch::randint(0, 10, {4, 3});
    // Disabled taxonomy: never throws regardless of inputs.
    REQUIRE_NOTHROW(check_concat_taxonomy(false, torch::Tensor(), torch::Tensor(), "x"));
    // Enabled taxonomy: both required.
    REQUIRE_NOTHROW(check_concat_taxonomy(true, g, f, "x"));
    REQUIRE_THROWS(check_concat_taxonomy(true, g, torch::Tensor(), "x"));
    REQUIRE_THROWS(check_concat_taxonomy(true, torch::Tensor(), f, "x"));
    REQUIRE_THROWS(check_concat_taxonomy(true, torch::Tensor(), torch::Tensor(), "x"));
}

// TaxonomyVocab::fit reserves <UNK>=0, so n_genera()/n_families() already count
// UNK; every encoding's taxonomy tables have exactly n_genera rows.
TEST_CASE("Hash+taxonomy embedding tables sized to n_genera (no +1 over-alloc)", "[model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;          // dataset convention: count includes <UNK>=0
    schema.n_families = 10;
    schema.dataset_config.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.n_taxonomy_slots = 3;
    config.hidden_dims = {32, 16};

    ResolveModel model(schema, config);

    auto gw = model->get_genus_weights();
    auto fw = model->get_family_weights();
    REQUIRE(gw.defined());
    REQUIRE(fw.defined());
    REQUIRE(gw.size(0) == 20);   // exactly n_genera, not 21
    REQUIRE(fw.size(0) == 10);   // exactly n_families, not 11
}

TEST_CASE("PlotEncoderEmbed forward", "[encoder]") {
    auto spec = encoder_spec(SpeciesEncodingMode::Embed, 5, 100, 20, 10);
    spec.config.species_embed_dim = 16;
    spec.config.genus_emb_dim = 8;
    spec.config.family_emb_dim = 8;
    spec.config.top_k_species = 5;
    spec.config.n_taxonomy_slots = 3;
    PlotEncoderEmbed encoder(spec);

    auto output = encoder->forward({.continuous = torch::randn({4, 5}),
                                    .genus_ids = torch::randint(1, 20, {4, 3}),
                                    .family_ids = torch::randint(1, 10, {4, 3}),
                                    .species_ids = torch::randint(1, 100, {4, 5})});

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 16);
}

TEST_CASE("FusedPositionalEmbedding zeroes padding/UNK at every position", "[encoder]") {
    // Guards the fix where the fused table's single padding_idx only zeroed
    // position 0; positions k>0 with id 0 returned a learnable row k*vocab_size.
    const int64_t vocab = 50;
    const int n_pos = 4, dim = 8;
    FusedPositionalEmbedding emb(vocab, n_pos, dim);
    emb->eval();

    // Positions 0 and 2 are real species; positions 1 and 3 are padding (id 0).
    auto ids = torch::tensor({{5, 0, 7, 0}}, torch::kInt64);  // (1, n_pos)
    auto out = emb->forward(ids);                              // (1, n_pos*dim)
    REQUIRE(out.size(1) == n_pos * dim);

    auto out2d = out.view({1, n_pos, dim});
    // Padding positions contribute exactly zero (regardless of position index).
    REQUIRE(out2d.index({0, 1}).abs().sum().item<float>() == 0.0f);
    REQUIRE(out2d.index({0, 3}).abs().sum().item<float>() == 0.0f);
    // Real positions are non-zero (rows are N(0,1)-initialized).
    REQUIRE(out2d.index({0, 0}).abs().sum().item<float>() > 0.0f);
    REQUIRE(out2d.index({0, 2}).abs().sum().item<float>() > 0.0f);
}

TEST_CASE("PlotEncoderSparse forward", "[encoder]") {
    auto spec = encoder_spec(SpeciesEncodingMode::Sparse, 5, 50, 20, 10);
    spec.config.species_embed_dim = 32;
    spec.config.genus_emb_dim = 8;
    spec.config.family_emb_dim = 8;
    spec.config.n_taxonomy_slots = 3;
    PlotEncoderSparse encoder(spec);

    auto output = encoder->forward({.continuous = torch::randn({4, 5}),
                                    .genus_ids = torch::randint(0, 20, {4, 3}),
                                    .family_ids = torch::randint(0, 10, {4, 3}),
                                    .species_vector = to_sparse_rows(torch::randn({4, 50}))});

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 16);
}

TEST_CASE("TaskHead regression", "[head]") {
    TaskHead head(64, TaskType::Regression, 0, TransformType::None);

    auto latent = torch::randn({4, 64});
    auto output = head->forward(latent);

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 1);
}

TEST_CASE("TaskHead classification", "[head]") {
    TaskHead head(64, TaskType::Classification, 5, TransformType::None);

    auto latent = torch::randn({4, 64});
    auto output = head->forward(latent);

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 5);  // num_classes

    auto pred = head->predict(latent);
    REQUIRE(pred.size(0) == 4);  // Class indices
}

TEST_CASE("TaskHead inverse transform log1p", "[head]") {
    TaskHead head(64, TaskType::Regression, 0, TransformType::Log1p);

    // expm1(log1p(x)) should give x back
    auto test_val = torch::tensor({1.0f});
    auto log_val = torch::log1p(test_val);
    auto back = head->inverse_transform(log_val);

    REQUIRE_THAT(back.item<float>(), Catch::Matchers::WithinAbs(1.0f, 1e-5));
}

TEST_CASE("TaskHead without hidden widths is the single output layer", "[head]") {
    TaskHead single(16, TaskType::Classification, 4);
    CHECK_FALSE(single->has_hidden_layers());
    CHECK(parameter_names(*single) == std::set<std::string>{"output.weight", "output.bias"});

    TaskHead deep(16, TaskType::Regression, 0, TransformType::None, std::vector<int64_t>{8, 6},
                  ActivationType::ReLU, 0.1f);
    CHECK(deep->has_hidden_layers());
    CHECK(any_starts_with(parameter_names(*deep), "head_mlp.0."));
    CHECK(deep->forward(torch::randn({3, 16})).sizes() == torch::IntArrayRef({3, 1}));
}

TEST_CASE("ResolveModel hash mode forward", "[model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.covariate_names = {};
    schema.dataset_config.track_unknown_fraction = true;

    // Add targets
    schema.targets.push_back({"area", TaskType::Regression, TransformType::Log1p, 0, 1.0f});
    schema.targets.push_back({"habitat", TaskType::Classification, TransformType::None, 5, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);

    // n_continuous = 2 (coords) + 0 (covariates) + 1 (unknown_fraction) + 32 (hash) = 35
    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1

    auto outputs = model->forward({.continuous = continuous, .genus_ids = genus_ids, .family_ids = family_ids});

    SECTION("outputs contain all targets") {
        REQUIRE(outputs.count("area") > 0);
        REQUIRE(outputs.count("habitat") > 0);
    }

    SECTION("output shapes are correct") {
        REQUIRE(outputs["area"].size(0) == 8);
        REQUIRE(outputs["area"].size(1) == 1);
        REQUIRE(outputs["habitat"].size(0) == 8);
        REQUIRE(outputs["habitat"].size(1) == 5);
    }

    SECTION("MLP does not require full-batch training") {
        REQUIRE_FALSE(model->requires_full_batch_training());
    }
}

TEST_CASE("ResolveModel embed mode forward", "[model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_species_vocab = 100;  // Required for embed mode
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 25;
    schema.n_families = 15;
    schema.covariate_names = {};
    schema.dataset_config.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Embed;
    config.species_embed_dim = 16;
    config.top_k_species = 5;
    config.n_taxonomy_slots = 3;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);

    // n_continuous = 2 (coords) + 0 (covariates) + 1 (unknown_fraction) = 3
    auto continuous = torch::randn({8, 3});
    auto species_ids = torch::randint(1, 100, {8, 5});
    auto genus_ids = torch::randint(1, 25, {8, 3});
    auto family_ids = torch::randint(1, 15, {8, 3});

    auto outputs = model->forward({.continuous = continuous, .genus_ids = genus_ids,
                                   .family_ids = family_ids, .species_ids = species_ids});

    REQUIRE(outputs.count("area") > 0);
    REQUIRE(outputs["area"].size(0) == 8);
}

TEST_CASE("ResolveModel sparse mode forward", "[model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_species_vocab = 50;  // Required for sparse mode
    schema.has_coordinates = true;
    schema.has_taxonomy = false;
    schema.covariate_names = {"temp", "precip"};
    schema.dataset_config.track_unknown_fraction = false;
    schema.targets.push_back({"biomass", TaskType::Regression, TransformType::Log1p, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;  // Will be overridden by uses_explicit_vector
    config.uses_explicit_vector = true;
    config.species_embed_dim = 32;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);

    // n_continuous = 2 (coords) + 2 (covariates) = 4
    auto continuous = torch::randn({8, 4});
    auto species_vector = torch::randn({8, 50});

    auto outputs = model->forward(
        {.continuous = continuous, .species_vector = to_sparse_rows(species_vector)});

    REQUIRE(outputs.count("biomass") > 0);
    REQUIRE(outputs["biomass"].size(0) == 8);
}

TEST_CASE("ResolveModel get latent", "[model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = false;
    schema.has_taxonomy = false;
    schema.n_genera = 0;
    schema.n_families = 0;
    schema.dataset_config.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);

    // n_continuous = 2 (coords) + 32 (hash) = 34
    auto continuous = torch::randn({8, 34});
    auto latent = model->get_latent({.continuous = continuous});

    REQUIRE(latent.size(0) == 8);
    REQUIRE(latent.size(1) == 32);  // Last hidden dim
}
