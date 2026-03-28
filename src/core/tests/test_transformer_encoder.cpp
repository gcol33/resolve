#include <catch2/catch_test_macros.hpp>
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"

using namespace resolve;

// ============================================================================
// PlotEncoderTransformer Tests
// ============================================================================

TEST_CASE("PlotEncoderTransformer attention pooling no self-attention", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/64,
        /*n_heads=*/4,
        /*n_attention_layers=*/0
    );

    auto continuous = torch::randn({4, 10});
    auto species_ids = torch::randint(1, 100, {4, 15}, torch::kInt64);

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids);

    REQUIRE(out.size(0) == 4);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(out.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderTransformer attention pooling with self-attention", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/10,
        /*n_species=*/100,
        /*n_genera=*/30,
        /*n_families=*/10,
        /*d_model=*/64,
        /*n_heads=*/4,
        /*n_attention_layers=*/2,
        /*transformer_ff_dim=*/128,
        /*transformer_pooling=*/"attention"
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

TEST_CASE("PlotEncoderTransformer CLS pooling", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/1,
        /*transformer_ff_dim=*/64,
        /*transformer_pooling=*/"cls"
    );

    auto continuous = torch::randn({2, 5});
    auto species_ids = torch::randint(1, 50, {2, 10}, torch::kInt64);

    encoder->eval();
    auto out = encoder->forward(continuous, species_ids);

    REQUIRE(out.size(0) == 2);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(out.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderTransformer forward_tokens returns pre-pooling tokens", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/1
    );

    auto species_ids = torch::randint(1, 50, {4, 10}, torch::kInt64);
    auto mask = (species_ids != 0);

    encoder->eval();
    auto tokens = encoder->forward_tokens(species_ids, {}, {}, mask.to(torch::kFloat32), mask);

    REQUIRE(tokens.size(0) == 4);
    REQUIRE(tokens.size(1) == 10);
    REQUIRE(tokens.size(2) == 32);  // d_model
}

TEST_CASE("PlotEncoderTransformer gradient flow", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/50,
        /*n_genera=*/0,
        /*n_families=*/0,
        /*d_model=*/32,
        /*n_heads=*/2,
        /*n_attention_layers=*/0
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

TEST_CASE("PlotEncoderTransformer embedding extraction shapes", "[transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/5,
        /*n_species=*/100,
        /*n_genera=*/30,
        /*n_families=*/10,
        /*d_model=*/64
    );

    auto sp_w = encoder->get_species_weights();
    auto g_w = encoder->get_genus_weights();
    auto f_w = encoder->get_family_weights();

    REQUIRE(sp_w.size(0) == 100);
    REQUIRE(sp_w.size(1) == 64);
    REQUIRE(g_w.size(0) == 30);
    REQUIRE(g_w.size(1) == 64);
    REQUIRE(f_w.size(0) == 10);
    REQUIRE(f_w.size(1) == 64);
}

TEST_CASE("ResolveModel with Transformer mode constructs and forwards", "[transformer][model]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 500;
    schema.n_species_vocab = 500;
    schema.has_coordinates = true;
    schema.has_taxonomy = false;
    schema.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Transformer;
    config.d_model = 64;
    config.n_heads = 4;
    config.n_attention_layers = 1;
    config.transformer_ff_dim = 128;
    config.transformer_pooling = "attention";
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);
    model->eval();

    auto continuous = torch::randn({4, 2});
    auto species_ids = torch::randint(1, 500, {4, 20}, torch::kInt64);
    auto pool_weights = torch::rand({4, 20});
    auto pool_mask = (species_ids != 0);
    auto pool_has_cover = torch::ones({4});

    auto outputs = model->forward(
        continuous, {}, {}, species_ids, {},
        {}, {}, pool_weights, pool_mask, pool_has_cover
    );

    REQUIRE(outputs.count("area") == 1);
    REQUIRE(outputs["area"].size(0) == 4);
}
