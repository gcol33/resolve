#include <catch2/catch_test_macros.hpp>
#include "resolve/enum_names.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_encoder_transformer.hpp"
#include "test_support.hpp"

using namespace resolve;
using namespace resolve::test;

namespace {

SpeciesEncoderSpec transformer_spec(int64_t n_continuous, int64_t n_species, int64_t n_genera,
                                    int64_t n_families, int d_model, int n_heads = 4,
                                    int n_attention_layers = 0, int ff_dim = 256,
                                    TransformerPooling pooling = TransformerPooling::Attention) {
    auto spec = encoder_spec(SpeciesEncodingMode::Transformer, n_continuous, n_species,
                             n_genera, n_families, {1024, 512});
    spec.config.d_model = d_model;
    spec.config.n_heads = n_heads;
    spec.config.n_attention_layers = n_attention_layers;
    spec.config.transformer_ff_dim = ff_dim;
    spec.config.transformer_pooling = pooling;
    return spec;
}

}  // namespace

// ============================================================================
// PlotEncoderTransformer Tests
// ============================================================================

TEST_CASE("PlotEncoderTransformer attention pooling no self-attention", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(10, 100, 0, 0, 64));

    encoder->eval();
    auto out = encoder->forward({.continuous = torch::randn({4, 10}),
                                 .species_ids = torch::randint(1, 100, {4, 15}, torch::kInt64)});

    REQUIRE(out.size(0) == 4);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(out.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderTransformer attention pooling with self-attention", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(10, 100, 30, 10, 64, 4, 2, 128));

    encoder->eval();
    auto out = encoder->forward({.continuous = torch::randn({4, 10}),
                                 .species_ids = torch::randint(1, 100, {4, 15}, torch::kInt64),
                                 .pool_genus_ids = torch::randint(1, 30, {4, 15}, torch::kInt64),
                                 .pool_family_ids = torch::randint(1, 10, {4, 15}, torch::kInt64),
                                 .pool_weights = torch::rand({4, 15})});

    REQUIRE(out.size(0) == 4);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(encoder->has_taxonomy());
}

TEST_CASE("PlotEncoderTransformer CLS pooling", "[transformer]") {
    PlotEncoderTransformer encoder(
        transformer_spec(5, 50, 0, 0, 32, 2, 1, 64, TransformerPooling::Cls));

    encoder->eval();
    auto out = encoder->forward({.continuous = torch::randn({2, 5}),
                                 .species_ids = torch::randint(1, 50, {2, 10}, torch::kInt64)});

    REQUIRE(out.size(0) == 2);
    REQUIRE(out.size(1) == encoder->latent_dim());
    REQUIRE(out.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderTransformer forward_tokens returns pre-pooling tokens", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(5, 50, 0, 0, 32, 2, 1));

    auto species_ids = torch::randint(1, 50, {4, 10}, torch::kInt64);
    auto mask = (species_ids != 0);

    encoder->eval();
    auto tokens = encoder->forward_tokens(species_ids, {}, {}, mask.to(torch::kFloat32), mask);

    REQUIRE(tokens.size(0) == 4);
    REQUIRE(tokens.size(1) == 10);
    REQUIRE(tokens.size(2) == 32);  // d_model
}

TEST_CASE("PlotEncoderTransformer guards a plot without species", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(3, 20, 0, 0, 16, 2, 1, 32));
    encoder->eval();
    auto species_ids = torch::tensor({{3, 5, 0}, {0, 0, 0}}, torch::kInt64);
    auto out = encoder->forward({.continuous = torch::randn({2, 3}), .species_ids = species_ids});
    REQUIRE(out.isfinite().all().item<bool>());
    auto tokens = encoder->forward_tokens(species_ids, {}, {}, {}, species_ids != 0);
    REQUIRE(tokens.isfinite().all().item<bool>());
}

TEST_CASE("PlotEncoderTransformer gradient flow", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(5, 50, 0, 0, 32, 2, 0));

    auto continuous = torch::randn({4, 5}, torch::requires_grad());

    encoder->train();
    auto out = encoder->forward({.continuous = continuous,
                                 .species_ids = torch::randint(1, 50, {4, 10}, torch::kInt64)});
    out.sum().backward();

    REQUIRE(continuous.grad().defined());
    REQUIRE(continuous.grad().abs().sum().item<float>() > 0);
}

TEST_CASE("PlotEncoderTransformer embedding extraction shapes", "[transformer]") {
    PlotEncoderTransformer encoder(transformer_spec(5, 100, 30, 10, 64));

    auto sp_w = encoder->species_weights();
    auto g_w = encoder->genus_weights();
    auto f_w = encoder->family_weights();

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
    schema.dataset_config.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Transformer;
    config.d_model = 64;
    config.n_heads = 4;
    config.n_attention_layers = 1;
    config.transformer_ff_dim = 128;
    config.transformer_pooling = TransformerPooling::Attention;
    config.hidden_dims = {64, 32};

    ResolveModel model(schema, config);
    model->eval();

    auto continuous = torch::randn({4, 2});
    auto species_ids = torch::randint(1, 500, {4, 20}, torch::kInt64);
    auto pool_weights = torch::rand({4, 20});
    auto pool_mask = (species_ids != 0);
    auto pool_has_cover = torch::ones({4});

    auto outputs = model->forward({.continuous = continuous,
                                   .species_ids = species_ids,
                                   .pool_weights = pool_weights,
                                   .pool_mask = pool_mask,
                                   .pool_has_cover = pool_has_cover});

    REQUIRE(outputs.count("area") == 1);
    REQUIRE(outputs["area"].size(0) == 4);
}

TEST_CASE("Transformer pooling is named attention or cls", "[transformer][pooling]") {
    CHECK(enum_from_name<TransformerPooling>("attention") == TransformerPooling::Attention);
    CHECK(enum_from_name<TransformerPooling>("cls") == TransformerPooling::Cls);
    CHECK(std::string(enum_to_name(TransformerPooling::Cls)) == "cls");
    CHECK_THROWS(enum_from_name<TransformerPooling>("mean"));
}

// CLS pooling with zero attention layers would emit a constant (the CLS
// parameter) and drop every species, so it is rejected.
TEST_CASE("Transformer CLS pooling requires >=1 attention layer", "[transformer]") {
    REQUIRE_THROWS_AS(
        PlotEncoderTransformer(transformer_spec(5, 50, 0, 0, 32, 2, 0, 64, TransformerPooling::Cls)),
        std::invalid_argument);

    // attention pooling with 0 layers stays valid.
    REQUIRE_NOTHROW(PlotEncoderTransformer(transformer_spec(5, 50, 0, 0, 32, 2, 0, 64)));
}

// With an attention layer, CLS pooling depends on the species input (a
// constant pooled vector would make every plot identical).
TEST_CASE("Transformer CLS pooling depends on species", "[transformer]") {
    PlotEncoderTransformer encoder(
        transformer_spec(4, 40, 0, 0, 16, 2, 1, 32, TransformerPooling::Cls));
    encoder->eval();

    auto continuous = torch::zeros({2, 4});
    auto out_a = encoder->forward(
        {.continuous = continuous, .species_ids = torch::randint(1, 40, {2, 6}, torch::kInt64)});
    auto out_b = encoder->forward(
        {.continuous = continuous, .species_ids = torch::randint(1, 40, {2, 6}, torch::kInt64)});
    REQUIRE(out_a.isfinite().all().item<bool>());
    REQUIRE_FALSE(torch::allclose(out_a, out_b));
}
