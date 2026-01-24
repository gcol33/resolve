#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/model.hpp"

using namespace resolve;

TEST_CASE("PlotEncoder forward without taxonomy", "[encoder]") {
    PlotEncoder encoder(10, 0, 0, 8, 8, 3, std::vector<int64_t>{32, 16}, 0.3f);

    auto continuous = torch::randn({4, 10});  // batch of 4, 10 features
    auto output = encoder->forward(continuous);

    REQUIRE(output.size(0) == 4);
    REQUIRE(output.size(1) == 16);  // Last hidden dim
}

TEST_CASE("PlotEncoder forward with taxonomy", "[encoder]") {
    PlotEncoder encoder(10, 100, 50, 8, 8, 3, std::vector<int64_t>{32, 16}, 0.3f);

    auto continuous = torch::randn({4, 10});
    auto genus_ids = torch::randint(0, 100, {4, 3});  // batch x top_k
    auto family_ids = torch::randint(0, 50, {4, 3});

    auto output = encoder->forward(continuous, genus_ids, family_ids);

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

TEST_CASE("SpaccModel full forward", "[model]") {
    SpaccSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_continuous = 34;  // 2 coords + 32 hash
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.covariate_names = {};

    // Add targets
    schema.targets.push_back({"area", TaskType::Regression, TransformType::Log1p, 0, 1.0f});
    schema.targets.push_back({"habitat", TaskType::Classification, TransformType::None, 5, 1.0f});

    ModelConfig config;
    config.hash_dim = 32;
    config.top_k = 3;
    config.hidden_dims = {64, 32};

    SpaccModel model(schema, config);

    // Create dummy input
    auto continuous = torch::randn({8, 34});  // 2 + 32
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});

    auto outputs = model->forward(continuous, genus_ids, family_ids);

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
}

TEST_CASE("SpaccModel get latent", "[model]") {
    SpaccSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_continuous = 34;
    schema.has_abundance = false;
    schema.has_taxonomy = false;
    schema.n_genera = 0;
    schema.n_families = 0;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.hidden_dims = {64, 32};

    SpaccModel model(schema, config);

    auto continuous = torch::randn({8, 34});
    auto latent = model->get_latent(continuous);

    REQUIRE(latent.size(0) == 8);
    REQUIRE(latent.size(1) == 32);  // Last hidden dim
}
