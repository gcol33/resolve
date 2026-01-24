#include <catch2/catch.hpp>

#include <resolve/model.hpp>
#include <resolve/types.hpp>

using namespace resolve;

TEST_CASE("ResolveModel construction", "[model]") {

    ModelConfig config;
    config.encoder_dim = 128;
    config.hidden_dim = 256;
    config.n_encoder_layers = 2;
    config.dropout = 0.1f;
    config.hash_dim = 32;
    config.genus_vocab_size = 100;
    config.family_vocab_size = 50;
    config.n_continuous = 10;
    config.top_k = 3;
    config.mode = SpeciesEncodingMode::Hash;

    std::vector<TargetConfig> targets = {
        {"ph", TaskType::Regression, 1, TransformType::None},
        {"nitrogen", TaskType::Regression, 1, TransformType::Log1p},
    };

    SECTION("model creates successfully") {
        ResolveModel model(config, targets);
        REQUIRE(model.ptr() != nullptr);
    }

    SECTION("model forward pass produces correct shapes") {
        ResolveModel model(config, targets);
        model->eval();

        int batch_size = 8;
        int n_continuous = config.hash_dim + config.n_continuous;  // hash embedding + continuous
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});

        auto outputs = model->forward(continuous, genus_ids, family_ids, {}, {});

        REQUIRE(outputs.count("ph") == 1);
        REQUIRE(outputs.count("nitrogen") == 1);

        REQUIRE(outputs["ph"].size(0) == batch_size);
        REQUIRE(outputs["ph"].size(1) == 1);

        REQUIRE(outputs["nitrogen"].size(0) == batch_size);
        REQUIRE(outputs["nitrogen"].size(1) == 1);
    }
}

TEST_CASE("ResolveModel with embed mode", "[model]") {

    ModelConfig config;
    config.encoder_dim = 128;
    config.hidden_dim = 256;
    config.n_encoder_layers = 2;
    config.dropout = 0.1f;
    config.species_vocab_size = 1000;
    config.genus_vocab_size = 100;
    config.family_vocab_size = 50;
    config.n_continuous = 10;
    config.top_k = 5;
    config.mode = SpeciesEncodingMode::Embed;

    std::vector<TargetConfig> targets = {
        {"carbon", TaskType::Regression, 1, TransformType::Log1p},
    };

    SECTION("embed mode forward pass") {
        ResolveModel model(config, targets);
        model->eval();

        int batch_size = 4;
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, config.n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});
        auto species_ids = torch::randint(0, config.species_vocab_size, {batch_size, config.top_k});

        auto outputs = model->forward(continuous, genus_ids, family_ids, species_ids, {});

        REQUIRE(outputs.count("carbon") == 1);
        REQUIRE(outputs["carbon"].size(0) == batch_size);
    }
}

TEST_CASE("ResolveModel with sparse mode", "[model]") {

    ModelConfig config;
    config.encoder_dim = 128;
    config.hidden_dim = 256;
    config.n_encoder_layers = 2;
    config.dropout = 0.1f;
    config.n_species_vector = 500;
    config.genus_vocab_size = 100;
    config.family_vocab_size = 50;
    config.n_continuous = 10;
    config.top_k = 3;
    config.mode = SpeciesEncodingMode::Sparse;

    std::vector<TargetConfig> targets = {
        {"clay", TaskType::Regression, 1, TransformType::None},
    };

    SECTION("sparse mode forward pass") {
        ResolveModel model(config, targets);
        model->eval();

        int batch_size = 4;
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, config.n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});
        auto species_vector = torch::rand({batch_size, config.n_species_vector});

        auto outputs = model->forward(continuous, genus_ids, family_ids, {}, species_vector);

        REQUIRE(outputs.count("clay") == 1);
        REQUIRE(outputs["clay"].size(0) == batch_size);
    }
}

TEST_CASE("ResolveModel classification task", "[model]") {

    ModelConfig config;
    config.encoder_dim = 64;
    config.hidden_dim = 128;
    config.n_encoder_layers = 1;
    config.dropout = 0.1f;
    config.hash_dim = 16;
    config.genus_vocab_size = 50;
    config.family_vocab_size = 25;
    config.n_continuous = 5;
    config.top_k = 2;
    config.mode = SpeciesEncodingMode::Hash;

    std::vector<TargetConfig> targets = {
        {"soil_type", TaskType::Classification, 5, TransformType::None},
    };

    SECTION("classification output has correct classes") {
        ResolveModel model(config, targets);
        model->eval();

        int batch_size = 4;
        int n_continuous = config.hash_dim + config.n_continuous;
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});

        auto outputs = model->forward(continuous, genus_ids, family_ids, {}, {});

        REQUIRE(outputs["soil_type"].size(0) == batch_size);
        REQUIRE(outputs["soil_type"].size(1) == 5);  // n_classes
    }
}

TEST_CASE("ResolveModel latent representation", "[model]") {

    ModelConfig config;
    config.encoder_dim = 128;
    config.hidden_dim = 256;
    config.n_encoder_layers = 2;
    config.dropout = 0.0f;
    config.hash_dim = 32;
    config.genus_vocab_size = 100;
    config.family_vocab_size = 50;
    config.n_continuous = 10;
    config.top_k = 3;
    config.mode = SpeciesEncodingMode::Hash;

    std::vector<TargetConfig> targets = {
        {"ph", TaskType::Regression, 1, TransformType::None},
    };

    SECTION("get_latent returns correct shape") {
        ResolveModel model(config, targets);
        model->eval();

        int batch_size = 4;
        int n_continuous = config.hash_dim + config.n_continuous;
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});

        auto latent = model->get_latent(continuous, genus_ids, family_ids, {}, {});

        REQUIRE(latent.size(0) == batch_size);
        REQUIRE(latent.size(1) == config.encoder_dim);
    }
}

TEST_CASE("ResolveModel serialization", "[model]") {

    ModelConfig config;
    config.encoder_dim = 64;
    config.hidden_dim = 128;
    config.n_encoder_layers = 1;
    config.dropout = 0.0f;
    config.hash_dim = 16;
    config.genus_vocab_size = 50;
    config.family_vocab_size = 25;
    config.n_continuous = 5;
    config.top_k = 2;
    config.mode = SpeciesEncodingMode::Hash;

    std::vector<TargetConfig> targets = {
        {"ph", TaskType::Regression, 1, TransformType::None},
    };

    SECTION("save and load preserves weights") {
        ResolveModel model1(config, targets);
        ResolveModel model2(config, targets);

        // Create test input
        int batch_size = 2;
        int n_continuous = config.hash_dim + config.n_continuous;
        int n_taxonomy = 2 * config.top_k;

        auto continuous = torch::randn({batch_size, n_continuous});
        auto genus_ids = torch::randint(0, config.genus_vocab_size, {batch_size, n_taxonomy});
        auto family_ids = torch::randint(0, config.family_vocab_size, {batch_size, n_taxonomy});

        // Get output from model1
        model1->eval();
        auto out1 = model1->forward(continuous, genus_ids, family_ids, {}, {});

        // Save model1
        std::stringstream ss;
        torch::serialize::OutputArchive out_archive;
        model1->save(out_archive);
        out_archive.save_to(ss);

        // Load into model2
        torch::serialize::InputArchive in_archive;
        in_archive.load_from(ss);
        model2->load(in_archive);

        // Get output from model2
        model2->eval();
        auto out2 = model2->forward(continuous, genus_ids, family_ids, {}, {});

        // Outputs should match
        REQUIRE(torch::allclose(out1["ph"], out2["ph"]));
    }
}
