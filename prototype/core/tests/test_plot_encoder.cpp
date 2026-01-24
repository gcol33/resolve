#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <resolve/plot_encoder.hpp>

using namespace resolve;

// ============================================================================
// Helper functions
// ============================================================================

std::vector<PlotRecord> make_plot_data() {
    return {
        {"plot1", {{"soil_type", "clay"}, {"climate", "temperate"}}, {{"elevation", 500.0f}, {"latitude", 45.0f}}},
        {"plot2", {{"soil_type", "sand"}, {"climate", "tropical"}}, {{"elevation", 100.0f}, {"latitude", 10.0f}}},
        {"plot3", {{"soil_type", "clay"}, {"climate", "temperate"}}, {{"elevation", 800.0f}, {"latitude", 50.0f}}},
    };
}

std::vector<ObservationRecord> make_obs_data() {
    return {
        // plot1: 3 species
        {"plot1", {{"species", "sp1"}, {"genus", "GenusA"}, {"family", "FamX"}}, {{"cover", 10.0f}}},
        {"plot1", {{"species", "sp2"}, {"genus", "GenusB"}, {"family", "FamX"}}, {{"cover", 5.0f}}},
        {"plot1", {{"species", "sp3"}, {"genus", "GenusA"}, {"family", "FamY"}}, {{"cover", 2.0f}}},
        // plot2: 2 species
        {"plot2", {{"species", "sp1"}, {"genus", "GenusA"}, {"family", "FamX"}}, {{"cover", 8.0f}}},
        {"plot2", {{"species", "sp4"}, {"genus", "GenusC"}, {"family", "FamZ"}}, {{"cover", 12.0f}}},
        // plot3: 2 species
        {"plot3", {{"species", "sp2"}, {"genus", "GenusB"}, {"family", "FamX"}}, {{"cover", 7.0f}}},
        {"plot3", {{"species", "sp5"}, {"genus", "GenusD"}, {"family", "FamY"}}, {{"cover", 3.0f}}},
    };
}

// ============================================================================
// CategoryVocab Tests
// ============================================================================

TEST_CASE("CategoryVocab basic operations", "[plot_encoder]") {
    CategoryVocab vocab;

    SECTION("fit and encode") {
        std::vector<std::string> values = {"cat", "dog", "bird", "cat"};
        vocab.fit(values);

        REQUIRE(vocab.size() == 4);  // 3 unique + UNK
        REQUIRE(vocab.contains("cat"));
        REQUIRE(vocab.contains("dog"));
        REQUIRE(vocab.contains("bird"));
        REQUIRE_FALSE(vocab.contains("fish"));

        // Known values get their index
        REQUIRE(vocab.encode("cat") > 0);
        REQUIRE(vocab.encode("dog") > 0);

        // Unknown values get UNK index (0)
        REQUIRE(vocab.encode("fish") == 0);
    }
}

// ============================================================================
// StandardScaler Tests
// ============================================================================

TEST_CASE("StandardScaler basic operations", "[plot_encoder]") {
    StandardScaler scaler;

    SECTION("fit and transform") {
        std::vector<float> values = {0.0f, 10.0f, 20.0f};
        scaler.fit(values);

        REQUIRE(scaler.mean() == Approx(10.0f));
        REQUIRE(scaler.std() > 0.0f);

        // Mean should transform to ~0
        REQUIRE(scaler.transform(10.0f) == Approx(0.0f));
    }

    SECTION("constant values") {
        std::vector<float> values = {5.0f, 5.0f, 5.0f};
        scaler.fit(values);

        // Should not divide by zero
        REQUIRE(scaler.transform(5.0f) == Approx(0.0f));
    }
}

// ============================================================================
// PlotEncoder Tests
// ============================================================================

TEST_CASE("PlotEncoder numeric encoding", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_numeric("coords", {"elevation", "latitude"}, DataSource::Plot);

    auto plots = make_plot_data();
    encoder.fit(plots, {});

    REQUIRE(encoder.is_fitted());
    REQUIRE(encoder.continuous_dim() == 2);

    auto encoded = encoder.transform(plots, {}, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 1);
    REQUIRE(encoded.columns[0].name == "coords");
    REQUIRE(encoded.columns[0].values.size(0) == 3);  // 3 plots
    REQUIRE(encoded.columns[0].values.size(1) == 2);  // 2 columns

    // Check auto-scaling worked (values should be standardized)
    auto cont = encoded.continuous_features();
    REQUIRE(cont.size(0) == 3);
    REQUIRE(cont.size(1) == 2);
}

TEST_CASE("PlotEncoder raw encoding (no scaling)", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_raw("raw_coords", {"elevation"}, DataSource::Plot);

    auto plots = make_plot_data();
    encoder.fit(plots, {});

    auto encoded = encoder.transform(plots, {}, {"plot1", "plot2", "plot3"});

    // Values should be unchanged
    float plot1_elev = encoded.columns[0].values[0][0].item<float>();
    REQUIRE(plot1_elev == Approx(500.0f));
}

TEST_CASE("PlotEncoder hash encoding", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_hash("species_hash", {"species"}, /*dim=*/16, /*top_k=*/2, /*bottom_k=*/0, /*rank_by=*/"cover", DataSource::Observation);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    auto encoded = encoder.transform(plots, obs, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 1);
    REQUIRE(encoded.columns[0].name == "species_hash");
    REQUIRE(encoded.columns[0].values.size(0) == 3);   // 3 plots
    REQUIRE(encoded.columns[0].values.size(1) == 16);  // hash_dim

    REQUIRE(encoder.continuous_dim() == 16);
}

TEST_CASE("PlotEncoder embed encoding (observation-level)", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_embed("genus_embed", {"genus"}, /*dim=*/8, /*top_k=*/2, /*bottom_k=*/0, /*rank_by=*/"cover", DataSource::Observation);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    auto encoded = encoder.transform(plots, obs, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 1);
    REQUIRE(encoded.columns[0].name == "genus_embed");
    REQUIRE(encoded.columns[0].is_embedding_ids);
    REQUIRE(encoded.columns[0].values.size(0) == 3);  // 3 plots
    REQUIRE(encoded.columns[0].values.size(1) == 2);  // top_k slots

    // Check vocab size
    REQUIRE(encoder.vocab_size("genus_embed") > 0);

    // Embedding configs for model construction
    auto configs = encoder.embedding_configs();
    REQUIRE(configs.size() == 1);
    auto [name, vocab_size, dim, n_slots] = configs[0];
    REQUIRE(name == "genus_embed");
    REQUIRE(dim == 8);
    REQUIRE(n_slots == 2);
}

TEST_CASE("PlotEncoder embed encoding (plot-level)", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_embed("climate_embed", {"climate"}, /*dim=*/4, /*top_k=*/0, /*bottom_k=*/0, /*rank_by=*/"", DataSource::Plot);

    auto plots = make_plot_data();
    encoder.fit(plots, {});

    auto encoded = encoder.transform(plots, {}, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 1);
    REQUIRE(encoded.columns[0].is_embedding_ids);
    REQUIRE(encoded.columns[0].values.size(0) == 3);  // 3 plots
    REQUIRE(encoded.columns[0].values.size(1) == 1);  // 1 slot (no top_k)
}

TEST_CASE("PlotEncoder onehot encoding", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_onehot("soil_onehot", {"soil_type"}, DataSource::Plot);

    auto plots = make_plot_data();
    encoder.fit(plots, {});

    auto encoded = encoder.transform(plots, {}, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 1);
    REQUIRE(encoded.columns[0].name == "soil_onehot");
    REQUIRE_FALSE(encoded.columns[0].is_embedding_ids);

    // 2 unique soil types + UNK = 3
    REQUIRE(encoded.columns[0].values.size(1) == 3);

    // Each row should have exactly one 1.0
    for (int64_t i = 0; i < 3; ++i) {
        float sum = encoded.columns[0].values[i].sum().item<float>();
        REQUIRE(sum == Approx(1.0f));
    }
}

TEST_CASE("PlotEncoder combined encoding", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_numeric("coords", {"elevation", "latitude"}, DataSource::Plot);
    encoder.add_hash("species_hash", {"species"}, 16, 2, 0, "cover", DataSource::Observation);
    encoder.add_embed("genus_embed", {"genus"}, 8, 2, 0, "cover", DataSource::Observation);
    encoder.add_onehot("soil_onehot", {"soil_type"}, DataSource::Plot);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    auto encoded = encoder.transform(plots, obs, {"plot1", "plot2", "plot3"});

    REQUIRE(encoded.columns.size() == 4);

    // Check continuous features (numeric + hash + onehot)
    auto cont = encoded.continuous_features();
    REQUIRE(cont.size(0) == 3);
    // 2 (numeric) + 16 (hash) + 3 (onehot) = 21
    REQUIRE(cont.size(1) == 21);

    // Check embedding IDs
    auto genus_ids = encoded.embedding_ids("genus_embed");
    REQUIRE(genus_ids.size(0) == 3);
    REQUIRE(genus_ids.size(1) == 2);
}

TEST_CASE("PlotEncoder top_k and bottom_k selection", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_embed("genus_top_bottom", {"genus"}, 8, /*top_k=*/1, /*bottom_k=*/1, "cover", DataSource::Observation);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    auto encoded = encoder.transform(plots, obs, {"plot1", "plot2", "plot3"});

    // n_slots = top_k + bottom_k = 2
    REQUIRE(encoded.columns[0].values.size(1) == 2);
}

TEST_CASE("PlotEncoder serialization", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_numeric("coords", {"elevation"}, DataSource::Plot);
    encoder.add_hash("species_hash", {"species"}, 16, 2, 0, "cover", DataSource::Observation);
    encoder.add_embed("genus_embed", {"genus"}, 8, 2, 0, "cover", DataSource::Observation);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    // Save
    std::string path = "test_plot_encoder.bin";
    encoder.save(path);

    // Load
    auto loaded = PlotEncoder::load(path);

    REQUIRE(loaded.is_fitted());
    REQUIRE(loaded.specs().size() == 3);
    REQUIRE(loaded.continuous_dim() == encoder.continuous_dim());
    REQUIRE(loaded.vocab_size("genus_embed") == encoder.vocab_size("genus_embed"));

    // Transform with loaded encoder should work
    auto encoded = loaded.transform(plots, obs, {"plot1", "plot2", "plot3"});
    REQUIRE(encoded.columns.size() == 3);

    // Cleanup
    std::remove(path.c_str());
}

TEST_CASE("PlotEncoder EncodedPlotData helpers", "[plot_encoder]") {
    PlotEncoder encoder;
    encoder.add_numeric("coords", {"elevation"}, DataSource::Plot);
    encoder.add_embed("genus_embed", {"genus"}, 8, 2, 0, "cover", DataSource::Observation);
    encoder.add_embed("family_embed", {"family"}, 4, 2, 0, "cover", DataSource::Observation);

    auto plots = make_plot_data();
    auto obs = make_obs_data();
    encoder.fit(plots, obs);

    auto encoded = encoder.transform(plots, obs, {"plot1", "plot2", "plot3"});

    // embedding_specs
    auto specs = encoded.embedding_specs();
    REQUIRE(specs.size() == 2);

    // embedding_ids
    auto genus_ids = encoded.embedding_ids("genus_embed");
    REQUIRE(genus_ids.defined());

    auto family_ids = encoded.embedding_ids("family_embed");
    REQUIRE(family_ids.defined());

    // Unknown embedding should throw
    REQUIRE_THROWS(encoded.embedding_ids("nonexistent"));
}
