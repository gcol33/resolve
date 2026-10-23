// TabularAdapter: one forward per adapter architecture (FT-Transformer,
// ExcelFormer, TabNet, SAINT, GNN, heterogeneous GNN) over a hand-built schema.

#include <catch2/catch_test_macros.hpp>

#include "resolve/adapter.hpp"
#include "resolve/sparse_rows.hpp"

#include <torch/torch.h>

using namespace resolve;

TEST_CASE("TabularAdapter FTTransformer forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.covariate_names = {};
    schema.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.hidden_dims = {64, 32};
    config.encoder_architecture = EncoderArchitecture::FTTransformer;
    config.ft_transformer.d_model = 64;
    config.ft_transformer.n_heads = 4;
    config.ft_transformer.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // n_continuous = 2 (coords) + 1 (unknown_fraction) + 32 (hash) = 35
    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);  // Latent dimension depends on config
}

TEST_CASE("TabularAdapter ExcelFormer forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = false;
    schema.covariate_names = {};
    schema.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.hidden_dims = {64, 32};
    config.encoder_architecture = EncoderArchitecture::ExcelFormer;
    config.excelformer.d_model = 64;
    config.excelformer.n_heads = 4;
    config.excelformer.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // n_continuous = 2 (coords) + 32 (hash) = 34
    auto continuous = torch::randn({8, 34});

    auto out = adapter->forward(continuous);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter TabNet forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::TabNet;
    config.tabnet.n_steps = 3;
    config.tabnet.n_d = 16;
    config.tabnet.n_a = 16;

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter SAINT forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::SAINT;
    config.saint.d_model = 64;
    config.saint.n_heads = 4;
    config.saint.n_layers = 2;

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter GNN forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::GNN;
    config.gnn.hidden_dim = 64;
    config.gnn.n_layers = 2;
    config.gnn.k_neighbors = 4;  // Must be < batch_size

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter HeterogeneousGNN forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_species_vocab = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.uses_explicit_vector = true;
    config.species_embed_dim = 16;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
    config.heterogeneous_gnn.hidden_dim = 64;
    config.heterogeneous_gnn.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // Build a simple species graph: 50 species, some co-occurrence edges
    auto edge_index = torch::tensor({{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 0}}, torch::kInt64);
    auto edge_type = torch::zeros({6}, torch::kInt64);  // All type 0
    adapter->set_species_graph(edge_index, edge_type);

    // n_continuous = 2 (coords) + 1 (unknown_fraction) = 3
    auto continuous = torch::randn({8, 3});
    auto genus_ids = torch::randint(0, 20, {8, 3});   // valid ids 0..n_genera-1
    auto family_ids = torch::randint(0, 10, {8, 3});  // valid ids 0..n_families-1
    auto species_vector = to_sparse_rows(torch::rand({8, 50}));  // Required for HeterogeneousGNN

    auto out = adapter->forward(continuous, genus_ids, family_ids, {}, species_vector);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}
