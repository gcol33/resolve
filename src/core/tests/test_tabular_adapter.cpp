// TabularAdapter: one forward per adapter architecture (FT-Transformer,
// ExcelFormer, TabNet, SAINT, GNN, heterogeneous GNN) over a hand-built schema,
// and what each architecture reads from the plot inputs.

#include <catch2/catch_test_macros.hpp>

#include "resolve/adapter.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

using namespace resolve;

namespace {

// Coordinates, an unknown-fraction column and taxonomy over 20 genera and 10
// families.
ResolveSchema taxonomy_schema() {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_species_vocab = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.dataset_config.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});
    return schema;
}

ModelConfig hash_config(EncoderArchitecture architecture) {
    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.hidden_dims = {64, 32};
    config.encoder_architecture = architecture;
    return config;
}

// Eight plots for a hash-mode adapter: the continuous block with the hash
// embedding at its tail, and valid taxonomy codes.
PlotInputs hash_inputs(const ResolveSchema& schema, const ModelConfig& config, int64_t n = 8) {
    PlotInputs inputs;
    inputs.continuous = torch::randn({n, schema.continuous_width() + config.hash_dim});
    if (schema.has_taxonomy) {
        inputs.genus_ids = torch::randint(0, schema.n_genera, {n, config.n_taxonomy_slots});
        inputs.family_ids = torch::randint(0, schema.n_families, {n, config.n_taxonomy_slots});
    }
    return inputs;
}

std::set<std::string> parameter_names(const torch::nn::Module& module) {
    std::set<std::string> names;
    for (const auto& named : module.named_parameters()) names.insert(named.key());
    return names;
}

}  // namespace

TEST_CASE("TabularAdapter FTTransformer forward", "[adapter]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::FTTransformer);
    config.ft_transformer.d_model = 64;
    config.ft_transformer.n_heads = 4;
    config.ft_transformer.n_layers = 2;

    TabularAdapter adapter(schema, config);
    auto out = adapter->forward(hash_inputs(schema, config));

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == 32);  // the last hidden dim
    REQUIRE(out.aux_losses.empty());
}

TEST_CASE("TabularAdapter ExcelFormer forward", "[adapter]") {
    auto schema = taxonomy_schema();
    schema.has_taxonomy = false;
    schema.dataset_config.track_unknown_fraction = false;
    auto config = hash_config(EncoderArchitecture::ExcelFormer);
    config.excelformer.d_model = 64;
    config.excelformer.n_heads = 4;
    config.excelformer.n_layers = 2;

    TabularAdapter adapter(schema, config);
    auto out = adapter->forward(hash_inputs(schema, config));

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == 32);
}

TEST_CASE("TabularAdapter TabNet forward", "[adapter][tabnet]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::TabNet);
    config.tabnet.n_steps = 3;
    config.tabnet.n_d = 16;
    config.tabnet.n_a = 16;

    TabularAdapter adapter(schema, config);
    auto out = adapter->forward(hash_inputs(schema, config));

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == 16);  // n_d
}

TEST_CASE("TabNet reads the taxonomy slots through per-slot embeddings", "[adapter][tabnet]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::TabNet);
    config.tabnet.n_steps = 2;
    config.tabnet.n_d = 8;
    config.tabnet.n_a = 8;

    torch::manual_seed(31);
    TabularAdapter adapter(schema, config);
    adapter->eval();

    const auto names = parameter_names(*adapter);
    for (int k = 0; k < 3; ++k) {
        CHECK(names.count("genus_emb_" + std::to_string(k) + ".weight") == 1);
        CHECK(names.count("family_emb_" + std::to_string(k) + ".weight") == 1);
    }

    auto inputs = hash_inputs(schema, config);
    auto other_taxonomy = inputs;
    other_taxonomy.genus_ids = (inputs.genus_ids + 7).remainder(schema.n_genera);
    torch::NoGradGuard no_grad;
    CHECK_FALSE(torch::allclose(adapter->forward(inputs).latent,
                                adapter->forward(other_taxonomy).latent, 1e-5, 1e-6));

    SECTION("a forward missing a taxonomy tensor is refused") {
        auto without = inputs;
        without.family_ids = torch::Tensor();
        CHECK_THROWS(adapter->forward(without));
    }
}

TEST_CASE("TabNet reports its sparsity term weighted by its coefficient", "[adapter][tabnet]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::TabNet);
    config.tabnet.n_steps = 2;
    config.tabnet.n_d = 8;
    config.tabnet.n_a = 8;

    auto inputs = hash_inputs(schema, config);
    auto term_at = [&](float coefficient) {
        auto cfg = config;
        cfg.tabnet.sparsity_coefficient = coefficient;
        torch::manual_seed(32);
        TabularAdapter adapter(schema, cfg);
        auto out = adapter->forward(inputs);
        REQUIRE(out.aux_losses.count(kTabNetSparsityLoss) == 1);
        return out.aux_losses.at(kTabNetSparsityLoss);
    };

    auto small = term_at(0.01f);
    auto large = term_at(0.1f);
    REQUIRE(small.dim() == 0);
    CHECK(small.item<float>() > 0.0f);
    CHECK(std::abs(large.item<float>() - 10.0f * small.item<float>()) <
          1e-5f * std::max(1.0f, large.item<float>()));
    CHECK(term_at(0.0f).item<float>() == 0.0f);

    SECTION("the term carries gradient into the attentive transformer") {
        torch::manual_seed(32);
        TabularAdapter adapter(schema, config);
        adapter->forward(inputs).aux_losses.at(kTabNetSparsityLoss).backward();
        bool reached = false;
        for (const auto& named : adapter->named_parameters()) {
            if (named.key().find("attention_fc") == std::string::npos) continue;
            if (named.value().grad().defined() &&
                named.value().grad().abs().sum().item<float>() > 0.0f) {
                reached = true;
            }
        }
        CHECK(reached);
    }
}

TEST_CASE("TabularAdapter SAINT forward", "[adapter][saint]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::SAINT);
    config.saint.d_model = 64;
    config.saint.n_heads = 4;
    config.saint.n_layers = 2;
    config.saint.row_attention_rows = 4;

    TabularAdapter adapter(schema, config);
    REQUIRE(adapter->row_reference_rows() == 3);
    auto out = adapter->forward(hash_inputs(schema, config));

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == 32);

    SECTION("without row attention there is no reference set") {
        auto cfg = config;
        cfg.saint.use_row_attention = false;
        CHECK(TabularAdapter(schema, cfg)->row_reference_rows() == 0);
    }
}

TEST_CASE("TabularAdapter GNN forward", "[adapter][gnn]") {
    auto schema = taxonomy_schema();
    auto config = hash_config(EncoderArchitecture::GNN);
    config.gnn.hidden_dim = 64;
    config.gnn.n_layers = 2;
    config.gnn.k_neighbors = 4;

    TabularAdapter adapter(schema, config);
    auto out = adapter->forward(hash_inputs(schema, config));

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == 64);
    CHECK(parameter_names(*adapter).count("genus_emb_0.weight") == 1);
}

TEST_CASE("TabularAdapter HeterogeneousGNN forward", "[adapter][hetero_gnn]") {
    auto schema = taxonomy_schema();

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.uses_explicit_vector = true;
    config.species_embed_dim = 16;
    config.n_taxonomy_slots = 3;
    config.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
    config.heterogeneous_gnn.hidden_dim = 64;
    config.heterogeneous_gnn.n_layers = 2;

    TabularAdapter adapter(schema, config);
    REQUIRE_FALSE(adapter->has_species_graph());
    REQUIRE(adapter->data_buffers().size() == 2);

    // A simple species graph: 50 species, some edges of type 0.
    adapter->set_species_graph(
        torch::tensor({{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 0}}, torch::kInt64),
        torch::zeros({6}, torch::kInt64));
    REQUIRE(adapter->has_species_graph());

    PlotInputs inputs;
    inputs.continuous = torch::randn({8, schema.continuous_width()});
    inputs.species_vector = to_sparse_rows(torch::rand({8, 50}));
    auto out = adapter->forward(inputs);

    REQUIRE(out.latent.size(0) == 8);
    REQUIRE(out.latent.size(1) == schema.continuous_width() + config.heterogeneous_gnn.output_dim);
}

TEST_CASE("TabularAdapter refuses the architectures ResolveModel builds itself", "[adapter]") {
    auto schema = taxonomy_schema();
    CHECK_THROWS_AS(TabularAdapter(schema, hash_config(EncoderArchitecture::MLP)),
                    std::invalid_argument);
    CHECK_THROWS_AS(TabularAdapter(schema, hash_config(EncoderArchitecture::TraitNet)),
                    std::invalid_argument);
}
