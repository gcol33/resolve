#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/attention.hpp"
#include "resolve/dataset.hpp"
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"
#include "resolve/role_mapping.hpp"

#include <torch/torch.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;

// ============================================================================
// Architecture knobs that reached no code
//
// Every field of every architecture sub-config round-trips through the
// checkpoint, the C ABI, nanobind, R and `resolve info` -- the field registry
// (issue #108) makes that automatic. What the registry cannot check is whether
// the ENGINE reads the field, and several did not: a run could set one, save
// it, print it back, and train a model shaped by none of it. Same defect as
// TabNetConfig::use_sparsemax (issue #103), SelectionMode outside the hash
// encoding (issue #113) and HeterogeneousGNNConfig::n_heads.
//
// These cases pin, per knob, that it changes what the engine builds or
// computes.
// ============================================================================

namespace {

// Dropout rate per module path, so a test can say WHERE a configured rate
// landed. A rate of 0 registers no module, so a path missing from the map means
// there is no dropout there.
std::map<std::string, double> dropout_rates(const torch::nn::Module& module) {
    std::map<std::string, double> rates;
    for (const auto& item : module.named_modules()) {
        if (auto dropout =
                std::dynamic_pointer_cast<torch::nn::DropoutImpl>(item.value())) {
            rates[item.key()] = dropout->options.p();
        }
    }
    return rates;
}

bool has_module(const torch::nn::Module& module, const std::string& name) {
    for (const auto& item : module.named_modules()) {
        if (item.key() == name) return true;
    }
    return false;
}

// A schema whose continuous block is exactly the hash embedding: no
// coordinates, no covariates, no novelty columns and the Zero missing-value
// policy, so a forward takes (batch, ModelConfig::hash_dim).
ResolveSchema hash_only_schema() {
    ResolveSchema schema;
    schema.n_plots = 64;
    schema.n_species = 20;
    schema.has_coordinates = false;
    schema.has_taxonomy = false;
    schema.track_unknown_fraction = false;
    schema.track_unknown_count = false;
    schema.targets.push_back({"y", TaskType::Regression, TransformType::None, 0, 1.0f});
    return schema;
}

ModelConfig hash_only_model(int64_t hash_dim) {
    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = static_cast<int>(hash_dim);
    config.hidden_dims = {24, 16};
    config.dropout = 0.0f;  // isolate the rate under test
    return config;
}

// --- a small dataset, for the cases that need real plots -------------------

class TempFile {
public:
    explicit TempFile(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_arch_knobs_" + std::to_string(counter_++) + ".csv");
        std::ofstream file(path_);
        file << content;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

constexpr int kPlots = 24;
constexpr int kSpeciesPerPlot = 4;
constexpr int kSpecies = 8;

// Plots on a line, so spatial neighbourhood is plot index order.
std::string graph_header_csv() {
    std::ostringstream out;
    out << "plot_id,lon,lat,elev,y\n";
    for (int i = 0; i < kPlots; ++i) {
        out << "p" << i << "," << (5.0 + 0.5 * i) << "," << (45.0 + 0.5 * i)
            << "," << (200 + 7 * i) << "," << (2.0 + 0.1 * i) << "\n";
    }
    return out.str();
}

// Composition cycles with a period of three, which cuts across the spatial
// order: two plots sharing composition are usually far apart on the line.
std::string graph_species_csv() {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < kPlots; ++i) {
        for (int j = 0; j < kSpeciesPerPlot; ++j) {
            const int species = ((i % 3) * kSpeciesPerPlot + j) % kSpecies;
            out << "p" << i << ",sp_" << species << "," << (1 + ((i + j) % 4))
                << ",gen_" << (species % 4) << ",fam_" << (species % 2) << "\n";
        }
    }
    return out.str();
}

RoleMapping graph_roles() {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    roles.longitude = "lon";
    roles.latitude = "lat";
    roles.covariates = {"elev"};
    return roles;
}

ResolveDataset graph_dataset(const std::string& header, const std::string& species) {
    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.use_taxonomy = true;
    return ResolveDataset::from_csv(header, species, graph_roles(),
                                    {TargetSpec::regression("y")}, config);
}

// A GNN model over the sparse encoding, reading its graph the way `mode` says.
ModelConfig gnn_model(GraphConstructionMode mode, bool use_edge_features = false) {
    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.hidden_dims = {16, 12};
    config.dropout = 0.0f;
    config.encoder_architecture = EncoderArchitecture::GNN;
    config.gnn.gnn_type = GNNType::GCN;
    config.gnn.n_layers = 1;
    config.gnn.hidden_dim = 8;
    config.gnn.edge_dropout = 0.0f;
    config.gnn.k_neighbors = 2;
    config.gnn.graph_mode = mode;
    config.gnn.use_edge_features = use_edge_features;
    return config;
}

torch::Tensor gnn_forward(ResolveModel& model, const ResolveDataset& dataset) {
    return model->forward(dataset.continuous_block(/*include_hash=*/false),
                          dataset.genus_ids(), dataset.family_ids(),
                          dataset.species_ids(), dataset.species_vector())
        .at("y");
}

}  // namespace

// ============================================================================
// The three transformer dropout rates
// ============================================================================

TEST_CASE("A transformer block drops at the three places it is configured for",
          "[transformer][dropout]") {
    TransformerBlockConfig config;
    config.d_model = 16;
    config.n_heads = 2;
    config.attention_dropout = 0.11f;
    config.ffn_dropout = 0.22f;
    config.residual_dropout = 0.33f;

    TransformerBlock block(config);
    auto rates = dropout_rates(*block);

    REQUIRE(rates.count("attention.dropout") == 1);
    REQUIRE(rates.count("ffn.dropout") == 1);
    REQUIRE(rates.count("dropout") == 1);
    CHECK_THAT(rates.at("attention.dropout"),
               Catch::Matchers::WithinAbs(0.11, 1e-6));
    CHECK_THAT(rates.at("ffn.dropout"), Catch::Matchers::WithinAbs(0.22, 1e-6));
    CHECK_THAT(rates.at("dropout"), Catch::Matchers::WithinAbs(0.33, 1e-6));
}

TEST_CASE("One rate at all three places is the uniform spelling",
          "[transformer][dropout]") {
    auto config = TransformerBlockConfig::uniform(16, 2, /*d_ff=*/0, 0.25f);
    CHECK(config.attention_dropout == 0.25f);
    CHECK(config.ffn_dropout == 0.25f);
    CHECK(config.residual_dropout == 0.25f);
    // d_ff 0 means the standard 4 * d_model.
    CHECK(config.ffn_dim() == 64);

    TransformerBlock block(config);
    auto rates = dropout_rates(*block);
    CHECK(rates.at("attention.dropout") == rates.at("ffn.dropout"));
    CHECK(rates.at("ffn.dropout") == rates.at("dropout"));
}

TEST_CASE("FTTransformerConfig::ffn_dropout reaches the feed-forward layer",
          "[ft_transformer][dropout]") {
    auto schema = hash_only_schema();
    auto config = hash_only_model(32);
    config.encoder_architecture = EncoderArchitecture::FTTransformer;
    config.ft_transformer.d_model = 16;
    config.ft_transformer.n_heads = 2;
    config.ft_transformer.n_layers = 1;
    config.ft_transformer.attention_dropout = 0.15f;
    config.ft_transformer.ffn_dropout = 0.45f;

    ResolveModel model(schema, config);
    auto rates = dropout_rates(*model);

    const std::string ffn = "adapter.ft_transformer.encoder.layers.0.ffn.dropout";
    const std::string attention =
        "adapter.ft_transformer.encoder.layers.0.attention.dropout";
    REQUIRE(rates.count(ffn) == 1);
    REQUIRE(rates.count(attention) == 1);
    // The configured attention rate used to drive all three places, so the FFN
    // ran at 0.15 and ffn_dropout was inert.
    CHECK_THAT(rates.at(ffn), Catch::Matchers::WithinAbs(0.45, 1e-6));
    CHECK_THAT(rates.at(attention), Catch::Matchers::WithinAbs(0.15, 1e-6));
}

// ============================================================================
// ExcelFormerConfig::pre_norm
// ============================================================================

TEST_CASE("ExcelFormerConfig::pre_norm chooses the normalization placement",
          "[excelformer][pre_norm]") {
    auto schema = hash_only_schema();

    auto pre_norm_config = hash_only_model(32);
    pre_norm_config.encoder_architecture = EncoderArchitecture::ExcelFormer;
    pre_norm_config.excelformer.d_model = 16;
    pre_norm_config.excelformer.n_heads = 2;
    pre_norm_config.excelformer.n_layers = 1;
    pre_norm_config.excelformer.attention_dropout = 0.0f;
    auto post_norm_config = pre_norm_config;
    post_norm_config.excelformer.pre_norm = false;

    torch::manual_seed(11);
    ResolveModel pre_norm(schema, pre_norm_config);
    torch::manual_seed(11);
    ResolveModel post_norm(schema, post_norm_config);
    pre_norm->eval();
    post_norm->eval();

    // The trailing normalization belongs to the pre-norm arrangement only.
    CHECK(has_module(*pre_norm, "adapter.excelformer.final_norm"));
    CHECK_FALSE(has_module(*post_norm, "adapter.excelformer.final_norm"));

    // And the two compute different functions: the knob was hardcoded to
    // pre-norm, so asking for post-norm changed nothing at all.
    auto continuous = torch::randn({6, 32});
    auto pre_out = pre_norm->forward(continuous).at("y");
    auto post_out = post_norm->forward(continuous).at("y");
    REQUIRE(pre_out.sizes() == post_out.sizes());
    CHECK_FALSE(torch::allclose(pre_out, post_out, 1e-4, 1e-5));
}

// ============================================================================
// TabNetConfig::virtual_batch_size (ghost batch normalization)
// ============================================================================

TEST_CASE("Ghost batch normalization normalizes each slice on its own",
          "[tabnet][ghost_bn]") {
    const int64_t channels = 3;
    torch::nn::BatchNorm1d bn(channels);
    bn->train();

    // Two halves with very different locations. Normalizing the batch in one
    // piece leaves that difference in the output; normalizing each half against
    // its own statistics removes it, which is what ghost batch norm is for.
    torch::manual_seed(4);
    auto first = torch::randn({4, channels});
    auto second = torch::randn({4, channels}) + 10.0f;
    auto x = torch::cat({first, second}, 0);

    auto ghost = apply_ghost_batch_norm(bn, x, /*virtual_batch_size=*/4);
    auto whole = apply_ghost_batch_norm(bn, x, /*virtual_batch_size=*/0);

    REQUIRE(ghost.sizes() == x.sizes());
    CHECK(ghost.narrow(0, 0, 4).mean().abs().item<float>() < 1e-4f);
    CHECK(ghost.narrow(0, 4, 4).mean().abs().item<float>() < 1e-4f);
    // The full-batch pass keeps the two halves apart.
    CHECK(whole.narrow(0, 0, 4).mean().item<float>() < -0.5f);
    CHECK(whole.narrow(0, 4, 4).mean().item<float>() > 0.5f);
}

TEST_CASE("A ghost size that cannot split the batch is plain batch normalization",
          "[tabnet][ghost_bn]") {
    torch::nn::BatchNorm1d bn(3);
    bn->train();
    auto x = torch::randn({8, 3});
    auto plain = bn->forward(x);

    CHECK(torch::allclose(apply_ghost_batch_norm(bn, x, 0), plain));
    CHECK(torch::allclose(apply_ghost_batch_norm(bn, x, 8), plain));
    CHECK(torch::allclose(apply_ghost_batch_norm(bn, x, 99), plain));
    // A trailing slice of one row would make BatchNorm1d throw, so it joins the
    // slice before it -- here the whole batch, in one piece.
    auto five = torch::randn({5, 3});
    CHECK(torch::allclose(apply_ghost_batch_norm(bn, five, 4), bn->forward(five)));
    // Nine rows at four is 4 + 5, not 4 + 4 + 1.
    auto nine = torch::randn({9, 3});
    REQUIRE_NOTHROW(apply_ghost_batch_norm(bn, nine, 4));
    CHECK(apply_ghost_batch_norm(bn, nine, 4).size(0) == 9);
}

TEST_CASE("Eval mode ignores the ghost size", "[tabnet][ghost_bn]") {
    torch::nn::BatchNorm1d bn(3);
    bn->train();
    bn->forward(torch::randn({32, 3}));  // populate the running statistics
    bn->eval();

    auto x = torch::randn({8, 3});
    // Eval normalization reads the running statistics, so a slice sees exactly
    // what the whole batch sees.
    CHECK(torch::allclose(apply_ghost_batch_norm(bn, x, 2), bn->forward(x)));
}

TEST_CASE("TabNet runs its blocks at the configured ghost size",
          "[tabnet][ghost_bn]") {
    const int64_t input_dim = 6;
    auto x = torch::cat({torch::randn({8, input_dim}),
                         torch::randn({8, input_dim}) + 8.0f}, 0);

    auto build = [&](int64_t virtual_batch_size) {
        torch::manual_seed(7);
        return TabNetEncoder(input_dim, /*n_steps=*/2, /*n_d=*/8, /*n_a=*/8,
                             /*relaxation_factor=*/1.5f,
                             /*sparsity_coefficient=*/1e-3f,
                             /*use_sparsemax=*/true, virtual_batch_size);
    };

    auto full = build(0);
    auto ghost = build(4);
    CHECK(full->virtual_batch_size() == 0);
    CHECK(ghost->virtual_batch_size() == 4);

    // An eval forward reads running statistics, so the ghost size is not part
    // of the function evaluated there. Checked first, while both encoders still
    // carry their initial statistics.
    full->eval();
    ghost->eval();
    CHECK(torch::allclose(full->forward(x).first, ghost->forward(x).first,
                          1e-5, 1e-6));

    full->train();
    ghost->train();
    auto full_out = full->forward(x).first;
    auto ghost_out = ghost->forward(x).first;
    REQUIRE(full_out.sizes() == ghost_out.sizes());
    // Same weights, same input: the only difference is how the blocks
    // normalize, and it has to show.
    CHECK_FALSE(torch::allclose(full_out, ghost_out, 1e-4, 1e-5));

    // Training at a ghost size also accumulates the running statistics slice by
    // slice, so the two encoders no longer agree in eval mode either -- the
    // knob is part of what the model learns, not only of one forward.
    full->eval();
    ghost->eval();
    CHECK_FALSE(torch::allclose(full->forward(x).first, ghost->forward(x).first,
                                1e-5, 1e-6));
}

TEST_CASE("TabNetConfig::virtual_batch_size reaches the encoder the model builds",
          "[tabnet][ghost_bn]") {
    auto schema = hash_only_schema();
    auto config = hash_only_model(16);
    config.encoder_architecture = EncoderArchitecture::TabNet;
    config.tabnet.n_steps = 2;
    config.tabnet.n_d = 8;
    config.tabnet.n_a = 8;

    auto build = [&](int virtual_batch_size) {
        auto cfg = config;
        cfg.tabnet.virtual_batch_size = virtual_batch_size;
        torch::manual_seed(21);
        return ResolveModel(schema, cfg);
    };

    // Sixteen rows in two groups of eight with different locations, so slicing
    // at eight changes the normalization the blocks apply.
    auto continuous = torch::cat({torch::randn({8, 16}),
                                  torch::randn({8, 16}) + 8.0f}, 0);

    auto full = build(0);
    auto ghost = build(8);

    // Identical as long as neither has trained: the ghost size acts on the
    // batch statistics a training forward computes.
    full->eval();
    ghost->eval();
    CHECK(torch::allclose(full->forward(continuous).at("y"),
                          ghost->forward(continuous).at("y"), 1e-5, 1e-6));

    full->train();
    ghost->train();
    auto full_out = full->forward(continuous).at("y");
    auto ghost_out = ghost->forward(continuous).at("y");
    CHECK_FALSE(torch::allclose(full_out, ghost_out, 1e-4, 1e-5));
}

// ============================================================================
// GNNConfig::graph_mode and GNNConfig::use_edge_features
// ============================================================================

TEST_CASE("A weighted edge carries its similarity, an unweighted one carries 1",
          "[gnn][edge_features]") {
    // Three nodes on a line, k = 2, so every node keeps both others and the
    // graph is the same whatever the weighting -- only the weights differ.
    auto coords = torch::tensor({{0.0f, 0.0f}, {1.0f, 0.0f}, {40.0f, 0.0f}});

    auto plain = build_knn_adjacency(coords, 2, GraphMetric::Euclidean, false);
    auto weighted = build_knn_adjacency(coords, 2, GraphMetric::Euclidean, true);

    // Unweighted, a near and a far neighbour count the same.
    CHECK_THAT(plain[0][1].item<float>(),
               Catch::Matchers::WithinAbs(plain[0][2].item<float>(), 1e-6));
    // Weighted, the near neighbour outweighs the far one.
    CHECK(weighted[0][1].item<float>() > weighted[0][2].item<float>());
    CHECK(weighted[0][2].item<float>() > 0.0f);
    // Both stay symmetric and self-looped, which is what the layers assume.
    CHECK_THAT(weighted[1][0].item<float>(),
               Catch::Matchers::WithinAbs(weighted[0][1].item<float>(), 1e-6));
    CHECK(weighted[0][0].item<float>() > 0.0f);
}

TEST_CASE("The cosine metric compares composition, the Euclidean metric distance",
          "[gnn][graph_mode]") {
    // Rows 0 and 1 point the same way at different magnitudes; row 2 is
    // orthogonal but closer to row 0 in straight-line distance. k = 2 makes the
    // graph complete, so the neighbour choice is not what is under test -- the
    // weights are.
    auto features = torch::tensor({{1.0f, 0.0f}, {6.0f, 0.0f}, {0.0f, 3.0f}});

    auto cosine = build_knn_adjacency(features, 2, GraphMetric::Cosine, true);
    auto euclidean = build_knn_adjacency(features, 2, GraphMetric::Euclidean, true);

    // Same direction, so the cosine graph ties rows 0 and 1 together...
    CHECK(cosine[0][1].item<float>() > cosine[0][2].item<float>());
    // ... while the straight-line graph prefers the nearer, orthogonal row 2
    // (distance 3.16 against 5).
    CHECK(euclidean[0][2].item<float>() > euclidean[0][1].item<float>());
}

TEST_CASE("Each graph mode builds its graph from the features it names",
          "[gnn][graph_mode]") {
    TempFile header(graph_header_csv());
    TempFile species(graph_species_csv());
    auto dataset = graph_dataset(header.path(), species.path());

    auto build = [&](GraphConstructionMode mode, bool weighted = false) {
        torch::manual_seed(5);
        auto model = ResolveModel(dataset.schema(), gnn_model(mode, weighted));
        model->eval();
        return model;
    };

    auto spatial = build(GraphConstructionMode::Spatial);
    auto taxonomic = build(GraphConstructionMode::Taxonomic);
    auto cooccurrence = build(GraphConstructionMode::CoOccurrence);

    auto spatial_out = gnn_forward(spatial, dataset);
    auto taxonomic_out = gnn_forward(taxonomic, dataset);
    auto cooccurrence_out = gnn_forward(cooccurrence, dataset);

    REQUIRE(spatial_out.size(0) == kPlots);
    REQUIRE(taxonomic_out.sizes() == spatial_out.sizes());
    REQUIRE(cooccurrence_out.sizes() == spatial_out.sizes());
    REQUIRE(std::isfinite(spatial_out.sum().item<float>()));
    REQUIRE(std::isfinite(taxonomic_out.sum().item<float>()));
    REQUIRE(std::isfinite(cooccurrence_out.sum().item<float>()));

    // Identical weights and identical features: only the graph differs, and
    // every mode used to build the spatial one.
    CHECK_FALSE(torch::allclose(spatial_out, taxonomic_out, 1e-4, 1e-5));
    CHECK_FALSE(torch::allclose(spatial_out, cooccurrence_out, 1e-4, 1e-5));
    CHECK_FALSE(torch::allclose(taxonomic_out, cooccurrence_out, 1e-4, 1e-5));

    // And the weighting is a second, independent choice.
    auto weighted = build(GraphConstructionMode::Spatial, true);
    CHECK_FALSE(torch::allclose(spatial_out, gnn_forward(weighted, dataset),
                                1e-4, 1e-5));
}

TEST_CASE("A graph mode the data cannot supply is refused by name",
          "[gnn][graph_mode]") {
    // No coordinates, no taxonomy, no species vector: every mode is short of
    // what it measures, and each says which.
    auto schema = hash_only_schema();
    auto continuous = torch::randn({8, 16});

    auto model_for = [&](GraphConstructionMode mode) {
        auto config = hash_only_model(16);
        config.encoder_architecture = EncoderArchitecture::GNN;
        config.gnn.gnn_type = GNNType::GCN;
        config.gnn.n_layers = 1;
        config.gnn.hidden_dim = 8;
        config.gnn.graph_mode = mode;
        return ResolveModel(schema, config);
    };

    auto spatial = model_for(GraphConstructionMode::Spatial);
    auto taxonomic = model_for(GraphConstructionMode::Taxonomic);
    auto cooccurrence = model_for(GraphConstructionMode::CoOccurrence);

    CHECK_THROWS(spatial->forward(continuous));
    CHECK_THROWS(taxonomic->forward(continuous));
    CHECK_THROWS(cooccurrence->forward(continuous));
}

// ============================================================================
// TraitNetConfig: interaction_dim, interaction, shared_trait_encoder
// ============================================================================

namespace {

// TraitNet reads the continuous block as its environment, so this schema
// carries coordinates and one covariate: three columns.
ResolveSchema trait_schema(int64_t n_species) {
    ResolveSchema schema;
    schema.n_plots = 32;
    schema.n_species = n_species;
    schema.has_coordinates = true;
    schema.covariate_names = {"elev"};
    schema.has_taxonomy = false;
    schema.track_unknown_fraction = false;
    schema.track_unknown_count = false;
    schema.targets.push_back({"y", TaskType::Regression, TransformType::None, 0, 1.0f});
    return schema;
}

ModelConfig trait_model(int64_t trait_dim, int64_t interaction_dim,
                        TraitInteractionMode mode = TraitInteractionMode::Bilinear,
                        bool shared = true) {
    ModelConfig config;
    config.encoder_architecture = EncoderArchitecture::TraitNet;
    config.dropout = 0.0f;
    config.hidden_dims = {16, 8};
    config.trait_net.env_dim = 12;  // the hidden width of both encoders
    config.trait_net.trait_dim = static_cast<int>(trait_dim);
    config.trait_net.interaction_dim = static_cast<int>(interaction_dim);
    config.trait_net.interaction = mode;
    config.trait_net.shared_trait_encoder = shared;
    return config;
}

std::map<std::string, std::vector<int64_t>> parameter_shapes(
        const torch::nn::Module& module) {
    std::map<std::string, std::vector<int64_t>> shapes;
    for (const auto& item : module.named_parameters()) {
        shapes[item.key()] = item.value().sizes().vec();
    }
    return shapes;
}

}  // namespace

TEST_CASE("TraitNetConfig::interaction_dim is the width the head reads",
          "[traitnet][interaction]") {
    const int64_t n_species = 5;
    const int64_t trait_dim = 4;
    auto schema = trait_schema(n_species);

    torch::manual_seed(31);
    ResolveModel wide(schema, trait_model(trait_dim, /*interaction_dim=*/24));
    auto shapes = parameter_shapes(*wide);
    // The interaction produces interaction_dim, which the output projection
    // reads; it used to be pinned to the width of the environment encoding.
    REQUIRE(shapes.count("trait_net_encoder.output_proj.weight") == 1);
    const std::vector<int64_t> head_in{1, 24};
    const std::vector<int64_t> bilinear_shape{24, 12, 12};
    CHECK(shapes.at("trait_net_encoder.output_proj.weight") == head_in);
    CHECK(shapes.at("trait_net_encoder.interaction.weight") == bilinear_shape);

    // 0 keeps the environment encoding width, which is what the encoder did
    // for every setting before.
    ResolveModel same(schema, trait_model(trait_dim, /*interaction_dim=*/0));
    const std::vector<int64_t> head_in_default{1, 12};
    CHECK(parameter_shapes(*same).at("trait_net_encoder.output_proj.weight") ==
          head_in_default);

    // And it runs: traits in, one value per species out.
    wide->eval();
    wide->set_traits(torch::randn({n_species, trait_dim}));
    auto out = wide->forward(torch::randn({6, 3})).at("y");
    CHECK(out.size(0) == 6);
    CHECK(std::isfinite(out.sum().item<float>()));
}

TEST_CASE("Each TraitNet interaction mode is the combination it names",
          "[traitnet][interaction]") {
    const int64_t n_species = 5;
    const int64_t trait_dim = 4;
    auto schema = trait_schema(n_species);
    auto traits = torch::randn({n_species, trait_dim});
    auto env = torch::randn({6, 3});

    auto run = [&](TraitInteractionMode mode) {
        torch::manual_seed(17);
        ResolveModel model(schema, trait_model(trait_dim, 16, mode));
        model->eval();
        model->set_traits(traits);
        auto shapes = parameter_shapes(*model);
        auto out = model->forward(env).at("y");
        return std::make_pair(shapes, out);
    };

    auto [bilinear_shapes, bilinear_out] = run(TraitInteractionMode::Bilinear);
    auto [mlp_shapes, mlp_out] = run(TraitInteractionMode::MLP);
    auto [attention_shapes, attention_out] = run(TraitInteractionMode::Attention);

    // The mode picks which combination is built -- and only that one.
    CHECK(bilinear_shapes.count("trait_net_encoder.interaction.weight") == 1);
    CHECK(bilinear_shapes.count("trait_net_encoder.interaction_mlp.weight") == 0);

    CHECK(mlp_shapes.count("trait_net_encoder.interaction_mlp.weight") == 1);
    // The two representations side by side: 2 * 12 in, interaction_dim out.
    const std::vector<int64_t> mlp_shape{16, 24};
    CHECK(mlp_shapes.at("trait_net_encoder.interaction_mlp.weight") == mlp_shape);
    CHECK(mlp_shapes.count("trait_net_encoder.interaction.weight") == 0);

    CHECK(attention_shapes.count("trait_net_encoder.interaction_query.weight") == 1);
    const std::vector<int64_t> value_shape{16, 12};
    CHECK(attention_shapes.at("trait_net_encoder.interaction_value.weight") ==
          value_shape);
    CHECK(attention_shapes.count("trait_net_encoder.interaction.weight") == 0);

    // All three run, and they are different functions.
    for (const auto& out : {bilinear_out, mlp_out, attention_out}) {
        REQUIRE(out.size(0) == 6);
        CHECK(std::isfinite(out.sum().item<float>()));
    }
    CHECK_FALSE(torch::allclose(bilinear_out, mlp_out, 1e-4, 1e-5));
    CHECK_FALSE(torch::allclose(bilinear_out, attention_out, 1e-4, 1e-5));
}

TEST_CASE("Gradients reach whichever interaction the mode selected",
          "[traitnet][interaction]") {
    const int64_t n_species = 5;
    const int64_t trait_dim = 4;
    auto schema = trait_schema(n_species);

    for (auto mode : {TraitInteractionMode::Bilinear, TraitInteractionMode::MLP,
                      TraitInteractionMode::Attention}) {
        torch::manual_seed(19);
        ResolveModel model(schema, trait_model(trait_dim, 16, mode));
        model->train();
        model->set_traits(torch::randn({n_species, trait_dim}));
        auto out = model->forward(torch::randn({6, 3})).at("y");
        out.sum().backward();

        bool reached = false;
        for (const auto& item : model->named_parameters()) {
            if (item.key().find("interaction") == std::string::npos) continue;
            if (item.value().grad().defined() &&
                item.value().grad().abs().sum().item<float>() > 0.0f) {
                reached = true;
            }
        }
        INFO("interaction mode " << static_cast<int>(mode));
        CHECK(reached);
    }
}

TEST_CASE("shared_trait_encoder chooses one trait encoder or one per species",
          "[traitnet][trait_encoder]") {
    const int64_t n_species = 5;
    const int64_t trait_dim = 4;
    auto schema = trait_schema(n_species);
    auto traits = torch::randn({n_species, trait_dim});
    auto env = torch::randn({6, 3});

    torch::manual_seed(23);
    ResolveModel shared(schema, trait_model(trait_dim, 16,
                                            TraitInteractionMode::Bilinear, true));
    torch::manual_seed(23);
    ResolveModel per_species(schema, trait_model(trait_dim, 16,
                                                 TraitInteractionMode::Bilinear, false));
    shared->eval();
    per_species->eval();
    shared->set_traits(traits);
    per_species->set_traits(traits);

    auto shared_shapes = parameter_shapes(*shared);
    auto per_species_shapes = parameter_shapes(*per_species);
    // Shared: one matrix for every species.
    const std::vector<int64_t> shared_shape{12, trait_dim};
    CHECK(shared_shapes.at("trait_net_encoder.trait_encoder.0.weight") ==
          shared_shape);
    CHECK(shared_shapes.count(
              "trait_net_encoder.per_species_trait_encoder.weight_0") == 0);
    // Per species: one matrix each, which is n_species times the parameters.
    const std::vector<int64_t> per_species_shape{n_species, trait_dim, 12};
    CHECK(per_species_shapes.at(
              "trait_net_encoder.per_species_trait_encoder.weight_0") ==
          per_species_shape);
    CHECK(per_species_shapes.count("trait_net_encoder.trait_encoder.0.weight") == 0);

    auto shared_out = shared->forward(env).at("y");
    auto per_species_out = per_species->forward(env).at("y");
    REQUIRE(shared_out.sizes() == per_species_out.sizes());
    CHECK_FALSE(torch::allclose(shared_out, per_species_out, 1e-4, 1e-5));
}

TEST_CASE("A per-species trait encoder reads only the weights of its own species",
          "[traitnet][trait_encoder]") {
    const int64_t n_species = 4;
    const int64_t trait_dim = 3;
    const int64_t hidden = 6;
    PerSpeciesTraitEncoder encoder(n_species, trait_dim, hidden, /*n_layers=*/1,
                                   /*dropout=*/0.0f);
    encoder->eval();

    auto traits = torch::randn({n_species, trait_dim});
    auto before = encoder->forward(traits).clone();

    // Move one first-layer column of one species and nothing else. A uniform
    // shift across that species's whole matrix would be removed again by the
    // layer normalization that follows, so it would prove nothing.
    {
        torch::NoGradGuard no_grad;
        for (auto& item : encoder->named_parameters()) {
            if (item.key() == "weight_0") {
                item.value().select(0, 2).select(1, 0).add_(2.0f);
            }
        }
    }
    auto after = encoder->forward(traits);

    // Only that row moves, which is what "not shared" means.
    CHECK_FALSE(torch::allclose(before.select(0, 2), after.select(0, 2), 1e-4, 1e-5));
    for (int64_t s : {0, 1, 3}) {
        INFO("species " << s);
        CHECK(torch::allclose(before.select(0, s), after.select(0, s), 1e-5, 1e-6));
    }
}

// ============================================================================
// ModelConfig::parallel_layers
// ============================================================================
//
// ParallelBlock -- branches, five aggregations, the residual path -- was fully
// implemented and never constructed: nothing in the engine read
// ModelConfig::parallel_layers, so the whole architecture was unreachable while
// the configuration round-tripped through the checkpoint and `resolve info`
// reported it. ParallelBranchConfig::branch_weight was unread even inside the
// block.

namespace {

ParallelBranchConfig branch(std::vector<int64_t> hidden_dims,
                            float weight = 1.0f, float dropout = 0.0f) {
    ParallelBranchConfig config;
    config.hidden_dims = std::move(hidden_dims);
    config.branch_weight = weight;
    config.dropout = dropout;
    return config;
}

ParallelLayersConfig parallel_layers(
        std::vector<ParallelBranchConfig> branches,
        ParallelAggregation aggregation = ParallelAggregation::Sum,
        bool use_residual = false) {
    ParallelLayersConfig config;
    config.enabled = true;
    config.branches = std::move(branches);
    config.aggregation = aggregation;
    config.use_residual = use_residual;
    // Attention aggregation splits a branch width across heads, and these
    // branches are ten wide.
    config.attention_heads = 2;
    return config;
}

std::set<std::string> parameter_names(const torch::nn::Module& module) {
    std::set<std::string> names;
    for (const auto& item : module.named_parameters()) names.insert(item.key());
    return names;
}

bool any_starts_with(const std::set<std::string>& names, const std::string& prefix) {
    for (const auto& name : names) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("A parallel tail is built for every species encoding",
          "[parallel][encoder]") {
    auto schema = hash_only_schema();
    // The pooled and embed encodings index a species vocabulary.
    schema.n_species_vocab = 8;
    schema.n_genera = 3;
    schema.n_families = 2;

    for (auto encoding : {SpeciesEncodingMode::Hash, SpeciesEncodingMode::Embed,
                          SpeciesEncodingMode::Sparse, SpeciesEncodingMode::RankPool,
                          SpeciesEncodingMode::Transformer}) {
        auto config = hash_only_model(16);
        config.species_encoding = encoding;
        config.uses_explicit_vector = encoding == SpeciesEncodingMode::Sparse;
        config.species_embed_dim = 8;
        config.d_model = 16;
        config.n_heads = 2;
        config.n_attention_layers = 1;
        config.transformer_ff_dim = 16;
        config.parallel_layers = parallel_layers(
            std::vector<ParallelBranchConfig>{branch({10}), branch({10})});

        torch::manual_seed(41);
        ResolveModel model(schema, config);
        auto names = parameter_names(*model);
        INFO("species encoding " << static_cast<int>(encoding));
        // The block is the tail: its branches are there and the plain MLP is
        // not built at all.
        CHECK(any_starts_with(names, "encoder.parallel.branch_0."));
        CHECK(any_starts_with(names, "encoder.parallel.branch_1."));
        CHECK_FALSE(any_starts_with(names, "encoder.mlp."));
    }
}

TEST_CASE("The aggregation decides the width the parallel tail reports",
          "[parallel][encoder]") {
    auto schema = hash_only_schema();

    auto latent_for = [&](ParallelAggregation aggregation,
                          std::vector<ParallelBranchConfig> branches) {
        auto config = hash_only_model(16);
        config.parallel_layers = parallel_layers(std::move(branches), aggregation);
        torch::manual_seed(43);
        ResolveModel model(schema, config);
        return model->latent_dim();
    };

    // Concatenation adds the branch widths up; every other mode keeps one.
    const std::vector<ParallelBranchConfig> uneven{branch({10}), branch({6})};
    const std::vector<ParallelBranchConfig> even{branch({10}), branch({10})};
    CHECK(latent_for(ParallelAggregation::Concat, uneven) == 16);
    CHECK(latent_for(ParallelAggregation::Sum, even) == 10);
    CHECK(latent_for(ParallelAggregation::Mean, even) == 10);
    CHECK(latent_for(ParallelAggregation::Attention, even) == 10);
    CHECK(latent_for(ParallelAggregation::Gated, even) == 10);

    // The modes that combine branches element-wise need them the same width.
    CHECK_THROWS_AS(latent_for(ParallelAggregation::Sum, uneven),
                    std::invalid_argument);
}

TEST_CASE("branch_weight is what a branch contributes to the aggregation",
          "[parallel][branch_weight]") {
    const int64_t input_dim = 12;
    auto x = torch::randn({8, input_dim});

    // branch_weight is not a parameter, so the same seed gives all three blocks
    // identical weights and the only difference is the scaling.
    auto block_for = [&](float second_weight) {
        torch::manual_seed(47);
        auto block = ParallelBlock(
            input_dim,
            parallel_layers(std::vector<ParallelBranchConfig>{
                                branch({6}), branch({6}, second_weight)},
                            ParallelAggregation::Sum));
        block->eval();
        return block;
    };

    auto both = block_for(1.0f)->forward(x);
    auto first_only = block_for(0.0f)->forward(x);
    auto half = block_for(0.5f)->forward(x);

    // A weight of 0 takes the branch out of the sum; 1 leaves it whole.
    CHECK_FALSE(torch::allclose(both, first_only, 1e-4, 1e-5));
    // ... and a weight of one half is exactly half of it, which is what
    // scaling the branch means.
    CHECK(torch::allclose(half, (both + first_only) * 0.5f, 1e-5, 1e-6));
}

TEST_CASE("Only one thing can be the encoder tail", "[parallel][refusal]") {
    auto schema = hash_only_schema();

    auto config = hash_only_model(16);
    config.parallel_layers = parallel_layers(
        std::vector<ParallelBranchConfig>{branch({10})});
    REQUIRE_NOTHROW(ResolveModel(schema, config));

    // TabM, a mixture in the tail and a parallel block all replace the MLP.
    auto with_tabm = config;
    with_tabm.tabm.enabled = true;
    CHECK_THROWS_AS(ResolveModel(schema, with_tabm), std::invalid_argument);

    auto with_moe = config;
    with_moe.moe_routing = MoERoutingType::Soft;
    with_moe.moe_placement = MoEPlacement::Tail;
    with_moe.n_experts = 2;
    CHECK_THROWS_AS(ResolveModel(schema, with_moe), std::invalid_argument);

    // The mixture off the tail leaves the tail to the branches, so the two
    // coexist there.
    auto post_moe = with_moe;
    post_moe.moe_placement = MoEPlacement::Post;
    CHECK_NOTHROW(ResolveModel(schema, post_moe));

    // Enabled with nothing to run is refused rather than silently ignored.
    auto empty = config;
    empty.parallel_layers.branches.clear();
    CHECK_THROWS_AS(ResolveModel(schema, empty), std::invalid_argument);
}

TEST_CASE("A parallel tail trains and round-trips through a checkpoint",
          "[parallel][encoder]") {
    auto schema = hash_only_schema();
    auto config = hash_only_model(16);
    config.parallel_layers = parallel_layers(
        std::vector<ParallelBranchConfig>{branch({10}), branch({10})},
        ParallelAggregation::Gated, /*use_residual=*/true);

    torch::manual_seed(53);
    ResolveModel model(schema, config);
    model->train();
    auto out = model->forward(torch::randn({8, 16})).at("y");
    REQUIRE(out.size(0) == 8);
    out.sum().backward();

    bool reached = false;
    for (const auto& item : model->named_parameters()) {
        if (item.key().find("parallel.branch_1") == std::string::npos) continue;
        if (item.value().grad().defined() &&
            item.value().grad().abs().sum().item<float>() > 0.0f) {
            reached = true;
        }
    }
    CHECK(reached);

    // The configuration is what rebuilds the same architecture, so a reloaded
    // model has the same parameters, name for name.
    torch::manual_seed(59);
    ResolveModel rebuilt(schema, config);
    CHECK(parameter_names(*model) == parameter_names(*rebuilt));
}
