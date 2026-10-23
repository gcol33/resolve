#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/species_graph.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

// ============================================================================
// The species graph a HeterogeneousGNN passes messages on
//
// The architecture reads a typed graph over the species vocabulary, and
// HeterogeneousGNNConfig carries four fields describing how to build one:
// use_taxonomic_edges, use_cooccurrence_edges, k_cooccurrence and
// cooccurrence_threshold. Trainer::prepare_data builds the graph from them and
// the checkpoint carries it.
//
// These cases pin what each field does, what the relations are, and that a
// model trains and scores on the graph the trainer builds and the checkpoint
// carries.
// ============================================================================

namespace {

constexpr int kPlots = 12;

std::string header_csv() {
    return line_header_csv({kPlots, 9.0, 0.3, 46.0, 0.3, 150.0, 11.0, 1.0, 0.2});
}

// Two groups of plots with disjoint composition, so co-occurrence is a
// different relation from taxonomy:
//
//   genus   g_0 = {sp_0, sp_1}   g_1 = {sp_2, sp_3}   g_2 = {sp_4, sp_5}
//   family  f_0 = {sp_0 .. sp_3}                      f_1 = {sp_4, sp_5}
//   plots   0-5  hold sp_0, sp_1, sp_3               6-11 hold sp_2, sp_4, sp_5
//
// so sp_2 and sp_3 share a genus but never a plot, while sp_0 and sp_3 share a
// family and every plot they appear in.
std::string species_csv() {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    const std::vector<std::vector<int>> groups = {{0, 1, 3}, {2, 4, 5}};
    const int genus_of[6] = {0, 0, 1, 1, 2, 2};
    const int family_of[6] = {0, 0, 0, 0, 1, 1};
    for (int plot = 0; plot < kPlots; ++plot) {
        const auto& species = groups[plot < kPlots / 2 ? 0 : 1];
        for (size_t j = 0; j < species.size(); ++j) {
            const int s = species[j];
            out << "p" << plot << ",sp_" << s << "," << (1 + static_cast<int>(j))
                << ",gen_" << genus_of[s] << ",fam_" << family_of[s] << "\n";
        }
    }
    return out.str();
}

ResolveDataset build_dataset(const std::string& header, const std::string& species,
                             SpeciesEncodingMode mode = SpeciesEncodingMode::Sparse,
                             bool use_taxonomy = true) {
    DatasetConfig config;
    config.species_encoding = mode;
    config.use_taxonomy = use_taxonomy;
    return ResolveDataset::from_csv(header, species,
                                    Roles().taxonomy().coordinates().covariates({"elev"}),
                                    {TargetSpec::regression("y")}, config);
}

// Species code of a name, so a test never assumes the frequency-ranked order
// the vocabulary happens to assign.
std::map<std::string, int64_t> codes_of(const ResolveDataset& dataset) {
    std::map<std::string, int64_t> codes;
    const auto& vocab = dataset.species_vocab();
    for (size_t i = 0; i < vocab.size(); ++i) {
        codes[vocab[i]] = static_cast<int64_t>(i);
    }
    return codes;
}

std::set<std::pair<int64_t, int64_t>> edges_of_type(const SpeciesGraph& graph,
                                                    SpeciesEdgeType type) {
    std::set<std::pair<int64_t, int64_t>> edges;
    if (graph.empty()) return edges;
    auto index = graph.edge_index.contiguous();
    auto types = graph.edge_type.contiguous();
    auto index_acc = index.accessor<int64_t, 2>();
    auto type_acc = types.accessor<int64_t, 1>();
    for (int64_t e = 0; e < graph.n_edges(); ++e) {
        if (type_acc[e] == static_cast<int64_t>(type)) {
            edges.insert({index_acc[0][e], index_acc[1][e]});
        }
    }
    return edges;
}

HeterogeneousGNNConfig taxonomy_only() {
    HeterogeneousGNNConfig config;
    config.use_taxonomic_edges = true;
    config.use_cooccurrence_edges = false;
    return config;
}

HeterogeneousGNNConfig cooccurrence_only(float threshold = 0.4f, int k = 5) {
    HeterogeneousGNNConfig config;
    config.use_taxonomic_edges = false;
    config.use_cooccurrence_edges = true;
    config.cooccurrence_threshold = threshold;
    config.k_cooccurrence = k;
    return config;
}

}  // namespace

// ============================================================================
// use_taxonomic_edges
// ============================================================================

TEST_CASE("Species sharing a genus or a family are joined, by the right relation",
          "[species_graph][taxonomy]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());
    auto code = codes_of(dataset);

    auto graph = build_species_graph(dataset, taxonomy_only());
    auto genus = edges_of_type(graph, SpeciesEdgeType::SameGenus);
    auto family = edges_of_type(graph, SpeciesEdgeType::SameFamily);

    // Three genera of two species each: one edge per ordered pair.
    CHECK(genus.size() == 6);
    CHECK(genus.count({code.at("sp_0"), code.at("sp_1")}) == 1);
    CHECK(genus.count({code.at("sp_1"), code.at("sp_0")}) == 1);
    CHECK(genus.count({code.at("sp_2"), code.at("sp_3")}) == 1);
    CHECK(genus.count({code.at("sp_4"), code.at("sp_5")}) == 1);
    // Different genera are not joined by this relation.
    CHECK(genus.count({code.at("sp_0"), code.at("sp_2")}) == 0);
    CHECK(genus.count({code.at("sp_1"), code.at("sp_4")}) == 0);

    // One family of four species (12 ordered pairs) and one of two (2).
    CHECK(family.size() == 14);
    CHECK(family.count({code.at("sp_0"), code.at("sp_3")}) == 1);
    CHECK(family.count({code.at("sp_2"), code.at("sp_1")}) == 1);
    CHECK(family.count({code.at("sp_0"), code.at("sp_4")}) == 0);

    // The relations are separate: a same-genus pair is not reported as a
    // same-family edge only, and both are present for it.
    CHECK(genus.count({code.at("sp_0"), code.at("sp_1")}) == 1);
    CHECK(family.count({code.at("sp_0"), code.at("sp_1")}) == 1);

    // <UNK> is a node with no edges.
    CHECK(edges_of_type(graph, SpeciesEdgeType::CoOccurrence).empty());
    auto index_acc = graph.edge_index.accessor<int64_t, 2>();
    for (int64_t e = 0; e < graph.n_edges(); ++e) {
        CHECK(index_acc[0][e] > 0);
        CHECK(index_acc[1][e] > 0);
    }
    CHECK(graph.n_species == dataset.schema().n_species_vocab);
}

// ============================================================================
// use_cooccurrence_edges, k_cooccurrence, cooccurrence_threshold
// ============================================================================

TEST_CASE("Co-occurrence joins species recorded in the same plots",
          "[species_graph][cooccurrence]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());
    auto code = codes_of(dataset);

    auto graph = build_species_graph(dataset, cooccurrence_only());
    auto shared = edges_of_type(graph, SpeciesEdgeType::CoOccurrence);

    // Two plot groups of three species: every ordered pair inside a group.
    CHECK(shared.size() == 12);
    CHECK(shared.count({code.at("sp_0"), code.at("sp_1")}) == 1);
    CHECK(shared.count({code.at("sp_3"), code.at("sp_0")}) == 1);
    CHECK(shared.count({code.at("sp_2"), code.at("sp_4")}) == 1);
    // sp_2 and sp_3 share a genus but never a plot.
    CHECK(shared.count({code.at("sp_2"), code.at("sp_3")}) == 0);
    CHECK(shared.count({code.at("sp_1"), code.at("sp_5")}) == 0);
    // This relation says nothing about taxonomy.
    CHECK(edges_of_type(graph, SpeciesEdgeType::SameGenus).empty());
    CHECK(edges_of_type(graph, SpeciesEdgeType::SameFamily).empty());
}

TEST_CASE("cooccurrence_threshold is the share of plots a pair has to share",
          "[species_graph][cooccurrence]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    // Every co-occurring pair here shares exactly half the plots.
    CHECK(build_species_graph(dataset, cooccurrence_only(0.49f)).n_edges() == 12);
    CHECK(build_species_graph(dataset, cooccurrence_only(0.51f)).n_edges() == 0);
}

TEST_CASE("k_cooccurrence caps how many partners a species keeps",
          "[species_graph][cooccurrence]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    auto all_partners = build_species_graph(dataset, cooccurrence_only(0.4f, 5));
    auto one_partner = build_species_graph(dataset, cooccurrence_only(0.4f, 1));
    CHECK(all_partners.n_edges() == 12);
    // One partner per species, symmetrized: fewer edges, and every species
    // still reaches someone.
    CHECK(one_partner.n_edges() < all_partners.n_edges());
    CHECK(one_partner.n_edges() > 0);

    std::set<int64_t> connected;
    auto index_acc = one_partner.edge_index.accessor<int64_t, 2>();
    for (int64_t e = 0; e < one_partner.n_edges(); ++e) {
        connected.insert(index_acc[0][e]);
    }
    CHECK(connected.size() == 6);

    // A cap of zero asks for no co-occurrence partners at all.
    CHECK(build_species_graph(dataset, cooccurrence_only(0.4f, 0)).n_edges() == 0);
}

namespace {

// The co-occurrence edges counted from the dense plots x species presence
// matrix: shared[a][b] = plots holding both, kept when the share of plots
// reaches the threshold and the pair shares a plot, the k partners with most
// shared plots per species (ties to the lower code), symmetrized.
std::set<std::pair<int64_t, int64_t>> dense_cooccurrence(const ResolveDataset& dataset,
                                                         float threshold, int64_t k) {
    const auto presence = (sparse_rows_to_dense(dataset.species_vector()) > 0).to(torch::kFloat32);
    const int64_t n_plots = presence.size(0);
    const int64_t n_species = presence.size(1);
    const auto shared = torch::matmul(presence.t(), presence).to(torch::kLong);
    const auto acc = shared.accessor<int64_t, 2>();
    std::set<std::pair<int64_t, int64_t>> edges;
    for (int64_t a = 1; a < n_species; ++a) {
        std::vector<int64_t> partners;
        for (int64_t b = 1; b < n_species; ++b) {
            if (b == a || acc[a][b] == 0) continue;
            const float share = static_cast<float>(acc[a][b]) / static_cast<float>(n_plots);
            if (share >= threshold) partners.push_back(b);
        }
        std::stable_sort(partners.begin(), partners.end(),
                         [&](int64_t x, int64_t y) { return acc[a][x] > acc[a][y]; });
        if (static_cast<int64_t>(partners.size()) > k) partners.resize(static_cast<size_t>(k));
        for (int64_t b : partners) {
            edges.insert({a, b});
            edges.insert({b, a});
        }
    }
    return edges;
}

}  // namespace

TEST_CASE("Co-occurrence counted from the plots' species lists equals the dense count",
          "[species_graph][cooccurrence]") {
    // 150 plots drawing from 40 species with skewed frequencies, so counts
    // range widely, ties occur and some species never meet.
    std::ostringstream species;
    species << "plot_id,sp,cover,genus,family\n";
    uint64_t state = 12345;
    auto next = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    };
    const int n_plots = 150;
    for (int p = 0; p < n_plots; ++p) {
        const int n = 1 + static_cast<int>(next() % 9);
        for (int j = 0; j < n; ++j) {
            const uint32_t r = next() % 1000;
            const int s = static_cast<int>((r * r) / 25000);  // 0..39, skewed to low codes
            species << "p" << p << ",sp_" << s << "," << (1 + next() % 5) << ",g_" << s / 4
                    << ",f_" << s / 10 << "\n";
        }
    }
    TempFile header(line_header_csv({n_plots, 9.0, 0.3, 46.0, 0.3, 150.0, 11.0, 1.0, 0.2}));
    TempFile species_file(species.str());
    auto dataset = build_dataset(header.path(), species_file.path());

    for (const auto& [threshold, k] : std::vector<std::pair<float, int>>{
             {0.0f, 3}, {0.01f, 5}, {0.05f, 2}, {0.02f, 40}, {0.2f, 4}}) {
        const auto graph = build_species_graph(dataset, cooccurrence_only(threshold, k));
        CHECK(edges_of_type(graph, SpeciesEdgeType::CoOccurrence) ==
              dense_cooccurrence(dataset, threshold, k));
    }
}

TEST_CASE("Both relations together carry one edge each", "[species_graph]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    HeterogeneousGNNConfig config;
    config.use_taxonomic_edges = true;
    config.use_cooccurrence_edges = true;
    config.cooccurrence_threshold = 0.4f;
    config.k_cooccurrence = 5;

    auto graph = build_species_graph(dataset, config);
    CHECK(edges_of_type(graph, SpeciesEdgeType::SameGenus).size() == 6);
    CHECK(edges_of_type(graph, SpeciesEdgeType::SameFamily).size() == 14);
    CHECK(edges_of_type(graph, SpeciesEdgeType::CoOccurrence).size() == 12);
    CHECK(graph.n_edges() == 32);
}

// ============================================================================
// What the builder refuses
// ============================================================================

TEST_CASE("A graph with no relation to build it from is refused",
          "[species_graph][refusal]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    HeterogeneousGNNConfig none;
    none.use_taxonomic_edges = false;
    none.use_cooccurrence_edges = false;
    CHECK_THROWS_AS(build_species_graph(dataset, none), std::invalid_argument);
}

TEST_CASE("A relation the dataset cannot supply is refused",
          "[species_graph][refusal]") {
    TempFile header(header_csv());
    TempFile species(species_csv());

    // No taxonomy loaded: the taxonomic relation has nothing to group by.
    auto no_taxonomy = build_dataset(header.path(), species.path(),
                                     SpeciesEncodingMode::Sparse,
                                     /*use_taxonomy=*/false);
    CHECK_THROWS(build_species_graph(no_taxonomy, taxonomy_only()));

    // Hash encoding produces no per-plot species vector, so co-occurrence
    // cannot be counted.
    auto hashed = build_dataset(header.path(), species.path(),
                                SpeciesEncodingMode::Hash);
    CHECK_THROWS(build_species_graph(hashed, cooccurrence_only()));
    // Its taxonomy is still there, so the other relation still builds.
    CHECK(build_species_graph(hashed, taxonomy_only()).n_edges() == 20);
}

TEST_CASE("A relation numbered past the encoder's edge-type table is refused",
          "[species_graph][refusal]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    // Co-occurrence is edge type 2, so it needs a table of at least three rows.
    auto config = cooccurrence_only();
    config.n_edge_types = 2;
    CHECK_THROWS_AS(build_species_graph(dataset, config), std::invalid_argument);
    config.n_edge_types = 3;
    CHECK_NOTHROW(build_species_graph(dataset, config));

    // Taxonomy alone needs two.
    auto taxonomy = taxonomy_only();
    taxonomy.n_edge_types = 1;
    CHECK_THROWS_AS(build_species_graph(dataset, taxonomy), std::invalid_argument);
}

// ============================================================================
// The dataset's own taxonomy of the vocabulary
// ============================================================================

TEST_CASE("The dataset reports which genus and family each species belongs to",
          "[species_graph][dataset]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());
    auto code = codes_of(dataset);

    const auto& genus = dataset.species_genus_ids();
    const auto& family = dataset.species_family_ids();
    REQUIRE(genus.defined());
    REQUIRE(family.defined());
    REQUIRE(genus.numel() == dataset.schema().n_species_vocab);
    REQUIRE(family.numel() == dataset.schema().n_species_vocab);

    auto genus_acc = genus.accessor<int64_t, 1>();
    auto family_acc = family.accessor<int64_t, 1>();
    // <UNK> belongs to nothing.
    CHECK(genus_acc[0] == 0);
    CHECK(family_acc[0] == 0);
    // Same genus, same code; different genus, different code.
    CHECK(genus_acc[code.at("sp_0")] == genus_acc[code.at("sp_1")]);
    CHECK(genus_acc[code.at("sp_0")] != genus_acc[code.at("sp_2")]);
    CHECK(genus_acc[code.at("sp_0")] > 0);
    // The family relation is coarser: sp_0 and sp_2 differ in genus, agree in
    // family.
    CHECK(family_acc[code.at("sp_0")] == family_acc[code.at("sp_2")]);
    CHECK(family_acc[code.at("sp_0")] != family_acc[code.at("sp_4")]);

    // A dataset loaded without taxonomy reports none.
    auto plain = build_dataset(header.path(), species.path(),
                               SpeciesEncodingMode::Sparse, /*use_taxonomy=*/false);
    CHECK(plain.species_genus_ids().numel() == 0);
}

// ============================================================================
// End to end: the architecture trains and scores
// ============================================================================

namespace {

ModelConfig hetero_model() {
    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.uses_explicit_vector = true;
    config.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
    config.hidden_dims = {12, 8};
    config.dropout = 0.0f;
    config.heterogeneous_gnn.hidden_dim = 8;
    config.heterogeneous_gnn.output_dim = 6;
    config.heterogeneous_gnn.n_layers = 1;
    config.heterogeneous_gnn.n_heads = 2;
    config.heterogeneous_gnn.dropout = 0.0f;
    config.heterogeneous_gnn.cooccurrence_threshold = 0.4f;
    config.heterogeneous_gnn.k_cooccurrence = 5;
    return config;
}

}  // namespace

TEST_CASE("A HeterogeneousGNN trains on the graph the trainer builds",
          "[species_graph][trainer]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    torch::manual_seed(3);
    ResolveModel model(dataset.schema(), hetero_model());
    REQUIRE(model->requires_species_graph());
    // Nothing has built one yet: prepare_data is where the graph comes from.
    REQUIRE_FALSE(model->has_species_graph());

    TrainConfig train;
    train.batch_size = 8;
    train.max_epochs = 2;
    train.patience = 2;
    train.lr = 1e-3f;
    Trainer trainer(model, train);
    trainer.prepare_data(dataset, /*test_size=*/0.25f, /*seed=*/0);

    // prepare_data is the one place with both the model and the data, so it is
    // where the graph comes from.
    REQUIRE(model->has_species_graph());
    REQUIRE(model->species_graph_edge_index().size(0) == 2);
    REQUIRE(model->species_graph_edge_type().numel() ==
            model->species_graph_edge_index().size(1));

    auto result = trainer.fit();
    REQUIRE_FALSE(result.test_loss_history.empty());
    CHECK(std::isfinite(result.test_loss_history.back()));
}

TEST_CASE("The trained graph travels in the checkpoint",
          "[species_graph][checkpoint]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    torch::manual_seed(4);
    ResolveModel model(dataset.schema(), hetero_model());
    TrainConfig train;
    train.batch_size = 8;
    train.max_epochs = 1;
    train.patience = 1;
    Trainer trainer(model, train);
    trainer.prepare_data(dataset, /*test_size=*/0.25f, /*seed=*/0);

    auto expected_index = model->species_graph_edge_index().clone();
    auto expected_type = model->species_graph_edge_type().clone();
    REQUIRE(expected_index.numel() > 0);

    const TempPath ckpt(".pt");
    const std::string path = ckpt.path();
    trainer.save(path);

    // Scoring passes messages on the same graph the weights were trained on,
    // which the training data is not around to rebuild.
    auto predictor = Predictor::load(path, torch::kCPU);
    auto predictions = predictor.predict(dataset, /*return_latent=*/false,
                                         /*batch_size=*/-1);
    REQUIRE(predictions.predictions.count("y") == 1);
    CHECK(std::isfinite(predictions.predictions.at("y").sum().item<float>()));

    // And it is the same graph, edge for edge.
    ResolveModel reloaded(dataset.schema(), hetero_model());
    TrainConfig fresh;
    Trainer reader(reloaded, fresh);
    reader.load_state(path, torch::kCPU);
    REQUIRE(reloaded->has_species_graph());
    CHECK(torch::equal(reloaded->species_graph_edge_index(), expected_index));
    CHECK(torch::equal(reloaded->species_graph_edge_type(), expected_type));
}

TEST_CASE("Only the architecture that reads a species graph accepts one",
          "[species_graph][refusal]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    auto dataset = build_dataset(header.path(), species.path());

    auto config = hetero_model();
    config.encoder_architecture = EncoderArchitecture::MLP;
    config.uses_explicit_vector = true;
    ResolveModel mlp(dataset.schema(), config);
    CHECK_FALSE(mlp->requires_species_graph());
    CHECK_FALSE(mlp->has_species_graph());
    CHECK_THROWS_AS(mlp->set_species_graph(torch::zeros({2, 2}, torch::kInt64),
                                           torch::zeros({2}, torch::kInt64)),
                    std::invalid_argument);
}
