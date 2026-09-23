#include "resolve/species_graph.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolve {

namespace {

// Species rows processed per pass of the co-occurrence count. The pass holds a
// (block, n_species) count matrix, so this bounds its memory independently of
// how large the vocabulary is.
constexpr int64_t kCooccurrenceBlock = 512;

// The directed edges of one relation, collected as source/target pairs.
struct EdgeList {
    std::vector<int64_t> source;
    std::vector<int64_t> target;

    void add(int64_t from, int64_t to) {
        source.push_back(from);
        target.push_back(to);
    }
    [[nodiscard]] size_t size() const { return source.size(); }
};

void check_edge_budget(size_t count, const char* relation) {
    if (static_cast<int64_t>(count) > kMaxSpeciesGraphEdges) {
        throw std::runtime_error(
            std::string("build_species_graph: the ") + relation + " relation "
            "would produce " + std::to_string(count) + " edges, past the " +
            std::to_string(kMaxSpeciesGraphEdges) + " this graph holds. Switch "
            "that relation off in HeterogeneousGNNConfig, or narrow the species "
            "vocabulary.");
    }
}

// Every ordered pair of distinct species sharing a group. Code 0 is the
// reserved <UNK> species and is not a member of any group.
void add_group_pairs(const torch::Tensor& group_of_species, EdgeList& edges,
                     const char* relation) {
    if (!group_of_species.defined() || group_of_species.numel() == 0) return;
    auto groups = group_of_species.to(torch::kCPU).contiguous();
    const auto* data = groups.data_ptr<int64_t>();
    const int64_t n = groups.numel();

    std::map<int64_t, std::vector<int64_t>> members;
    for (int64_t code = 1; code < n; ++code) {
        if (data[code] > 0) members[data[code]].push_back(code);
    }

    size_t wanted = edges.size();
    for (const auto& [group, species] : members) {
        (void)group;
        wanted += species.size() * (species.size() - 1);
    }
    check_edge_budget(wanted, relation);

    for (const auto& [group, species] : members) {
        (void)group;
        for (size_t a = 0; a < species.size(); ++a) {
            for (size_t b = 0; b < species.size(); ++b) {
                if (a != b) edges.add(species[a], species[b]);
            }
        }
    }
}

// The strongest co-occurrence partners of every species, as a symmetric
// relation: a pair kept from either end is an edge, so the k-per-species cut
// does not depend on which of the two ran first.
EdgeList cooccurrence_pairs(const torch::Tensor& species_vector, int64_t k,
                            float threshold) {
    const int64_t n_plots = species_vector.size(0);
    const int64_t n_species = species_vector.size(1);
    EdgeList edges;
    if (n_plots == 0 || n_species <= 1) return edges;

    // Presence, not abundance: the relation is "recorded in the same plot".
    auto presence = (species_vector.to(torch::kCPU) > 0).to(torch::kFloat32);
    const int64_t k_eff = std::min<int64_t>(std::max<int64_t>(k, 0), n_species - 1);
    if (k_eff == 0) return edges;

    // (from, to) pairs, deduplicated through a sorted key list at the end.
    std::vector<int64_t> keys;
    for (int64_t start = 0; start < n_species; start += kCooccurrenceBlock) {
        const int64_t rows = std::min(kCooccurrenceBlock, n_species - start);
        auto block = presence.narrow(/*dim=*/1, start, rows);
        // (rows, n_species): how many plots hold both species, as a share of
        // all plots.
        auto shared = torch::matmul(block.transpose(0, 1), presence) /
                      static_cast<float>(n_plots);
        // A species neither co-occurs with itself nor with <UNK>.
        for (int64_t row = 0; row < rows; ++row) {
            shared[row][start + row] = 0.0f;
        }
        shared.select(/*dim=*/1, 0).zero_();
        if (start == 0) shared.select(/*dim=*/0, 0).zero_();

        auto [values, indices] = shared.topk(k_eff, /*dim=*/1);
        auto kept = values >= threshold;
        auto values_acc = kept.accessor<bool, 2>();
        auto index_acc = indices.accessor<int64_t, 2>();
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t from = start + row;
            if (from == 0) continue;
            for (int64_t slot = 0; slot < k_eff; ++slot) {
                if (!values_acc[row][slot]) continue;
                const int64_t to = index_acc[row][slot];
                if (to == from || to == 0) continue;
                keys.push_back(from * n_species + to);
                keys.push_back(to * n_species + from);  // symmetric
            }
        }
        check_edge_budget(keys.size(), "co-occurrence");
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    for (int64_t key : keys) {
        edges.add(key / n_species, key % n_species);
    }
    return edges;
}

}  // namespace

SpeciesGraph build_species_graph(const ResolveDataset& dataset,
                                 const HeterogeneousGNNConfig& config) {
    const auto& schema = dataset.schema();
    const int64_t n_species = schema.n_species_vocab > 0
                                  ? schema.n_species_vocab
                                  : schema.n_species;
    if (n_species <= 0) {
        throw std::runtime_error(
            "build_species_graph: the dataset carries no species vocabulary, so "
            "there are no nodes to build a species graph over.");
    }
    if (!config.use_taxonomic_edges && !config.use_cooccurrence_edges) {
        throw std::invalid_argument(
            "build_species_graph: HeterogeneousGNNConfig switches off both "
            "use_taxonomic_edges and use_cooccurrence_edges, which leaves no "
            "relation to connect the species by. Enable at least one.");
    }

    EdgeList genus_edges;
    EdgeList family_edges;
    if (config.use_taxonomic_edges) {
        const bool has_genus = dataset.species_genus_ids().defined() &&
                               dataset.species_genus_ids().numel() > 0;
        const bool has_family = dataset.species_family_ids().defined() &&
                                dataset.species_family_ids().numel() > 0;
        if (!has_genus && !has_family) {
            throw std::runtime_error(
                "build_species_graph: use_taxonomic_edges joins species by "
                "shared genus and family, but the dataset carries no taxonomy. "
                "Provide genus/family roles with DatasetConfig::use_taxonomy, or "
                "switch use_taxonomic_edges off.");
        }
        add_group_pairs(dataset.species_genus_ids(), genus_edges, "same-genus");
        add_group_pairs(dataset.species_family_ids(), family_edges, "same-family");
    }

    EdgeList cooccurrence_edges;
    if (config.use_cooccurrence_edges) {
        const auto& species_vector = dataset.species_vector();
        if (!species_vector.defined() || species_vector.numel() == 0 ||
            species_vector.dim() != 2) {
            throw std::runtime_error(
                "build_species_graph: use_cooccurrence_edges counts how often "
                "two species share a plot, which needs the per-plot species "
                "vector that only the sparse species encoding produces. Set "
                "DatasetConfig::species_encoding = sparse, or switch "
                "use_cooccurrence_edges off.");
        }
        cooccurrence_edges = cooccurrence_pairs(
            species_vector, config.k_cooccurrence, config.cooccurrence_threshold);
    }

    const size_t total = genus_edges.size() + family_edges.size() +
                         cooccurrence_edges.size();
    check_edge_budget(total, "species");

    // The encoder looks an edge's type up in an embedding table sized
    // n_edge_types, so a relation numbered past the table would index out of
    // bounds. Say which relation needs how many rather than letting the lookup
    // fail inside the forward pass.
    int64_t highest_type = -1;
    if (!genus_edges.size() && !family_edges.size() && !cooccurrence_edges.size()) {
        highest_type = -1;
    } else if (!cooccurrence_edges.size()) {
        highest_type = family_edges.size()
            ? static_cast<int64_t>(SpeciesEdgeType::SameFamily)
            : static_cast<int64_t>(SpeciesEdgeType::SameGenus);
    } else {
        highest_type = static_cast<int64_t>(SpeciesEdgeType::CoOccurrence);
    }
    if (highest_type >= config.n_edge_types) {
        throw std::invalid_argument(
            "build_species_graph: the requested relations use edge type " +
            std::to_string(highest_type) + ", but HeterogeneousGNNConfig only "
            "carries n_edge_types = " + std::to_string(config.n_edge_types) +
            ". Raise n_edge_types to at least " +
            std::to_string(highest_type + 1) + ", or switch off the relations "
            "past it (same genus is 0, same family 1, co-occurrence 2).");
    }

    std::vector<int64_t> sources;
    std::vector<int64_t> targets;
    std::vector<int64_t> types;
    sources.reserve(total);
    targets.reserve(total);
    types.reserve(total);
    auto append = [&](const EdgeList& edges, SpeciesEdgeType type) {
        for (size_t i = 0; i < edges.size(); ++i) {
            sources.push_back(edges.source[i]);
            targets.push_back(edges.target[i]);
            types.push_back(static_cast<int64_t>(type));
        }
    };
    append(genus_edges, SpeciesEdgeType::SameGenus);
    append(family_edges, SpeciesEdgeType::SameFamily);
    append(cooccurrence_edges, SpeciesEdgeType::CoOccurrence);

    SpeciesGraph graph;
    graph.n_species = n_species;
    const auto options = torch::TensorOptions().dtype(torch::kInt64);
    graph.edge_index = torch::empty({2, static_cast<int64_t>(total)}, options);
    graph.edge_type = torch::empty({static_cast<int64_t>(total)}, options);
    if (total > 0) {
        graph.edge_index.select(0, 0).copy_(torch::tensor(sources, options));
        graph.edge_index.select(0, 1).copy_(torch::tensor(targets, options));
        graph.edge_type.copy_(torch::tensor(types, options));
    }
    return graph;
}

}  // namespace resolve
