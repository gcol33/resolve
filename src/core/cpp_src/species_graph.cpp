#include "resolve/species_graph.hpp"
#include "resolve/sparse_rows.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolve {

namespace {

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
//
// Counted from the plots' species lists rather than a plots x species matrix:
// species a meets species b once for every plot listing both, so walking the
// plots of a (an inverted index) and the species of each of those plots counts
// every partner of a in work proportional to the records touched, and memory
// stays O(records + edges). A partner is any species sharing at least one plot
// whose share of all plots reaches `threshold`; the k with the most shared
// plots are kept, ties to the lower species code.
EdgeList cooccurrence_pairs(const torch::Tensor& species_vector, int64_t k,
                            float threshold) {
    const int64_t n_plots = species_vector.size(0);
    const int64_t n_species = species_vector.size(1);
    EdgeList edges;
    if (n_plots == 0 || n_species <= 1) return edges;
    const int64_t k_eff = std::min<int64_t>(std::max<int64_t>(k, 0), n_species - 1);
    if (k_eff == 0) return edges;

    // Presence, not abundance: the relation is "recorded in the same plot".
    // plot p lists species plot_species[plot_start[p] .. plot_start[p + 1]).
    const auto rows = species_vector.to(torch::kCPU);
    const auto crow = rows.crow_indices().contiguous();
    const auto col = rows.col_indices().contiguous();
    const auto val = rows.values().contiguous().to(torch::kFloat32);
    const auto* crow_p = crow.data_ptr<int64_t>();
    const auto* col_p = col.data_ptr<int64_t>();
    const auto* val_p = val.data_ptr<float>();
    std::vector<int64_t> plot_start(static_cast<size_t>(n_plots) + 1, 0);
    std::vector<int32_t> plot_species;
    plot_species.reserve(static_cast<size_t>(col.numel()));
    for (int64_t p = 0; p < n_plots; ++p) {
        for (int64_t i = crow_p[p]; i < crow_p[p + 1]; ++i) {
            if (val_p[i] > 0.0f && col_p[i] > 0) {
                plot_species.push_back(static_cast<int32_t>(col_p[i]));
            }
        }
        plot_start[static_cast<size_t>(p) + 1] = static_cast<int64_t>(plot_species.size());
    }

    // species a is listed in plots species_plots[species_start[a] .. species_start[a + 1]).
    std::vector<int64_t> species_start(static_cast<size_t>(n_species) + 1, 0);
    for (int32_t s : plot_species) ++species_start[static_cast<size_t>(s) + 1];
    for (size_t s = 0; s < static_cast<size_t>(n_species); ++s) {
        species_start[s + 1] += species_start[s];
    }
    std::vector<int32_t> species_plots(plot_species.size());
    {
        std::vector<int64_t> cursor(species_start.begin(), species_start.end() - 1);
        for (int64_t p = 0; p < n_plots; ++p) {
            for (int64_t i = plot_start[static_cast<size_t>(p)];
                 i < plot_start[static_cast<size_t>(p) + 1]; ++i) {
                const auto s = static_cast<size_t>(plot_species[static_cast<size_t>(i)]);
                species_plots[static_cast<size_t>(cursor[s]++)] = static_cast<int32_t>(p);
            }
        }
    }

    // Each species' kept partners, filled independently per species.
    std::vector<std::vector<int32_t>> partners(static_cast<size_t>(n_species));
    at::parallel_for(1, n_species, /*grain_size=*/64, [&](int64_t begin, int64_t end) {
        std::vector<int64_t> shared(static_cast<size_t>(n_species), 0);
        std::vector<int32_t> touched;
        std::vector<int32_t> candidates;
        for (int64_t a = begin; a < end; ++a) {
            touched.clear();
            for (int64_t j = species_start[static_cast<size_t>(a)];
                 j < species_start[static_cast<size_t>(a) + 1]; ++j) {
                const auto p = static_cast<size_t>(species_plots[static_cast<size_t>(j)]);
                for (int64_t i = plot_start[p]; i < plot_start[p + 1]; ++i) {
                    const int32_t b = plot_species[static_cast<size_t>(i)];
                    if (b == a) continue;
                    if (shared[static_cast<size_t>(b)]++ == 0) touched.push_back(b);
                }
            }
            candidates.clear();
            for (int32_t b : touched) {
                const float share = static_cast<float>(shared[static_cast<size_t>(b)]) /
                                    static_cast<float>(n_plots);
                if (share >= threshold) candidates.push_back(b);
            }
            const auto keep = std::min<size_t>(candidates.size(), static_cast<size_t>(k_eff));
            std::partial_sort(candidates.begin(),
                              candidates.begin() + static_cast<std::ptrdiff_t>(keep),
                              candidates.end(), [&](int32_t x, int32_t y) {
                                  const int64_t cx = shared[static_cast<size_t>(x)];
                                  const int64_t cy = shared[static_cast<size_t>(y)];
                                  return cx != cy ? cx > cy : x < y;
                              });
            partners[static_cast<size_t>(a)].assign(
                candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep));
            for (int32_t b : touched) shared[static_cast<size_t>(b)] = 0;
        }
    });

    // (from, to) pairs in both directions, deduplicated through a sorted key
    // list.
    std::vector<int64_t> keys;
    for (int64_t from = 1; from < n_species; ++from) {
        for (int32_t to : partners[static_cast<size_t>(from)]) {
            keys.push_back(from * n_species + to);
            keys.push_back(static_cast<int64_t>(to) * n_species + from);
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
    const int64_t n_species = dataset.schema().n_species_vocab;
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
        if (!is_sparse_rows(species_vector) || species_vector.size(0) == 0) {
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
