#include "resolve/species_encoding.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>

namespace resolve {

uint32_t murmur_hash(const std::string& key, uint32_t seed) {
    uint32_t h = seed;
    for (char c : key) {
        h ^= static_cast<uint32_t>(c);
        h *= 0x5bd1e995;
        h ^= h >> 15;
    }
    return h;
}

void hash_species(
    const std::vector<std::pair<std::string, float>>& species_abundances,
    float* embedding,
    int hash_dim
) {
    std::fill(embedding, embedding + hash_dim, 0.0f);

    for (const auto& [species, abundance] : species_abundances) {
        uint32_t h1 = murmur_hash(species, 0);
        uint32_t h2 = murmur_hash(species, 1);

        int idx = h1 % hash_dim;
        float sign = (h2 % 2 == 0) ? 1.0f : -1.0f;
        embedding[idx] += sign * abundance;
    }
}

static std::vector<std::pair<std::string, float>> select_k(
    std::vector<std::pair<std::string, float>> species,
    int k,
    bool descending
) {
    if (static_cast<int>(species.size()) <= k) {
        return species;
    }

    std::partial_sort(
        species.begin(),
        species.begin() + k,
        species.end(),
        [descending](const auto& a, const auto& b) {
            return descending ? a.second > b.second : a.second < b.second;
        }
    );

    species.resize(k);
    return species;
}

std::vector<std::pair<std::string, float>> select_top_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    return select_k(std::move(species), k, /*descending=*/true);
}

std::vector<std::pair<std::string, float>> select_bottom_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    return select_k(std::move(species), k, /*descending=*/false);
}

std::vector<std::pair<std::string, float>> apply_selection(
    std::vector<std::pair<std::string, float>> species,
    SelectionMode mode,
    int k
) {
    switch (mode) {
        case SelectionMode::Top:
            return select_top_k(std::move(species), k);
        case SelectionMode::Bottom:
            return select_bottom_k(std::move(species), k);
        case SelectionMode::TopBottom: {
            auto top = select_top_k(species, k);
            auto bottom = select_bottom_k(species, k);
            // Merge, avoiding duplicates
            for (const auto& s : bottom) {
                if (std::find_if(top.begin(), top.end(),
                        [&s](const auto& x) { return x.first == s.first; }) == top.end()) {
                    top.push_back(s);
                }
            }
            return top;
        }
        case SelectionMode::All:
        default:
            return species;
    }
}

void apply_normalization(
    std::vector<std::pair<std::string, float>>& species,
    NormalizationMode mode
) {
    if (mode == NormalizationMode::Norm) {
        float total = 0.0f;
        for (const auto& [sp, ab] : species) total += ab;
        if (total > 0) {
            for (auto& [sp, ab] : species) ab /= total;
        }
    } else if (mode == NormalizationMode::Log1p) {
        for (auto& [sp, ab] : species) ab = std::log1p(ab);
    }
}

// =============================================================================
// SpeciesVocab
// =============================================================================

SpeciesVocab SpeciesVocab::from_records(
    const std::vector<SpeciesRecord>& records, int min_count
) {
    // Count species occurrences
    std::unordered_map<std::string, int> counts;
    for (const auto& r : records) {
        counts[r.species_id]++;
    }

    // Filter by min_count and sort alphabetically for deterministic IDs
    std::vector<std::string> species;
    species.reserve(counts.size());
    for (const auto& [name, count] : counts) {
        if (count >= min_count) {
            species.push_back(name);
        }
    }
    std::sort(species.begin(), species.end());

    // Build 1-indexed mapping (0 = unknown)
    SpeciesVocab vocab;
    for (size_t i = 0; i < species.size(); ++i) {
        vocab.species_to_id_[species[i]] = static_cast<int64_t>(i + 1);
    }
    return vocab;
}

int64_t SpeciesVocab::encode(const std::string& species) const {
    auto it = species_to_id_.find(species);
    return it != species_to_id_.end() ? it->second : 0;
}

SpeciesVocab SpeciesVocab::from_map(std::unordered_map<std::string, int64_t> species_to_id) {
    SpeciesVocab vocab;
    vocab.species_to_id_ = std::move(species_to_id);
    return vocab;
}

// =============================================================================
// Shared helper: build taxonomy vocab and species-to-genus/family maps
// =============================================================================

static void build_taxonomy_maps(
    const std::vector<SpeciesRecord>& records,
    TaxonomyVocab& taxonomy_vocab,
    std::unordered_map<std::string, std::string>& species_to_genus,
    std::unordered_map<std::string, std::string>& species_to_family
) {
    // TaxonomyVocab::fit() builds genus/family maps from records directly
    taxonomy_vocab = TaxonomyVocab();
    taxonomy_vocab.fit(records);

    // Build species -> genus/family lookup (first occurrence wins)
    species_to_genus.clear();
    species_to_family.clear();
    for (const auto& r : records) {
        if (!r.genus.empty()) {
            species_to_genus.emplace(r.species_id, r.genus);
        }
        if (!r.family.empty()) {
            species_to_family.emplace(r.species_id, r.family);
        }
    }
}

// =============================================================================
// RankPoolEncoder
// =============================================================================

RankPoolEncoder::RankPoolEncoder(PoolWeighting weighting, int min_frequency)
    : weighting_(weighting), min_frequency_(min_frequency) {}

void RankPoolEncoder::fit(const std::vector<SpeciesRecord>& records) {
    species_vocab_ = SpeciesVocab::from_records(records, min_frequency_);
    build_taxonomy_maps(records, taxonomy_vocab_, species_to_genus_, species_to_family_);
    fitted_ = true;
}

void RankPoolEncoder::set_vocabs(SpeciesVocab species_vocab, TaxonomyVocab taxonomy_vocab) {
    // Why: only swap the vocabs in place — do NOT set fitted_=true here.
    // The previous behaviour silently fitted the encoder against whatever
    // (possibly empty) vocab the caller handed in, masking a missing fit()
    // upstream. The current invariant is "fit() with real records is the
    // only thing that flips fitted_". from_csv_with_schema follows that
    // ordering: fit() runs first to populate species_to_genus_/_family_
    // from the test records (transform() needs those for the genus/family
    // string lookup), then set_vocabs swaps the species + taxonomy vocab to
    // the training-set ones. A future caller that skips fit() will hit the
    // "must be fit before transform" error in transform(), which is the
    // correct failure mode for an unfit encoder.
    if (!fitted_) {
        throw std::runtime_error(
            "RankPoolEncoder::set_vocabs called before fit(); call fit() first so "
            "species_to_genus_/_family_ are populated before swapping vocabs");
    }
    species_vocab_ = std::move(species_vocab);
    taxonomy_vocab_ = std::move(taxonomy_vocab);
}

// Helper: compute a single weight for one species entry
static float compute_pool_weight(
    PoolWeighting weighting,
    float abundance,
    float total_abundance
) {
    switch (weighting) {
        case PoolWeighting::Binary:
            return 1.0f;
        case PoolWeighting::Abundance:
            return abundance;
        case PoolWeighting::Log1p:
            return std::log1p(abundance);
        case PoolWeighting::Norm:
            return (total_abundance > 0.0f) ? abundance / total_abundance : 1.0f;
        case PoolWeighting::Rank:
            // Rank weighting requires the full species list — handled externally
            return 0.0f;
    }
    return 1.0f;
}

// Helper: assign rank-based weights (1/rank, dense ranking by descending abundance)
static void assign_rank_weights(
    const std::vector<float>& abundances,
    std::vector<float>& weights
) {
    const auto n = abundances.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return abundances[a] > abundances[b];
    });

    // Dense ranking: ties share the same rank
    std::vector<int> ranks(n);
    int rank = 1;
    for (size_t j = 0; j < n; ++j) {
        if (j > 0 && abundances[order[j]] < abundances[order[j - 1]]) {
            rank = static_cast<int>(j + 1);
        }
        ranks[order[j]] = rank;
    }

    weights.resize(n);
    for (size_t i = 0; i < n; ++i) {
        weights[i] = 1.0f / static_cast<float>(ranks[i]);
    }
}

RankPoolEncodedData RankPoolEncoder::transform(
    const std::vector<SpeciesRecord>& records,
    const std::vector<std::string>& plot_ids,
    int species_cap
) const {
    if (!fitted_) {
        throw std::runtime_error("RankPoolEncoder must be fit before transform");
    }

    const int64_t n_plots = static_cast<int64_t>(plot_ids.size());
    const bool has_taxonomy = taxonomy_vocab_.n_genera() > 1;  // >1 because index 0 is <UNK>

    // Group record indices by plot_id
    std::unordered_map<std::string, std::vector<size_t>> plot_to_indices;
    for (size_t i = 0; i < records.size(); ++i) {
        plot_to_indices[records[i].plot_id].push_back(i);
    }

    // Per-plot intermediate data
    struct PlotData {
        std::vector<int64_t> sp_ids, g_ids, f_ids;
        std::vector<float> weights;
        float unknown_abd = 0.0f;
        float total_abd = 0.0f;
        bool has_abundance = false;
    };

    std::vector<PlotData> plot_data(n_plots);
    int64_t max_species = 0;

    for (int64_t pi = 0; pi < n_plots; ++pi) {
        const auto& pid = plot_ids[pi];
        auto it = plot_to_indices.find(pid);
        if (it == plot_to_indices.end()) continue;

        auto& pd = plot_data[pi];
        const auto& indices = it->second;

        // Collect raw entries for this plot
        std::vector<float> abundances;
        abundances.reserve(indices.size());

        for (size_t idx : indices) {
            const auto& r = records[idx];
            int64_t sp_id = species_vocab_.encode(r.species_id);

            auto git = species_to_genus_.find(r.species_id);
            int64_t g_id = (git != species_to_genus_.end() && has_taxonomy)
                           ? taxonomy_vocab_.encode_genus(git->second) : 0;
            auto fit_it = species_to_family_.find(r.species_id);
            int64_t f_id = (fit_it != species_to_family_.end() && has_taxonomy)
                           ? taxonomy_vocab_.encode_family(fit_it->second) : 0;

            pd.sp_ids.push_back(sp_id);
            pd.g_ids.push_back(g_id);
            pd.f_ids.push_back(f_id);
            abundances.push_back(r.abundance);

            if (r.abundance != 1.0f) pd.has_abundance = true;
            pd.total_abd += r.abundance;
            if (sp_id == 0) pd.unknown_abd += r.abundance;
        }

        // Compute weights
        if (weighting_ == PoolWeighting::Rank) {
            assign_rank_weights(abundances, pd.weights);
        } else {
            pd.weights.reserve(abundances.size());
            for (float abd : abundances) {
                pd.weights.push_back(compute_pool_weight(weighting_, abd, pd.total_abd));
            }
        }

        max_species = std::max(max_species, static_cast<int64_t>(pd.sp_ids.size()));
    }

    if (max_species == 0) max_species = 1;  // Avoid 0-width tensors

    // Resolve species_cap (mirrors DatasetConfig::pool_species_cap):
    //   0  -> no cap (use the global per-plot max we just computed).
    //   -1 -> auto p99 over the per-plot species-count distribution.
    //   >0 -> use the value as-is.
    // Then, if the resolved cap is smaller than max_species, truncate each
    // plot's per-species buffers to the first `cap` entries (matching the
    // POC's `a[:cap]` slice in `_data.py`) and shrink max_species. Print a
    // one-line summary so users see the drop in n_padding.
    int64_t resolved_cap = max_species;  // default: no cap
    if (species_cap == -1) {
        // Build a sorted copy of per-plot lengths to compute the percentile.
        // n_plots is in the millions for production datasets but this is a
        // one-shot int64 sort per dataset load (not per epoch), so the cost
        // is dwarfed by the row-scan that produced the data.
        std::vector<int64_t> lengths;
        lengths.reserve(static_cast<size_t>(n_plots));
        for (const auto& pd : plot_data) {
            lengths.push_back(static_cast<int64_t>(pd.sp_ids.size()));
        }
        if (!lengths.empty()) {
            // p99 via nth_element on the 99th-percentile index. Matches
            // numpy.percentile's "linear" default closely enough for the
            // padding-cap use case (we round down to an integer either way).
            size_t p99_idx = static_cast<size_t>(0.99 * (lengths.size() - 1));
            std::nth_element(lengths.begin(),
                             lengths.begin() + p99_idx,
                             lengths.end());
            resolved_cap = std::max<int64_t>(lengths[p99_idx], 1);
        }
    } else if (species_cap > 0) {
        resolved_cap = species_cap;
    }

    if (resolved_cap < max_species) {
        const int64_t old_max = max_species;
        for (auto& pd : plot_data) {
            if (static_cast<int64_t>(pd.sp_ids.size()) > resolved_cap) {
                pd.sp_ids.resize(static_cast<size_t>(resolved_cap));
                pd.g_ids.resize(static_cast<size_t>(resolved_cap));
                pd.f_ids.resize(static_cast<size_t>(resolved_cap));
                pd.weights.resize(static_cast<size_t>(resolved_cap));
            }
        }
        max_species = resolved_cap;
        const double saved = 1.0 - static_cast<double>(max_species) /
                                   static_cast<double>(old_max);
        std::cout << "  rank_pool: capping species at "
                  << (species_cap == -1 ? "p99=" : "cap=")
                  << resolved_cap
                  << " (max=" << old_max
                  << ", saves " << static_cast<int>(saved * 100.0 + 0.5)
                  << "% padding)" << std::endl;
    }

    // Build padded tensors
    auto sp_ids = torch::zeros({n_plots, max_species}, torch::kInt64);
    auto g_ids  = torch::zeros({n_plots, max_species}, torch::kInt64);
    auto f_ids  = torch::zeros({n_plots, max_species}, torch::kInt64);
    auto wts    = torch::zeros({n_plots, max_species}, torch::kFloat32);
    auto msk    = torch::zeros({n_plots, max_species}, torch::kBool);
    auto has_cov  = torch::zeros({n_plots}, torch::kFloat32);
    auto unk_frac = torch::zeros({n_plots}, torch::kFloat32);

    auto sp_a  = sp_ids.accessor<int64_t, 2>();
    auto g_a   = g_ids.accessor<int64_t, 2>();
    auto f_a   = f_ids.accessor<int64_t, 2>();
    auto w_a   = wts.accessor<float, 2>();
    auto m_a   = msk.accessor<bool, 2>();
    auto hc_a  = has_cov.accessor<float, 1>();
    auto uf_a  = unk_frac.accessor<float, 1>();

    for (int64_t pi = 0; pi < n_plots; ++pi) {
        const auto& pd = plot_data[pi];
        const int64_t n_sp = static_cast<int64_t>(pd.sp_ids.size());
        for (int64_t j = 0; j < n_sp; ++j) {
            sp_a[pi][j] = pd.sp_ids[j];
            g_a[pi][j]  = pd.g_ids[j];
            f_a[pi][j]  = pd.f_ids[j];
            w_a[pi][j]  = pd.weights[j];
            m_a[pi][j]  = true;
        }
        hc_a[pi] = pd.has_abundance ? 1.0f : 0.0f;
        uf_a[pi] = (pd.total_abd > 0.0f) ? pd.unknown_abd / pd.total_abd : 0.0f;
    }

    RankPoolEncodedData result;
    result.species_ids      = sp_ids;
    result.genus_ids        = g_ids;
    result.family_ids       = f_ids;
    result.weights          = wts;
    result.mask             = msk;
    result.has_cover        = has_cov;
    result.unknown_fraction = unk_frac;
    result.n_species_vocab  = species_vocab_.size();
    result.n_genera_vocab   = taxonomy_vocab_.n_genera();
    result.n_families_vocab = taxonomy_vocab_.n_families();
    return result;
}

// =============================================================================
// EmbeddingEncoder
// =============================================================================

EmbeddingEncoder::EmbeddingEncoder(int top_k_species, int top_k_taxonomy, SelectionMode selection)
    : top_k_species_(top_k_species), top_k_taxonomy_(top_k_taxonomy), selection_(selection) {}

void EmbeddingEncoder::fit(const std::vector<SpeciesRecord>& records) {
    species_vocab_ = SpeciesVocab::from_records(records, /*min_count=*/1);
    build_taxonomy_maps(records, taxonomy_vocab_, species_to_genus_, species_to_family_);
    fitted_ = true;
}

EmbeddingEncodedData EmbeddingEncoder::transform(
    const std::vector<SpeciesRecord>& records,
    const std::vector<std::string>& plot_ids
) const {
    if (!fitted_) {
        throw std::runtime_error("EmbeddingEncoder must be fit before transform");
    }

    const int64_t n_plots = static_cast<int64_t>(plot_ids.size());
    const bool has_taxonomy = taxonomy_vocab_.n_genera() > 1;

    // Group record indices by plot_id
    std::unordered_map<std::string, std::vector<size_t>> plot_to_indices;
    for (size_t i = 0; i < records.size(); ++i) {
        plot_to_indices[records[i].plot_id].push_back(i);
    }

    // Allocate output tensors (fixed size per plot)
    auto sp_ids   = torch::zeros({n_plots, static_cast<int64_t>(top_k_species_)}, torch::kInt64);
    auto g_ids    = torch::zeros({n_plots, static_cast<int64_t>(top_k_taxonomy_)}, torch::kInt64);
    auto f_ids    = torch::zeros({n_plots, static_cast<int64_t>(top_k_taxonomy_)}, torch::kInt64);
    auto unk_frac = torch::zeros({n_plots}, torch::kFloat32);

    auto sp_a = sp_ids.accessor<int64_t, 2>();
    auto g_a  = g_ids.accessor<int64_t, 2>();
    auto f_a  = f_ids.accessor<int64_t, 2>();
    auto uf_a = unk_frac.accessor<float, 1>();

    for (int64_t pi = 0; pi < n_plots; ++pi) {
        const auto& pid = plot_ids[pi];
        auto it = plot_to_indices.find(pid);
        if (it == plot_to_indices.end()) continue;

        const auto& indices = it->second;

        // Build species abundance pairs for selection
        std::vector<std::pair<std::string, float>> species_abd;
        species_abd.reserve(indices.size());
        float total_abd = 0.0f;
        float unknown_abd = 0.0f;

        for (size_t idx : indices) {
            const auto& r = records[idx];
            species_abd.emplace_back(r.species_id, r.abundance);
            total_abd += r.abundance;
            if (species_vocab_.encode(r.species_id) == 0) {
                unknown_abd += r.abundance;
            }
        }

        uf_a[pi] = (total_abd > 0.0f) ? unknown_abd / total_abd : 0.0f;

        // Select top-k species using the existing apply_selection helper
        auto selected_species = apply_selection(species_abd, selection_, top_k_species_);

        // Encode selected species IDs (pad with 0 if fewer than top_k)
        for (int j = 0; j < top_k_species_ && j < static_cast<int>(selected_species.size()); ++j) {
            sp_a[pi][j] = species_vocab_.encode(selected_species[j].first);
        }

        // Build genus/family aggregation for taxonomy top-k
        // Aggregate abundance per genus and family, then select top-k
        if (has_taxonomy) {
            std::unordered_map<std::string, float> genus_abd, family_abd;
            for (size_t idx : indices) {
                const auto& r = records[idx];
                auto git = species_to_genus_.find(r.species_id);
                if (git != species_to_genus_.end()) {
                    genus_abd[git->second] += r.abundance;
                }
                auto fit_it = species_to_family_.find(r.species_id);
                if (fit_it != species_to_family_.end()) {
                    family_abd[fit_it->second] += r.abundance;
                }
            }

            // Sort genera by abundance descending and take top-k
            std::vector<std::pair<std::string, float>> genus_list(genus_abd.begin(), genus_abd.end());
            std::sort(genus_list.begin(), genus_list.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            for (int j = 0; j < top_k_taxonomy_ && j < static_cast<int>(genus_list.size()); ++j) {
                g_a[pi][j] = taxonomy_vocab_.encode_genus(genus_list[j].first);
            }

            // Sort families by abundance descending and take top-k
            std::vector<std::pair<std::string, float>> family_list(family_abd.begin(), family_abd.end());
            std::sort(family_list.begin(), family_list.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            for (int j = 0; j < top_k_taxonomy_ && j < static_cast<int>(family_list.size()); ++j) {
                f_a[pi][j] = taxonomy_vocab_.encode_family(family_list[j].first);
            }
        }
    }

    EmbeddingEncodedData result;
    result.species_ids      = sp_ids;
    result.genus_ids        = g_ids;
    result.family_ids       = f_ids;
    result.unknown_fraction = unk_frac;
    result.n_species_vocab  = species_vocab_.size();
    result.n_genera_vocab   = taxonomy_vocab_.n_genera();
    result.n_families_vocab = taxonomy_vocab_.n_families();
    return result;
}

} // namespace resolve
