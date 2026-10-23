#include "resolve/species_encoding.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace resolve {

// =============================================================================
// PlotAssemblages
// =============================================================================

namespace {

// Keeps the lexicographically smallest non-empty name seen for a species.
void keep_smallest(std::string& current, const std::string& candidate) {
    if (candidate.empty()) return;
    if (current.empty() || candidate < current) current = candidate;
}

}  // namespace

PlotAssemblages PlotAssemblages::build(std::vector<SpeciesRecord> records,
                                       const std::vector<std::string>& plot_ids,
                                       RepresentationMode representation) {
    PlotAssemblages out;
    out.plot_ids_ = plot_ids;
    out.representation_ = representation;
    const auto n_plots = static_cast<int64_t>(plot_ids.size());

    std::unordered_map<std::string_view, int64_t> plot_index;
    plot_index.reserve(plot_ids.size());
    for (int64_t p = 0; p < n_plots; ++p) {
        plot_index.emplace(out.plot_ids_[static_cast<size_t>(p)], p);
    }

    // The plot each record belongs to (-1: not one of plot_ids), and every
    // species' taxonomy resolved over all kept records.
    std::vector<int64_t> record_plot(records.size(), -1);
    std::vector<int64_t> counts(static_cast<size_t>(n_plots), 0);
    std::unordered_map<std::string, std::pair<std::string, std::string>> taxa;
    size_t n_kept = 0;
    for (size_t r = 0; r < records.size(); ++r) {
        auto it = plot_index.find(records[r].plot_id);
        if (it == plot_index.end()) continue;
        record_plot[r] = it->second;
        ++counts[static_cast<size_t>(it->second)];
        ++n_kept;
        auto& [genus, family] = taxa[records[r].species_id];
        keep_smallest(genus, records[r].genus);
        keep_smallest(family, records[r].family);
    }

    // Record indices bucketed by plot, file order kept within a plot.
    std::vector<size_t> starts(static_cast<size_t>(n_plots) + 1, 0);
    for (size_t p = 0; p < static_cast<size_t>(n_plots); ++p) {
        starts[p + 1] = starts[p] + static_cast<size_t>(counts[p]);
    }
    std::vector<size_t> by_plot(n_kept);
    {
        std::vector<size_t> cursor(starts.begin(), starts.end() - 1);
        for (size_t r = 0; r < records.size(); ++r) {
            if (record_plot[r] < 0) continue;
            by_plot[cursor[static_cast<size_t>(record_plot[r])]++] = r;
        }
    }

    // The entries are reserved up front so they never move: `position` keys
    // view their species names.
    out.entries_.reserve(n_kept);
    out.ranking_.reserve(n_kept);
    out.offsets_.reserve(static_cast<size_t>(n_plots) + 1);
    std::unordered_map<std::string_view, size_t> position;
    for (size_t p = 0; p < static_cast<size_t>(n_plots); ++p) {
        position.clear();
        for (size_t i = starts[p]; i < starts[p + 1]; ++i) {
            SpeciesRecord& record = records[by_plot[i]];
            auto found = position.find(record.species_id);
            if (found != position.end()) {
                out.ranking_[found->second] += record.abundance;
                continue;
            }
            const auto& taxon = taxa.at(record.species_id);
            out.ranking_.push_back(record.abundance);
            out.entries_.push_back(std::move(record));
            SpeciesRecord& entry = out.entries_.back();
            entry.genus = taxon.first;
            entry.family = taxon.second;
            entry.plot_id = out.plot_ids_[p];
            position.emplace(entry.species_id, out.entries_.size() - 1);
        }
        out.offsets_.push_back(static_cast<int64_t>(out.entries_.size()));
    }

    for (size_t e = 0; e < out.entries_.size(); ++e) {
        out.entries_[e].abundance =
            representation == RepresentationMode::PresenceAbsence ? 1.0f : out.ranking_[e];
    }
    return out;
}

PlotAssemblages PlotAssemblages::build(std::vector<SpeciesRecord> records,
                                       RepresentationMode representation) {
    std::vector<std::string> plot_ids;
    {
        std::unordered_map<std::string_view, char> seen;
        for (const auto& record : records) {
            if (seen.emplace(record.plot_id, 0).second) plot_ids.push_back(record.plot_id);
        }
    }
    return build(std::move(records), plot_ids, representation);
}

std::span<const SpeciesRecord> PlotAssemblages::plot(int64_t p) const {
    const auto begin = static_cast<size_t>(offsets_.at(static_cast<size_t>(p)));
    const auto end = static_cast<size_t>(offsets_.at(static_cast<size_t>(p) + 1));
    return std::span<const SpeciesRecord>(entries_).subspan(begin, end - begin);
}

std::span<const float> PlotAssemblages::ranking(int64_t p) const {
    const auto begin = static_cast<size_t>(offsets_.at(static_cast<size_t>(p)));
    const auto end = static_cast<size_t>(offsets_.at(static_cast<size_t>(p) + 1));
    return std::span<const float>(ranking_).subspan(begin, end - begin);
}

// =============================================================================
// Species selection
// =============================================================================

namespace {

std::vector<size_t> all_indices(size_t n) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t{0});
    return idx;
}

// The plot's entries ranked by `key` (descending or ascending), ties by
// species name, truncated to the first k when there are more.
std::vector<size_t> ranked_indices(std::span<const SpeciesRecord> plot,
                                   std::span<const float> key,
                                   int k,
                                   bool descending) {
    std::vector<size_t> idx = all_indices(plot.size());
    const size_t keep = k < 0 ? idx.size() : std::min(idx.size(), static_cast<size_t>(k));
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(keep), idx.end(),
        [&](size_t a, size_t b) {
            if (key[a] != key[b]) return descending ? key[a] > key[b] : key[a] < key[b];
            return plot[a].species_id < plot[b].species_id;
        });
    idx.resize(keep);
    return idx;
}

}  // namespace

std::vector<size_t> selection_indices(const PlotAssemblages& assemblages, int64_t p,
                                      SelectionMode mode, int k) {
    const auto plot = assemblages.plot(p);
    const auto key = assemblages.ranking(p);
    switch (mode) {
        case SelectionMode::Top:
            return ranked_indices(plot, key, k, /*descending=*/true);
        case SelectionMode::Bottom:
            return ranked_indices(plot, key, k, /*descending=*/false);
        case SelectionMode::TopBottom: {
            auto out = ranked_indices(plot, key, k, /*descending=*/true);
            std::vector<char> taken(plot.size(), 0);
            for (size_t i : out) taken[i] = 1;
            for (size_t i : ranked_indices(plot, key, k, /*descending=*/false)) {
                if (!taken[i]) out.push_back(i);
            }
            return out;
        }
        case SelectionMode::All:
            break;
    }
    return all_indices(plot.size());
}

std::vector<size_t> species_budget_indices(const PlotAssemblages& assemblages, int64_t p,
                                           SelectionMode mode, int budget) {
    if (budget <= 0 || mode == SelectionMode::All) {
        return all_indices(assemblages.plot(p).size());
    }
    auto idx = selection_indices(assemblages, p, mode, budget);
    std::sort(idx.begin(), idx.end());
    return idx;
}

std::vector<float> normalized_values(const PlotAssemblages& assemblages, int64_t p,
                                     const std::vector<size_t>& selected,
                                     NormalizationMode mode) {
    const auto plot = assemblages.plot(p);
    std::vector<float> values;
    values.reserve(selected.size());
    for (size_t i : selected) values.push_back(plot[i].abundance);
    switch (mode) {
        case NormalizationMode::Norm: {
            float total = 0.0f;
            for (float v : values) total += v;
            if (total > 0.0f) {
                for (float& v : values) v /= total;
            }
            break;
        }
        case NormalizationMode::Log1p:
            for (float& v : values) v = std::log1p(v);
            break;
        case NormalizationMode::Raw:
            break;
    }
    return values;
}

void accumulate_feature_hash(std::string_view species, float weight, float* embedding,
                             int hash_dim) {
    const HashBucketSign slot = feature_hash_bucket_sign(species, static_cast<int32_t>(hash_dim));
    embedding[slot.bucket] += slot.sign * weight;
}

int64_t percentile_linear_trunc(std::vector<int64_t>& values, double q) {
    if (values.empty()) return 0;
    const double rank = (q / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    const double frac = rank - static_cast<double>(lo);

    // nth_element puts the lo-th order statistic in place with everything
    // smaller to its left; the hi-th (lo+1) order statistic is then the
    // minimum of the remaining right partition.
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo), values.end());
    const int64_t a_lo = values[lo];
    int64_t a_hi = a_lo;
    // hi can reach values.size() when q -> 100; guard the right partition so
    // min_element is never called on an empty range.
    if (hi > lo && (lo + 1) < values.size()) {
        a_hi = *std::min_element(values.begin() + static_cast<std::ptrdiff_t>(lo) + 1, values.end());
    }

    const double interp = static_cast<double>(a_lo) +
                          frac * static_cast<double>(a_hi - a_lo);
    return static_cast<int64_t>(interp);  // int() truncation toward zero
}

// =============================================================================
// SpeciesVocab / EncodingVocab
// =============================================================================

namespace {
constexpr const char* kUnknownName = "<UNK>";
}  // namespace

SpeciesVocab::SpeciesVocab() : names_{kUnknownName} {}

SpeciesVocab::SpeciesVocab(std::vector<std::string> names) : names_(std::move(names)) {
    if (names_.empty()) names_.emplace_back(kUnknownName);
    species_to_id_.reserve(names_.size());
    for (size_t code = 1; code < names_.size(); ++code) {
        species_to_id_.emplace(names_[code], static_cast<int64_t>(code));
    }
}

SpeciesVocab SpeciesVocab::fit(const PlotAssemblages& assemblages, int min_count) {
    // A plot lists each species once, so an entry count is a plot count.
    std::unordered_map<std::string_view, int64_t> plots_with;
    for (const auto& entry : assemblages.entries()) ++plots_with[entry.species_id];

    std::vector<std::pair<std::string_view, int64_t>> ranked;
    ranked.reserve(plots_with.size());
    for (const auto& [name, count] : plots_with) {
        if (count >= min_count) ranked.emplace_back(name, count);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });

    std::vector<std::string> names;
    names.reserve(ranked.size() + 1);
    names.emplace_back(kUnknownName);
    for (const auto& item : ranked) names.emplace_back(item.first);
    return SpeciesVocab(std::move(names));
}

SpeciesVocab SpeciesVocab::from_records(const std::vector<SpeciesRecord>& records,
                                        int min_count) {
    return fit(PlotAssemblages::build(records), min_count);
}

SpeciesVocab SpeciesVocab::from_ordered(std::vector<std::string> names) {
    return SpeciesVocab(std::move(names));
}

int64_t SpeciesVocab::encode(const std::string& species) const {
    auto it = species_to_id_.find(species);
    return it != species_to_id_.end() ? it->second : 0;
}

EncodingVocab EncodingVocab::fit(const PlotAssemblages& assemblages, int min_count,
                                 bool with_taxonomy) {
    EncodingVocab vocab;
    vocab.species = SpeciesVocab::fit(assemblages, min_count);
    if (with_taxonomy) vocab.taxonomy.fit(assemblages.entries());
    return vocab;
}

// =============================================================================
// Unknown-species statistics and recognition
// =============================================================================

namespace {

// Per-plot totals against a vocabulary: the one pass both the model's novelty
// features and the recognition report are computed from. Values accumulate in
// double so a species-rich plot keeps the tail of its sum.
struct NoveltyTally {
    std::vector<double> total;
    std::vector<double> unknown;
    std::vector<int64_t> n_species;
    std::vector<int64_t> n_unknown;
};

NoveltyTally tally_novelty(const PlotAssemblages& assemblages, const SpeciesVocab& vocab) {
    const auto n_plots = static_cast<size_t>(assemblages.n_plots());
    NoveltyTally t;
    t.total.assign(n_plots, 0.0);
    t.unknown.assign(n_plots, 0.0);
    t.n_species.assign(n_plots, 0);
    t.n_unknown.assign(n_plots, 0);
    for (size_t p = 0; p < n_plots; ++p) {
        for (const auto& entry : assemblages.plot(static_cast<int64_t>(p))) {
            const double value = static_cast<double>(entry.abundance);
            t.total[p] += value;
            ++t.n_species[p];
            if (vocab.encode(entry.species_id) == 0) {
                t.unknown[p] += value;
                ++t.n_unknown[p];
            }
        }
    }
    return t;
}

}  // namespace

UnknownSpeciesStats compute_unknown_species_stats(const PlotAssemblages& assemblages,
                                                  const SpeciesVocab& vocab) {
    const int64_t n_plots = assemblages.n_plots();
    UnknownSpeciesStats out;
    out.fraction = torch::zeros({n_plots}, torch::kFloat32);
    out.count = torch::zeros({n_plots}, torch::kFloat32);
    if (n_plots == 0) return out;

    const NoveltyTally tally = tally_novelty(assemblages, vocab);
    auto fraction = out.fraction.accessor<float, 1>();
    auto count = out.count.accessor<float, 1>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto i = static_cast<size_t>(p);
        count[p] = static_cast<float>(tally.n_unknown[i]);
        if (tally.total[i] > 0.0) {
            fraction[p] = static_cast<float>(tally.unknown[i] / tally.total[i]);
        }
    }
    return out;
}

SpeciesRecognition compute_species_recognition(const PlotAssemblages& assemblages,
                                               const SpeciesVocab& vocab) {
    const int64_t n_plots = assemblages.n_plots();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    SpeciesRecognition out;
    out.n_species = torch::zeros({n_plots}, torch::kFloat32);
    out.n_recognised = torch::zeros({n_plots}, torch::kFloat32);
    out.count_share = torch::full({n_plots}, nan, torch::kFloat32);
    out.abundance_share = torch::full({n_plots}, nan, torch::kFloat32);
    if (n_plots == 0) return out;

    const NoveltyTally tally = tally_novelty(assemblages, vocab);
    auto n_species = out.n_species.accessor<float, 1>();
    auto n_recognised = out.n_recognised.accessor<float, 1>();
    auto count_share = out.count_share.accessor<float, 1>();
    auto abundance_share = out.abundance_share.accessor<float, 1>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto i = static_cast<size_t>(p);
        const int64_t n = tally.n_species[i];
        const int64_t recognised = n - tally.n_unknown[i];
        n_species[p] = static_cast<float>(n);
        n_recognised[p] = static_cast<float>(recognised);
        if (n > 0) {
            count_share[p] = static_cast<float>(recognised) / static_cast<float>(n);
        }
        if (tally.total[i] > 0.0) {
            abundance_share[p] =
                static_cast<float>((tally.total[i] - tally.unknown[i]) / tally.total[i]);
        }
    }
    return out;
}

// =============================================================================
// Taxonomy
// =============================================================================

SpeciesTaxonomy resolve_species_taxonomy(const PlotAssemblages& assemblages,
                                         const SpeciesVocab& species_vocab,
                                         const TaxonomyVocab& taxonomy_vocab) {
    SpeciesTaxonomy taxonomy;
    const auto n_codes = static_cast<size_t>(species_vocab.size());
    taxonomy.genus.assign(n_codes, 0);
    taxonomy.family.assign(n_codes, 0);
    // Every entry of a species carries the same resolved taxonomy, so any one
    // of them answers for the species.
    for (const auto& entry : assemblages.entries()) {
        const int64_t code = species_vocab.encode(entry.species_id);
        if (code == 0) continue;
        taxonomy.genus[static_cast<size_t>(code)] = taxonomy_vocab.encode_genus(entry.genus);
        taxonomy.family[static_cast<size_t>(code)] = taxonomy_vocab.encode_family(entry.family);
    }
    return taxonomy;
}

namespace {

// The first k names by descending weight, ties by name.
std::vector<std::string_view> top_k_by_weight(
    const std::unordered_map<std::string_view, float>& weight, int k) {
    std::vector<std::pair<std::string_view, float>> items(weight.begin(), weight.end());
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    const size_t n = std::min(items.size(), static_cast<size_t>(std::max(k, 0)));
    std::vector<std::string_view> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(items[i].first);
    return out;
}

}  // namespace

TaxonomySlots encode_taxonomy_slots(const PlotAssemblages& assemblages,
                                    const TaxonomyVocab& taxonomy,
                                    AggregationMode aggregation,
                                    int n_slots) {
    const int64_t n_plots = assemblages.n_plots();
    const auto width = static_cast<int64_t>(std::max(n_slots, 0));
    TaxonomySlots slots;
    slots.genus_ids = torch::zeros({n_plots, width}, torch::kInt64);
    slots.family_ids = torch::zeros({n_plots, width}, torch::kInt64);
    auto genus_acc = slots.genus_ids.accessor<int64_t, 2>();
    auto family_acc = slots.family_ids.accessor<int64_t, 2>();

    std::unordered_map<std::string_view, float> genus_weight;
    std::unordered_map<std::string_view, float> family_weight;
    std::string name;
    for (int64_t p = 0; p < n_plots; ++p) {
        genus_weight.clear();
        family_weight.clear();
        for (const auto& entry : assemblages.plot(p)) {
            const float w = aggregation == AggregationMode::Count ? 1.0f : entry.abundance;
            if (!entry.genus.empty()) genus_weight[entry.genus] += w;
            if (!entry.family.empty()) family_weight[entry.family] += w;
        }
        const auto genera = top_k_by_weight(genus_weight, n_slots);
        for (size_t s = 0; s < genera.size(); ++s) {
            name.assign(genera[s]);
            genus_acc[p][static_cast<int64_t>(s)] = taxonomy.encode_genus(name);
        }
        const auto families = top_k_by_weight(family_weight, n_slots);
        for (size_t s = 0; s < families.size(); ++s) {
            name.assign(families[s]);
            family_acc[p][static_cast<int64_t>(s)] = taxonomy.encode_family(name);
        }
    }
    return slots;
}

// =============================================================================
// SpeciesVocabEncoder
// =============================================================================

void SpeciesVocabEncoder::fit(const std::vector<SpeciesRecord>& records) {
    fit(PlotAssemblages::build(records));
}

void SpeciesVocabEncoder::fit(const PlotAssemblages& assemblages) {
    vocab_ = EncodingVocab::fit(assemblages, min_frequency_);
    fitted_ = true;
}

void SpeciesVocabEncoder::set_vocab(EncodingVocab vocab) {
    vocab_ = std::move(vocab);
    fitted_ = true;
}

void SpeciesVocabEncoder::require_vocab(const char* encoder) const {
    if (!fitted_) {
        throw std::runtime_error(std::string(encoder) +
                                 ": fit() or set_vocab() must run before transform()");
    }
}

// =============================================================================
// RankPoolEncoder
// =============================================================================

namespace {

// 1/rank, dense-ranked by descending value: ties share a rank and ranks have
// no gaps, so [10, 5, 5, 2] gives [1, 1/2, 1/2, 1/3].
std::vector<float> dense_rank_weights(const std::vector<float>& values) {
    const size_t n = values.size();
    std::vector<size_t> order = all_indices(n);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return values[a] > values[b]; });
    std::vector<float> weights(n);
    int rank = 0;
    for (size_t j = 0; j < n; ++j) {
        if (j == 0 || values[order[j]] < values[order[j - 1]]) ++rank;
        weights[order[j]] = 1.0f / static_cast<float>(rank);
    }
    return weights;
}

std::vector<float> pool_weights(std::vector<float> values, PoolWeighting weighting) {
    switch (weighting) {
        case PoolWeighting::Binary:
            std::fill(values.begin(), values.end(), 1.0f);
            return values;
        case PoolWeighting::Abundance:
            return values;
        case PoolWeighting::Log1p:
            for (float& v : values) v = std::log1p(v);
            return values;
        case PoolWeighting::Norm: {
            float total = 0.0f;
            for (float v : values) total += v;
            for (float& v : values) v = total > 0.0f ? v / total : 1.0f;
            return values;
        }
        case PoolWeighting::Rank:
            return dense_rank_weights(values);
    }
    return values;
}

// The width a pool species cap resolves to over these per-plot lengths:
// `widest` without a cap, the cap itself when positive, the requested
// percentile of the lengths (at least 1) when negative.
int64_t resolve_species_cap(int species_cap, std::vector<int64_t> lengths, int64_t widest) {
    if (species_cap <= -100) {
        throw std::invalid_argument(
            "pool_species_cap " + std::to_string(species_cap) + " names the " +
            std::to_string(100 + species_cap) + "th percentile; a negative cap -x "
            "means the (100 - x)th percentile and takes x in 1..99 (0 = no cap)");
    }
    if (species_cap > 0) return species_cap;
    if (species_cap == 0 || lengths.empty()) return widest;
    return std::max<int64_t>(
        percentile_linear_trunc(lengths, static_cast<double>(100 + species_cap)), 1);
}

}  // namespace

RankPoolEncoder::RankPoolEncoder(PoolWeighting weighting, int min_frequency,
                                 SelectionMode selection, int species_budget)
    : SpeciesVocabEncoder(min_frequency),
      weighting_(weighting),
      selection_(selection),
      species_budget_(species_budget) {}

RankPoolEncodedData RankPoolEncoder::transform(const PlotAssemblages& assemblages,
                                               int species_cap,
                                               bool has_abundance_column) const {
    require_vocab("RankPoolEncoder");
    const int64_t n_plots = assemblages.n_plots();

    struct PlotSlots {
        std::vector<int64_t> species, genus, family;
        std::vector<float> weights;
    };
    std::vector<PlotSlots> slots(static_cast<size_t>(n_plots));
    std::vector<int64_t> lengths(static_cast<size_t>(n_plots), 0);
    int64_t widest = 0;
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto plot = assemblages.plot(p);
        const auto kept = species_budget_indices(assemblages, p, selection_, species_budget_);
        auto& s = slots[static_cast<size_t>(p)];
        std::vector<float> values;
        values.reserve(kept.size());
        for (size_t i : kept) {
            const auto& entry = plot[i];
            s.species.push_back(vocab_.species.encode(entry.species_id));
            s.genus.push_back(vocab_.taxonomy.encode_genus(entry.genus));
            s.family.push_back(vocab_.taxonomy.encode_family(entry.family));
            values.push_back(entry.abundance);
        }
        s.weights = pool_weights(std::move(values), weighting_);
        lengths[static_cast<size_t>(p)] = static_cast<int64_t>(kept.size());
        widest = std::max(widest, static_cast<int64_t>(kept.size()));
    }
    widest = std::max<int64_t>(widest, 1);

    const int64_t width = std::min(resolve_species_cap(species_cap, lengths, widest), widest);
    if (width < widest) {
        const double saved = 1.0 - static_cast<double>(width) / static_cast<double>(widest);
        std::cout << "  rank_pool: capping species at "
                  << (species_cap < 0 ? "p" + std::to_string(100 + species_cap) + "=" : "cap=")
                  << width << " (max=" << widest
                  << ", saves " << static_cast<int>(saved * 100.0 + 0.5)
                  << "% padding)" << std::endl;
    }

    RankPoolEncodedData out;
    out.species_ids = torch::zeros({n_plots, width}, torch::kInt64);
    out.genus_ids = torch::zeros({n_plots, width}, torch::kInt64);
    out.family_ids = torch::zeros({n_plots, width}, torch::kInt64);
    out.weights = torch::zeros({n_plots, width}, torch::kFloat32);
    out.mask = torch::zeros({n_plots, width}, torch::kBool);
    // A column-presence flag: abundances reach the pool only when a column
    // was mapped and the representation keeps them.
    const bool has_cover = has_abundance_column &&
                           assemblages.representation() == RepresentationMode::Abundance;
    out.has_cover = torch::full({n_plots}, has_cover ? 1.0f : 0.0f, torch::kFloat32);

    auto species = out.species_ids.accessor<int64_t, 2>();
    auto genus = out.genus_ids.accessor<int64_t, 2>();
    auto family = out.family_ids.accessor<int64_t, 2>();
    auto weights = out.weights.accessor<float, 2>();
    auto mask = out.mask.accessor<bool, 2>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto& s = slots[static_cast<size_t>(p)];
        const int64_t n = std::min(static_cast<int64_t>(s.species.size()), width);
        for (int64_t j = 0; j < n; ++j) {
            const auto jj = static_cast<size_t>(j);
            species[p][j] = s.species[jj];
            genus[p][j] = s.genus[jj];
            family[p][j] = s.family[jj];
            weights[p][j] = s.weights[jj];
            mask[p][j] = true;
        }
    }
    return out;
}

RankPoolEncodedData RankPoolEncoder::transform(const std::vector<SpeciesRecord>& records,
                                               const std::vector<std::string>& plot_ids,
                                               int species_cap,
                                               bool has_abundance_column,
                                               RepresentationMode representation) const {
    return transform(PlotAssemblages::build(records, plot_ids, representation),
                     species_cap, has_abundance_column);
}

// =============================================================================
// EmbeddingEncoder
// =============================================================================

EmbeddingEncoder::EmbeddingEncoder(int top_k_species, int top_k_taxonomy,
                                   SelectionMode selection, AggregationMode aggregation)
    : SpeciesVocabEncoder(/*min_frequency=*/1),
      top_k_species_(top_k_species),
      top_k_taxonomy_(top_k_taxonomy),
      selection_(selection),
      aggregation_(aggregation) {}

EmbeddingEncodedData EmbeddingEncoder::transform(const PlotAssemblages& assemblages) const {
    require_vocab("EmbeddingEncoder");
    if (selection_ == SelectionMode::All) {
        throw std::invalid_argument(
            "EmbeddingEncoder: SelectionMode::All cannot be encoded into a fixed "
            "top_k_species slots. Use Top, Bottom or TopBottom, or switch to the "
            "rank_pool / transformer / sparse species encoding, which encode "
            "every species a plot records.");
    }
    // TopBottom fills one fixed-width row from both ends, so each end takes
    // half the slots (rounded up); an odd slot count drops the last bottom
    // species instead of overflowing the row.
    const int per_side_k = selection_ == SelectionMode::TopBottom
                               ? (top_k_species_ + 1) / 2
                               : top_k_species_;

    const int64_t n_plots = assemblages.n_plots();
    EmbeddingEncodedData out;
    out.species_ids = torch::zeros({n_plots, static_cast<int64_t>(top_k_species_)}, torch::kInt64);
    auto species = out.species_ids.accessor<int64_t, 2>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto plot = assemblages.plot(p);
        const auto selected = selection_indices(assemblages, p, selection_, per_side_k);
        const size_t n = std::min(selected.size(), static_cast<size_t>(top_k_species_));
        for (size_t j = 0; j < n; ++j) {
            species[p][static_cast<int64_t>(j)] = vocab_.species.encode(plot[selected[j]].species_id);
        }
    }

    if (vocab_.has_taxonomy()) {
        auto taxonomy = encode_taxonomy_slots(assemblages, vocab_.taxonomy, aggregation_,
                                              top_k_taxonomy_);
        out.genus_ids = std::move(taxonomy.genus_ids);
        out.family_ids = std::move(taxonomy.family_ids);
    } else {
        const auto width = static_cast<int64_t>(top_k_taxonomy_);
        out.genus_ids = torch::zeros({n_plots, width}, torch::kInt64);
        out.family_ids = torch::zeros({n_plots, width}, torch::kInt64);
    }
    return out;
}

EmbeddingEncodedData EmbeddingEncoder::transform(const std::vector<SpeciesRecord>& records,
                                                 const std::vector<std::string>& plot_ids,
                                                 RepresentationMode representation) const {
    return transform(PlotAssemblages::build(records, plot_ids, representation));
}

} // namespace resolve
