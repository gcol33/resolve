#include "resolve/species_encoding.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>

namespace resolve {

// =============================================================================
// PlotAssemblages
// =============================================================================

namespace {

// Keeps the code of the lexicographically smallest name seen (-1 = none yet).
void keep_smallest(int32_t& current, int32_t candidate, const NamePool& names) {
    if (candidate < 0 || candidate == current) return;
    if (current < 0 || names.name(candidate) < names.name(current)) current = candidate;
}

// rank[c] = the position of name c in the lexicographic order of the pool.
std::vector<int32_t> name_ranks(const NamePool& names) {
    std::vector<int32_t> order(static_cast<size_t>(names.size()));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return names.name(a) < names.name(b); });
    std::vector<int32_t> rank(order.size());
    for (size_t i = 0; i < order.size(); ++i) rank[static_cast<size_t>(order[i])] = static_cast<int32_t>(i);
    return rank;
}

template <typename T>
void release(std::vector<T>& v) {
    std::vector<T>().swap(v);
}

}  // namespace

void PlotAssemblages::Builder::add(int32_t plot, std::string_view species, float abundance,
                                   std::string_view genus, std::string_view family) {
    plot_.push_back(plot);
    species_.push_back(species_names_.intern(species));
    genus_.push_back(genus.empty() ? -1 : genus_names_.intern(genus));
    family_.push_back(family.empty() ? -1 : family_names_.intern(family));
    abundance_.push_back(abundance);
}

PlotAssemblages PlotAssemblages::Builder::finish(int64_t n_plots,
                                                 std::span<const int32_t> plot_of,
                                                 RepresentationMode representation) && {
    PlotAssemblages out;
    out.representation_ = representation;
    const size_t n_records = species_.size();
    const auto n_species = static_cast<size_t>(species_names_.size());
    auto destination = [&](size_t r) -> int64_t {
        const int32_t code = plot_[r];
        if (code < 0 || static_cast<size_t>(code) >= plot_of.size()) return -1;
        const int32_t p = plot_of[static_cast<size_t>(code)];
        return p >= 0 && p < n_plots ? p : -1;
    };

    // Every species' taxonomy, resolved over the records of kept plots.
    out.genus_of_.assign(n_species, -1);
    out.family_of_.assign(n_species, -1);
    for (size_t r = 0; r < n_records; ++r) {
        if (destination(r) < 0) continue;
        const auto s = static_cast<size_t>(species_[r]);
        keep_smallest(out.genus_of_[s], genus_[r], genus_names_);
        keep_smallest(out.family_of_[s], family_[r], family_names_);
    }
    release(genus_);
    release(family_);

    // The kept records bucketed by plot, the order they were added kept
    // within a plot.
    std::vector<int64_t> starts(static_cast<size_t>(n_plots) + 1, 0);
    for (size_t r = 0; r < n_records; ++r) {
        const int64_t p = destination(r);
        if (p >= 0) ++starts[static_cast<size_t>(p) + 1];
    }
    for (size_t p = 0; p < static_cast<size_t>(n_plots); ++p) starts[p + 1] += starts[p];
    const auto n_kept = static_cast<size_t>(starts.back());
    std::vector<int32_t> by_plot_species(n_kept);
    std::vector<float> by_plot_abundance(n_kept);
    {
        std::vector<int64_t> cursor(starts.begin(), starts.end() - 1);
        for (size_t r = 0; r < n_records; ++r) {
            const int64_t p = destination(r);
            if (p < 0) continue;
            const auto i = static_cast<size_t>(cursor[static_cast<size_t>(p)]++);
            by_plot_species[i] = species_[r];
            by_plot_abundance[i] = abundance_[r];
        }
    }
    release(plot_);
    release(species_);
    release(abundance_);

    // One entry per species per plot, where its first record stood, carrying
    // the sum of its records' abundances (summed in record order).
    std::vector<int64_t> stamp(n_species, -1);
    std::vector<size_t> position(n_species, 0);
    out.offsets_.assign(1, 0);
    out.offsets_.reserve(static_cast<size_t>(n_plots) + 1);
    out.species_.reserve(n_kept);
    out.ranking_.reserve(n_kept);
    for (size_t p = 0; p < static_cast<size_t>(n_plots); ++p) {
        for (auto i = static_cast<size_t>(starts[p]); i < static_cast<size_t>(starts[p + 1]); ++i) {
            const auto s = static_cast<size_t>(by_plot_species[i]);
            if (stamp[s] == static_cast<int64_t>(p)) {
                out.ranking_[position[s]] += by_plot_abundance[i];
                continue;
            }
            stamp[s] = static_cast<int64_t>(p);
            position[s] = out.species_.size();
            out.species_.push_back(by_plot_species[i]);
            out.ranking_.push_back(by_plot_abundance[i]);
        }
        out.offsets_.push_back(static_cast<int64_t>(out.species_.size()));
    }
    out.species_.shrink_to_fit();
    out.ranking_.shrink_to_fit();

    out.plots_per_species_.assign(n_species, 0);
    for (int32_t s : out.species_) ++out.plots_per_species_[static_cast<size_t>(s)];
    for (size_t s = 0; s < n_species; ++s) {
        if (out.plots_per_species_[s] > 0 && (out.genus_of_[s] >= 0 || out.family_of_[s] >= 0)) {
            out.names_taxa_ = true;
            break;
        }
    }

    out.species_names_ = std::move(species_names_);
    out.genus_names_ = std::move(genus_names_);
    out.family_names_ = std::move(family_names_);
    out.species_rank_ = name_ranks(out.species_names_);
    out.genus_rank_ = name_ranks(out.genus_names_);
    out.family_rank_ = name_ranks(out.family_names_);
    return out;
}

PlotAssemblages PlotAssemblages::build(const std::vector<SpeciesRecord>& records,
                                       const std::vector<std::string>& plot_ids,
                                       RepresentationMode representation) {
    // A plot ID listed twice keeps its first position; the later one holds no
    // entries.
    NamePool plots;
    std::vector<int32_t> plot_of;
    for (size_t p = 0; p < plot_ids.size(); ++p) {
        if (plots.insert(plot_ids[p]).second) plot_of.push_back(static_cast<int32_t>(p));
    }
    Builder builder;
    for (const auto& record : records) {
        const int32_t code = plots.find(record.plot_id);
        if (code >= 0) {
            builder.add(code, record.species_id, record.abundance, record.genus, record.family);
        }
    }
    return std::move(builder).finish(static_cast<int64_t>(plot_ids.size()), plot_of,
                                     representation);
}

PlotAssemblages PlotAssemblages::build(const std::vector<SpeciesRecord>& records,
                                       RepresentationMode representation) {
    NamePool plots;
    Builder builder;
    for (const auto& record : records) {
        builder.add(plots.intern(record.plot_id), record.species_id, record.abundance,
                    record.genus, record.family);
    }
    std::vector<int32_t> plot_of(static_cast<size_t>(plots.size()));
    std::iota(plot_of.begin(), plot_of.end(), 0);
    return std::move(builder).finish(plots.size(), plot_of, representation);
}

std::span<const int32_t> PlotAssemblages::species(int64_t p) const {
    const auto begin = static_cast<size_t>(offset(p));
    return std::span<const int32_t>(species_).subspan(begin, static_cast<size_t>(plot_size(p)));
}

std::span<const float> PlotAssemblages::ranking(int64_t p) const {
    const auto begin = static_cast<size_t>(offset(p));
    return std::span<const float>(ranking_).subspan(begin, static_cast<size_t>(plot_size(p)));
}

namespace {

// The vocabulary code of each interned name: entry i is encode(names.name(i)).
template <typename Encode>
std::vector<int64_t> codes_of(const NamePool& names, Encode&& encode) {
    std::vector<int64_t> codes(static_cast<size_t>(names.size()));
    for (int32_t i = 0; i < names.size(); ++i)
        codes[static_cast<size_t>(i)] = encode(names.name(i));
    return codes;
}

}  // namespace

std::vector<int64_t> PlotAssemblages::species_codes(const SpeciesVocab& vocab) const {
    return codes_of(species_names_, [&](const std::string& name) { return vocab.encode(name); });
}

std::vector<int64_t> PlotAssemblages::genus_codes(const TaxonomyVocab& taxonomy) const {
    return codes_of(genus_names_,
                    [&](const std::string& name) { return taxonomy.encode_genus(name); });
}

std::vector<int64_t> PlotAssemblages::family_codes(const TaxonomyVocab& taxonomy) const {
    return codes_of(family_names_,
                    [&](const std::string& name) { return taxonomy.encode_family(name); });
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

// Plot p's entries ranked by their summed abundance (descending or
// ascending), ties by species name, truncated to the first k when there are
// more.
std::vector<size_t> ranked_indices(const PlotAssemblages& assemblages, int64_t p, int k,
                                   bool descending) {
    const auto species = assemblages.species(p);
    const auto key = assemblages.ranking(p);
    std::vector<size_t> idx = all_indices(species.size());
    const size_t keep = k < 0 ? idx.size() : std::min(idx.size(), static_cast<size_t>(k));
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(keep), idx.end(),
        [&](size_t a, size_t b) {
            if (key[a] != key[b]) return descending ? key[a] > key[b] : key[a] < key[b];
            return assemblages.species_rank(species[a]) < assemblages.species_rank(species[b]);
        });
    idx.resize(keep);
    return idx;
}

}  // namespace

std::vector<size_t> selection_indices(const PlotAssemblages& assemblages, int64_t p,
                                      SelectionMode mode, int k) {
    switch (mode) {
        case SelectionMode::Top:
            return ranked_indices(assemblages, p, k, /*descending=*/true);
        case SelectionMode::Bottom:
            return ranked_indices(assemblages, p, k, /*descending=*/false);
        case SelectionMode::TopBottom: {
            auto out = ranked_indices(assemblages, p, k, /*descending=*/true);
            std::vector<char> taken(static_cast<size_t>(assemblages.plot_size(p)), 0);
            for (size_t i : out) taken[i] = 1;
            for (size_t i : ranked_indices(assemblages, p, k, /*descending=*/false)) {
                if (!taken[i]) out.push_back(i);
            }
            return out;
        }
        case SelectionMode::All:
            break;
    }
    return all_indices(static_cast<size_t>(assemblages.plot_size(p)));
}

std::vector<size_t> species_budget_indices(const PlotAssemblages& assemblages, int64_t p,
                                           SelectionMode mode, int budget) {
    if (budget <= 0 || mode == SelectionMode::All) {
        return all_indices(static_cast<size_t>(assemblages.plot_size(p)));
    }
    auto idx = selection_indices(assemblages, p, mode, budget);
    std::sort(idx.begin(), idx.end());
    return idx;
}

std::vector<float> normalized_values(const PlotAssemblages& assemblages, int64_t p,
                                     const std::vector<size_t>& selected,
                                     NormalizationMode mode) {
    const auto ranking = assemblages.ranking(p);
    std::vector<float> values;
    values.reserve(selected.size());
    for (size_t i : selected) values.push_back(assemblages.value(ranking[i]));
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
    // A plot lists each species once, so the number of plots a species is
    // recorded in is its entry count.
    const auto& plots_with = assemblages.plots_per_species();
    std::vector<int32_t> ranked;
    for (int32_t s = 0; s < assemblages.n_species(); ++s) {
        const int64_t count = plots_with[static_cast<size_t>(s)];
        if (count > 0 && count >= min_count) ranked.push_back(s);
    }
    std::sort(ranked.begin(), ranked.end(), [&](int32_t a, int32_t b) {
        const int64_t ca = plots_with[static_cast<size_t>(a)];
        const int64_t cb = plots_with[static_cast<size_t>(b)];
        return ca != cb ? ca > cb : assemblages.species_rank(a) < assemblages.species_rank(b);
    });

    std::vector<std::string> names;
    names.reserve(ranked.size() + 1);
    names.emplace_back(kUnknownName);
    for (int32_t s : ranked) names.push_back(assemblages.species_name(s));
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
    if (with_taxonomy) {
        // The genera and families of the species the plots record.
        std::set<std::string> genera;
        std::set<std::string> families;
        for (int32_t s = 0; s < assemblages.n_species(); ++s) {
            if (assemblages.plots_per_species()[static_cast<size_t>(s)] == 0) continue;
            if (assemblages.genus_of(s) >= 0) genera.insert(assemblages.genus_name(assemblages.genus_of(s)));
            if (assemblages.family_of(s) >= 0) families.insert(assemblages.family_name(assemblages.family_of(s)));
        }
        vocab.taxonomy.fit(genera, families);
    }
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
    const std::vector<int64_t> codes = assemblages.species_codes(vocab);
    NoveltyTally t;
    t.total.assign(n_plots, 0.0);
    t.unknown.assign(n_plots, 0.0);
    t.n_species.assign(n_plots, 0);
    t.n_unknown.assign(n_plots, 0);
    for (size_t p = 0; p < n_plots; ++p) {
        const auto species = assemblages.species(static_cast<int64_t>(p));
        const auto ranking = assemblages.ranking(static_cast<int64_t>(p));
        for (size_t i = 0; i < species.size(); ++i) {
            const double value = static_cast<double>(assemblages.value(ranking[i]));
            t.total[p] += value;
            ++t.n_species[p];
            if (codes[static_cast<size_t>(species[i])] == 0) {
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

namespace {

// The code `codes` gives taxon t of a species (taxon_of), 0 for none.
int64_t taxon_code(int32_t taxon, const std::vector<int64_t>& codes) {
    return taxon >= 0 ? codes[static_cast<size_t>(taxon)] : 0;
}

}  // namespace

SpeciesTaxonomy resolve_species_taxonomy(const PlotAssemblages& assemblages,
                                         const SpeciesVocab& species_vocab,
                                         const TaxonomyVocab& taxonomy_vocab) {
    SpeciesTaxonomy taxonomy;
    const auto n_codes = static_cast<size_t>(species_vocab.size());
    taxonomy.genus.assign(n_codes, 0);
    taxonomy.family.assign(n_codes, 0);
    const std::vector<int64_t> codes = assemblages.species_codes(species_vocab);
    const std::vector<int64_t> genus_codes = assemblages.genus_codes(taxonomy_vocab);
    const std::vector<int64_t> family_codes = assemblages.family_codes(taxonomy_vocab);
    for (int32_t s = 0; s < assemblages.n_species(); ++s) {
        const int64_t code = codes[static_cast<size_t>(s)];
        if (code == 0 || assemblages.plots_per_species()[static_cast<size_t>(s)] == 0) continue;
        taxonomy.genus[static_cast<size_t>(code)] = taxon_code(assemblages.genus_of(s), genus_codes);
        taxonomy.family[static_cast<size_t>(code)] = taxon_code(assemblages.family_of(s), family_codes);
    }
    return taxonomy;
}

namespace {

// The weight of each distinct taxon of one plot, in the order the plot first
// names it, ranked and cut to the first k slots.
class TaxonTally {
public:
    explicit TaxonTally(int32_t n_taxa) : slot_(static_cast<size_t>(n_taxa), -1) {}

    void add(int32_t taxon, float weight) {
        if (taxon < 0) return;
        int32_t& slot = slot_[static_cast<size_t>(taxon)];
        if (slot < 0) {
            slot = static_cast<int32_t>(taxa_.size());
            taxa_.push_back(taxon);
            weights_.push_back(0.0f);
        }
        weights_[static_cast<size_t>(slot)] += weight;
    }

    // The first k taxa by descending weight, ties by `rank` (name order);
    // resets the tally.
    template <typename Rank>
    std::vector<int32_t> top_k(int k, Rank rank) {
        std::vector<size_t> order = all_indices(taxa_.size());
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (weights_[a] != weights_[b]) return weights_[a] > weights_[b];
            return rank(taxa_[a]) < rank(taxa_[b]);
        });
        const size_t n = std::min(order.size(), static_cast<size_t>(std::max(k, 0)));
        std::vector<int32_t> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.push_back(taxa_[order[i]]);
        for (int32_t taxon : taxa_) slot_[static_cast<size_t>(taxon)] = -1;
        taxa_.clear();
        weights_.clear();
        return out;
    }

private:
    std::vector<int32_t> slot_;
    std::vector<int32_t> taxa_;
    std::vector<float> weights_;
};

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

    const std::vector<int64_t> genus_codes = assemblages.genus_codes(taxonomy);
    const std::vector<int64_t> family_codes = assemblages.family_codes(taxonomy);
    TaxonTally genera(assemblages.n_genera());
    TaxonTally families(assemblages.n_families());
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto species = assemblages.species(p);
        const auto ranking = assemblages.ranking(p);
        for (size_t i = 0; i < species.size(); ++i) {
            const float w = aggregation == AggregationMode::Count ? 1.0f
                                                                  : assemblages.value(ranking[i]);
            genera.add(assemblages.genus_of(species[i]), w);
            families.add(assemblages.family_of(species[i]), w);
        }
        const auto top_genera =
            genera.top_k(n_slots, [&](int32_t g) { return assemblages.genus_rank(g); });
        for (size_t s = 0; s < top_genera.size(); ++s) {
            genus_acc[p][static_cast<int64_t>(s)] = genus_codes[static_cast<size_t>(top_genera[s])];
        }
        const auto top_families =
            families.top_k(n_slots, [&](int32_t f) { return assemblages.family_rank(f); });
        for (size_t s = 0; s < top_families.size(); ++s) {
            family_acc[p][static_cast<int64_t>(s)] = family_codes[static_cast<size_t>(top_families[s])];
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
    const std::vector<int64_t> species_codes = assemblages.species_codes(vocab_.species);
    const std::vector<int64_t> genus_codes = assemblages.genus_codes(vocab_.taxonomy);
    const std::vector<int64_t> family_codes = assemblages.family_codes(vocab_.taxonomy);

    // Each plot's kept entries as indices into its plot, with their weights.
    std::vector<std::vector<size_t>> kept(static_cast<size_t>(n_plots));
    std::vector<std::vector<float>> weights(static_cast<size_t>(n_plots));
    std::vector<int64_t> lengths(static_cast<size_t>(n_plots), 0);
    int64_t widest = 0;
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto pi = static_cast<size_t>(p);
        kept[pi] = species_budget_indices(assemblages, p, selection_, species_budget_);
        const auto ranking = assemblages.ranking(p);
        std::vector<float> values;
        values.reserve(kept[pi].size());
        for (size_t i : kept[pi]) values.push_back(assemblages.value(ranking[i]));
        weights[pi] = pool_weights(std::move(values), weighting_);
        lengths[pi] = static_cast<int64_t>(kept[pi].size());
        widest = std::max(widest, lengths[pi]);
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
    auto weight = out.weights.accessor<float, 2>();
    auto mask = out.mask.accessor<bool, 2>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto pi = static_cast<size_t>(p);
        const auto plot_species = assemblages.species(p);
        const int64_t n = std::min(lengths[pi], width);
        for (int64_t j = 0; j < n; ++j) {
            const auto jj = static_cast<size_t>(j);
            const int32_t s = plot_species[kept[pi][jj]];
            species[p][j] = species_codes[static_cast<size_t>(s)];
            genus[p][j] = taxon_code(assemblages.genus_of(s), genus_codes);
            family[p][j] = taxon_code(assemblages.family_of(s), family_codes);
            weight[p][j] = weights[pi][jj];
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
    const std::vector<int64_t> species_codes = assemblages.species_codes(vocab_.species);
    EmbeddingEncodedData out;
    out.species_ids = torch::zeros({n_plots, static_cast<int64_t>(top_k_species_)}, torch::kInt64);
    auto species = out.species_ids.accessor<int64_t, 2>();
    for (int64_t p = 0; p < n_plots; ++p) {
        const auto plot_species = assemblages.species(p);
        const auto selected = selection_indices(assemblages, p, selection_, per_side_k);
        const size_t n = std::min(selected.size(), static_cast<size_t>(top_k_species_));
        for (size_t j = 0; j < n; ++j) {
            species[p][static_cast<int64_t>(j)] =
                species_codes[static_cast<size_t>(plot_species[selected[j]])];
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
