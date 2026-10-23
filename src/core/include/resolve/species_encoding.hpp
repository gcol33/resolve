#pragma once

#include "resolve/hash_core.hpp"
#include "resolve/name_pool.hpp"
#include "resolve/types.hpp"
#include <torch/torch.h>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

namespace resolve {

class SpeciesVocab;

// =============================================================================
// PlotAssemblages: the species records every encoding reads
// =============================================================================

// The species records of a set of plots, grouped by plot and merged so that a
// plot lists each species once. Every species encoding, the vocabulary fit,
// the unknown-species statistics and the recognition report read this one
// table, so they all agree on what a plot contains.
//
// Storage is one CSR table: plot p owns entries [offset(p), offset(p + 1)) of
// two flat arrays, the species of each entry as an interned code
// (species_name() spells it) and its summed abundance. Genus and family are
// properties of a species, held once per interned species. A table over
// millions of plots costs two numbers per entry plus the distinct names.
//
// Records of one species in one plot (an archive records a species once per
// vegetation layer) become one entry, placed where the species' first record
// in the plot stood:
//
//   abundance  the sum of the records' abundances, the species' total cover
//              across layers;
//   genus,     the lexicographically smallest genus and family any record of
//   family     the species names, over the records of all kept plots. Every
//              entry of a species therefore carries the same taxonomy,
//              whatever order the rows arrive in.
//
// Representation then sets the value each entry carries into the encodings:
// the summed abundance under RepresentationMode::Abundance, 1.0 under
// PresenceAbsence. The summed abundance stays available as the plot's ranking,
// which is what the species selection orders by in either mode, so
// presence/absence with Top still keeps the dominant species.
class PlotAssemblages {
public:
    // Accumulates species records as they are read, each naming its plot by
    // a caller's integer code, and groups them into a PlotAssemblages. Names
    // are interned on arrival, so a record costs a few integers.
    class Builder {
    public:
        // One record of the plot coded `plot` (>= 0). An empty genus or family
        // names none.
        void add(int32_t plot, std::string_view species, float abundance,
                 std::string_view genus, std::string_view family);

        // The table over `n_plots` plots: plot p holds the records of every
        // plot code c with plot_of[c] == p, in the order they were added. A
        // record whose code maps to -1, or lies past the end of `plot_of`, is
        // dropped and names no taxonomy. The builder is consumed.
        [[nodiscard]] PlotAssemblages finish(int64_t n_plots, std::span<const int32_t> plot_of,
                                             RepresentationMode representation) &&;

        [[nodiscard]] size_t n_records() const noexcept { return species_.size(); }

    private:
        NamePool species_names_;
        NamePool genus_names_;
        NamePool family_names_;
        std::vector<int32_t> plot_;
        std::vector<int32_t> species_;
        std::vector<int32_t> genus_;   // -1: none named
        std::vector<int32_t> family_;  // -1: none named
        std::vector<float> abundance_;
    };

    PlotAssemblages() = default;

    // Plots in the order of `plot_ids`. A record naming a plot outside
    // `plot_ids` is dropped; a plot without records holds no entries.
    static PlotAssemblages build(const std::vector<SpeciesRecord>& records,
                                 const std::vector<std::string>& plot_ids,
                                 RepresentationMode representation = RepresentationMode::Abundance);

    // Plots in the order their first record appears.
    static PlotAssemblages build(const std::vector<SpeciesRecord>& records,
                                 RepresentationMode representation = RepresentationMode::Abundance);

    [[nodiscard]] int64_t n_plots() const noexcept {
        return static_cast<int64_t>(offsets_.size()) - 1;
    }
    [[nodiscard]] int64_t n_entries() const noexcept {
        return static_cast<int64_t>(species_.size());
    }
    // Plot p owns entries [offset(p), offset(p + 1)).
    [[nodiscard]] int64_t offset(int64_t p) const { return offsets_.at(static_cast<size_t>(p)); }
    [[nodiscard]] int64_t plot_size(int64_t p) const { return offset(p + 1) - offset(p); }
    // The interned species of plot p's entries.
    [[nodiscard]] std::span<const int32_t> species(int64_t p) const;
    // The summed recorded abundances of plot p's entries, parallel to species(p).
    [[nodiscard]] std::span<const float> ranking(int64_t p) const;
    // The value an entry with summed abundance `ranking` carries into the
    // encodings under this table's representation.
    [[nodiscard]] float value(float ranking) const noexcept {
        return representation_ == RepresentationMode::PresenceAbsence ? 1.0f : ranking;
    }
    [[nodiscard]] RepresentationMode representation() const noexcept { return representation_; }

    // Interned species, genera and families, coded 0 .. n-1. A species may be
    // interned without entries (all its records fell in dropped plots).
    [[nodiscard]] int32_t n_species() const noexcept { return species_names_.size(); }
    [[nodiscard]] const std::string& species_name(int32_t s) const { return species_names_.name(s); }
    // The position of species s in the lexicographic order of the names: the
    // tie-break every ranking of species uses.
    [[nodiscard]] int32_t species_rank(int32_t s) const { return species_rank_[static_cast<size_t>(s)]; }
    // The genus / family of species s as a code into genus_name() /
    // family_name(), -1 when its records name none.
    [[nodiscard]] int32_t genus_of(int32_t s) const { return genus_of_[static_cast<size_t>(s)]; }
    [[nodiscard]] int32_t family_of(int32_t s) const { return family_of_[static_cast<size_t>(s)]; }
    [[nodiscard]] int32_t n_genera() const noexcept { return genus_names_.size(); }
    [[nodiscard]] int32_t n_families() const noexcept { return family_names_.size(); }
    [[nodiscard]] const std::string& genus_name(int32_t g) const { return genus_names_.name(g); }
    [[nodiscard]] const std::string& family_name(int32_t f) const { return family_names_.name(f); }
    [[nodiscard]] int32_t genus_rank(int32_t g) const { return genus_rank_[static_cast<size_t>(g)]; }
    [[nodiscard]] int32_t family_rank(int32_t f) const { return family_rank_[static_cast<size_t>(f)]; }
    // The number of plots each species is recorded in (a plot lists a species
    // once), indexed by the interned code.
    [[nodiscard]] const std::vector<int64_t>& plots_per_species() const noexcept {
        return plots_per_species_;
    }
    // Whether any species with entries names a genus or a family.
    [[nodiscard]] bool names_taxa() const noexcept { return names_taxa_; }

    // The code `vocab` gives each interned species, genus or family (0 for a
    // name it does not hold), indexed by the interned code.
    [[nodiscard]] std::vector<int64_t> species_codes(const SpeciesVocab& vocab) const;
    [[nodiscard]] std::vector<int64_t> genus_codes(const TaxonomyVocab& taxonomy) const;
    [[nodiscard]] std::vector<int64_t> family_codes(const TaxonomyVocab& taxonomy) const;

private:
    NamePool species_names_;
    NamePool genus_names_;
    NamePool family_names_;
    std::vector<int32_t> species_rank_;
    std::vector<int32_t> genus_of_;
    std::vector<int32_t> family_of_;
    std::vector<int32_t> genus_rank_;
    std::vector<int32_t> family_rank_;
    std::vector<int64_t> plots_per_species_;
    bool names_taxa_ = false;
    std::vector<int64_t> offsets_{0};  // plot p owns entries [offsets_[p], offsets_[p + 1])
    std::vector<int32_t> species_;
    std::vector<float> ranking_;
    RepresentationMode representation_ = RepresentationMode::Abundance;
};

// =============================================================================
// Species selection
// =============================================================================

// Indices into plot(p) that `mode` keeps, in the order the selection ranks
// them: idx[0] is the most abundant under Top and the least abundant under
// Bottom; TopBottom lists the top k, most abundant first, then the bottom k,
// least abundant first, skipping any species already taken. Abundance ties
// break by species name, so the order is independent of row order. A plot is
// ranked whatever its size and truncated only when it holds more than k
// species. All returns every index in plot order and ignores k.
std::vector<size_t> selection_indices(const PlotAssemblages& assemblages, int64_t p,
                                      SelectionMode mode, int k);

// Indices a per-plot species BUDGET keeps, in plot order. `budget <= 0` and
// SelectionMode::All both mean "no budget" and return every index. For the
// variable-width encodings (rank_pool, transformer, sparse); hash and embed
// carry their budget in their width and rank with selection_indices.
std::vector<size_t> species_budget_indices(const PlotAssemblages& assemblages, int64_t p,
                                           SelectionMode mode, int budget);

// The values the `selected` entries of plot(p) carry, then normalized: Norm
// divides by their sum, Log1p takes log(1 + value), Raw leaves them.
std::vector<float> normalized_values(const PlotAssemblages& assemblages, int64_t p,
                                     const std::vector<size_t>& selected,
                                     NormalizationMode mode);

// Integer percentile matching numpy's `int(np.percentile(values, q,
// interpolation="linear"))`: linear interpolation between the two bracketing
// order statistics, then truncation toward zero. `values` is partially sorted
// in place (nth_element); empty input returns 0. q is a percentage in [0, 100].
int64_t percentile_linear_trunc(std::vector<int64_t>& values, double q);

// =============================================================================
// SpeciesVocab: species name -> integer code, 0 = unknown
// =============================================================================

class SpeciesVocab {
public:
    // An empty vocabulary: only the reserved "<UNK>" code 0.
    SpeciesVocab();

    // The vocabulary policy of the engine: species recorded in at least
    // `min_count` plots, by descending plot count, ties by name, coded 1..K.
    static SpeciesVocab fit(const PlotAssemblages& assemblages, int min_count = 1);

    // fit() over records grouped by their own plot IDs.
    static SpeciesVocab from_records(const std::vector<SpeciesRecord>& records,
                                     int min_count = 1);

    // A vocabulary fitted elsewhere, as its ordered names: names[c] encodes to
    // c, and names[0] is the reserved "<UNK>".
    static SpeciesVocab from_ordered(std::vector<std::string> names);

    [[nodiscard]] int64_t encode(const std::string& species) const;
    // Number of codes, "<UNK>" included.
    [[nodiscard]] int64_t size() const noexcept { return static_cast<int64_t>(names_.size()); }
    // True when the vocabulary holds no species besides "<UNK>".
    [[nodiscard]] bool empty() const noexcept { return names_.size() <= 1; }
    [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }
    // Name -> code of every species, "<UNK>" excluded.
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& species_to_id() const noexcept {
        return species_to_id_;
    }

private:
    explicit SpeciesVocab(std::vector<std::string> names);

    std::vector<std::string> names_;
    std::unordered_map<std::string, int64_t> species_to_id_;
};

// The species and taxonomy vocabularies an encoding codes a plot with. Fitted
// once per dataset and handed to every encoder, or adopted from a checkpoint.
struct EncodingVocab {
    SpeciesVocab species;
    TaxonomyVocab taxonomy;

    // SpeciesVocab::fit, plus the taxonomy vocabulary of the entries' genus and
    // family names when `with_taxonomy` is set (an empty one otherwise).
    static EncodingVocab fit(const PlotAssemblages& assemblages, int min_count = 1,
                             bool with_taxonomy = true);

    // Whether the taxonomy holds a genus or a family besides "<UNK>". A
    // family-only dataset has taxonomy.
    [[nodiscard]] bool has_taxonomy() const noexcept {
        return taxonomy.n_genera() > 1 || taxonomy.n_families() > 1;
    }
};

// =============================================================================
// Unknown-species statistics and recognition
// =============================================================================

// Per-plot novelty of an assemblage, measured against a vocabulary. An entry
// whose species the vocabulary lacks encodes to the reserved UNK code 0 and is
// unknown:
//
//   fraction[p] = sum(value of p's unknown entries) / sum(value of p's entries)
//   count[p]    = number of unknown species in p
//
// "value" is the entry's abundance, or 1 under presence/absence, so the
// fraction is the unknown share of what the encodings read. Both describe the
// plot's full assemblage, before any selection or species cap. A plot with no
// entries, or whose values sum to <= 0, reports 0 for both.
//
// Non-zero only where the vocabulary was fitted on other data: the
// vocab-reusing loaders (ResolveDataset::from_*_with_schema / _with_vocabs) and
// encoders fitted on one split and applied to another.
struct UnknownSpeciesStats {
    torch::Tensor fraction;  // (n_plots,) float32, in [0, 1]
    torch::Tensor count;     // (n_plots,) float32, whole numbers
};

UnknownSpeciesStats compute_unknown_species_stats(const PlotAssemblages& assemblages,
                                                  const SpeciesVocab& vocab);

// How much of each plot's assemblage a vocabulary recognises, reported beside
// a prediction (not a model input). Same notion of "unknown" as
// compute_unknown_species_stats:
//
//   n_species[p]       = species p records
//   n_recognised[p]    = how many of those the vocabulary holds
//   count_share[p]     = n_recognised / n_species;       NaN when p has none
//   abundance_share[p] = recognised value / total value; NaN when the total
//                        is <= 0, which is 1 - UnknownSpeciesStats::fraction
//                        wherever that is defined
//
// All four are (n_plots,) float32.
struct SpeciesRecognition {
    torch::Tensor n_species;
    torch::Tensor n_recognised;
    torch::Tensor count_share;
    torch::Tensor abundance_share;
};

SpeciesRecognition compute_species_recognition(const PlotAssemblages& assemblages,
                                               const SpeciesVocab& vocab);

// =============================================================================
// Taxonomy
// =============================================================================

// The taxonomy of a species vocabulary: `genus[c]` is the taxonomy-vocab genus
// of the species coded c, 0 (<UNK>) where no entry names one. Both vectors are
// as long as the species vocabulary; code 0 resolves to 0.
struct SpeciesTaxonomy {
    std::vector<int64_t> genus;
    std::vector<int64_t> family;
};

SpeciesTaxonomy resolve_species_taxonomy(const PlotAssemblages& assemblages,
                                         const SpeciesVocab& species_vocab,
                                         const TaxonomyVocab& taxonomy_vocab);

// The fixed genus and family slots of every plot: the plot's distinct genera
// (families) ranked by `aggregation` and coded through `taxonomy`, the first
// `n_slots` of them, 0 past the end. AggregationMode::Abundance ranks a genus
// by the summed value of its species in the plot, Count by how many of the
// plot's species belong to it; ties break by name. The one taxonomy block of
// the fixed-width encodings (hash, sparse, embed).
struct TaxonomySlots {
    torch::Tensor genus_ids;   // (n_plots, n_slots) int64
    torch::Tensor family_ids;  // (n_plots, n_slots) int64
};

TaxonomySlots encode_taxonomy_slots(const PlotAssemblages& assemblages,
                                    const TaxonomyVocab& taxonomy,
                                    AggregationMode aggregation,
                                    int n_slots);

// =============================================================================
// Encoders over a shared vocabulary
// =============================================================================

// The vocabulary half the rank-pool and embedding encoders share: fit or adopt
// an EncodingVocab, report its sizes.
class SpeciesVocabEncoder {
public:
    // Fit the vocabulary on records grouped by their own plot IDs.
    void fit(const std::vector<SpeciesRecord>& records);
    void fit(const PlotAssemblages& assemblages);
    // Adopt a vocabulary fitted elsewhere (a training split, a checkpoint).
    void set_vocab(EncodingVocab vocab);

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] const EncodingVocab& vocab() const noexcept { return vocab_; }
    [[nodiscard]] const SpeciesVocab& species_vocab() const noexcept { return vocab_.species; }
    [[nodiscard]] const TaxonomyVocab& taxonomy_vocab() const noexcept { return vocab_.taxonomy; }
    [[nodiscard]] int64_t n_species_vocab() const noexcept { return vocab_.species.size(); }
    [[nodiscard]] int64_t n_genera() const noexcept { return vocab_.taxonomy.n_genera(); }
    [[nodiscard]] int64_t n_families() const noexcept { return vocab_.taxonomy.n_families(); }

protected:
    explicit SpeciesVocabEncoder(int min_frequency) : min_frequency_(min_frequency) {}
    void require_vocab(const char* encoder) const;

    EncodingVocab vocab_;
    int min_frequency_;
    bool fitted_ = false;
};

// =============================================================================
// RankPoolEncoder: every species of a plot, weighted, padded
// =============================================================================

struct RankPoolEncodedData {
    torch::Tensor species_ids;  // (n_plots, max_species) int64
    torch::Tensor genus_ids;    // (n_plots, max_species) int64
    torch::Tensor family_ids;   // (n_plots, max_species) int64
    torch::Tensor weights;      // (n_plots, max_species) float32
    torch::Tensor mask;         // (n_plots, max_species) bool
    torch::Tensor has_cover;    // (n_plots,) float32
};

class RankPoolEncoder : public SpeciesVocabEncoder {
public:
    // `selection` + `species_budget` are the per-plot species budget
    // (DatasetConfig::selection / species_budget): each plot is narrowed to
    // `species_budget` species at the requested end of the abundance ranking
    // before weights, taxonomy IDs and the padded width are computed. The
    // defaults (All / 0) encode every species. The species cap passed to
    // transform() then slices what survives, in plot order.
    explicit RankPoolEncoder(PoolWeighting weighting = PoolWeighting::Log1p,
                             int min_frequency = 1,
                             SelectionMode selection = SelectionMode::All,
                             int species_budget = 0);

    // species_cap mirrors DatasetConfig::pool_species_cap:
    //   0  -> no cap (pad to the widest plot).
    //   -x -> the (100 - x)th percentile of per-plot species counts, x in
    //         1..99 (-1 = p99, -5 = p95); -100 or below throws.
    //   >0 -> manual cap; each plot keeps its first `cap` species in plot order.
    // A cap that bites prints a one-line summary of the padding it saves.
    //
    // has_cover is 1 for every plot when an abundance column was mapped
    // (RoleMapping::has_abundance) and the assemblages carry abundances, 0
    // under presence/absence or without an abundance column: a column-presence
    // flag, never inferred from the values.
    [[nodiscard]] RankPoolEncodedData transform(const PlotAssemblages& assemblages,
                                                int species_cap = 0,
                                                bool has_abundance_column = false) const;

    // transform() over `records` grouped in `plot_ids` order.
    [[nodiscard]] RankPoolEncodedData transform(
        const std::vector<SpeciesRecord>& records,
        const std::vector<std::string>& plot_ids,
        int species_cap = 0,
        bool has_abundance_column = false,
        RepresentationMode representation = RepresentationMode::Abundance) const;

    [[nodiscard]] PoolWeighting weighting() const noexcept { return weighting_; }
    [[nodiscard]] SelectionMode selection() const noexcept { return selection_; }
    [[nodiscard]] int species_budget() const noexcept { return species_budget_; }

private:
    PoolWeighting weighting_;
    SelectionMode selection_;
    int species_budget_;
};

// =============================================================================
// EmbeddingEncoder: fixed ranked species slots plus fixed taxonomy slots
// =============================================================================

struct EmbeddingEncodedData {
    torch::Tensor species_ids;  // (n_plots, top_k_species) int64
    torch::Tensor genus_ids;    // (n_plots, top_k_taxonomy) int64
    torch::Tensor family_ids;   // (n_plots, top_k_taxonomy) int64
};

class EmbeddingEncoder : public SpeciesVocabEncoder {
public:
    // `selection` decides which species fill the top_k_species slots, in the
    // order selection_indices ranks them: the most abundant first (Top), the
    // least abundant first (Bottom), or ceil(k/2) from each end (TopBottom).
    // SelectionMode::All has no fixed-width encoding and transform() rejects
    // it. `aggregation` ranks the taxonomy slots (encode_taxonomy_slots).
    explicit EmbeddingEncoder(int top_k_species = 10, int top_k_taxonomy = 3,
                              SelectionMode selection = SelectionMode::Top,
                              AggregationMode aggregation = AggregationMode::Abundance);

    [[nodiscard]] EmbeddingEncodedData transform(const PlotAssemblages& assemblages) const;

    // transform() over `records` grouped in `plot_ids` order.
    [[nodiscard]] EmbeddingEncodedData transform(
        const std::vector<SpeciesRecord>& records,
        const std::vector<std::string>& plot_ids,
        RepresentationMode representation = RepresentationMode::Abundance) const;

    [[nodiscard]] SelectionMode selection() const noexcept { return selection_; }
    [[nodiscard]] AggregationMode aggregation() const noexcept { return aggregation_; }

private:
    int top_k_species_;
    int top_k_taxonomy_;
    SelectionMode selection_;
    AggregationMode aggregation_;
};

} // namespace resolve
