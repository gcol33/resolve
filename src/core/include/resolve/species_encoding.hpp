#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace resolve {

// MurmurHash3 finalizer for feature hashing
uint32_t murmur_hash(const std::string& key, uint32_t seed = 0);

// Resolved feature-hash slot for one species: signed-feature-hashing bucket
// and sign. The scheme mirrors the CUDA kernel (cuda/kernels.cu) exactly so the
// CPU-aggregated embedding matches the GPU kernel output for the same data.
struct HashBucketSign {
    int bucket;
    float sign;
};

// Compute the (bucket, sign) for a species under the canonical feature-hash
// scheme shared by the CPU and CUDA paths. Single definition of the contract.
HashBucketSign feature_hash_bucket_sign(const std::string& species, int hash_dim);

// Integer percentile matching numpy's `int(np.percentile(values, q,
// interpolation="linear"))`: linear interpolation between the two bracketing
// order statistics, then truncation toward zero. `values` is partially sorted
// in place (nth_element); empty input returns 0. q is a percentage in [0, 100].
// Used by RankPoolEncoder::transform for the auto (p99) species cap, where the
// previous floor-rank index (no interpolation) over-truncated skewed per-plot
// length distributions relative to the POC's int(np.percentile(...)).
int64_t percentile_linear_trunc(std::vector<int64_t>& values, double q);

// Top-k distinct names by descending aggregated abundance, ties broken by name
// ascending (deterministic; the previous ad-hoc sorts relied on unordered_map
// iteration order for ties). Single source for embed-mode taxonomy slot
// selection, shared by the dataset loader (encode_species) and EmbeddingEncoder
// so both produce identical genus/family slots.
std::vector<std::string> topk_by_abundance(
    const std::unordered_map<std::string, float>& name_to_abundance, int k);

// Feature hashing for species
void hash_species(
    const std::vector<std::pair<std::string, float>>& species_abundances,
    float* embedding,
    int hash_dim
);

// Select top-k species by abundance
std::vector<std::pair<std::string, float>> select_top_k(
    std::vector<std::pair<std::string, float>> species,
    int k
);

// Select bottom-k species by abundance
std::vector<std::pair<std::string, float>> select_bottom_k(
    std::vector<std::pair<std::string, float>> species,
    int k
);

// Apply selection mode to species list
std::vector<std::pair<std::string, float>> apply_selection(
    std::vector<std::pair<std::string, float>> species,
    SelectionMode mode,
    int k
);

// Apply normalization to abundance values
void apply_normalization(
    std::vector<std::pair<std::string, float>>& species,
    NormalizationMode mode
);

// =============================================================================
// SpeciesVocab — maps species names to integer IDs (1-indexed, 0 = unknown)
// =============================================================================

class SpeciesVocab {
public:
    SpeciesVocab() = default;

    // Build from species records, filtering by min_count
    static SpeciesVocab from_records(
        const std::vector<SpeciesRecord>& records, int min_count = 1);

    // Build from a pre-fit string->id map (1-indexed; UNK at code 0 is
    // implicit). Used when reusing a training-set vocab on a held-out test
    // set (see RankPoolEncoder::set_vocabs and ResolveDataset::from_csv_with_schema).
    static SpeciesVocab from_map(std::unordered_map<std::string, int64_t> species_to_id);

    [[nodiscard]] int64_t encode(const std::string& species) const;
    [[nodiscard]] int64_t size() const noexcept { return static_cast<int64_t>(species_to_id_.size()) + 1; }
    [[nodiscard]] bool empty() const noexcept { return species_to_id_.empty(); }
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& species_to_id() const noexcept { return species_to_id_; }

private:
    std::unordered_map<std::string, int64_t> species_to_id_;  // 1-indexed, 0=unknown
};

// =============================================================================
// RankPoolEncoder — variable-length species lists with weighted pooling
// =============================================================================

struct RankPoolEncodedData {
    torch::Tensor species_ids;       // (n_plots, max_species) int64
    torch::Tensor genus_ids;         // (n_plots, max_species) int64
    torch::Tensor family_ids;        // (n_plots, max_species) int64
    torch::Tensor weights;           // (n_plots, max_species) float32
    torch::Tensor mask;              // (n_plots, max_species) bool
    torch::Tensor has_cover;         // (n_plots,) float32
    torch::Tensor unknown_fraction;  // (n_plots,) float32
    int64_t n_species_vocab = 0;
    int64_t n_genera_vocab = 0;
    int64_t n_families_vocab = 0;
};

enum class PoolWeighting { Binary, Abundance, Log1p, Norm, Rank };

class RankPoolEncoder {
public:
    explicit RankPoolEncoder(PoolWeighting weighting = PoolWeighting::Log1p, int min_frequency = 1);

    void fit(const std::vector<SpeciesRecord>& records);

    // species_cap mirrors DatasetConfig::pool_species_cap exactly:
    //   0  -> no cap (default; pad to global per-plot max).
    //   -1 -> auto p99 (compute 99th percentile of per-plot species counts).
    //   >0 -> manual cap; per-plot lists are truncated to the first `cap`
    //         records in original CSV order (matching the POC's `a[:cap]`).
    // When the cap kicks in we print a one-line summary so users see the
    // drop ("rank_pool: capping species at p99=X (max=Y, saves Z% padding)").
    // has_abundance_column mirrors the POC's `roles.has_abundance`: has_cover is
    // 1 for every plot when an abundance/cover column was mapped, else 0. It is
    // a column-presence flag, NOT inferred from values (a plot whose covers are
    // all exactly 1.0 still has cover data), so pass whether the dataset mapped
    // an abundance column rather than letting the encoder guess.
    [[nodiscard]] RankPoolEncodedData transform(
        const std::vector<SpeciesRecord>& records,
        const std::vector<std::string>& plot_ids,
        int species_cap = 0,
        bool has_abundance_column = false) const;

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] int64_t n_species_vocab() const noexcept { return species_vocab_.size(); }
    [[nodiscard]] int64_t n_genera_vocab() const noexcept { return taxonomy_vocab_.n_genera(); }
    [[nodiscard]] int64_t n_families_vocab() const noexcept { return taxonomy_vocab_.n_families(); }
    [[nodiscard]] const SpeciesVocab& species_vocab() const noexcept { return species_vocab_; }
    [[nodiscard]] const TaxonomyVocab& taxonomy_vocab() const noexcept { return taxonomy_vocab_; }

    // Replace the fitted species + taxonomy vocabs with externally-supplied
    // ones. Requires fit() to have been called first (throws otherwise);
    // leaves species_to_genus_/species_to_family_ as whatever fit() last set
    // so transform() can still look up each species's genus/family string
    // and resolve it through the supplied taxonomy_vocab. Used by
    // ResolveDataset::from_csv_with_schema for cross-split evaluation.
    // Does NOT toggle fitted_ — that flag tracks "fit() ran on real records"
    // and must not be set by a vocab swap (an empty external vocab would
    // otherwise leave the encoder silently fitted-with-nothing).
    void set_vocabs(SpeciesVocab species_vocab, TaxonomyVocab taxonomy_vocab);

private:
    SpeciesVocab species_vocab_;
    TaxonomyVocab taxonomy_vocab_;
    std::unordered_map<std::string, std::string> species_to_genus_;
    std::unordered_map<std::string, std::string> species_to_family_;
    PoolWeighting weighting_;
    int min_frequency_;
    bool fitted_ = false;
};

// =============================================================================
// EmbeddingEncoder — fixed-size top-k species/taxonomy ID encoding
// =============================================================================

struct EmbeddingEncodedData {
    torch::Tensor species_ids;       // (n_plots, top_k_species) int64
    torch::Tensor genus_ids;         // (n_plots, top_k_taxonomy) int64
    torch::Tensor family_ids;        // (n_plots, top_k_taxonomy) int64
    torch::Tensor unknown_fraction;  // (n_plots,) float32
    int64_t n_species_vocab = 0;
    int64_t n_genera_vocab = 0;
    int64_t n_families_vocab = 0;
};

class EmbeddingEncoder {
public:
    explicit EmbeddingEncoder(
        int top_k_species = 10, int top_k_taxonomy = 3,
        SelectionMode selection = SelectionMode::Top);

    void fit(const std::vector<SpeciesRecord>& records);

    [[nodiscard]] EmbeddingEncodedData transform(
        const std::vector<SpeciesRecord>& records,
        const std::vector<std::string>& plot_ids) const;

    [[nodiscard]] bool is_fitted() const noexcept { return fitted_; }
    [[nodiscard]] int64_t n_species_vocab() const noexcept { return species_vocab_.size(); }
    [[nodiscard]] int64_t n_genera_vocab() const noexcept { return taxonomy_vocab_.n_genera(); }
    [[nodiscard]] int64_t n_families_vocab() const noexcept { return taxonomy_vocab_.n_families(); }

private:
    SpeciesVocab species_vocab_;
    TaxonomyVocab taxonomy_vocab_;
    std::unordered_map<std::string, std::string> species_to_genus_;
    std::unordered_map<std::string, std::string> species_to_family_;
    int top_k_species_;
    int top_k_taxonomy_;
    SelectionMode selection_;
    bool fitted_ = false;
};

} // namespace resolve
