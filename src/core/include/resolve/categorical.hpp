#pragma once

// Categorical-covariate support for the RESOLVE C++ engine.
//
// This module provides:
//
//   - CategoricalVocab  : per-column string -> int64 code map, with reserved
//                         code 0 for unknown / missing. Fit from raw strings
//                         (auto-factorize sorted-unique non-NA), encode at
//                         CSV-load time, serialize/deserialize from a
//                         torch::serialize archive.
//
//   - CategoricalEmbedderImpl : torch::nn::Module holding one nn::Embedding
//                               table per categorical column. forward()
//                               takes (B, N) int64 ids and returns
//                               (B, N * embed_dim) float32, the concatenated
//                               embeddings ready to fuse into the latent
//                               vector before the MLP head.
//
// Design notes:
//   * Code 0 is always reserved for <UNK>. Vocab sizes reported to the
//     model are K + 1 where K is the number of distinct non-NA strings.
//   * NA-string detection goes through is_na_string (categorical.cpp), the
//     same matcher the target and covariate loaders use, so a raw cell is
//     classified identically whatever role its column carries.
//   * Embedding dim is shared across all columns. It could be made per-column
//     if that becomes a tuning lever.
//   * No public copy/move ctors on CategoricalEmbedder beyond what
//     torch::nn::Module gives us — clone/serialize via state_dict like any
//     other Module.

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

// Strings treated as NA when factorizing a categorical column. The single
// matcher every loader shares, so one raw cell is classified the same way
// whatever role its column carries.
[[nodiscard]] bool is_na_string(const std::string& s) noexcept;

// =============================================================================
// CategoricalVocab
// =============================================================================
//
// Holds per-column string -> int64 code maps. Code 0 is reserved for
// unknown/NA. Codes 1..K are assigned by sorting the unique non-NA values
// lexicographically (matches Python's `sorted(set(...))` factorize path).
//
// Typical lifecycle:
//
//   CategoricalVocab vocab;
//   vocab.fit(column_names, raw_values_by_column);   // build all maps
//   auto tensor = vocab.encode(raw_values_by_column); // (n_rows, n_columns)
//
// At load time:
//
//   CategoricalVocab vocab = CategoricalVocab::load(archive);
//
// At save time:
//
//   vocab.save(archive);
//
// `vocab_size(col)` returns K + 1 (the size to use for the column's
// nn::Embedding table — includes the UNK slot).
class CategoricalVocab {
public:
    CategoricalVocab() = default;

    // Fit one column at a time. Builds the {string: code} map (code 0 = UNK).
    // raw values are scanned once for unique non-NA values; sorted; assigned
    // codes 1..K.
    void fit_column(const std::string& column_name,
                    const std::vector<std::string>& raw_values);

    // Fit all columns from a parallel layout: column_names[i] is fit from
    // raw_values_per_column[i]. raw_values_per_column[i].size() == n_rows
    // (same n_rows for every column).
    void fit(const std::vector<std::string>& column_names,
             const std::vector<std::vector<std::string>>& raw_values_per_column);

    // Encode a single raw value for a fitted column. Returns 0 if the column
    // is unknown, or the value is NA, or the value is not in the column's
    // vocab. Never throws.
    [[nodiscard]] int64_t encode(const std::string& column_name,
                                 const std::string& raw_value) const noexcept;

    // Encode all columns into a single (n_rows, n_columns) int64 tensor.
    // column_names must be a subset of (or equal to) the fitted columns and
    // the order defines the output column order. raw_values_per_column has
    // shape parallel to column_names; every inner vector must be the same
    // length (= n_rows).
    [[nodiscard]] torch::Tensor encode_batch(
        const std::vector<std::string>& column_names,
        const std::vector<std::vector<std::string>>& raw_values_per_column) const;

    // Vocab size (including reserved UNK slot at code 0). Returns 1 if the
    // column has been fit with no non-NA values, or 1 if the column is not
    // known to the vocab (just the UNK slot).
    [[nodiscard]] int64_t vocab_size(const std::string& column_name) const noexcept;

    // True if `fit_column` has been called for `column_name`.
    [[nodiscard]] bool has_column(const std::string& column_name) const noexcept;

    // Ordered list of fitted column names (in fit order).
    [[nodiscard]] const std::vector<std::string>& column_names() const noexcept {
        return column_order_;
    }

    // Parallel list of vocab sizes (K + 1 per column) in column_names() order.
    [[nodiscard]] std::vector<int64_t> vocab_sizes() const;

    // Serialize the entire vocab to `archive` under `prefix`. Layout mirrors
    // TaxonomyVocab (length-prefixed UInt8 byte arrays) so it survives the
    // LibTorch archive roundtrip on Windows + Linux.
    void save(torch::serialize::OutputArchive& archive,
              const std::string& prefix = "categorical_") const;

    // Inverse of save. The same prefix must be used. Returns an empty
    // vocab if no entries were saved (back-compat with checkpoints written
    // before categorical support landed).
    [[nodiscard]] static CategoricalVocab load(
        torch::serialize::InputArchive& archive,
        const std::string& prefix = "categorical_");

    // Direct accessor for one column's full string->code map. Mainly used
    // by tests + downstream tooling that wants to expose the mapping to
    // users.
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& column_map(
        const std::string& column_name) const;

    // Install a column's string -> code map verbatim, appending the column to
    // the fit order if it is new. The counterpart to column_map(), for
    // RESTORING a persisted vocabulary rather than fitting one: the codes come
    // from the checkpoint and must be reproduced exactly, whereas fit_column
    // re-derives them from the sorted unique values. Used by the C-ABI
    // ExternalVocabs carrier (issue #102). Code 0 stays reserved for UNK and
    // must not appear in `map`.
    void set_column_map(const std::string& column_name,
                        const std::unordered_map<std::string, int64_t>& map);

private:
    // Insertion order of fitted columns. Defines the canonical column order
    // for `encode_batch` and serialization.
    std::vector<std::string> column_order_;

    // column_name -> (string -> code). Code 0 is implicit (UNK) and not
    // stored in this map.
    std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
        maps_;
};

// =============================================================================
// CategoricalEmbedder
// =============================================================================
//
// One nn::Embedding table per categorical column, all with the same
// embedding dimension. Forward:
//
//   ids   :  (B, N)        int64,   values in [0, vocab_sizes[i])
//   out   :  (B, N*D)      float32, table_i(ids[:, i]) concatenated along
//                                    the feature axis.
//
// Construct with empty vocab_sizes vector when there are no categoricals;
// in that case `output_dim() == 0` and `forward()` returns an
// (B, 0) tensor matching the convention used elsewhere in the encoder.
class CategoricalEmbedderImpl : public torch::nn::Module {
public:
    // vocab_sizes[i] = size of column i's embedding table (must include
    // the reserved UNK slot, i.e. K + 1). embed_dim is the per-column
    // embedding dimension (shared across columns).
    CategoricalEmbedderImpl(const std::vector<int64_t>& vocab_sizes,
                            int64_t embed_dim);

    // Forward pass. Returns a (B, n_columns * embed_dim) float tensor.
    // If n_columns == 0, returns a (B, 0) float tensor placed on the same
    // device as `ids` (or CPU if ids is empty too).
    [[nodiscard]] torch::Tensor forward(torch::Tensor ids);

    // Total output dim = n_columns * embed_dim.
    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }

    // Number of categorical columns held.
    [[nodiscard]] int64_t n_columns() const noexcept {
        return static_cast<int64_t>(tables_.size());
    }

    // Per-column embedding dim (uniform across columns).
    [[nodiscard]] int64_t embed_dim() const noexcept { return embed_dim_; }

    // Direct accessor for downstream weight-extraction code. Returns
    // a (vocab_size_i, embed_dim) tensor (detached, copy).
    [[nodiscard]] torch::Tensor get_table_weights(int64_t column_idx) const;

private:
    int64_t embed_dim_;
    int64_t output_dim_;
    std::vector<torch::nn::Embedding> tables_;  // registered as submodules
};

TORCH_MODULE(CategoricalEmbedder);

}  // namespace resolve
