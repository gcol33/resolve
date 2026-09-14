#pragma once

#include "resolve/scalers.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <utility>
#include <vector>

namespace resolve {

// The per-plot numeric inputs the model reads as one continuous block, before
// the categorical embeddings are fused onto it. A missing coordinate or
// covariate is NaN; every tensor is optional (undefined or empty when the data
// carries no such input).
struct ContinuousInputs {
    torch::Tensor coordinates;       // (n, 2)
    torch::Tensor covariates;        // (n, n_covariates)
    torch::Tensor unknown_fraction;  // (n,)
    torch::Tensor unknown_count;     // (n,)
    torch::Tensor hash_embedding;    // (n, hash_dim); appended last when present
};

// Assemble the continuous block, column order:
//
//   coordinates (2) | coordinate flag (1) | covariates (k) | covariate flags (k)
//   | unknown fraction (1) | unknown count (1) | hash embedding (hash_dim)
//
// Under MissingValuePolicy::Indicate the value columns keep their NaN, which
// fit_continuous_scalers / standardize_continuous fill, and each flag column is
// 1.0 where its value is missing (the coordinate flag covers the pair). Under
// Zero there are no flag columns and a NaN is read as 0.0. The flag columns are
// present exactly when their value columns are, so the width agrees with
// ResolveSchema::missing_flag_width() for data built from that schema.
[[nodiscard]] torch::Tensor assemble_continuous(const ContinuousInputs& inputs,
                                                MissingValuePolicy policy,
                                                int64_t n_rows);

// Fit the fill, mean and scale of every column on `fitting_rows`, a slice of an
// assembled block. The fill is the mean of a column's recorded (non-NaN) values,
// 0.0 when it has none; the mean and scale are taken after filling.
void fit_continuous_scalers(Scalers& scalers, const torch::Tensor& fitting_rows);

// Fill missing cells and standardise with fitted scalers. A scaler without a
// fill (a checkpoint written before missing values were filled) reads a NaN as
// 0.0, which is what such a model was trained on.
[[nodiscard]] torch::Tensor standardize_continuous(const torch::Tensor& block,
                                                   const Scalers& scalers);

// Undo standardize_continuous: rescale, then mark every cell whose flag says it
// was missing as NaN again, so a refit on a different set of rows (a
// cross-validation fold) fills from that fold's recorded values alone.
[[nodiscard]] torch::Tensor unstandardize_continuous(const torch::Tensor& block,
                                                     const Scalers& scalers,
                                                     const ResolveSchema& schema);

// For each flag column of a schema's block, the value columns it marks and its
// own index. Empty under Zero.
[[nodiscard]] std::vector<std::pair<std::vector<int64_t>, int64_t>>
missing_flag_columns(const ResolveSchema& schema);

}  // namespace resolve
