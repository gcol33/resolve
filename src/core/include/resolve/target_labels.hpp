#pragma once

// Which plots carry a label for a target.
//
// A target whose missing labels are masked (TargetMissingPolicy::Mask) keeps
// its unlabelled plots in the dataset with a sentinel in the target tensor:
//
//   - classification: class code kUnlabelledClass (-1), which no class takes,
//     since every class code is >= 0;
//   - regression: NaN, which no recorded value takes, since the loaders reject
//     a non-finite value as missing.
//
// The sentinel is the only record of which plots are unlabelled. Every split,
// fold and batch is an index_select of the target tensor, so the record travels
// with the rows and cannot fall out of step with them. Everything that fits or
// scores a target -- scalers, metrics, baselines, calibration, residuals,
// test-fold predictions -- reads the labelled rows through labelled_rows(); the
// per-batch loss weights its rows by labelled_mask() instead, so a training
// batch never waits on the host to count them.
//
// A Drop target never holds a sentinel: its unlabelled plots are not in the
// dataset. labelled_rows() therefore reports every row of a Drop target as
// labelled without scanning it, which keeps the Drop path the computation it
// has always been.

#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace resolve {

// Class code a masked classification target holds for an unlabelled plot.
inline constexpr int64_t kUnlabelledClass = -1;

// (n,) bool: true where `target` carries a label. Classification: code >= 0.
// Regression: a finite value. Reads the tensor whatever the target's policy.
[[nodiscard]] torch::Tensor labelled_mask(const torch::Tensor& target, TaskType task);

// The rows of one target's tensor that carry a label.
struct LabelledRows {
    // Positions (int64, on the target's device) of the labelled rows; undefined
    // when every row is labelled, so select() is then the identity and no copy
    // is made.
    torch::Tensor index;
    int64_t n_labelled = 0;
    int64_t n_rows = 0;

    [[nodiscard]] bool all_labelled() const noexcept { return n_labelled == n_rows; }

    // `rows` restricted to the labelled rows along dim 0. `rows` has n_rows
    // rows: the target itself, or a prediction aligned with it.
    [[nodiscard]] torch::Tensor select(const torch::Tensor& rows) const;

    // (n_labelled,) int64 on CPU: the position of each labelled row.
    [[nodiscard]] torch::Tensor positions() const;
};

// Name of the per-target count of labelled plots the trainer reports beside
// the metrics of a masked target.
inline constexpr const char* kLabelledMetric = "n_labelled";

// The labelled rows of `target` for `cfg`. A Drop target is labelled on every
// row (see above); a Mask target is scanned with labelled_mask().
[[nodiscard]] LabelledRows labelled_rows(const TargetConfig& cfg, const torch::Tensor& target);

}  // namespace resolve
