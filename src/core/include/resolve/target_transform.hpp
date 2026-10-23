#pragma once

// The map from a regression target's model space back to its original units.
//
// Training stores a regression target transformed (log1p for TransformType::
// Log1p) and then standardised, ((x - mean) / scale), and the model predicts in
// that space. Every place that reports, scores or returns a regression value in
// original units -- the phased loss's SMAPE and band terms, the metrics, the
// residuals, the baselines, Trainer::predict, the Predictor and TaskHead --
// goes through to_original_units, so they agree on one inverse.

#include "resolve/types.hpp"

#include <torch/torch.h>

namespace resolve {

// `values` in original units: unscaled with (mean, scale) when both are
// defined, then the transform inverted. The log1p inverse clamps its exponent
// to [kExpClampMin, kExpClampMax] before expm1, so one wild prediction cannot
// overflow to inf; a recorded target is a finite log1p value well inside the
// clamp, so the clamp leaves it unchanged.
[[nodiscard]] inline torch::Tensor to_original_units(
    torch::Tensor values,
    TransformType transform,
    const torch::Tensor& mean = {},
    const torch::Tensor& scale = {}
) {
    if (mean.defined() && scale.defined()) {
        values = values * scale.to(values.device()) + mean.to(values.device());
    }
    if (transform == TransformType::Log1p) {
        values = torch::expm1(torch::clamp(values, kExpClampMin, kExpClampMax));
    }
    return values;
}

}  // namespace resolve
