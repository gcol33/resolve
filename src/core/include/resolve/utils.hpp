#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace resolve {

// Half-cosine ramp: 1 at progress 0, 0 at progress 1, clamped outside [0, 1].
// The shared shape behind the LR cosine-annealing schedule (Trainer) and the
// JEPA EMA-decay schedule (JEPAPretrainer). std::numbers keeps the arithmetic
// in float end to end; M_PI is a double, so the older per-file expression
// promoted the whole schedule to double and narrowed back on assignment.
inline float cosine_ramp(float progress) noexcept {
    progress = std::clamp(progress, 0.0f, 1.0f);
    return 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * progress));
}

// Push tensor to vector if it's defined and non-empty
inline void push_if_defined(std::vector<torch::Tensor>& parts, const torch::Tensor& t) {
    if (t.defined() && t.numel() > 0) {
        parts.push_back(t);
    }
}

// Overload for tensor that needs unsqueeze
inline void push_if_defined(std::vector<torch::Tensor>& parts, const torch::Tensor& t, int unsqueeze_dim) {
    if (t.defined() && t.numel() > 0) {
        parts.push_back(t.unsqueeze(unsqueeze_dim));
    }
}

// Move tensor to device if defined
inline torch::Tensor to_device_if_defined(const torch::Tensor& t, torch::Device device) {
    return t.defined() ? t.to(device) : t;
}

// Select batch from tensor and move to device (returns empty tensor if input not defined)
inline torch::Tensor select_batch(const torch::Tensor& t, const torch::Tensor& idx, torch::Device device) {
    return t.defined() ? t.index_select(0, idx).to(device) : torch::Tensor{};
}

} // namespace resolve
