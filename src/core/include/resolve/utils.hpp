#pragma once

#include <torch/torch.h>
#include <vector>

namespace resolve {

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
