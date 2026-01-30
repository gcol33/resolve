#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>
#include <vector>
#include <random>

namespace resolve {

// Create shuffled train/test split indices
std::pair<torch::Tensor, torch::Tensor> create_split_indices(
    int64_t n_samples,
    float test_size,
    int seed
);

// Build continuous feature tensor from components
torch::Tensor build_continuous_features(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor unknown_fraction,
    torch::Tensor unknown_count,
    torch::Tensor hash_embedding,
    SpeciesEncodingMode encoding_mode,
    bool uses_explicit_vector
);

// Compute and apply standardization scalers
struct StandardizationResult {
    torch::Tensor scaled_data;
    torch::Tensor mean;
    torch::Tensor scale;
};

StandardizationResult compute_standardization(
    torch::Tensor data,
    torch::Tensor train_indices
);

// Split tensor by indices
std::pair<torch::Tensor, torch::Tensor> split_tensor(
    torch::Tensor tensor,
    torch::Tensor train_idx,
    torch::Tensor test_idx
);

} // namespace resolve
