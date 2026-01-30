#include "resolve/data_prep.hpp"
#include "resolve/utils.hpp"
#include <algorithm>
#include <numeric>

namespace resolve {

std::pair<torch::Tensor, torch::Tensor> create_split_indices(
    int64_t n_samples,
    float test_size,
    int seed
) {
    // Create indices and shuffle
    std::vector<int64_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Split indices
    int64_t n_test = static_cast<int64_t>(n_samples * test_size);
    int64_t n_train = n_samples - n_test;

    auto train_idx = torch::tensor(std::vector<int64_t>(indices.begin(), indices.begin() + n_train));
    auto test_idx = torch::tensor(std::vector<int64_t>(indices.begin() + n_train, indices.end()));

    return {train_idx, test_idx};
}

torch::Tensor build_continuous_features(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor unknown_fraction,
    torch::Tensor unknown_count,
    torch::Tensor hash_embedding,
    SpeciesEncodingMode encoding_mode,
    bool uses_explicit_vector
) {
    // Determine n_samples from first defined tensor
    int64_t n_samples = 0;
    if (coordinates.defined() && coordinates.numel() > 0) {
        n_samples = coordinates.size(0);
    } else if (hash_embedding.defined() && hash_embedding.numel() > 0) {
        n_samples = hash_embedding.size(0);
    } else if (covariates.defined() && covariates.numel() > 0) {
        n_samples = covariates.size(0);
    }

    if (n_samples == 0) {
        return torch::zeros({0, 0}, torch::kFloat32);
    }

    // Build continuous features
    std::vector<torch::Tensor> continuous_parts;
    push_if_defined(continuous_parts, coordinates);
    push_if_defined(continuous_parts, covariates);
    push_if_defined(continuous_parts, unknown_fraction, 1);
    if (unknown_count.defined() && unknown_count.numel() > 0) {
        continuous_parts.push_back(unknown_count.to(torch::kFloat32).unsqueeze(1));
    }

    // For hash mode, include hash embedding in continuous
    if (encoding_mode == SpeciesEncodingMode::Hash && !uses_explicit_vector) {
        push_if_defined(continuous_parts, hash_embedding);
    }

    if (!continuous_parts.empty()) {
        return torch::cat(continuous_parts, /*dim=*/1);
    } else {
        return torch::zeros({n_samples, 0}, torch::kFloat32);
    }
}

StandardizationResult compute_standardization(
    torch::Tensor data,
    torch::Tensor train_indices
) {
    StandardizationResult result;

    if (data.size(1) == 0) {
        result.scaled_data = data;
        result.mean = torch::Tensor();
        result.scale = torch::Tensor();
        return result;
    }

    auto train_data = data.index_select(0, train_indices);
    result.mean = train_data.mean(0);
    result.scale = train_data.std(0) + 1e-8f;

    result.scaled_data = (data - result.mean) / result.scale;

    return result;
}

std::pair<torch::Tensor, torch::Tensor> split_tensor(
    torch::Tensor tensor,
    torch::Tensor train_idx,
    torch::Tensor test_idx
) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return {torch::Tensor(), torch::Tensor()};
    }
    return {
        tensor.index_select(0, train_idx),
        tensor.index_select(0, test_idx)
    };
}

} // namespace resolve
