#pragma once

#include "resolve/types.hpp"  // TabMAggregation

#include <torch/torch.h>
#include <vector>

namespace resolve {

// =============================================================================
// BatchEnsemble Linear Layer (TabM core)
// =============================================================================

// BatchEnsemble: N implicit ensemble members share a base weight W,
// each perturbed by rank-1 factors: y_i = (r_i * x) @ W.T * s_i + b
// Effectively runs N MLPs at the cost of ~1 MLP + O(N*d) parameters.
//
// Reference: Wen et al., "BatchEnsemble: An Alternative Approach to
// Efficient Ensemble and Lifelong Learning" (ICLR 2020)
// Applied to tabular data in: Gorishniy et al., "TabM" (2024)
class BatchEnsembleLinearImpl : public torch::nn::Module {
public:
    BatchEnsembleLinearImpl(
        int64_t in_features,
        int64_t out_features,
        int n_ensembles,
        bool bias = true
    );

    // Forward pass
    // x: (batch, in_features) or (batch, n_ensembles, in_features)
    // Returns: (batch, n_ensembles, out_features)
    torch::Tensor forward(torch::Tensor x);

    [[nodiscard]] int64_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] int64_t out_features() const noexcept { return out_features_; }
    [[nodiscard]] int n_ensembles() const noexcept { return n_ensembles_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    int n_ensembles_;

    // Shared base weight: (out_features, in_features)
    torch::nn::Linear base_linear_{nullptr};

    // Per-ensemble rank-1 perturbation factors
    // r: (n_ensembles, in_features) - input scaling
    // s: (n_ensembles, out_features) - output scaling
    torch::Tensor r_;
    torch::Tensor s_;

    // Per-ensemble bias, added AFTER the output scaling s so it is a free
    // parameter (BatchEnsemble: y = s (W (r x)) + b). Undefined when bias=false.
    // The base_linear carries no bias of its own; folding it in would scale the
    // shared bias by s, coupling it to the per-member factor.
    torch::Tensor ensemble_bias_;
    bool has_bias_ = false;
};

TORCH_MODULE(BatchEnsembleLinear);

// =============================================================================
// TabM Encoder: MLP with BatchEnsemble layers
// =============================================================================

// TabM replaces standard Linear layers in the MLP backbone with
// BatchEnsembleLinear layers. During forward pass, N ensemble members
// run in parallel. Predictions are aggregated (mean/median) for output.
//
// It is one of the encoder tails (EncoderTail in encoder.hpp), open to every
// species encoder, not an EncoderArchitecture of its own.
class TabMEncoderImpl : public torch::nn::Module {
public:
    TabMEncoderImpl(
        int64_t input_dim,
        const std::vector<int64_t>& hidden_dims,
        int n_ensembles = 16,
        float dropout = 0.0f,
        TabMAggregation aggregation = TabMAggregation::Mean
    );

    // Forward pass
    // x: (batch, input_dim)
    // Returns: (batch, output_dim) - aggregated across ensembles
    torch::Tensor forward(torch::Tensor x);

    // Forward without aggregation (for analysis)
    // Returns: (batch, n_ensembles, output_dim)
    torch::Tensor forward_all(torch::Tensor x);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int n_ensembles() const noexcept { return n_ensembles_; }

private:
    int64_t output_dim_;
    int n_ensembles_;
    TabMAggregation aggregation_;

    // Stack of BatchEnsemble layers with activations
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::ModuleList norms_{nullptr};
    torch::nn::ModuleList dropouts_{nullptr};
};

TORCH_MODULE(TabMEncoder);

} // namespace resolve
