#pragma once

#include "resolve/mlp_block.hpp"
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

// TabM replaces the Linear layers of the MLP with BatchEnsembleLinear layers,
// so N ensemble members run in parallel; their outputs are aggregated (mean or
// median) at the end. Each hidden layer is built as MLPBlockConfig says --
// BatchEnsembleLinear, normalization, activation, dropout, and with
// use_residual a skip connection (projected by a shared Linear where the
// widths differ) -- with the normalization and activation applied to every
// member's features alike.
//
// It is one of the encoder tails (EncoderTail in encoder.hpp), open to every
// species encoder, not an EncoderArchitecture of its own.
class TabMEncoderImpl : public torch::nn::Module {
public:
    TabMEncoderImpl(
        int64_t input_dim,
        const std::vector<int64_t>& hidden_dims,
        int n_ensembles,
        const MLPBlockConfig& block,
        TabMAggregation aggregation = TabMAggregation::Mean
    );

    // x: (batch, input_dim) -> (batch, output_dim), aggregated across the
    // ensemble. With `activations`, each layer's activation is appended to it
    // as (batch * n_ensembles, width).
    torch::Tensor forward(torch::Tensor x, std::vector<torch::Tensor>* activations = nullptr);

    // Forward without aggregation: (batch, n_ensembles, output_dim).
    torch::Tensor forward_all(torch::Tensor x, std::vector<torch::Tensor>* activations = nullptr);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int n_ensembles() const noexcept { return n_ensembles_; }

private:
    int64_t output_dim_;
    int n_ensembles_;
    TabMAggregation aggregation_;

    torch::nn::ModuleList layers_{nullptr};       // BatchEnsembleLinear per layer
    torch::nn::ModuleList norms_{nullptr};        // one per layer, or none
    torch::nn::ModuleList activations_{nullptr};  // one per layer
    torch::nn::ModuleList dropouts_{nullptr};     // one per layer, or none
    torch::nn::ModuleList projections_{nullptr};  // with use_residual: one per layer
    std::vector<torch::nn::AnyModule> norm_fns_;
    std::vector<torch::nn::AnyModule> activation_fns_;
    std::vector<torch::nn::AnyModule> projection_fns_;
};

TORCH_MODULE(TabMEncoder);

} // namespace resolve
