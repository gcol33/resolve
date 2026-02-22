#pragma once

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
};

TORCH_MODULE(BatchEnsembleLinear);

// =============================================================================
// TabM Encoder: MLP with BatchEnsemble layers
// =============================================================================

// TabM replaces standard Linear layers in the MLP backbone with
// BatchEnsembleLinear layers. During forward pass, N ensemble members
// run in parallel. Predictions are aggregated (mean/median) for output.
//
// This is NOT a new EncoderArchitecture: it modifies the MLP backbone
// used by any PlotEncoder variant (hash/embed/sparse).
class TabMEncoderImpl : public torch::nn::Module {
public:
    // aggregation: "mean" or "median"
    TabMEncoderImpl(
        int64_t input_dim,
        const std::vector<int64_t>& hidden_dims,
        int n_ensembles = 16,
        float dropout = 0.0f,
        const std::string& aggregation = "mean"
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
    std::string aggregation_;

    // Stack of BatchEnsemble layers with activations
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::ModuleList norms_{nullptr};
    torch::nn::ModuleList dropouts_{nullptr};
    std::vector<int64_t> dims_;
};

TORCH_MODULE(TabMEncoder);

} // namespace resolve
