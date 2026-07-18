#include "resolve/tabm.hpp"
#include <stdexcept>
#include <cmath>

namespace resolve {

// =============================================================================
// BatchEnsembleLinear Implementation
// =============================================================================

BatchEnsembleLinearImpl::BatchEnsembleLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int n_ensembles,
    bool bias
) : in_features_(in_features),
    out_features_(out_features),
    n_ensembles_(n_ensembles)
{
    if (n_ensembles < 1) {
        throw std::invalid_argument(
            "n_ensembles must be >= 1, got " + std::to_string(n_ensembles));
    }

    // Shared base weight. The base linear never carries a bias: in
    // BatchEnsemble the bias is added after the per-ensemble output scaling s,
    // so folding it into base_linear would scale the shared bias by s (coupling
    // it to the per-member factor). A separate per-ensemble bias is used below.
    base_linear_ = register_module("base_linear",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false)));

    has_bias_ = bias;
    if (has_bias_) {
        ensemble_bias_ = register_parameter("ensemble_bias",
            torch::zeros({n_ensembles, out_features}));
    }

    // Per-ensemble rank-1 factors, initialized near 1.0: ones + N(0, 0.1) so
    // each ensemble member starts as a small perturbation of the shared weight.
    r_ = register_parameter("r",
        torch::ones({n_ensembles, in_features}) +
        torch::randn({n_ensembles, in_features}) * 0.1f);
    s_ = register_parameter("s",
        torch::ones({n_ensembles, out_features}) +
        torch::randn({n_ensembles, out_features}) * 0.1f);
}

torch::Tensor BatchEnsembleLinearImpl::forward(torch::Tensor x) {
    // x: (batch, in_features) or (batch, n_ensembles, in_features)
    auto batch_size = x.size(0);

    if (x.dim() == 2) {
        // First layer: expand input for all ensemble members
        // x: (batch, in_features) -> (batch, 1, in_features) * r: (1, n_ensembles, in_features)
        // = (batch, n_ensembles, in_features)
        x = x.unsqueeze(1) * r_.unsqueeze(0);
    } else {
        // Subsequent layers: x already has ensemble dimension
        // x: (batch, n_ensembles, in_features) * r: (1, n_ensembles, in_features)
        x = x * r_.unsqueeze(0);
    }

    // Apply shared linear: (batch, n_ensembles, in_features) -> (batch, n_ensembles, out_features)
    // base_linear expects 2D input, so reshape
    auto x_flat = x.reshape({batch_size * n_ensembles_, in_features_});
    auto out_flat = base_linear_->forward(x_flat);
    auto out = out_flat.reshape({batch_size, n_ensembles_, out_features_});

    // Apply output scaling: (batch, n_ensembles, out_features) * s: (1, n_ensembles, out_features)
    out = out * s_.unsqueeze(0);

    // Per-ensemble bias, added after scaling so it stays a free parameter.
    if (has_bias_) {
        out = out + ensemble_bias_.unsqueeze(0);
    }

    return out;
}

// =============================================================================
// TabMEncoder Implementation
// =============================================================================

TabMEncoderImpl::TabMEncoderImpl(
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    int n_ensembles,
    float dropout,
    const std::string& aggregation
) : n_ensembles_(n_ensembles),
    aggregation_(aggregation),
    dims_(hidden_dims)
{
    if (hidden_dims.empty()) {
        throw std::invalid_argument("hidden_dims must not be empty");
    }
    if (aggregation != "mean" && aggregation != "median") {
        throw std::invalid_argument(
            "aggregation must be 'mean' or 'median', got '" + aggregation + "'");
    }

    output_dim_ = hidden_dims.back();

    layers_ = register_module("layers", torch::nn::ModuleList());
    norms_ = register_module("norms", torch::nn::ModuleList());
    dropouts_ = register_module("dropouts", torch::nn::ModuleList());

    int64_t prev_dim = input_dim;
    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        layers_->push_back(
            BatchEnsembleLinear(prev_dim, hidden_dims[i], n_ensembles));

        // LayerNorm per ensemble member (applied to last dim)
        norms_->push_back(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dims[i]})));

        if (dropout > 0.0f) {
            dropouts_->push_back(torch::nn::Dropout(dropout));
        }

        prev_dim = hidden_dims[i];
    }
}

torch::Tensor TabMEncoderImpl::forward_all(torch::Tensor x) {
    // x: (batch, input_dim)
    // Returns: (batch, n_ensembles, output_dim)

    for (size_t i = 0; i < layers_->size(); ++i) {
        // BatchEnsembleLinear: produces (batch, n_ensembles, dim)
        x = layers_->ptr(static_cast<int64_t>(i))
            ->as<BatchEnsembleLinearImpl>()->forward(x);

        // LayerNorm (applied to last dim, works with 3D tensors)
        x = norms_->ptr(static_cast<int64_t>(i))
            ->as<torch::nn::LayerNormImpl>()->forward(x);

        // GELU activation
        x = torch::gelu(x);

        // Dropout
        if (dropouts_->size() > 0) {
            x = dropouts_->ptr(static_cast<int64_t>(i))
                ->as<torch::nn::DropoutImpl>()->forward(x);
        }
    }

    return x;
}

torch::Tensor TabMEncoderImpl::forward(torch::Tensor x) {
    // Get all ensemble outputs: (batch, n_ensembles, output_dim)
    auto all_outputs = forward_all(x);

    // Aggregate across ensemble dimension
    if (aggregation_ == "mean") {
        return all_outputs.mean(/*dim=*/1);
    } else {
        // Median aggregation
        return std::get<0>(all_outputs.median(/*dim=*/1));
    }
}

} // namespace resolve
