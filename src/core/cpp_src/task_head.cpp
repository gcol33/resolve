#include "resolve/encoder.hpp"
#include <cmath>
#include <stdexcept>

namespace resolve {

// =============================================================================
// TaskHead Implementation
// =============================================================================

// Helper to initialize output layer
void TaskHeadImpl::init_output_layer(int64_t input_dim, int num_classes) {
    int64_t out_features = (task_ == TaskType::Classification) ? num_classes : 1;
    output_ = register_module("output", torch::nn::Linear(input_dim, out_features));
}

// Legacy constructor (single linear layer)
TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform
) : task_(task), transform_(transform)
{
    init_output_layer(latent_dim, num_classes);
}

// New constructor with configurable hidden layers
TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform,
    const std::vector<int64_t>& hidden_dims,
    ActivationType activation,
    float dropout
) : task_(task), transform_(transform), hidden_dims_(hidden_dims)
{
    int64_t prev_dim = latent_dim;

    // Build hidden layers if specified
    if (!hidden_dims.empty()) {
        torch::nn::Sequential mlp;
        for (size_t i = 0; i < hidden_dims.size(); ++i) {
            mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
            // Add activation
            auto act = make_activation(activation, hidden_dims[i]);
            mlp->push_back(act);
            // Add dropout if specified
            if (dropout > 0) {
                mlp->push_back(torch::nn::Dropout(dropout));
            }
            prev_dim = hidden_dims[i];
        }
        head_mlp_ = register_module("head_mlp", mlp);
    }

    // Final output layer
    init_output_layer(prev_dim, num_classes);
}

torch::Tensor TaskHeadImpl::forward(torch::Tensor latent) {
    auto x = latent;
    if (head_mlp_) {
        x = head_mlp_->forward(x);
    }
    return output_->forward(x);
}

torch::Tensor TaskHeadImpl::predict(torch::Tensor latent) {
    auto output = forward(latent);

    if (task_ == TaskType::Classification) {
        return torch::argmax(output, /*dim=*/1);
    } else {
        output = output.squeeze(-1);
        return inverse_transform(output);
    }
}

torch::Tensor TaskHeadImpl::inverse_transform(torch::Tensor predictions) {
    if (transform_ == TransformType::Log1p) {
        return torch::expm1(torch::clamp(predictions, kExpClampMin, kExpClampMax));
    }
    return predictions;
}

} // namespace resolve
