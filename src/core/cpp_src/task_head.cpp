#include "resolve/task_head.hpp"
#include "resolve/encoder.hpp"  // make_activation
#include "resolve/target_transform.hpp"

namespace resolve {

TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform,
    const std::vector<int64_t>& hidden_dims,
    ActivationType activation,
    float dropout
) : task_(task), transform_(transform)
{
    int64_t prev_dim = latent_dim;
    if (!hidden_dims.empty()) {
        torch::nn::Sequential mlp;
        for (const int64_t width : hidden_dims) {
            mlp->push_back(torch::nn::Linear(prev_dim, width));
            mlp->push_back(make_activation(activation, width));
            if (dropout > 0) {
                mlp->push_back(torch::nn::Dropout(dropout));
            }
            prev_dim = width;
        }
        head_mlp_ = register_module("head_mlp", mlp);
    }

    const int64_t out_features = task_ == TaskType::Classification ? num_classes : 1;
    output_ = register_module("output", torch::nn::Linear(prev_dim, out_features));
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
    return to_original_units(std::move(predictions), transform_);
}

} // namespace resolve
