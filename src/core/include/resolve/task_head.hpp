#pragma once

#include "resolve/types.hpp"

#include <torch/torch.h>

#include <vector>

namespace resolve {

// The prediction head of one target: optional hidden layers (Linear,
// activation, dropout) and a Linear to one output (regression) or one logit
// per class. Without hidden_dims the head is the single Linear.
class TaskHeadImpl : public torch::nn::Module {
public:
    TaskHeadImpl(
        int64_t latent_dim,
        TaskType task,
        int num_classes = 0,
        TransformType transform = TransformType::None,
        const std::vector<int64_t>& hidden_dims = {},
        ActivationType activation = ActivationType::GELU,
        float dropout = 0.0f
    );

    // Forward pass - returns raw output
    torch::Tensor forward(torch::Tensor latent);

    // Predict with inverse transform
    torch::Tensor predict(torch::Tensor latent);

    // Inverse transform for predictions
    torch::Tensor inverse_transform(torch::Tensor predictions);

    [[nodiscard]] TaskType task() const noexcept { return task_; }
    [[nodiscard]] TransformType transform() const noexcept { return transform_; }
    [[nodiscard]] bool has_hidden_layers() const noexcept { return static_cast<bool>(head_mlp_); }

private:
    TaskType task_;
    TransformType transform_;
    torch::nn::Sequential head_mlp_{nullptr};  // Hidden layers (if any)
    torch::nn::Linear output_{nullptr};         // Final projection
};

TORCH_MODULE(TaskHead);

} // namespace resolve
