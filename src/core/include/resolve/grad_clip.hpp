#pragma once

#include <torch/torch.h>

#include <optional>
#include <vector>

namespace resolve {

// Scales the gradients of `params` so their global L2 norm is at most
// `max_norm`: torch::nn::utils::clip_grad_norm_'s computation (the same norms,
// scale min(1, max_norm / (norm + 1e-6)) as a tensor), so the clipped
// gradients agree bit for bit. Unlike it, nothing is read back to the host
// (torch's returns the norm as a double, which waits for the whole backward
// pass), so a training step queues behind its backward pass instead.
// Parameters without a gradient are skipped.
inline void clip_grad_norm(const std::vector<torch::Tensor>& params, double max_norm) {
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> grads;
    std::vector<torch::Tensor> norms;
    grads.reserve(params.size());
    norms.reserve(params.size());
    for (const auto& param : params) {
        const auto& grad = param.grad();
        if (!grad.defined()) continue;
        grads.push_back(grad);
        norms.push_back(grad.norm(2));
    }
    if (grads.empty()) return;
    const auto total = norms.size() == 1 ? norms[0] : torch::stack(norms).norm(2);
    const auto scale = torch::clamp(max_norm / (total + 1e-6), std::nullopt, 1.0);
    for (auto& grad : grads) {
        grad.mul_(scale);
    }
}

}  // namespace resolve
