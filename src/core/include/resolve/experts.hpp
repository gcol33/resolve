#pragma once

#include <cmath>
#include <torch/torch.h>
#include "resolve/types.hpp"

namespace resolve {

// MoE forward result
struct MoEResult {
    torch::Tensor output;
    torch::Tensor aux_loss;
    torch::Tensor gate_probs;
};

// ---------------------------------------------------------------------------
// Mixture of Experts module — full implementation.
//
// Expert weights are stored as stacked 3-D tensors and computed via batched
// matmul (single cuBLAS GEMM call per layer) instead of sequential
// per-expert forward passes.  For E experts this reduces GPU kernel launches
// from E to 1 per layer.
//
// Supports three routing modes:
//   None  – bypass, single linear projection (no experts)
//   Soft  – all experts contribute, weighted by softmax gate probs
//   TopK  – sparse routing, only top-k experts per sample
// ---------------------------------------------------------------------------
class MixtureOfExpertsImpl : public torch::nn::Module {
public:
    MixtureOfExpertsImpl(
        int64_t input_dim,
        const std::vector<int64_t>& expert_hidden_dims,
        int64_t output_dim,
        int n_experts,
        MoERoutingType routing,
        int top_k,
        float noise_std,
        float dropout
    ) : input_dim_(input_dim),
        output_dim_(output_dim),
        n_experts_(n_experts),
        routing_(routing),
        top_k_(std::min(top_k, n_experts)),
        noise_std_(noise_std),
        dropout_(dropout) {

        if (routing_ == MoERoutingType::None) {
            // Bypass mode: single linear layer, no expert routing
            bypass_linear_ = register_module(
                "bypass_linear", torch::nn::Linear(input_dim, output_dim));
            return;
        }

        // Gating network: Linear → Softmax (softmax applied in forward)
        gate_ = register_module("gate", torch::nn::Linear(input_dim, n_experts));

        // Build stacked expert weights.
        // dims = [input_dim, hidden_0, hidden_1, ..., output_dim]
        std::vector<int64_t> dims;
        dims.push_back(input_dim);
        dims.insert(dims.end(), expert_hidden_dims.begin(),
                     expert_hidden_dims.end());
        dims.push_back(output_dim);
        n_layers_ = static_cast<int>(dims.size()) - 1;

        // Each layer stores weights (E, O, I) and biases (E, O)
        // with Kaiming uniform init per expert slice.
        for (int i = 0; i < n_layers_; ++i) {
            auto W = torch::empty({n_experts, dims[i + 1], dims[i]});
            auto b = torch::zeros({n_experts, dims[i + 1]});

            for (int e = 0; e < n_experts; ++e) {
                // Kaiming uniform (fan_in = dims[i]), matching nn.Linear default
                torch::nn::init::kaiming_uniform_(W[e], std::sqrt(5.0));
                double bound = 1.0 / std::sqrt(static_cast<double>(dims[i]));
                torch::nn::init::uniform_(b[e], -bound, bound);
            }

            layer_weights_.push_back(
                register_parameter("layer_weight_" + std::to_string(i), W));
            layer_biases_.push_back(
                register_parameter("layer_bias_" + std::to_string(i), b));
        }
    }

    // Forward returns MoEResult with output, aux_loss, gate_probs
    MoEResult forward(torch::Tensor x) {
        if (routing_ == MoERoutingType::None) {
            auto output = bypass_linear_->forward(x);
            auto aux_loss = torch::tensor(0.0f, x.options());
            auto gate_probs = torch::ones(
                {x.size(0), 1}, x.options());
            return {output, aux_loss, gate_probs};
        }

        // Gate logits with optional noise for load balancing during training
        auto logits = gate_->forward(x);  // (B, E)
        if (is_training() && noise_std_ > 0.0f) {
            logits = logits + torch::randn_like(logits) * noise_std_;
        }
        auto gate_probs = torch::softmax(logits, /*dim=*/-1);  // (B, E)

        // All expert outputs via batched matmul: (E, B, D_out)
        auto expert_out = batched_experts(x);

        torch::Tensor output;
        torch::Tensor aux_loss;

        if (routing_ == MoERoutingType::Soft) {
            // Weighted sum across all experts
            // gate_probs.T: (E, B), unsqueeze → (E, B, 1)
            auto weights = gate_probs.t().unsqueeze(-1);  // (E, B, 1)
            output = (weights * expert_out).sum(/*dim=*/0);  // (B, D_out)

            // Load-balancing: CV² of expert importance
            auto importance = gate_probs.sum(/*dim=*/0);  // (E,)
            auto mean_imp = importance.mean();
            aux_loss = importance.var() / (mean_imp * mean_imp + 1e-8f);
        } else {
            // TopK: select top-k experts per sample and re-normalize
            auto topk_result = torch::topk(gate_probs, top_k_, /*dim=*/-1);
            auto top_k_probs = std::get<0>(topk_result);  // (B, K)
            auto top_k_idx = std::get<1>(topk_result);    // (B, K)
            top_k_probs = top_k_probs /
                top_k_probs.sum(/*dim=*/-1, /*keepdim=*/true);

            // Gather selected expert outputs
            // (E, B, D) → (B, E, D), then gather along expert dim
            auto expert_out_bt = expert_out.permute({1, 0, 2});  // (B, E, D)
            auto idx_expanded = top_k_idx.unsqueeze(-1).expand(
                {-1, -1, output_dim_});  // (B, K, D)
            auto selected = torch::gather(
                expert_out_bt, /*dim=*/1, idx_expanded);  // (B, K, D)
            output = (top_k_probs.unsqueeze(-1) * selected).sum(/*dim=*/1);

            // Load-balancing: Switch Transformer loss = E * sum(f_i * P_i)
            auto flat_idx = top_k_idx.reshape({-1});
            auto counts = torch::zeros({n_experts_},
                torch::TensorOptions().dtype(torch::kFloat32)
                                      .device(x.device()));
            counts.scatter_add_(
                0, flat_idx,
                torch::ones_like(flat_idx, torch::kFloat32));
            auto f = counts / counts.sum();
            auto P = gate_probs.mean(/*dim=*/0);  // (E,)
            aux_loss = static_cast<float>(n_experts_) * (f * P).sum();
        }

        return {output, aux_loss, gate_probs};
    }

    // Simplified forward without aux loss (for inference)
    torch::Tensor forward_simple(torch::Tensor x) {
        return forward(x).output;
    }

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int n_experts() const noexcept { return n_experts_; }

private:
    // Run all experts on all inputs via batched matmul.
    //
    // First layer uses broadcast matmul to avoid copying x E times.
    // Subsequent layers use torch::bmm over the already-batched activations.
    //
    // Input:  x (B, D_in)
    // Output: (E, B, D_out) expert outputs for all samples.
    torch::Tensor batched_experts(const torch::Tensor& x) {
        // First layer: (1, B, I) @ (E, I, H0) → (E, B, H0) via broadcast
        auto h = torch::matmul(
            x.unsqueeze(0),
            layer_weights_[0].transpose(1, 2));
        h = h + layer_biases_[0].unsqueeze(1);

        if (n_layers_ > 1) {
            h = torch::gelu(h);
            if (dropout_ > 0.0f && is_training()) {
                h = torch::dropout(h, dropout_, /*train=*/true);
            }
        }

        // Remaining layers: (E, B, H_prev) @ (E, H_prev, H_next)
        for (int i = 1; i < n_layers_; ++i) {
            h = torch::bmm(h, layer_weights_[i].transpose(1, 2));
            h = h + layer_biases_[i].unsqueeze(1);
            // Apply GELU + dropout for all but the last layer
            if (i < n_layers_ - 1) {
                h = torch::gelu(h);
                if (dropout_ > 0.0f && is_training()) {
                    h = torch::dropout(h, dropout_, /*train=*/true);
                }
            }
        }

        return h;  // (E, B, D_out)
    }

    int64_t input_dim_;
    int64_t output_dim_;
    int n_experts_;
    MoERoutingType routing_;
    int top_k_;
    float noise_std_;
    float dropout_;
    int n_layers_ = 0;

    // Bypass mode (routing == None)
    torch::nn::Linear bypass_linear_{nullptr};

    // Expert routing mode
    torch::nn::Linear gate_{nullptr};
    std::vector<torch::Tensor> layer_weights_;
    std::vector<torch::Tensor> layer_biases_;
};

TORCH_MODULE(MixtureOfExperts);

} // namespace resolve
