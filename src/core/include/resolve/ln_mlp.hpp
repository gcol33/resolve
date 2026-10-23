#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// An MLP of Linear -> LayerNorm -> GELU [-> Dropout] stages, one per entry of
// `hidden_dims`, starting from `in_dim`. Dropout is added only when
// `dropout` > 0. TraitNet's environment and trait encoders and the species
// VAE's encoder and decoder are built from it.
[[nodiscard]] inline torch::nn::Sequential make_ln_gelu_mlp(
    int64_t in_dim, const std::vector<int64_t>& hidden_dims, float dropout) {
    torch::nn::Sequential mlp;
    for (int64_t hidden : hidden_dims) {
        mlp->push_back(torch::nn::Linear(in_dim, hidden));
        mlp->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})));
        mlp->push_back(torch::nn::GELU());
        if (dropout > 0.0f) {
            mlp->push_back(torch::nn::Dropout(dropout));
        }
        in_dim = hidden;
    }
    return mlp;
}

}  // namespace resolve
