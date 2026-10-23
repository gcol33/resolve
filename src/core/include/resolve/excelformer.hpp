#pragma once

#include "resolve/ft_transformer.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// ExcelFormer: an FT-Transformer whose attention is semi-permeable. Every token
// carries a learned importance in (0, 1); token i attends freely to token j
// when j is informative (importance above importance_threshold) or at least as
// important as i, and is down-weighted towards j otherwise. The rule enters
// every block as an additive bias on the attention scores, built from smooth
// comparisons so the importance logits train by gradient.
// Chen et al., "ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data"
class ExcelFormerEncoderImpl : public torch::nn::Module {
public:
    ExcelFormerEncoderImpl(
        int64_t n_numerical,
        const std::vector<int64_t>& cat_cardinalities,
        const TransformerBlockConfig& block,
        int64_t n_layers = 3,
        float importance_threshold = 0.5f
    );

    // numerical: (batch, n_numerical); categoricals as in FeatureTokenizer.
    // Returns: (batch, d_model), the CLS token.
    torch::Tensor forward(torch::Tensor numerical, torch::Tensor categoricals = {});

    [[nodiscard]] int64_t output_dim() const noexcept { return d_model_; }

private:
    // The semi-permeable rule as a (1, n_tokens, n_tokens) additive log-bias on
    // the pre-softmax scores: ~0 where the gate is open, large-negative where
    // it is closed. The leading 1 broadcasts over the batch.
    [[nodiscard]] torch::Tensor build_attention_bias() const;

    int64_t d_model_;
    float importance_threshold_;

    FeatureTokenizer tokenizer_{nullptr};
    torch::Tensor importance_logits_;  // (n_tokens,), before the sigmoid
    TransformerEncoder encoder_{nullptr};
};

TORCH_MODULE(ExcelFormerEncoder);

}  // namespace resolve
