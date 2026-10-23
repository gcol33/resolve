#pragma once

// The transformer blocks the tabular attention encoders share (multi-head
// attention, feed-forward, block, block stack, feature tokenizer, CLS readout)
// and the FT-Transformer encoder built from them.

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// =============================================================================
// Multi-Head Attention
// =============================================================================

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(
        int64_t d_model,
        int64_t n_heads,
        float dropout = 0.0f,
        bool bias = true
    );

    // query: (batch, seq_len_q, d_model); key, value: (batch, seq_len_k, d_model)
    // mask: optional additive bias on the pre-softmax scores, 0 = attend
    //   freely and large-negative = blocked, shaped (batch, seq_len_k),
    //   (batch, seq_len_q, seq_len_k) or anything broadcastable to
    //   (batch, n_heads, seq_len_q, seq_len_k) after the leading dims are
    //   expanded as below. A bias with requires_grad receives gradient, which is
    //   how ExcelFormer's importance gate trains.
    // Returns: (batch, seq_len_q, d_model)
    torch::Tensor forward(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        torch::Tensor mask = {}
    );

    // Every query position attends over the positions of `memory` and over
    // itself: position i of `query` scores the keys of every memory position
    // plus its own key, and mixes the matching values. This is the attention a
    // position would receive in forward(x, x, x) if x were `memory` with it
    // appended, without the memory positions attending to it in turn.
    // query: (batch, seq_len_q, d_model); memory: (batch, seq_len_m, d_model)
    // Returns: (batch, seq_len_q, d_model)
    torch::Tensor attend_memory_and_self(torch::Tensor query, torch::Tensor memory);

    // The dropout rate on the attention weights while training.
    [[nodiscard]] float dropout() const noexcept { return dropout_; }

private:
    // (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
    torch::Tensor split_heads(const torch::Tensor& x) const;
    // (batch, n_heads, seq, d_k) -> projected (batch, seq, d_model)
    torch::Tensor merge_heads(const torch::Tensor& x);

    int64_t d_model_;
    int64_t n_heads_;
    int64_t d_k_;  // d_model / n_heads
    float dropout_;

    torch::nn::Linear W_q_{nullptr};
    torch::nn::Linear W_k_{nullptr};
    torch::nn::Linear W_v_{nullptr};
    torch::nn::Linear W_o_{nullptr};
};

TORCH_MODULE(MultiHeadAttention);

// =============================================================================
// Position-wise Feed-Forward Network
// =============================================================================

class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(
        int64_t d_model,
        int64_t d_ff,
        float dropout = 0.0f,
        bool use_gelu = true
    );

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    bool use_gelu_;
};

TORCH_MODULE(FeedForward);

// =============================================================================
// Transformer Block Configuration
// =============================================================================

// How one transformer block is shaped and where it drops activations. The three
// rates act at different places -- attention dropout on the attention weights,
// FFN dropout inside the feed-forward hidden layer, and residual dropout on a
// sublayer's output before it joins the residual stream -- and FT-Transformer
// (Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
// NeurIPS 2021, arXiv:2106.11959) tunes them separately. An architecture whose
// configuration carries one rate builds this with `uniform`.
struct TransformerBlockConfig {
    int64_t d_model = 0;
    int64_t n_heads = 8;
    int64_t d_ff = 0;  // 0 = 4 * d_model
    float attention_dropout = 0.1f;
    float ffn_dropout = 0.1f;
    float residual_dropout = 0.1f;
    bool pre_norm = true;  // Pre-LN (more stable) vs Post-LN

    // One rate at all three places.
    static TransformerBlockConfig uniform(int64_t d_model, int64_t n_heads,
                                          int64_t d_ff, float dropout,
                                          bool pre_norm = true) {
        return {d_model, n_heads, d_ff, dropout, dropout, dropout, pre_norm};
    }

    // The feed-forward width this block runs at: d_ff, or the standard
    // 4 * d_model when d_ff is left at 0.
    [[nodiscard]] int64_t ffn_dim() const {
        return d_ff == 0 ? 4 * d_model : d_ff;
    }
};

// =============================================================================
// Transformer Encoder Block
// =============================================================================

class TransformerBlockImpl : public torch::nn::Module {
public:
    explicit TransformerBlockImpl(const TransformerBlockConfig& config);

    // x: (batch, seq_len, d_model)
    // mask: optional additive attention bias (MultiHeadAttention::forward)
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    MultiHeadAttention attention_{nullptr};
    FeedForward ffn_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    bool pre_norm_;
};

TORCH_MODULE(TransformerBlock);

// =============================================================================
// Transformer Encoder (stack of blocks)
// =============================================================================

// n_layers blocks, followed by a final LayerNorm under the pre-norm
// arrangement (a post-norm block already ends in one).
class TransformerEncoderImpl : public torch::nn::Module {
public:
    TransformerEncoderImpl(const TransformerBlockConfig& config, int64_t n_layers);

    // x: (batch, seq_len, d_model); mask as in TransformerBlock::forward,
    // applied in every block.
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::LayerNorm final_norm_{nullptr};
};

TORCH_MODULE(TransformerEncoder);

// =============================================================================
// Feature Tokenizer
// =============================================================================

// Turns a row of tabular features into a sequence of d_model tokens:
// [CLS, one token per numerical feature, one token per categorical feature].
// Numerical feature j becomes x_j * W_j + b_j with its own (d_model) weight
// and bias (Gorishniy et al. 2021, Eq. 2); every numerical feature is
// tokenized in one broadcast product over the (n_numerical, d_model) weight.
// The categorical features share one embedding table holding every feature's
// rows back to back, so code c of feature j reads row offset_j + c.
class FeatureTokenizerImpl : public torch::nn::Module {
public:
    FeatureTokenizerImpl(
        int64_t n_numerical,
        const std::vector<int64_t>& cat_cardinalities,
        int64_t d_model
    );

    // numerical: (batch, n_numerical)
    // categoricals: (batch, n_categorical) int64 codes, code c of feature j in
    //   [0, cat_cardinalities[j]); undefined when there are no categoricals.
    // Returns: (batch, n_tokens, d_model)
    torch::Tensor forward(torch::Tensor numerical, torch::Tensor categoricals = {});

    [[nodiscard]] int64_t n_tokens() const noexcept { return n_tokens_; }

private:
    int64_t n_numerical_;
    int64_t n_categorical_;
    int64_t d_model_;
    int64_t n_tokens_;

    torch::Tensor numerical_weight_;     // (n_numerical, d_model)
    torch::Tensor numerical_bias_;       // (n_numerical, d_model)
    torch::nn::Embedding categorical_embedding_{nullptr};  // (sum of cardinalities, d_model)
    torch::Tensor categorical_offsets_;  // (n_categorical,) first row of each feature
    torch::Tensor cls_token_;            // (1, 1, d_model)
};

TORCH_MODULE(FeatureTokenizer);

// The row representation of a token sequence the FeatureTokenizer built: the
// CLS token, (batch, n_tokens, d_model) -> (batch, d_model).
[[nodiscard]] inline torch::Tensor cls_readout(const torch::Tensor& tokens) {
    return tokens.select(/*dim=*/1, /*index=*/0);
}

// =============================================================================
// FT-Transformer Encoder (Feature Tokenizer + Transformer)
// =============================================================================

// Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
class FTTransformerEncoderImpl : public torch::nn::Module {
public:
    FTTransformerEncoderImpl(
        int64_t n_numerical,
        const std::vector<int64_t>& cat_cardinalities,
        const TransformerBlockConfig& block,
        int64_t n_layers = 3
    );

    // numerical: (batch, n_numerical); categoricals as in FeatureTokenizer.
    // Returns: (batch, d_model), the CLS token.
    torch::Tensor forward(torch::Tensor numerical, torch::Tensor categoricals = {});

    [[nodiscard]] int64_t output_dim() const noexcept { return d_model_; }
    [[nodiscard]] int64_t n_tokens() const { return tokenizer_->n_tokens(); }

private:
    int64_t d_model_;

    FeatureTokenizer tokenizer_{nullptr};
    TransformerEncoder encoder_{nullptr};
};

TORCH_MODULE(FTTransformerEncoder);

}  // namespace resolve
