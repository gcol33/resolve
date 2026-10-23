#include "resolve/ft_transformer.hpp"

#include <cmath>
#include <numeric>

namespace resolve {

// =============================================================================
// Multi-Head Attention
// =============================================================================

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
    int64_t d_model,
    int64_t n_heads,
    float dropout,
    bool bias
) : d_model_(d_model), n_heads_(n_heads), d_k_(0), dropout_(dropout) {
    TORCH_CHECK(n_heads > 0 && d_model % n_heads == 0,
        "d_model (", d_model, ") must be divisible by n_heads (", n_heads, ")");
    d_k_ = d_model / n_heads;

    W_q_ = register_module("W_q", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_k_ = register_module("W_k", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_v_ = register_module("W_v", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_o_ = register_module("W_o", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
}

torch::Tensor MultiHeadAttentionImpl::split_heads(const torch::Tensor& x) const {
    return x.view({x.size(0), x.size(1), n_heads_, d_k_}).transpose(1, 2);
}

torch::Tensor MultiHeadAttentionImpl::merge_heads(const torch::Tensor& x) {
    const int64_t batch = x.size(0);
    const int64_t seq = x.size(2);
    return W_o_->forward(x.transpose(1, 2).contiguous().view({batch, seq, d_model_}));
}

torch::Tensor MultiHeadAttentionImpl::forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask
) {
    auto Q = split_heads(W_q_->forward(query));
    auto K = split_heads(W_k_->forward(key));
    auto V = split_heads(W_v_->forward(value));

    std::optional<torch::Tensor> bias;
    if (mask.defined()) {
        if (mask.dim() == 2) {
            // (batch, seq_len_k) -> (batch, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2);
        } else if (mask.dim() == 3) {
            // (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1);
        }
        // An additive bias has to share the query dtype (an fp16 query under
        // autocast meets an fp32 bias here).
        bias = mask.to(Q.scalar_type());
    }

    // Scaled by 1/sqrt(d_k), with dropout on the attention weights while
    // training.
    auto context = at::scaled_dot_product_attention(
        Q, K, V, bias, is_training() ? static_cast<double>(dropout_) : 0.0);
    return merge_heads(context);
}

torch::Tensor MultiHeadAttentionImpl::attend_memory_and_self(
    torch::Tensor query, torch::Tensor memory
) {
    auto Q = split_heads(W_q_->forward(query));     // (B, H, Lq, d_k)
    auto K_mem = split_heads(W_k_->forward(memory));  // (B, H, Lm, d_k)
    auto V_mem = split_heads(W_v_->forward(memory));
    auto K_self = split_heads(W_k_->forward(query));  // (B, H, Lq, d_k)
    auto V_self = split_heads(W_v_->forward(query));

    const double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));
    const int64_t n_memory = K_mem.size(2);

    // Scores over the memory positions and over each query's own key, then
    // one softmax over both.
    auto scores = torch::cat({
        torch::matmul(Q, K_mem.transpose(-2, -1)),
        (Q * K_self).sum(/*dim=*/-1, /*keepdim=*/true)}, /*dim=*/-1) * scale;
    auto weights = torch::softmax(scores, /*dim=*/-1);
    if (is_training() && dropout_ > 0.0f) {
        weights = torch::dropout(weights, dropout_, /*train=*/true);
    }

    auto context = torch::matmul(weights.narrow(-1, 0, n_memory), V_mem) +
                   weights.narrow(-1, n_memory, 1) * V_self;
    return merge_heads(context);
}

// =============================================================================
// Feed-Forward Network
// =============================================================================

FeedForwardImpl::FeedForwardImpl(
    int64_t d_model,
    int64_t d_ff,
    float dropout,
    bool use_gelu
) : use_gelu_(use_gelu) {
    linear1_ = register_module("linear1", torch::nn::Linear(d_model, d_ff));
    linear2_ = register_module("linear2", torch::nn::Linear(d_ff, d_model));
    if (dropout > 0.0f) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
    x = linear1_->forward(x);
    x = use_gelu_ ? torch::gelu(x) : torch::relu(x);
    if (dropout_) {
        x = dropout_->forward(x);
    }
    return linear2_->forward(x);
}

// =============================================================================
// Transformer Block
// =============================================================================

TransformerBlockImpl::TransformerBlockImpl(const TransformerBlockConfig& config)
    : pre_norm_(config.pre_norm) {
    const int64_t d_model = config.d_model;

    attention_ = register_module("attention",
        MultiHeadAttention(d_model, config.n_heads, config.attention_dropout));
    ffn_ = register_module("ffn",
        FeedForward(d_model, config.ffn_dim(), config.ffn_dropout));
    norm1_ = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    norm2_ = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    if (config.residual_dropout > 0.0f) {
        dropout_ = register_module("dropout",
                                   torch::nn::Dropout(config.residual_dropout));
    }
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
    auto residual = [this](torch::Tensor out) {
        return dropout_ ? dropout_->forward(out) : out;
    };
    if (pre_norm_) {
        auto normed = norm1_->forward(x);
        x = x + residual(attention_->forward(normed, normed, normed, mask));
        x = x + residual(ffn_->forward(norm2_->forward(x)));
    } else {
        x = norm1_->forward(x + residual(attention_->forward(x, x, x, mask)));
        x = norm2_->forward(x + residual(ffn_->forward(x)));
    }
    return x;
}

// =============================================================================
// Transformer Encoder
// =============================================================================

TransformerEncoderImpl::TransformerEncoderImpl(
    const TransformerBlockConfig& config,
    int64_t n_layers
) {
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(TransformerBlock(config));
    }
    if (config.pre_norm) {
        final_norm_ = register_module("final_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.d_model})));
    }
}

torch::Tensor TransformerEncoderImpl::forward(torch::Tensor x, torch::Tensor mask) {
    for (auto& layer : *layers_) {
        x = layer->as<TransformerBlock>()->forward(x, mask);
    }
    return final_norm_ ? final_norm_->forward(x) : x;
}

// =============================================================================
// Feature Tokenizer
// =============================================================================

FeatureTokenizerImpl::FeatureTokenizerImpl(
    int64_t n_numerical,
    const std::vector<int64_t>& cat_cardinalities,
    int64_t d_model
) : n_numerical_(n_numerical),
    n_categorical_(static_cast<int64_t>(cat_cardinalities.size())),
    d_model_(d_model),
    n_tokens_(1 + n_numerical + static_cast<int64_t>(cat_cardinalities.size())) {
    TORCH_CHECK(n_numerical >= 0, "FeatureTokenizer: n_numerical must be >= 0, got ",
                n_numerical);

    // Each numerical token is a Linear(1, d_model) of its feature, initialised
    // as nn::Linear initialises one: uniform on +-1/sqrt(fan_in) = +-1.
    numerical_weight_ = register_parameter("numerical_weight",
        torch::rand({n_numerical, d_model}) * 2.0 - 1.0);
    numerical_bias_ = register_parameter("numerical_bias",
        torch::rand({n_numerical, d_model}) * 2.0 - 1.0);

    if (n_categorical_ > 0) {
        std::vector<int64_t> offsets(cat_cardinalities.size(), 0);
        std::exclusive_scan(cat_cardinalities.begin(), cat_cardinalities.end(),
                            offsets.begin(), int64_t{0});
        const int64_t total_rows = std::accumulate(
            cat_cardinalities.begin(), cat_cardinalities.end(), int64_t{0});
        categorical_embedding_ = register_module("categorical_embedding",
            torch::nn::Embedding(total_rows, d_model));
        categorical_offsets_ = register_buffer("categorical_offsets",
            torch::tensor(offsets, torch::kInt64));
    }

    cls_token_ = register_parameter("cls_token", torch::randn({1, 1, d_model}) * 0.02f);
}

torch::Tensor FeatureTokenizerImpl::forward(torch::Tensor numerical,
                                            torch::Tensor categoricals) {
    const bool has_categoricals = categoricals.defined() && categoricals.numel() > 0;
    TORCH_CHECK(has_categoricals == (n_categorical_ > 0),
        "FeatureTokenizer: built for ", n_categorical_, " categorical features, got ",
        has_categoricals ? std::to_string(categoricals.size(-1)) : std::string("none"));
    TORCH_CHECK(numerical.defined() && numerical.dim() == 2 &&
                    numerical.size(1) == n_numerical_,
        "FeatureTokenizer: numerical must be (batch, ", n_numerical_, ")");

    const int64_t batch = numerical.size(0);
    std::vector<torch::Tensor> tokens;
    tokens.reserve(3);
    tokens.push_back(cls_token_.expand({batch, 1, d_model_}));
    // (batch, n_numerical, 1) * (n_numerical, d_model) + (n_numerical, d_model)
    tokens.push_back(numerical.unsqueeze(-1) * numerical_weight_ + numerical_bias_);
    if (has_categoricals) {
        TORCH_CHECK(categoricals.dim() == 2 && categoricals.size(0) == batch &&
                        categoricals.size(1) == n_categorical_,
            "FeatureTokenizer: categoricals must be (", batch, ", ", n_categorical_,
            "), got ", categoricals.sizes());
        tokens.push_back(categorical_embedding_->forward(
            categoricals.to(torch::kLong) + categorical_offsets_));
    }
    return torch::cat(tokens, /*dim=*/1);
}

// =============================================================================
// FT-Transformer Encoder
// =============================================================================

FTTransformerEncoderImpl::FTTransformerEncoderImpl(
    int64_t n_numerical,
    const std::vector<int64_t>& cat_cardinalities,
    const TransformerBlockConfig& block,
    int64_t n_layers
) : d_model_(block.d_model) {
    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, block.d_model));
    encoder_ = register_module("encoder", TransformerEncoder(block, n_layers));
}

torch::Tensor FTTransformerEncoderImpl::forward(torch::Tensor numerical,
                                                torch::Tensor categoricals) {
    return cls_readout(encoder_->forward(tokenizer_->forward(numerical, categoricals)));
}

}  // namespace resolve
