#include "resolve/attention.hpp"
#include "resolve/types.hpp"
#include <torch/autograd.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace resolve {

// =============================================================================
// Multi-Head Attention Implementation
// =============================================================================

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
    int64_t d_model,
    int64_t n_heads,
    float dropout,
    bool bias
) : d_model_(d_model), n_heads_(n_heads) {

    TORCH_CHECK(d_model % n_heads == 0,
        "d_model (", d_model, ") must be divisible by n_heads (", n_heads, ")");

    d_k_ = d_model / n_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(d_k_));

    W_q_ = register_module("W_q", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_k_ = register_module("W_k", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_v_ = register_module("W_v", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));
    W_o_ = register_module("W_o", torch::nn::Linear(
        torch::nn::LinearOptions(d_model, d_model).bias(bias)));

    if (dropout > 0.0f) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor MultiHeadAttentionImpl::forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask
) {
    auto batch_size = query.size(0);
    auto seq_len_q = query.size(1);
    auto seq_len_k = key.size(1);

    // Linear projections
    auto Q = W_q_->forward(query);  // (batch, seq_len_q, d_model)
    auto K = W_k_->forward(key);    // (batch, seq_len_k, d_model)
    auto V = W_v_->forward(value);  // (batch, seq_len_k, d_model)

    // Reshape for multi-head attention: (batch, n_heads, seq_len, d_k)
    Q = Q.view({batch_size, seq_len_q, n_heads_, d_k_}).transpose(1, 2);
    K = K.view({batch_size, seq_len_k, n_heads_, d_k_}).transpose(1, 2);
    V = V.view({batch_size, seq_len_k, n_heads_, d_k_}).transpose(1, 2);

    // Scaled dot-product attention
    // scores: (batch, n_heads, seq_len_q, seq_len_k)
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale_;

    // Apply an additive attention bias if provided: 0 = attend freely,
    // large-negative = blocked. A differentiable bias lets a learnable mask
    // (ExcelFormer's importance gate) receive gradient, which a hard
    // masked_fill(mask == 0) cannot. This custom attention is used only by the
    // tabular transformer family (FT-Transformer / SAINT pass no mask,
    // ExcelFormer passes the importance bias); the species transformer uses
    // torch::nn::TransformerEncoder, so no other caller relies on 0/1 masks.
    if (mask.defined()) {
        if (mask.dim() == 2) {
            // (batch, seq_len_k) -> (batch, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2);
        } else if (mask.dim() == 3) {
            // (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1);
        }
        scores = scores + mask;
    }

    // Attention weights
    auto attn_weights = torch::softmax(scores, /*dim=*/-1);
    if (dropout_.is_empty() == false) {
        attn_weights = dropout_->forward(attn_weights);
    }

    // Apply attention to values
    // (batch, n_heads, seq_len_q, d_k)
    auto context = torch::matmul(attn_weights, V);

    // Reshape back: (batch, seq_len_q, d_model)
    context = context.transpose(1, 2).contiguous().view({batch_size, seq_len_q, d_model_});

    // Output projection
    return W_o_->forward(context);
}

// =============================================================================
// Feed-Forward Network Implementation
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
    if (dropout_.is_empty() == false) {
        x = dropout_->forward(x);
    }
    x = linear2_->forward(x);
    return x;
}

// =============================================================================
// Transformer Block Implementation
// =============================================================================

TransformerBlockImpl::TransformerBlockImpl(
    int64_t d_model,
    int64_t n_heads,
    int64_t d_ff,
    float dropout,
    bool pre_norm
) : pre_norm_(pre_norm) {

    if (d_ff == 0) {
        d_ff = 4 * d_model;  // Standard transformer multiplier
    }

    attention_ = register_module("attention",
        MultiHeadAttention(d_model, n_heads, dropout));
    ffn_ = register_module("ffn",
        FeedForward(d_model, d_ff, dropout));
    norm1_ = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    norm2_ = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    if (dropout > 0.0f) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
    if (pre_norm_) {
        // Pre-LayerNorm (more stable training)
        auto normed = norm1_->forward(x);
        auto attn_out = attention_->forward(normed, normed, normed, mask);
        if (dropout_.is_empty() == false) {
            attn_out = dropout_->forward(attn_out);
        }
        x = x + attn_out;

        normed = norm2_->forward(x);
        auto ffn_out = ffn_->forward(normed);
        if (dropout_.is_empty() == false) {
            ffn_out = dropout_->forward(ffn_out);
        }
        x = x + ffn_out;
    } else {
        // Post-LayerNorm (original transformer)
        auto attn_out = attention_->forward(x, x, x, mask);
        if (dropout_.is_empty() == false) {
            attn_out = dropout_->forward(attn_out);
        }
        x = norm1_->forward(x + attn_out);

        auto ffn_out = ffn_->forward(x);
        if (dropout_.is_empty() == false) {
            ffn_out = dropout_->forward(ffn_out);
        }
        x = norm2_->forward(x + ffn_out);
    }
    return x;
}

// =============================================================================
// Transformer Encoder Implementation
// =============================================================================

TransformerEncoderImpl::TransformerEncoderImpl(
    int64_t d_model,
    int64_t n_heads,
    int64_t n_layers,
    int64_t d_ff,
    float dropout,
    bool pre_norm
) : pre_norm_(pre_norm) {

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(TransformerBlock(d_model, n_heads, d_ff, dropout, pre_norm));
    }

    if (pre_norm) {
        // Final LayerNorm for pre-norm architecture
        final_norm_ = register_module("final_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    }
}

torch::Tensor TransformerEncoderImpl::forward(torch::Tensor x, torch::Tensor mask) {
    for (auto& layer : *layers_) {
        x = layer->as<TransformerBlock>()->forward(x, mask);
    }
    if (pre_norm_ && final_norm_.is_empty() == false) {
        x = final_norm_->forward(x);
    }
    return x;
}

// =============================================================================
// Feature Tokenizer Implementation
// =============================================================================

FeatureTokenizerImpl::FeatureTokenizerImpl(
    int64_t n_numerical,
    std::vector<int64_t> cat_cardinalities,
    int64_t d_model,
    bool use_cls_token
) : n_numerical_(n_numerical),
    n_categorical_(static_cast<int64_t>(cat_cardinalities.size())),
    d_model_(d_model),
    use_cls_token_(use_cls_token) {

    n_tokens_ = n_numerical_ + n_categorical_;
    if (use_cls_token_) {
        n_tokens_ += 1;
    }

    // Numerical embeddings: one linear projection per feature
    if (n_numerical > 0) {
        numerical_embeddings_ = register_module("numerical_embeddings",
            torch::nn::ModuleList());
        for (int64_t i = 0; i < n_numerical; ++i) {
            numerical_embeddings_->push_back(
                torch::nn::Linear(torch::nn::LinearOptions(1, d_model)));
        }
    }

    // Categorical embeddings: one embedding table per categorical feature
    if (n_categorical_ > 0) {
        categorical_embeddings_ = register_module("categorical_embeddings",
            torch::nn::ModuleList());
        for (auto cardinality : cat_cardinalities) {
            categorical_embeddings_->push_back(
                torch::nn::Embedding(cardinality, d_model));
        }
    }

    // CLS token
    if (use_cls_token_) {
        cls_token_ = register_parameter("cls_token",
            torch::randn({1, 1, d_model}) * 0.02f);
    }
}

torch::Tensor FeatureTokenizerImpl::forward(
    torch::Tensor numerical,
    std::vector<torch::Tensor> categoricals
) {
    auto batch_size = numerical.defined() ? numerical.size(0) :
                      (categoricals.empty() ? 0 : categoricals[0].size(0));

    std::vector<torch::Tensor> tokens;

    // CLS token (expanded to batch size)
    if (use_cls_token_) {
        tokens.push_back(cls_token_.expand({batch_size, 1, d_model_}));
    }

    // Numerical features: each becomes a token
    if (numerical.defined() && n_numerical_ > 0) {
        for (int64_t i = 0; i < n_numerical_; ++i) {
            auto feat = numerical.select(1, i).unsqueeze(1);  // (batch, 1)
            auto linear_module = numerical_embeddings_->ptr(i)->as<torch::nn::LinearImpl>();
            auto emb = linear_module->forward(feat);
            tokens.push_back(emb.unsqueeze(1));  // (batch, 1, d_model)
        }
    }

    // Categorical features. Validate the count so passing more/fewer tensors
    // than tables gives a clear error instead of an opaque ModuleList out-of-range
    // (too many) or a silently shorter token sequence (too few).
    TORCH_CHECK(static_cast<int64_t>(categoricals.size()) == n_categorical_,
        "FeatureTokenizer: expected ", n_categorical_, " categorical tensors, got ",
        categoricals.size());
    for (size_t i = 0; i < categoricals.size(); ++i) {
        auto embedding_module = categorical_embeddings_->ptr(static_cast<int64_t>(i))->as<torch::nn::EmbeddingImpl>();
        auto emb = embedding_module->forward(categoricals[i]);
        tokens.push_back(emb.unsqueeze(1));  // (batch, 1, d_model)
    }

    // Concatenate all tokens
    return torch::cat(tokens, /*dim=*/1);  // (batch, n_tokens, d_model)
}

// =============================================================================
// Sparsemax Implementation
// =============================================================================

torch::Tensor sparsemax(torch::Tensor input, int64_t dim) {
    // Move dim to last position for easier processing
    auto original_dim = dim;
    if (dim < 0) {
        dim = input.dim() + dim;
    }

    // Transpose if needed
    if (dim != input.dim() - 1) {
        input = input.transpose(dim, -1);
    }

    auto original_shape = input.sizes().vec();
    auto n = input.size(-1);

    // Flatten to 2D: (batch, n)
    input = input.reshape({-1, n});

    // Sort in descending order
    auto [sorted, _] = input.sort(/*dim=*/-1, /*descending=*/true);

    // Compute cumulative sum
    auto cumsum = sorted.cumsum(/*dim=*/-1);

    // Compute k(z): number of elements in support
    auto range = torch::arange(1, n + 1, input.options());
    auto bound = 1.0f + range * sorted;
    auto is_in_support = (cumsum < bound).to(torch::kFloat32);
    auto k = is_in_support.sum(/*dim=*/-1, /*keepdim=*/true);

    // Compute tau (threshold). tau = (cumsum_{k} - 1) / k, where cumsum_{k} is
    // the cumulative sum AT the support boundary index k -- a single gathered
    // value, not the sum of every in-support cumsum. Summing the in-support
    // cumsums (cumsum_1 + ... + cumsum_k) over-counts by a factor that grows
    // with k, giving a too-large tau and a non-normalized output for any
    // support size > 1 (only k == 1 was correct).
    auto k_idx = (k.to(torch::kLong) - 1).clamp_min(0);
    auto cumsum_k = cumsum.gather(/*dim=*/-1, k_idx);
    auto tau = (cumsum_k - 1.0f) / k;

    // Compute output
    auto output = (input - tau).clamp_min(0.0f);

    // Reshape back
    output = output.reshape(original_shape);

    // Transpose back if needed
    if (original_dim < 0) {
        original_dim = static_cast<int64_t>(original_shape.size()) + original_dim;
    }
    if (original_dim != static_cast<int64_t>(original_shape.size()) - 1) {
        output = output.transpose(original_dim, -1);
    }

    return output;
}

// =============================================================================
// Entmax-1.5 Implementation
// =============================================================================
//
// alpha-entmax (Peters, Niculae & Martins, "Sparse Sequence-to-Sequence
// Models", ACL 2019; arXiv:1905.05702) is the family of simplex mappings
//
//     alpha-entmax(z) := argmax_{p in simplex} p^T z + H^T_alpha(p)     (Eq. 10)
//
// over the Tsallis alpha-entropies, with softmax (alpha = 1) and sparsemax
// (alpha = 2) as its endpoints. The solution has the closed form
//
//     alpha-entmax(z) = [ (alpha - 1) z - tau * 1 ]_+ ^ (1/(alpha - 1))  (Eq. 13)
//
// where tau is the unique threshold making the entries sum to one; entries with
// z_j <= tau/(alpha - 1) get exactly zero. At alpha = 1.5 this reads
// [ z/2 - tau ]_+^2. The (alpha - 1) = 1/2 rescaling of z is what separates
// 1.5-entmax from a plain squared-hinge simplex projection: dropping it runs the
// operator at a different temperature and collapses its support towards
// sparsemax's.
//
// alpha = 1.5 admits an EXACT sort-based solution -- Algorithm 2 of the paper
// (Sec. 3.3, derived in Appendix C.2) -- reproduced here verbatim:
//
//     Algorithm 2  Compute 1.5-entmax(z) exactly
//     1  Sort z, yielding z_[d] <= ... <= z_[1]; set z <- z/2
//     2  for rho in 1, ..., d do
//     3      M(rho) <- 1/rho sum_{j=1}^{rho} z_[j]
//     4      S(rho) <- sum_{j=1}^{rho} (z_[j] - M(rho))^2
//     5      tau(rho) <- M(rho) - sqrt( 1/rho (1 - S(rho)) )
//     6      if z_[rho+1] <= tau(rho) <= z_[rho] then
//     7          return p* = [z - tau * 1]_+^2
//
// which is what the reference implementation (github.com/deep-spin/entmax,
// `entmax15`) computes. The paper's Algorithm 1 (bisection) is the
// general-alpha routine (`entmax_bisect`) and is deliberately not implemented
// here: alpha = 1.5 is the only value the engine uses, and Algorithm 2 solves
// it exactly and in closed form.
//
// The backward pass uses the closed-form Jacobian of Proposition 1 (Sec. 3.4):
// with s_i = (p*_i)^(2 - alpha) on the support and 0 elsewhere -- so
// s = sqrt(p*) at alpha = 1.5 --
//
//     d alpha-entmax(z) / dz = diag(s) - (1 / ||s||_1) s s^T.
//
// It is a hand-written autograd Function rather than autograd through the
// sort/sqrt because line 5's discriminant has to be clamped for the rho past
// the support, where sqrt's derivative is not finite.

namespace {

// Algorithm 2, lines 1-6, vectorized over rho and over every slice along `dim`.
// `z_half` is (alpha - 1) * z = z / 2 (line 1). Returns tau (keepdim) and the
// support size rho*.
std::pair<torch::Tensor, torch::Tensor> entmax15_threshold(
    const torch::Tensor& z_half, int64_t dim
) {
    const int64_t n = z_half.size(dim);

    // Line 1: z_[1] >= ... >= z_[d].
    auto sorted = std::get<0>(z_half.sort(dim, /*descending=*/true));

    // rho = 1, ..., d, laid out along `dim` so it broadcasts against `sorted`.
    std::vector<int64_t> rho_shape(static_cast<size_t>(z_half.dim()), 1);
    rho_shape[static_cast<size_t>(dim)] = n;
    auto rho = torch::arange(1, n + 1, z_half.options()).view(rho_shape);

    // Lines 3-4. S(rho) = sum_{j<=rho} (z_[j] - M(rho))^2 = rho (E[z^2] - E[z]^2)
    // over the top-rho prefix.
    auto mean = sorted.cumsum(dim) / rho;
    auto mean_sq = sorted.square().cumsum(dim) / rho;
    auto ss = rho * (mean_sq - mean.square());

    // Line 5. Past the true support 1 - S(rho) turns negative; clamping keeps
    // sqrt finite, and those rho fail the line-6 test regardless.
    auto delta = ((1.0 - ss) / rho).clamp_min(0.0);
    auto tau_all = mean - delta.sqrt();

    // Line 6. tau(rho) <= z_[rho] holds exactly for rho = 1, ..., rho*, so
    // counting it recovers rho* (the paired z_[rho*+1] <= tau(rho*) then follows
    // from the descending order).
    auto support = (tau_all <= sorted).to(torch::kLong).sum(dim, /*keepdim=*/true);
    auto tau = tau_all.gather(dim, (support - 1).clamp_min(0));
    return {tau, support};
}

class Entmax15Function : public torch::autograd::Function<Entmax15Function> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        int64_t dim
    ) {
        // alpha-entmax is invariant to adding a constant to every score, so
        // centering on the row max is free and keeps the squaring in range.
        auto shifted = input - std::get<0>(input.max(dim, /*keepdim=*/true));

        // Eq. 13 at alpha = 1.5: the (alpha - 1) = 1/2 rescaling of z (line 1).
        auto z_half = shifted / 2.0;

        // Line 7.
        auto tau = entmax15_threshold(z_half, dim).first;
        auto output = (z_half - tau).clamp_min(0.0).square();

        ctx->save_for_backward({output});
        ctx->saved_data["dim"] = dim;
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        const auto& p = saved[0];
        const int64_t dim = ctx->saved_data["dim"].toInt();

        // Proposition 1: s_i = (p*_i)^(2 - alpha) = sqrt(p*_i) at alpha = 1.5,
        // and 0 off the support -- which sqrt(0) = 0 already gives. The Jacobian
        // diag(s) - s s^T / ||s||_1 applied to dY is s . dY - <s, dY>/||s||_1 s.
        auto s = p.sqrt();
        auto ds = grad_outputs[0] * s;
        auto q = ds.sum(dim, /*keepdim=*/true) / s.sum(dim, /*keepdim=*/true);
        return {ds - q * s, torch::Tensor()};
    }
};

}  // namespace

torch::Tensor entmax15(torch::Tensor input, int64_t dim) {
    const int64_t ndim = input.dim();
    TORCH_CHECK(ndim > 0, "entmax15: input must have at least 1 dimension");
    TORCH_CHECK(input.is_floating_point(),
        "entmax15: input must be a floating-point tensor, got ",
        input.scalar_type());
    TORCH_CHECK(dim >= -ndim && dim < ndim,
        "entmax15: dim ", dim, " is out of range for a ", ndim, "-D input");
    if (dim < 0) {
        dim += ndim;
    }

    // The threshold search squares the scores and differences near-equal
    // cumulative moments; run it at >= float32 and hand the result back in the
    // caller's dtype so an autocast fp16 call site stays well conditioned.
    const auto dtype = input.scalar_type();
    const bool promote = (dtype == torch::kHalf || dtype == torch::kBFloat16);

    auto output =
        Entmax15Function::apply(promote ? input.to(torch::kFloat) : input, dim);
    return promote ? output.to(dtype) : output;
}

// =============================================================================
// Row Attention Implementation
// =============================================================================

RowAttentionImpl::RowAttentionImpl(
    int64_t d_model,
    int64_t n_heads,
    float dropout
) {
    attention_ = register_module("attention",
        MultiHeadAttention(d_model, n_heads, dropout));
    norm_ = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
}

torch::Tensor RowAttentionImpl::forward(torch::Tensor x) {
    // x: (batch, n_features, d_model)
    // We want attention across batch dim for each feature

    // Transpose to (n_features, batch, d_model) - treat features as batch
    x = x.transpose(0, 1);

    // Apply self-attention (attends across "batch" which is now samples)
    auto normed = norm_->forward(x);
    auto attn_out = attention_->forward(normed, normed, normed);

    // Residual connection
    x = x + attn_out;

    // Transpose back to (batch, n_features, d_model)
    return x.transpose(0, 1);
}

// =============================================================================
// FT-Transformer Encoder Implementation
// =============================================================================

FTTransformerEncoderImpl::FTTransformerEncoderImpl(
    int64_t n_numerical,
    std::vector<int64_t> cat_cardinalities,
    int64_t d_model,
    int64_t n_heads,
    int64_t n_layers,
    int64_t d_ff,
    float dropout,
    bool use_cls_token,
    bool pre_norm
) : d_model_(d_model), use_cls_token_(use_cls_token) {

    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, d_model, use_cls_token));

    encoder_ = register_module("encoder",
        TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout, pre_norm));
}

torch::Tensor FTTransformerEncoderImpl::forward(
    torch::Tensor numerical,
    std::vector<torch::Tensor> categoricals
) {
    // Tokenize features: (batch, n_tokens, d_model)
    auto tokens = tokenizer_->forward(numerical, categoricals);

    // Apply transformer encoder
    auto encoded = encoder_->forward(tokens);

    if (use_cls_token_) {
        // Return CLS token representation: (batch, d_model)
        return encoded.select(1, 0);
    } else {
        // Return mean-pooled representation
        return encoded.mean(/*dim=*/1);
    }
}

// =============================================================================
// TabNet Step Implementation
// =============================================================================

TabNetGLUBlockImpl::TabNetGLUBlockImpl(int64_t in_dim, int64_t out_dim) {
    fc_ = register_module("fc", torch::nn::Linear(in_dim, 2 * out_dim));
    bn_ = register_module("bn", torch::nn::BatchNorm1d(2 * out_dim));
}

torch::Tensor TabNetGLUBlockImpl::forward(torch::Tensor x) {
    // GLU halves the last dimension: out = a * sigmoid(b) with [a,b] = FC(x).
    return torch::glu(bn_->forward(fc_->forward(x)), /*dim=*/1);
}

TabNetStepImpl::TabNetStepImpl(
    int64_t input_dim, int64_t n_d, int64_t n_a, int64_t n_independent,
    bool use_sparsemax
) : input_dim_(input_dim), n_d_(n_d), n_a_(n_a), use_sparsemax_(use_sparsemax) {
    // Attentive transformer: maps the previous attention split (n_a) to a
    // per-feature logit (input_dim), batch-normalized, then masked by the prior
    // scale and projected onto the simplex by `attentive_forward`.
    attention_fc_ = register_module("attention_fc", torch::nn::Linear(n_a, input_dim));
    bn_attention_ = register_module("bn_attention", torch::nn::BatchNorm1d(input_dim));

    // Step-specific feature-transformer GLU blocks (dim n_d + n_a throughout).
    independent_ = register_module("independent", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_independent; ++i) {
        independent_->push_back(TabNetGLUBlock(n_d + n_a, n_d + n_a));
    }
}

torch::Tensor TabNetStepImpl::attentive_forward(
    torch::Tensor att_prev, torch::Tensor prior_scales
) {
    auto logits = bn_attention_->forward(attention_fc_->forward(att_prev));
    logits = logits * prior_scales;  // Prior scale down features already used.
    // TabNetConfig::use_sparsemax picks the sparse simplex mapping: sparsemax
    // (alpha = 2, Arik & Pfister's default) or 1.5-entmax, which is strictly
    // less aggressive and keeps a larger mask support.
    return use_sparsemax_ ? sparsemax(logits, /*dim=*/-1)
                          : entmax15(logits, /*dim=*/-1);
}

torch::Tensor TabNetStepImpl::feature_independent(torch::Tensor h) {
    const float scale = std::sqrt(0.5f);
    for (size_t i = 0; i < independent_->size(); ++i) {
        auto block = independent_->ptr(i)->as<TabNetGLUBlockImpl>();
        h = (h + block->forward(h)) * scale;
    }
    return h;
}

// =============================================================================
// TabNet Encoder Implementation
// =============================================================================

namespace {
constexpr int64_t kTabNetNShared = 2;       // Shared feature-transformer blocks
constexpr int64_t kTabNetNIndependent = 2;  // Step-specific blocks
}  // namespace

TabNetEncoderImpl::TabNetEncoderImpl(
    int64_t input_dim,
    int64_t n_steps,
    int64_t n_d,
    int64_t n_a,
    float relaxation_factor,
    float sparsity_coefficient,
    bool use_sparsemax
) : input_dim_(input_dim), n_steps_(n_steps), n_d_(n_d), n_a_(n_a),
    relaxation_factor_(relaxation_factor), sparsity_coefficient_(sparsity_coefficient),
    use_sparsemax_(use_sparsemax) {

    // Batch-normalize the raw input features (Arik & Pfister, Sec. 3.2).
    initial_bn_ = register_module("initial_bn", torch::nn::BatchNorm1d(input_dim));

    // Shared feature-transformer GLU blocks, reused across every step. The first
    // maps input_dim -> n_d + n_a; the rest keep n_d + n_a.
    shared_ = register_module("shared", torch::nn::ModuleList());
    for (int64_t i = 0; i < kTabNetNShared; ++i) {
        int64_t in_dim = (i == 0) ? input_dim : (n_d + n_a);
        shared_->push_back(TabNetGLUBlock(in_dim, n_d + n_a));
    }

    // Decision steps (attentive transformer + independent feature-transformer).
    steps_ = register_module("steps", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_steps; ++i) {
        steps_->push_back(
            TabNetStep(input_dim, n_d, n_a, kTabNetNIndependent, use_sparsemax));
    }
}

torch::Tensor TabNetEncoderImpl::run_shared(torch::Tensor x) const {
    const float scale = std::sqrt(0.5f);
    auto first = shared_->ptr(0)->as<TabNetGLUBlockImpl>();
    torch::Tensor h = first->forward(x);  // dim change; no residual on the first
    for (size_t i = 1; i < shared_->size(); ++i) {
        auto block = shared_->ptr(i)->as<TabNetGLUBlockImpl>();
        h = (h + block->forward(h)) * scale;
    }
    return h;
}

std::pair<torch::Tensor, torch::Tensor> TabNetEncoderImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);

    // Original (batch-normalized) features. These are what each step's mask is
    // applied to -- masking is per-step over the ORIGINAL features (M[i] * f),
    // not cumulative over a running product of masks.
    auto features = initial_bn_->forward(x);

    // Seed the first attention split a[0] from an initial (shared) feature
    // transform over the full features.
    auto att = run_shared(features).slice(/*dim=*/1, n_d_, n_d_ + n_a_);

    // All features equally available at the start.
    auto prior_scales = torch::ones({batch_size, input_dim_}, features.options());
    auto aggregated_output = torch::zeros({batch_size, n_d_}, features.options());
    auto total_entropy = torch::zeros({batch_size}, features.options());
    auto feature_importance = torch::zeros({batch_size, input_dim_}, features.options());

    for (int64_t step = 0; step < n_steps_; ++step) {
        auto step_module = steps_->ptr(step)->as<TabNetStepImpl>();

        // Attentive transformer -> sparse simplex mask over the original
        // features (sparsemax or 1.5-entmax, per use_sparsemax_).
        auto mask = step_module->attentive_forward(att, prior_scales);

        // Relaxation: features selected now are less available to later steps.
        prior_scales = prior_scales * (relaxation_factor_ - mask);

        // Feature transformer over the masked original features.
        auto ft = step_module->feature_independent(run_shared(mask * features));
        auto decision = torch::relu(ft.slice(/*dim=*/1, 0, n_d_));
        att = ft.slice(/*dim=*/1, n_d_, n_d_ + n_a_);

        aggregated_output = aggregated_output + decision;
        feature_importance = feature_importance + mask;

        // Sparsity loss: entropy of attention masks.
        auto mask_entropy = -mask * torch::log(mask + 1e-15f);
        total_entropy = total_entropy + mask_entropy.sum(/*dim=*/-1);
    }

    feature_importance = feature_importance / static_cast<float>(n_steps_);
    sparsity_loss_ = sparsity_coefficient_ * total_entropy.mean();

    return std::make_pair(aggregated_output, feature_importance);
}

// =============================================================================
// SAINT Block Implementation
// =============================================================================

SAINTBlockImpl::SAINTBlockImpl(
    int64_t d_model,
    int64_t n_heads,
    int64_t d_ff,
    float dropout,
    bool use_row_attention
) : use_row_attention_(use_row_attention) {

    col_attention_ = register_module("col_attention",
        TransformerBlock(d_model, n_heads, d_ff, dropout, /*pre_norm=*/true));

    if (use_row_attention) {
        row_attention_ = register_module("row_attention",
            RowAttention(d_model, n_heads, dropout));
    }
}

torch::Tensor SAINTBlockImpl::forward(torch::Tensor x) {
    // Column attention: attend across features (tokens)
    x = col_attention_->forward(x);

    // Row attention: attend across samples
    if (use_row_attention_) {
        x = row_attention_->forward(x);
    }

    return x;
}

// =============================================================================
// SAINT Encoder Implementation
// =============================================================================

SAINTEncoderImpl::SAINTEncoderImpl(
    int64_t n_numerical,
    std::vector<int64_t> cat_cardinalities,
    int64_t d_model,
    int64_t n_heads,
    int64_t n_layers,
    int64_t d_ff,
    float dropout,
    bool use_row_attention,
    bool use_cls_token
) : d_model_(d_model), use_cls_token_(use_cls_token) {

    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, d_model, use_cls_token));

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(SAINTBlock(d_model, n_heads, d_ff, dropout, use_row_attention));
    }

    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
}

torch::Tensor SAINTEncoderImpl::forward(
    torch::Tensor numerical,
    std::vector<torch::Tensor> categoricals
) {
    // Tokenize features
    auto tokens = tokenizer_->forward(numerical, categoricals);

    // Apply SAINT blocks
    for (auto& layer : *layers_) {
        tokens = layer->as<SAINTBlockImpl>()->forward(tokens);
    }

    // Final normalization
    tokens = final_norm_->forward(tokens);

    if (use_cls_token_) {
        // Return CLS token
        return tokens.select(1, 0);
    } else {
        // Mean pooling
        return tokens.mean(/*dim=*/1);
    }
}

// =============================================================================
// Bilinear Trait Interaction Implementation
// =============================================================================

BilinearTraitInteractionImpl::BilinearTraitInteractionImpl(
    int64_t env_dim,
    int64_t trait_dim,
    int64_t output_dim
) : env_dim_(env_dim), trait_dim_(trait_dim), output_dim_(output_dim) {

    // Bilinear weight: out = env^T * W * traits
    weight_ = register_parameter("weight",
        torch::randn({output_dim, env_dim, trait_dim}) * 0.01f);
    bias_ = register_parameter("bias",
        torch::zeros({output_dim}));
}

torch::Tensor BilinearTraitInteractionImpl::forward(
    torch::Tensor env,
    torch::Tensor traits
) {
    // env: (batch, env_dim)
    // traits: (n_species, trait_dim)
    // weight: (output_dim, env_dim, trait_dim)

    // Compute bilinear: env @ W @ traits^T
    // Result: (batch, output_dim, n_species)

    // First: env @ W -> (batch, output_dim, trait_dim)
    auto env_transformed = torch::einsum("be,oet->bot", {env, weight_});

    // Then: (batch, output_dim, trait_dim) @ traits^T -> (batch, output_dim, n_species)
    auto output = torch::einsum("bot,st->bos", {env_transformed, traits});

    // Add bias: (batch, output_dim, n_species)
    output = output + bias_.view({1, output_dim_, 1});

    // Permute to (batch, n_species, output_dim)
    return output.permute({0, 2, 1});
}

// =============================================================================
// TraitNet Encoder Implementation
// =============================================================================

TraitNetEncoderImpl::TraitNetEncoderImpl(
    int64_t env_dim,
    int64_t trait_dim,
    int64_t n_species,
    int64_t hidden_dim,
    int64_t n_layers,
    float dropout
) : env_dim_(env_dim), trait_dim_(trait_dim), n_species_(n_species), hidden_dim_(hidden_dim) {

    // Environment encoder
    env_encoder_ = register_module("env_encoder", torch::nn::Sequential());
    int64_t current_dim = env_dim;
    for (int64_t i = 0; i < n_layers; ++i) {
        env_encoder_->push_back(torch::nn::Linear(current_dim, hidden_dim));
        env_encoder_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        env_encoder_->push_back(torch::nn::GELU());
        if (dropout > 0) {
            env_encoder_->push_back(torch::nn::Dropout(dropout));
        }
        current_dim = hidden_dim;
    }

    // Trait encoder
    trait_encoder_ = register_module("trait_encoder", torch::nn::Sequential());
    current_dim = trait_dim;
    for (int64_t i = 0; i < n_layers; ++i) {
        trait_encoder_->push_back(torch::nn::Linear(current_dim, hidden_dim));
        trait_encoder_->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        trait_encoder_->push_back(torch::nn::GELU());
        if (dropout > 0) {
            trait_encoder_->push_back(torch::nn::Dropout(dropout));
        }
        current_dim = hidden_dim;
    }

    // Bilinear interaction
    interaction_ = register_module("interaction",
        BilinearTraitInteraction(hidden_dim, hidden_dim, hidden_dim));

    // Output projection
    output_proj_ = register_module("output_proj",
        torch::nn::Linear(hidden_dim, 1));
}

void TraitNetEncoderImpl::set_traits(torch::Tensor traits) {
    TORCH_CHECK(traits.size(0) == n_species_,
        "traits must have ", n_species_, " species, got ", traits.size(0));
    TORCH_CHECK(traits.size(1) == trait_dim_,
        "traits must have ", trait_dim_, " features, got ", traits.size(1));
    traits_ = traits;
}

torch::Tensor TraitNetEncoderImpl::forward(
    torch::Tensor env,
    torch::Tensor traits
) {
    // Use stored traits if not provided
    if (!traits.defined()) {
        TORCH_CHECK(traits_.defined(),
            "traits not provided and not set via set_traits()");
        traits = traits_;
    }

    // Encode environment: (batch, env_dim) -> (batch, hidden_dim)
    auto env_encoded = env_encoder_->forward(env);

    // Encode traits: (n_species, trait_dim) -> (n_species, hidden_dim)
    auto traits_encoded = trait_encoder_->forward(traits);

    // Bilinear interaction: (batch, n_species, hidden_dim)
    auto interaction = interaction_->forward(env_encoded, traits_encoded);

    // Apply GELU
    interaction = torch::gelu(interaction);

    // Project to predictions: (batch, n_species, 1) -> (batch, n_species)
    auto output = output_proj_->forward(interaction).squeeze(-1);

    return output;
}

// =============================================================================
// ExcelFormer Encoder
// =============================================================================

ExcelFormerEncoderImpl::ExcelFormerEncoderImpl(
    int64_t n_numerical,
    std::vector<int64_t> cat_cardinalities,
    int64_t d_model,
    int64_t n_heads,
    int64_t n_layers,
    int64_t d_ff,
    float dropout,
    float importance_threshold,
    bool use_cls_token
) : d_model_(d_model),
    importance_threshold_(importance_threshold),
    use_cls_token_(use_cls_token)
{
    // Create feature tokenizer
    tokenizer_ = register_module("tokenizer",
        FeatureTokenizer(n_numerical, cat_cardinalities, d_model, use_cls_token));

    n_tokens_ = tokenizer_->n_tokens();

    // Learnable feature importance scores (one per token)
    importance_logits_ = register_parameter("importance_logits",
        torch::zeros({n_tokens_}));

    // Transformer layers
    if (d_ff == 0) d_ff = 4 * d_model;
    // Register blocks only through the ModuleList. Wrapping each block in its
    // own register_module("block_i", ...) as well would register it twice (as a
    // child of *this AND of layers_), so named_parameters() would yield every
    // block weight twice -- doubling optimizer updates and checkpoint size.
    // Matches SAINTEncoder / TransformerEncoder above.
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(
            TransformerBlock(d_model, n_heads, d_ff, dropout, /*pre_norm=*/true));
    }

    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
}

torch::Tensor ExcelFormerEncoderImpl::build_attention_mask() const {
    // Learned feature importance in (0, 1)
    auto importance = torch::sigmoid(importance_logits_);  // (n_tokens,)

    // Semi-permeable rule: feature i can attend to feature j if
    //   - importance(j) >= importance_threshold (j is informative, visible to all)
    //   - OR importance(j) >= importance(i) (j is at least as important as i)
    auto imp_row = importance.unsqueeze(1);  // (n_tokens, 1) - querying token i
    auto imp_col = importance.unsqueeze(0);  // (1, n_tokens) - key token j

    // Build the rule as a SMOOTH gate so gradient flows into importance_logits_.
    // A hard boolean cast (imp >= thr) has zero gradient, so the importance
    // parameter never trains and, at the default threshold, every gate opens and
    // ExcelFormer degenerates to a plain full-attention transformer.
    const float tau = 0.1f;  // temperature for the soft comparisons
    auto is_informative    = torch::sigmoid((imp_col - importance_threshold_) / tau);
    auto is_more_important = torch::sigmoid((imp_col - imp_row) / tau);
    // Soft OR (a + b - a*b), in (0, 1)
    auto can_attend = is_informative + is_more_important - is_informative * is_more_important;

    // Convert the [0,1] gate to an additive log-space attention bias: ~0 where
    // the gate is open, large-negative where it is closed. MultiHeadAttention
    // adds this to the pre-softmax scores. Shape (1, n_tokens, n_tokens) so the
    // leading 1 broadcasts across the batch dimension.
    auto bias = torch::log(can_attend + 1e-9f);

    return bias.unsqueeze(0);  // (1, n_tokens, n_tokens)
}

torch::Tensor ExcelFormerEncoderImpl::forward(
    torch::Tensor numerical,
    std::vector<torch::Tensor> categoricals
) {
    // Tokenize features
    auto tokens = tokenizer_->forward(numerical, categoricals);  // (batch, n_tokens, d_model)

    // Build semi-permeable attention bias (additive log-bias, broadcasts over batch)
    auto attn_mask = build_attention_mask();  // (1, n_tokens, n_tokens)

    // Pass through transformer layers with attention mask
    auto x = tokens;
    for (const auto& layer : *layers_) {
        x = layer->as<TransformerBlockImpl>()->forward(x, attn_mask);
    }

    x = final_norm_->forward(x);

    // Extract CLS token or mean pool
    if (use_cls_token_) {
        return x.select(1, 0);  // (batch, d_model) - CLS token
    } else {
        return x.mean(1);  // (batch, d_model) - mean pool
    }
}

torch::Tensor ExcelFormerEncoderImpl::feature_importance() const {
    return torch::sigmoid(importance_logits_).detach();
}

}  // namespace resolve
