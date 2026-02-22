#include "resolve/attention.hpp"
#include "resolve/types.hpp"
#include <algorithm>

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

    // Apply mask if provided
    if (mask.defined()) {
        if (mask.dim() == 2) {
            // (batch, seq_len_k) -> (batch, 1, 1, seq_len_k)
            mask = mask.unsqueeze(1).unsqueeze(2);
        } else if (mask.dim() == 3) {
            // (batch, seq_len_q, seq_len_k) -> (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1);
        }
        scores = scores.masked_fill(mask == 0, -1e9f);
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

    // Categorical features
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

    // Compute tau (threshold)
    auto cumsum_k = (cumsum * is_in_support).sum(/*dim=*/-1, /*keepdim=*/true);
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

torch::Tensor entmax15(torch::Tensor input, int64_t dim) {
    // Entmax with alpha=1.5
    // This is a simplified iterative implementation

    auto original_dim = dim;
    if (dim < 0) {
        dim = input.dim() + dim;
    }

    if (dim != input.dim() - 1) {
        input = input.transpose(dim, -1);
    }

    auto original_shape = input.sizes().vec();
    auto n = input.size(-1);
    input = input.reshape({-1, n});

    // Iterative bisection to find threshold tau
    auto tau_lo = std::get<0>(input.min(/*dim=*/-1, /*keepdim=*/true)) - 1.0f;
    auto tau_hi = std::get<0>(input.max(/*dim=*/-1, /*keepdim=*/true));

    for (int iter = 0; iter < 20; ++iter) {
        auto tau_mid = (tau_lo + tau_hi) / 2.0f;
        auto p = torch::relu(input - tau_mid).pow(2);
        auto sum_p = p.sum(/*dim=*/-1, /*keepdim=*/true);

        auto mask = (sum_p > 1.0f);
        tau_lo = torch::where(mask, tau_mid, tau_lo);
        tau_hi = torch::where(mask, tau_hi, tau_mid);
    }

    auto tau = (tau_lo + tau_hi) / 2.0f;
    auto output = torch::relu(input - tau).pow(2);

    // Normalize
    output = output / output.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1e-10f);

    output = output.reshape(original_shape);
    if (original_dim < 0) {
        original_dim = static_cast<int64_t>(original_shape.size()) + original_dim;
    }
    if (original_dim != static_cast<int64_t>(original_shape.size()) - 1) {
        output = output.transpose(original_dim, -1);
    }

    return output;
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

    auto batch_size = x.size(0);
    auto n_features = x.size(1);
    auto d_model = x.size(2);

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

TabNetStepImpl::TabNetStepImpl(
    int64_t input_dim,
    int64_t n_d,
    int64_t n_a,
    float relaxation_factor
) : input_dim_(input_dim), n_d_(n_d), n_a_(n_a), relaxation_factor_(relaxation_factor) {

    // Shared fully connected layer
    shared_fc_ = register_module("shared_fc",
        torch::nn::Linear(input_dim, n_d + n_a));
    bn_shared_ = register_module("bn_shared",
        torch::nn::BatchNorm1d(n_d + n_a));

    // Decision layer (for output)
    decision_fc_ = register_module("decision_fc",
        torch::nn::Linear(n_d, n_d));
    bn_decision_ = register_module("bn_decision",
        torch::nn::BatchNorm1d(n_d));

    // Attention layer (for feature selection)
    attention_fc_ = register_module("attention_fc",
        torch::nn::Linear(n_a, input_dim));
    bn_attention_ = register_module("bn_attention",
        torch::nn::BatchNorm1d(input_dim));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> TabNetStepImpl::forward(
    torch::Tensor x,
    torch::Tensor prior_scales
) {
    // Shared transformation
    auto shared = bn_shared_->forward(shared_fc_->forward(x));
    shared = torch::relu(shared);

    // Split into decision and attention parts
    auto decision_input = shared.slice(/*dim=*/1, 0, n_d_);
    auto attention_input = shared.slice(/*dim=*/1, n_d_, n_d_ + n_a_);

    // Decision output
    auto decision_out = bn_decision_->forward(decision_fc_->forward(decision_input));
    decision_out = torch::relu(decision_out);

    // Attention mask using sparsemax
    auto attention_logits = bn_attention_->forward(attention_fc_->forward(attention_input));
    attention_logits = attention_logits * prior_scales;  // Mask by prior scales
    auto mask = sparsemax(attention_logits, /*dim=*/-1);

    // Update prior scales
    auto new_prior_scales = prior_scales * (relaxation_factor_ - mask);

    // Masked input for next step
    auto masked_x = x * mask;

    return std::make_tuple(decision_out, mask, new_prior_scales);
}

// =============================================================================
// TabNet Encoder Implementation
// =============================================================================

TabNetEncoderImpl::TabNetEncoderImpl(
    int64_t input_dim,
    int64_t n_steps,
    int64_t n_d,
    int64_t n_a,
    float relaxation_factor,
    float sparsity_coefficient
) : input_dim_(input_dim), n_steps_(n_steps), n_d_(n_d), n_a_(n_a),
    sparsity_coefficient_(sparsity_coefficient) {

    // Initial batch normalization (using linear for simplicity)
    initial_bn_ = register_module("initial_bn",
        torch::nn::Linear(torch::nn::LinearOptions(input_dim, input_dim).bias(false)));

    // Initialize as identity
    torch::NoGradGuard no_grad;
    initial_bn_->weight.copy_(torch::eye(input_dim));

    // Decision steps
    steps_ = register_module("steps", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_steps; ++i) {
        steps_->push_back(TabNetStep(input_dim, n_d, n_a, relaxation_factor));
    }
}

std::pair<torch::Tensor, torch::Tensor> TabNetEncoderImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);

    // Initial batch norm
    x = initial_bn_->forward(x);

    // Initialize prior scales (all features equally available)
    auto prior_scales = torch::ones({batch_size, input_dim_}, x.options());

    // Aggregate outputs
    auto aggregated_output = torch::zeros({batch_size, n_d_}, x.options());
    auto total_entropy = torch::zeros({batch_size}, x.options());
    auto feature_importance = torch::zeros({batch_size, input_dim_}, x.options());

    for (int64_t step = 0; step < n_steps_; ++step) {
        auto step_module = steps_->ptr(step)->as<TabNetStepImpl>();
        auto [decision_out, mask, new_prior_scales] = step_module->forward(x, prior_scales);

        // Aggregate decision outputs
        aggregated_output = aggregated_output + decision_out;

        // Track feature importance
        feature_importance = feature_importance + mask;

        // Sparsity loss: entropy of attention masks
        auto mask_entropy = -mask * torch::log(mask + 1e-15f);
        total_entropy = total_entropy + mask_entropy.sum(/*dim=*/-1);

        // Update for next step
        prior_scales = new_prior_scales;
        x = x * mask;  // Apply mask for next step
    }

    // Normalize feature importance
    feature_importance = feature_importance / static_cast<float>(n_steps_);

    // Store sparsity loss
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

    auto batch_size = env.size(0);
    auto n_species = traits.size(0);

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
// GCN Layer Implementation
// =============================================================================

GCNLayerImpl::GCNLayerImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias
) {
    linear_ = register_module("linear",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
}

torch::Tensor GCNLayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    // GCN formula: H' = sigma(A_hat * H * W)
    // where A_hat is normalized adjacency (D^-0.5 * A * D^-0.5)

    // First apply linear transformation
    auto transformed = linear_->forward(x);

    // Then message passing via adjacency
    return torch::matmul(adj, transformed);
}

// =============================================================================
// GAT Layer Implementation
// =============================================================================

GATLayerImpl::GATLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_heads,
    float dropout,
    bool concat
) : in_features_(in_features), out_features_(out_features),
    n_heads_(n_heads), concat_(concat) {

    W_ = register_module("W",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features * n_heads).bias(false)));

    // Attention parameters
    a_src_ = register_parameter("a_src", torch::randn({n_heads, out_features}) * 0.01f);
    a_dst_ = register_parameter("a_dst", torch::randn({n_heads, out_features}) * 0.01f);

    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
    leaky_relu_ = register_module("leaky_relu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.2)));
}

torch::Tensor GATLayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    auto n_nodes = x.size(0);

    // Linear transformation: (n_nodes, in_features) -> (n_nodes, n_heads, out_features)
    auto Wh = W_->forward(x).view({n_nodes, n_heads_, out_features_});

    // Compute attention scores
    // Source attention: (n_nodes, n_heads)
    auto attn_src = torch::einsum("nho,ho->nh", {Wh, a_src_});
    // Destination attention: (n_nodes, n_heads)
    auto attn_dst = torch::einsum("nho,ho->nh", {Wh, a_dst_});

    // Pairwise attention: (n_nodes, n_nodes, n_heads)
    auto attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0);
    attn = leaky_relu_->forward(attn);

    // Mask out non-edges using adjacency
    auto mask = (adj == 0).unsqueeze(-1).expand_as(attn);
    attn = attn.masked_fill(mask, -1e9f);

    // Softmax over neighbors
    attn = torch::softmax(attn, /*dim=*/1);

    if (dropout_.is_empty() == false) {
        attn = dropout_->forward(attn);
    }

    // Aggregate: (n_nodes, n_heads, out_features)
    auto output = torch::einsum("nmh,mho->nho", {attn, Wh});

    if (concat_) {
        // Concat heads: (n_nodes, n_heads * out_features)
        return output.view({n_nodes, n_heads_ * out_features_});
    } else {
        // Average heads: (n_nodes, out_features)
        return output.mean(/*dim=*/1);
    }
}

// =============================================================================
// GraphSAGE Layer Implementation
// =============================================================================

GraphSAGELayerImpl::GraphSAGELayerImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias
) {
    linear_self_ = register_module("linear_self",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias)));
    linear_neighbor_ = register_module("linear_neighbor",
        torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false)));
}

torch::Tensor GraphSAGELayerImpl::forward(torch::Tensor x, torch::Tensor adj) {
    // GraphSAGE: h_v = sigma(W_self * h_v + W_neighbor * mean(h_neighbors))

    // Self transformation
    auto self_out = linear_self_->forward(x);

    // Neighbor aggregation (mean)
    // Normalize adjacency by degree for mean aggregation
    auto degree = adj.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(1.0f);
    auto adj_norm = adj / degree;

    auto neighbor_agg = torch::matmul(adj_norm, x);
    auto neighbor_out = linear_neighbor_->forward(neighbor_agg);

    return self_out + neighbor_out;
}

// =============================================================================
// GNN Encoder Implementation
// =============================================================================

GNNEncoderImpl::GNNEncoderImpl(
    int64_t in_features,
    int64_t hidden_dim,
    int64_t out_features,
    int64_t n_layers,
    GNNType gnn_type,
    int64_t n_heads,
    float dropout
) : out_features_(out_features), gnn_type_(gnn_type) {

    layers_ = register_module("layers", torch::nn::ModuleList());

    int64_t current_dim = in_features;

    for (int64_t i = 0; i < n_layers; ++i) {
        int64_t next_dim = (i == n_layers - 1) ? out_features : hidden_dim;

        switch (gnn_type) {
            case GNNType::GCN:
                layers_->push_back(GCNLayer(current_dim, next_dim));
                break;
            case GNNType::GAT: {
                bool concat = (i < n_layers - 1);  // Don't concat on last layer
                layers_->push_back(GATLayer(current_dim, next_dim, n_heads, dropout, concat));
                if (concat) {
                    next_dim = next_dim * n_heads;
                }
                break;
            }
            case GNNType::GraphSAGE:
                layers_->push_back(GraphSAGELayer(current_dim, next_dim));
                break;
        }

        current_dim = next_dim;
    }

    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor GNNEncoderImpl::forward(torch::Tensor x, torch::Tensor adj) {
    bool batched = x.dim() == 3;

    if (batched) {
        // Process each graph in the batch
        auto batch_size = x.size(0);
        std::vector<torch::Tensor> outputs;

        for (int64_t b = 0; b < batch_size; ++b) {
            auto x_b = x.select(0, b);
            auto adj_b = adj.select(0, b);

            for (size_t i = 0; i < layers_->size(); ++i) {
                switch (gnn_type_) {
                    case GNNType::GCN:
                        x_b = layers_->ptr(static_cast<int64_t>(i))->as<GCNLayerImpl>()->forward(x_b, adj_b);
                        break;
                    case GNNType::GAT:
                        x_b = layers_->ptr(static_cast<int64_t>(i))->as<GATLayerImpl>()->forward(x_b, adj_b);
                        break;
                    case GNNType::GraphSAGE:
                        x_b = layers_->ptr(static_cast<int64_t>(i))->as<GraphSAGELayerImpl>()->forward(x_b, adj_b);
                        break;
                }

                // Apply ReLU between layers (not after last)
                if (i < layers_->size() - 1) {
                    x_b = torch::relu(x_b);
                    if (dropout_.is_empty() == false) {
                        x_b = dropout_->forward(x_b);
                    }
                }
            }
            outputs.push_back(x_b);
        }

        return torch::stack(outputs);
    } else {
        // Single graph
        for (size_t i = 0; i < layers_->size(); ++i) {
            switch (gnn_type_) {
                case GNNType::GCN:
                    x = layers_->ptr(static_cast<int64_t>(i))->as<GCNLayerImpl>()->forward(x, adj);
                    break;
                case GNNType::GAT:
                    x = layers_->ptr(static_cast<int64_t>(i))->as<GATLayerImpl>()->forward(x, adj);
                    break;
                case GNNType::GraphSAGE:
                    x = layers_->ptr(static_cast<int64_t>(i))->as<GraphSAGELayerImpl>()->forward(x, adj);
                    break;
            }

            if (i < layers_->size() - 1) {
                x = torch::relu(x);
                if (dropout_.is_empty() == false) {
                    x = dropout_->forward(x);
                }
            }
        }

        return x;
    }
}

// =============================================================================
// Typed Message Passing Layer Implementation
// =============================================================================

TypedMessagePassingLayerImpl::TypedMessagePassingLayerImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t n_edge_types,
    int64_t n_heads,
    float dropout
) : n_edge_types_(n_edge_types),
    in_features_(in_features),
    out_features_(out_features)
{
    // Per-edge-type message function: concat(src, tgt) -> message
    torch::nn::ModuleList msg_fns;
    for (int64_t t = 0; t < n_edge_types; ++t) {
        auto seq = torch::nn::Sequential();
        seq->push_back(torch::nn::Linear(2 * in_features, out_features));
        seq->push_back(torch::nn::GELU());
        seq->push_back(torch::nn::Linear(out_features, out_features));
        msg_fns->push_back(seq);
    }
    message_fns_ = register_module("message_fns", msg_fns);

    // Attention mechanism for aggregating incoming messages
    attn_query_ = register_module("attn_q",
        torch::nn::Linear(in_features, out_features));
    attn_key_ = register_module("attn_k",
        torch::nn::Linear(out_features, out_features));

    // Output projection + norm
    output_ = register_module("output",
        torch::nn::Linear(out_features, out_features));
    norm_ = register_module("norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({out_features})));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
}

torch::Tensor TypedMessagePassingLayerImpl::forward(
    torch::Tensor node_features,
    torch::Tensor edge_index,
    torch::Tensor edge_type
) {
    int64_t n_nodes = node_features.size(0);
    int64_t n_edges = edge_index.size(1);

    if (n_edges == 0) {
        // No edges: return projected features (or zeros if dim mismatch)
        if (in_features_ == out_features_) {
            return norm_->forward(node_features);
        }
        return norm_->forward(output_->forward(
            torch::zeros({n_nodes, out_features_}, node_features.options())));
    }

    // Gather source and target features
    auto src_idx = edge_index[0];  // (n_edges,)
    auto tgt_idx = edge_index[1];  // (n_edges,)

    auto src_feats = node_features.index_select(0, src_idx);  // (n_edges, in_features)
    auto tgt_feats = node_features.index_select(0, tgt_idx);  // (n_edges, in_features)

    // Compute messages per edge type
    auto messages = torch::zeros({n_edges, out_features_}, node_features.options());
    auto edge_input = torch::cat({src_feats, tgt_feats}, /*dim=*/1);  // (n_edges, 2*in_features)

    for (int64_t t = 0; t < n_edge_types_; ++t) {
        auto mask = (edge_type == t);  // (n_edges,)
        if (!mask.any().item<bool>()) continue;

        auto type_input = edge_input.index({mask});
        auto type_msgs = message_fns_[t]->as<torch::nn::Sequential>()->forward(type_input);
        messages.index_put_({mask}, type_msgs);
    }

    // Attention-weighted aggregation of messages to target nodes
    auto query = attn_query_->forward(node_features);  // (n_nodes, out_features)
    auto key = attn_key_->forward(messages);            // (n_edges, out_features)

    // Gather query for each edge's target node
    auto tgt_query = query.index_select(0, tgt_idx);  // (n_edges, out_features)

    // Attention scores: dot product / sqrt(d)
    auto attn_scores = (tgt_query * key).sum(-1) /
        std::sqrt(static_cast<float>(out_features_));  // (n_edges,)

    // Scatter softmax: exp + scatter_add + normalize
    auto attn_exp = attn_scores.exp();  // (n_edges,)
    auto attn_sum = torch::zeros({n_nodes}, node_features.options());
    attn_sum.scatter_add_(0, tgt_idx, attn_exp);
    auto attn_weights = attn_exp /
        (attn_sum.index_select(0, tgt_idx) + kEpsilon);  // (n_edges,)

    // Weight messages and scatter to target nodes
    auto weighted_msgs = messages * attn_weights.unsqueeze(1);  // (n_edges, out_features)
    auto aggregated = torch::zeros({n_nodes, out_features_}, node_features.options());
    aggregated.scatter_add_(0,
        tgt_idx.unsqueeze(1).expand_as(weighted_msgs),
        weighted_msgs);

    // Output projection + dropout
    auto out = output_->forward(aggregated);
    out = dropout_->forward(out);

    // Residual connection (if dimensions match)
    if (in_features_ == out_features_) {
        out = out + node_features;
    }
    out = norm_->forward(out);

    return out;
}

// =============================================================================
// Heterogeneous GNN Encoder Implementation
// =============================================================================

HeterogeneousGNNEncoderImpl::HeterogeneousGNNEncoderImpl(
    int64_t n_species,
    int64_t hidden_dim,
    int64_t output_dim,
    int64_t n_layers,
    int64_t n_edge_types,
    int64_t n_heads,
    float dropout
) : n_species_(n_species),
    output_dim_(output_dim)
{
    // Learnable species node embeddings
    species_embeddings_ = register_parameter("species_embeddings",
        torch::randn({n_species, hidden_dim}) * 0.02f);

    // Input projection (identity if hidden_dim matches)
    input_proj_ = register_module("input_proj",
        torch::nn::Linear(hidden_dim, hidden_dim));

    // Stack of typed message passing layers
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(
            register_module("mp_" + std::to_string(i),
                TypedMessagePassingLayer(
                    hidden_dim, hidden_dim, n_edge_types, n_heads, dropout)));
    }

    // Final projection to output dimension
    output_proj_ = register_module("output_proj",
        torch::nn::Linear(hidden_dim, output_dim));
    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({output_dim})));
}

torch::Tensor HeterogeneousGNNEncoderImpl::forward(
    torch::Tensor edge_index,
    torch::Tensor edge_type
) {
    // Start with learnable species embeddings
    auto x = input_proj_->forward(species_embeddings_);

    // Apply message passing layers
    for (auto& layer : *layers_) {
        x = layer->as<TypedMessagePassingLayerImpl>()->forward(
            x, edge_index, edge_type);
    }

    // Project to output dimension
    x = output_proj_->forward(x);
    x = final_norm_->forward(x);

    return x;  // (n_species, output_dim)
}

torch::Tensor HeterogeneousGNNEncoderImpl::aggregate_for_plots(
    torch::Tensor species_embeddings,
    torch::Tensor species_vector
) {
    // species_embeddings: (n_species, output_dim)
    // species_vector: (batch, n_species)
    //
    // Weighted sum of species embeddings per plot:
    // plot_feat = sum_s(abundance_s * embedding_s) / sum_s(abundance_s)

    // Normalize species vector (so it sums to 1 per plot)
    auto weights = species_vector / (species_vector.sum(/*dim=*/1, /*keepdim=*/true) + kEpsilon);

    // Weighted sum: (batch, n_species) @ (n_species, output_dim) = (batch, output_dim)
    return torch::matmul(weights, species_embeddings);
}

// =============================================================================
// Utility: k-NN Adjacency Construction
// =============================================================================

torch::Tensor build_knn_adjacency(torch::Tensor coords, int64_t k) {
    // coords: (n_nodes, 2)
    auto n_nodes = coords.size(0);

    // Compute pairwise distances
    auto diff = coords.unsqueeze(0) - coords.unsqueeze(1);  // (n, n, 2)
    auto dist = diff.pow(2).sum(-1).sqrt();  // (n, n)

    // Find k nearest neighbors (excluding self)
    dist.fill_diagonal_(std::numeric_limits<float>::infinity());
    auto [_, indices] = dist.topk(k, /*dim=*/1, /*largest=*/false);

    // Build adjacency matrix
    auto adj = torch::zeros({n_nodes, n_nodes}, coords.options());

    // Set edges
    auto rows = torch::arange(n_nodes, coords.options()).unsqueeze(1).expand_as(indices);
    adj.index_put_({rows, indices}, 1.0f);

    // Symmetrize
    adj = (adj + adj.transpose(0, 1)).clamp_max(1.0f);

    // Normalize (symmetric normalization: D^-0.5 * A * D^-0.5)
    auto degree = adj.sum(/*dim=*/1);
    auto d_inv_sqrt = torch::pow(degree.clamp_min(1.0f), -0.5f);
    adj = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0);

    return adj;
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
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_layers; ++i) {
        layers_->push_back(
            register_module("block_" + std::to_string(i),
                TransformerBlock(d_model, n_heads, d_ff, dropout, /*pre_norm=*/true)));
    }

    final_norm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
}

torch::Tensor ExcelFormerEncoderImpl::build_attention_mask() const {
    // Feature importance via sigmoid
    auto importance = torch::sigmoid(importance_logits_);  // (n_tokens,)

    // Sort features by importance (descending)
    auto [sorted_imp, sorted_idx] = importance.sort(/*dim=*/0, /*descending=*/true);

    // Build permeable attention mask:
    // Feature i can attend to feature j if:
    //   - importance(j) >= importance_threshold (j is informative, visible to all)
    //   - OR importance(j) >= importance(i) (j is more important than i)
    auto imp_row = importance.unsqueeze(1);  // (n_tokens, 1)
    auto imp_col = importance.unsqueeze(0);  // (1, n_tokens)

    // informative features are visible to everyone
    auto is_informative = (imp_col >= importance_threshold_);  // (1, n_tokens)

    // feature j is at least as important as feature i
    auto is_more_important = (imp_col >= imp_row);  // (n_tokens, n_tokens)

    // Combine: can attend if target is informative OR more important
    auto can_attend = is_informative | is_more_important;  // (n_tokens, n_tokens)

    // Convert to boolean-style mask compatible with MultiHeadAttention:
    // 1 = can attend, 0 = blocked. MHA uses masked_fill(mask == 0, -1e9).
    // Shape (1, n_tokens, n_tokens) so MHA treats as (batch, seq_q, seq_k)
    // and the leading 1 broadcasts across the batch dimension.
    auto mask = can_attend.to(torch::kFloat).unsqueeze(0);

    return mask;  // (1, n_tokens, n_tokens)
}

torch::Tensor ExcelFormerEncoderImpl::forward(
    torch::Tensor numerical,
    std::vector<torch::Tensor> categoricals
) {
    // Tokenize features
    auto tokens = tokenizer_->forward(numerical, categoricals);  // (batch, n_tokens, d_model)

    // Build semi-permeable attention mask
    auto attn_mask = build_attention_mask();  // (n_tokens, n_tokens)

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
