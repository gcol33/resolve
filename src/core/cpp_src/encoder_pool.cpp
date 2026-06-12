#include "resolve/encoder.hpp"
#include <cmath>
#include <stdexcept>

namespace resolve {

// =============================================================================
// Cover Dropout Helper
// =============================================================================

void apply_cover_dropout(
    bool training,
    float cover_dropout,
    int64_t batch_size,
    torch::Device device,
    torch::Tensor& weights,
    const torch::Tensor& mask,
    torch::Tensor& has_cover
) {
    if (!training || cover_dropout <= 0.0f) return;

    // Select samples to drop cover info
    auto drop_mask = torch::rand({batch_size}, torch::TensorOptions().device(device)) < cover_dropout;

    if (drop_mask.any().item<bool>()) {
        weights = weights.clone();
        has_cover = has_cover.clone();

        // Replace weights with binary mask (uniform 1/0) for dropped samples
        weights.index_put_({drop_mask}, mask.index({drop_mask}).to(torch::kFloat32));
        has_cover.index_put_({drop_mask}, 0.0f);
    }
}

// =============================================================================
// PlotEncoderRankPool Implementation
// =============================================================================

PlotEncoderRankPoolImpl::PlotEncoderRankPoolImpl(
    int64_t n_continuous,
    int64_t n_species,
    int64_t n_genera,
    int64_t n_families,
    int species_embed_dim,
    int genus_embed_dim,
    int family_embed_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& mlp_config,
    float cover_dropout,
    const TabMConfig& tabm_config
) : cover_dropout_(cover_dropout) {
    has_taxonomy_ = (n_genera > 0 && n_families > 0);

    // Single shared embedding tables with padding_idx=0
    species_embedding_ = register_module("species_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(n_species, species_embed_dim).padding_idx(0)));

    int64_t embed_dim = species_embed_dim;

    if (has_taxonomy_) {
        genus_embedding_ = register_module("genus_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_genera, genus_embed_dim).padding_idx(0)));
        family_embedding_ = register_module("family_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_families, family_embed_dim).padding_idx(0)));
        embed_dim += genus_embed_dim + family_embed_dim;
    }

    // MLP input: continuous + pooled embedding + has_cover flag
    int64_t input_dim = n_continuous + embed_dim + 1;

    auto bb = build_and_register_backbone(
        *this, mlp_, tabm_encoder_, input_dim, hidden_dims, mlp_config, tabm_config);
    use_tabm_ = bb.use_tabm;
    latent_dim_ = bb.latent_dim;
}

torch::Tensor PlotEncoderRankPoolImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor weights,
    torch::Tensor mask,
    torch::Tensor has_cover
) {
    int64_t batch_size = continuous.size(0);
    auto device = continuous.device();

    // Default has_cover to 1.0
    if (!has_cover.defined() || has_cover.numel() == 0) {
        has_cover = torch::ones({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Build mask from species_ids if not provided
    if (!mask.defined() || mask.numel() == 0) {
        mask = (species_ids != 0);
    }
    auto mask_float = mask.to(torch::kFloat32);

    // Default weights to mask (binary)
    if (!weights.defined() || weights.numel() == 0) {
        weights = mask_float;
    }

    // Apply cover dropout during training
    apply_cover_dropout(is_training(), cover_dropout_, batch_size, device,
                        weights, mask, has_cover);

    // Weighted mean pool weights: normalize per plot. Padding and UNK (id 0)
    // positions carry zero weight via `mask_float`, so they drop out of the
    // reduction; a plot with no species reduces to the zero vector (w_sum
    // clamped to kEpsilon, numerator 0).
    auto w = weights * mask_float;
    auto w_sum = w.sum(/*dim=*/1, /*keepdim=*/true).clamp_min(kEpsilon);
    auto w_normed = w / w_sum;  // (batch, max_sp)

    // Fused weighted-sum pooling via embedding_bag (mode=sum, per-sample
    // weights = w_normed) computes sum_j w_normed[:, j] * emb[ids[:, j]] in a
    // single kernel per table, WITHOUT materializing the
    // (batch, max_sp, embed_dim) gather that the explicit multiply-then-sum
    // path allocates. That transient scales with max_species and is the
    // dominant training-time VRAM spike on species-rich targets; on a WDDM GPU
    // it spilled into shared memory and stalled the driver past the watchdog
    // (issue #6). padding_idx=0 excludes pad/UNK rows from the reduction and
    // freezes row 0's gradient, matching each table's padding_idx and the
    // explicit path exactly (emb[0] == 0 contributes nothing there either).
    namespace F = torch::nn::functional;
    const auto bag_opts = F::EmbeddingBagFuncOptions()
        .mode(torch::kSum)
        .per_sample_weights(w_normed)
        .padding_idx(0);

    auto pooled = F::embedding_bag(species_ids, species_embedding_->weight, bag_opts);

    // Concat taxonomy pools along the feature dim. Pooling is linear, so the
    // weighted mean of the concatenation equals the concatenation of the
    // per-table weighted means.
    if (has_taxonomy_ && genus_ids.defined() && genus_ids.numel() > 0) {
        auto pooled_g = F::embedding_bag(genus_ids, genus_embedding_->weight, bag_opts);
        auto pooled_f = F::embedding_bag(family_ids, family_embedding_->weight, bag_opts);
        pooled = torch::cat({pooled, pooled_g, pooled_f}, /*dim=*/-1);
    }

    // Concatenate: [continuous | pooled | has_cover]
    auto has_cover_col = has_cover.unsqueeze(-1);  // (batch, 1)
    auto x = torch::cat({continuous, pooled, has_cover_col}, /*dim=*/1);

    // MLP forward
    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
}

torch::Tensor PlotEncoderRankPoolImpl::get_species_weights() const {
    return species_embedding_->weight.detach();
}

torch::Tensor PlotEncoderRankPoolImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return genus_embedding_->weight.detach();
}

torch::Tensor PlotEncoderRankPoolImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return family_embedding_->weight.detach();
}


// =============================================================================
// PlotEncoderTransformer Implementation
// =============================================================================

PlotEncoderTransformerImpl::PlotEncoderTransformerImpl(
    int64_t n_continuous,
    int64_t n_species,
    int64_t n_genera,
    int64_t n_families,
    int d_model,
    int n_heads,
    int n_attention_layers,
    int transformer_ff_dim,
    const std::string& transformer_pooling,
    float transformer_dropout,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& mlp_config,
    float cover_dropout,
    const TabMConfig& tabm_config
) : d_model_(d_model),
    n_attention_layers_(n_attention_layers),
    transformer_pooling_(transformer_pooling),
    cover_dropout_(cover_dropout)
{
    has_taxonomy_ = (n_genera > 0 && n_families > 0);

    // Additive embeddings (all d_model-dimensional) with padding_idx=0
    species_embedding_ = register_module("species_embedding",
        torch::nn::Embedding(torch::nn::EmbeddingOptions(n_species, d_model).padding_idx(0)));

    if (has_taxonomy_) {
        genus_embedding_ = register_module("genus_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_genera, d_model).padding_idx(0)));
        family_embedding_ = register_module("family_embedding",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_families, d_model).padding_idx(0)));
    }

    // Weight projection: scalar -> d_model
    weight_proj_ = register_module("weight_proj",
        torch::nn::Linear(torch::nn::LinearOptions(1, d_model).bias(false)));

    // Mask embedding for MLM pretraining
    mask_embedding_ = register_parameter("mask_embedding", torch::zeros({d_model}));

    // Initialize embeddings with std=0.02 (BERT convention)
    {
        torch::NoGradGuard no_grad;
        species_embedding_->weight.normal_(0, kBertInitStd);
        species_embedding_->weight[0].zero_();  // Re-zero padding
        if (has_taxonomy_) {
            genus_embedding_->weight.normal_(0, kBertInitStd);
            genus_embedding_->weight[0].zero_();
            family_embedding_->weight.normal_(0, kBertInitStd);
            family_embedding_->weight[0].zero_();
        }
        weight_proj_->weight.normal_(0, kBertInitStd);
    }

    // Self-attention layers (optional)
    // NOTE: libtorch C++ API TransformerEncoderLayerOptions does not expose
    // batch_first or norm_first. We use the default (seq, batch, d_model) layout
    // and transpose manually before/after calling forward.
    if (n_attention_layers > 0) {
        auto layer_opts = torch::nn::TransformerEncoderLayerOptions(d_model, n_heads)
            .dim_feedforward(transformer_ff_dim)
            .dropout(transformer_dropout)
            .activation(torch::kGELU);
        auto layer = torch::nn::TransformerEncoderLayer(layer_opts);
        auto enc_opts = torch::nn::TransformerEncoderOptions(layer, n_attention_layers);
        transformer_encoder_ = register_module("transformer_encoder",
            torch::nn::TransformerEncoder(enc_opts));
    }

    // Pooling
    // NOTE: libtorch C++ API MultiheadAttentionOptions does not expose batch_first.
    // We transpose manually before/after calling forward.
    if (transformer_pooling == "attention") {
        pool_query_ = register_parameter("pool_query",
            torch::randn({1, 1, d_model}) * kBertInitStd);
        pool_attn_ = register_module("pool_attn",
            torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(d_model, n_heads)));
        pool_norm_ = register_module("pool_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    } else {
        // CLS pooling
        cls_token_ = register_parameter("cls_token",
            torch::randn({1, 1, d_model}) * kBertInitStd);
    }

    // MLP input: continuous + pooled (d_model) + has_cover flag
    int64_t input_dim = n_continuous + d_model + 1;

    auto bb = build_and_register_backbone(
        *this, mlp_, tabm_encoder_, input_dim, hidden_dims, mlp_config, tabm_config);
    use_tabm_ = bb.use_tabm;
    latent_dim_ = bb.latent_dim;
}

torch::Tensor PlotEncoderTransformerImpl::build_tokens(
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor weights,
    torch::Tensor masked_positions
) {
    // Additive embeddings in d_model space
    auto tokens = species_embedding_->forward(species_ids);  // (B, max_sp, d_model)

    if (has_taxonomy_ && genus_ids.defined() && genus_ids.numel() > 0) {
        tokens = tokens + genus_embedding_->forward(genus_ids);
        tokens = tokens + family_embedding_->forward(family_ids);
    }

    // Project scalar weights to d_model
    if (weights.defined() && weights.numel() > 0) {
        auto w_proj = weight_proj_->forward(weights.unsqueeze(-1));  // (B, max_sp, d_model)
        tokens = tokens + w_proj;
    }

    // Apply mask embedding for MLM pretraining
    if (masked_positions.defined() && masked_positions.numel() > 0) {
        tokens.index_put_({masked_positions}, mask_embedding_);
    }

    return tokens;
}

torch::Tensor PlotEncoderTransformerImpl::forward_tokens(
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor weights,
    torch::Tensor mask,
    torch::Tensor masked_positions
) {
    auto tokens = build_tokens(species_ids, genus_ids, family_ids, weights, masked_positions);

    // Self-attention (libtorch expects (seq, batch, d_model) — transpose around call)
    if (transformer_encoder_) {
        auto padding_mask = ~mask;  // True = ignore in attention
        tokens = tokens.transpose(0, 1);  // (B, seq, d) -> (seq, B, d)
        tokens = transformer_encoder_->forward(tokens, /*mask=*/{}, padding_mask);
        tokens = tokens.transpose(0, 1);  // (seq, B, d) -> (B, seq, d)
    }

    return tokens;  // (B, max_sp, d_model)
}

torch::Tensor PlotEncoderTransformerImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor weights,
    torch::Tensor mask,
    torch::Tensor has_cover,
    torch::Tensor masked_positions
) {
    int64_t batch_size = continuous.size(0);
    auto device = continuous.device();

    // Default has_cover
    if (!has_cover.defined() || has_cover.numel() == 0) {
        has_cover = torch::ones({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Build mask
    if (!mask.defined() || mask.numel() == 0) {
        mask = (species_ids != 0);
    }

    // Default weights
    if (!weights.defined() || weights.numel() == 0) {
        weights = mask.to(torch::kFloat32);
    }

    // Cover dropout
    apply_cover_dropout(is_training(), cover_dropout_, batch_size, device,
                        weights, mask, has_cover);

    // Build token embeddings and run self-attention
    auto tokens = forward_tokens(species_ids, genus_ids, family_ids, weights, mask, masked_positions);

    // Pooling
    torch::Tensor pooled;
    auto padding_mask = ~mask;  // True = ignore

    if (transformer_pooling_ == "attention") {
        // MultiheadAttention expects (seq, batch, d_model) — transpose around call
        auto query = pool_query_.expand({batch_size, -1, -1});  // (B, 1, d_model)
        auto query_t = query.transpose(0, 1);   // (1, B, d_model)
        auto tokens_t = tokens.transpose(0, 1);  // (seq, B, d_model)
        auto attn_result = pool_attn_->forward(query_t, tokens_t, tokens_t,
            /*key_padding_mask=*/padding_mask);
        // Output: (1, B, d_model) -> transpose back and squeeze
        pooled = pool_norm_->forward(std::get<0>(attn_result).transpose(0, 1).squeeze(1));  // (B, d_model)
    } else {
        // CLS pooling: prepend CLS token, run through transformer, extract pos 0
        auto cls = cls_token_.expand({batch_size, -1, -1});
        auto tokens_with_cls = torch::cat({cls, tokens}, /*dim=*/1);

        if (transformer_encoder_) {
            // TransformerEncoder expects (seq, batch, d_model) — transpose around call
            auto cls_pad = torch::zeros({batch_size, 1},
                torch::TensorOptions().dtype(torch::kBool).device(device));
            auto extended_mask = torch::cat({cls_pad, padding_mask}, /*dim=*/1);
            tokens_with_cls = tokens_with_cls.transpose(0, 1);  // (B, seq+1, d) -> (seq+1, B, d)
            tokens_with_cls = transformer_encoder_->forward(tokens_with_cls, /*mask=*/{}, extended_mask);
            tokens_with_cls = tokens_with_cls.transpose(0, 1);  // (seq+1, B, d) -> (B, seq+1, d)
        }
        pooled = tokens_with_cls.index({torch::indexing::Slice(), 0});  // (B, d_model)
    }

    // Concatenate and run MLP
    auto has_cover_col = has_cover.unsqueeze(-1);
    auto x = torch::cat({continuous, pooled, has_cover_col}, /*dim=*/1);

    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
}

torch::Tensor PlotEncoderTransformerImpl::get_species_weights() const {
    return species_embedding_->weight.detach();
}

torch::Tensor PlotEncoderTransformerImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return genus_embedding_->weight.detach();
}

torch::Tensor PlotEncoderTransformerImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return family_embedding_->weight.detach();
}

} // namespace resolve
