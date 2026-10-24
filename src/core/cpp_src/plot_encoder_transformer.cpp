#include "resolve/plot_encoder_transformer.hpp"

#include "resolve/encoder.hpp"
#include "resolve/species_pool.hpp"

#include <stdexcept>

namespace resolve {

namespace {
bool present(const torch::Tensor& t) { return t.defined() && t.numel() > 0; }
}  // namespace

// libtorch's TransformerEncoderLayer and MultiheadAttention take no batch_first
// option, so both are run on (seq, batch, d_model) and the tokens are
// transposed around each call.
class SpeciesTokenFeatures final : public SpeciesFeatures {
public:
    SpeciesTokenFeatures(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : tables_(owner, spec, spec.config.d_model, spec.config.d_model, spec.config.d_model),
          d_model_(spec.config.d_model),
          pooling_(spec.config.transformer_pooling),
          aggregation_(spec.aggregation),
          cover_dropout_(spec.config.cover_dropout) {
        const auto& config = spec.config;
        weight_proj_ = owner.register_module(
            "weight_proj", torch::nn::Linear(torch::nn::LinearOptions(1, d_model_).bias(false)));
        mask_embedding_ = owner.register_parameter("mask_embedding", torch::zeros({d_model_}));

        // BERT initialisation (std 0.02), the padding rows kept at zero.
        {
            torch::NoGradGuard no_grad;
            auto init = [](const torch::nn::Embedding& table) {
                table->weight.normal_(0, kBertInitStd);
                table->weight[0].zero_();
            };
            init(tables_.species);
            if (tables_.has_taxonomy()) {
                init(tables_.genus);
                init(tables_.family);
            }
            weight_proj_->weight.normal_(0, kBertInitStd);
        }

        if (config.n_attention_layers > 0) {
            auto layer = torch::nn::TransformerEncoderLayer(
                torch::nn::TransformerEncoderLayerOptions(d_model_, config.n_heads)
                    .dim_feedforward(config.transformer_ff_dim)
                    .dropout(config.transformer_dropout)
                    .activation(torch::kGELU));
            self_attention_ = owner.register_module(
                "transformer_encoder",
                torch::nn::TransformerEncoder(
                    torch::nn::TransformerEncoderOptions(layer, config.n_attention_layers)));
        }

        // CLS pooling reads the CLS position after the self-attention mixed it
        // with the species tokens; with no attention layer nothing mixes it and
        // the pooled vector is the constant CLS parameter, whatever the species.
        if (pooling_ == TransformerPooling::Cls && config.n_attention_layers == 0) {
            throw std::invalid_argument(
                "PlotEncoderTransformer: transformer_pooling=cls requires "
                "n_attention_layers >= 1 (with 0 layers the CLS vector is a constant "
                "that ignores all species). Use transformer_pooling=attention or "
                "set n_attention_layers >= 1.");
        }

        switch (pooling_) {
            case TransformerPooling::Attention:
                pool_query_ = owner.register_parameter(
                    "pool_query", torch::randn({1, 1, d_model_}) * kBertInitStd);
                pool_attn_ = owner.register_module(
                    "pool_attn", torch::nn::MultiheadAttention(
                                     torch::nn::MultiheadAttentionOptions(d_model_,
                                                                          config.n_heads)));
                pool_norm_ = owner.register_module(
                    "pool_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model_})));
                break;
            case TransformerPooling::Cls:
                cls_token_ = owner.register_parameter(
                    "cls_token", torch::randn({1, 1, d_model_}) * kBertInitStd);
                break;
        }
    }

    [[nodiscard]] int64_t width() const noexcept override { return d_model_ + 1; }
    [[nodiscard]] int d_model() const noexcept { return static_cast<int>(d_model_); }

    torch::Tensor forward(const PlotInputs& inputs, bool training) override {
        auto pool = PoolInputs::from(inputs);
        pool.mask = pool.mask_with_a_key();
        pool.apply_cover_dropout(training, cover_dropout_);
        const auto padding = pool.mask.logical_not();
        const int64_t batch = pool.species_ids.size(0);

        torch::Tensor pooled;
        switch (pooling_) {
            case TransformerPooling::Attention: {
                auto tokens = self_attend(
                    build_tokens(pool.species_ids, pool.genus_ids, pool.family_ids,
                                 pool.weights, {}),
                    padding);
                auto query = pool_query_.expand({batch, -1, -1}).transpose(0, 1);
                auto keys = tokens.transpose(0, 1);
                // In float32 under autocast: the key-padding fill of
                // multi_head_attention_forward has no Half kernel.
                auto attended = run_fp32_under_autocast(
                    [this](const std::vector<torch::Tensor>& in) {
                        return std::get<0>(pool_attn_->forward(in[0], in[1], in[1],
                                                               /*key_padding_mask=*/in[2]));
                    },
                    {query, keys, padding});
                pooled = pool_norm_->forward(attended.transpose(0, 1).squeeze(1));
                break;
            }
            case TransformerPooling::Cls: {
                // The CLS token joins the raw species tokens and the
                // self-attention runs once over both.
                auto tokens = build_tokens(pool.species_ids, pool.genus_ids, pool.family_ids,
                                           pool.weights, {});
                auto with_cls = torch::cat({cls_token_.expand({batch, -1, -1}), tokens}, 1);
                auto cls_padding = torch::zeros(
                    {batch, 1},
                    torch::TensorOptions().dtype(torch::kBool).device(padding.device()));
                with_cls = self_attend(with_cls, torch::cat({cls_padding, padding}, 1));
                pooled = with_cls.index({torch::indexing::Slice(), 0});
                break;
            }
        }
        if (taxonomy_pooled()) add_taxonomy_summary(pooled, pool);
        return torch::cat({pooled, pool.has_cover.unsqueeze(-1)}, /*dim=*/1);
    }

    torch::Tensor forward_tokens(const torch::Tensor& species_ids, const torch::Tensor& genus_ids,
                                 const torch::Tensor& family_ids, const torch::Tensor& weights,
                                 const torch::Tensor& mask,
                                 const torch::Tensor& masked_positions) {
        PoolInputs pool;
        pool.mask = present(mask) ? mask : (species_ids != 0);
        return self_attend(build_tokens(species_ids, genus_ids, family_ids, weights,
                                        masked_positions),
                           pool.mask_with_a_key().logical_not());
    }

    [[nodiscard]] torch::Tensor species_weights() const override {
        return tables_.species_weights();
    }
    [[nodiscard]] torch::Tensor genus_weights() const override { return tables_.genus_weights(); }
    [[nodiscard]] torch::Tensor family_weights() const override {
        return tables_.family_weights();
    }
    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        tables_.append_composition_parameters(out);
    }

private:
    // Under Count the taxonomy is pooled beside the attention, not in the tokens.
    [[nodiscard]] bool taxonomy_pooled() const noexcept {
        return aggregation_ == AggregationMode::Count && tables_.has_taxonomy();
    }

    torch::Tensor build_tokens(const torch::Tensor& species_ids, const torch::Tensor& genus_ids,
                               const torch::Tensor& family_ids, const torch::Tensor& weights,
                               const torch::Tensor& masked_positions) {
        auto tokens = tables_.species->forward(species_ids);  // (batch, max_species, d_model)
        // Genus and family are added independently: either may be absent.
        if (tables_.has_taxonomy() && !taxonomy_pooled()) {
            if (present(genus_ids)) tokens = tokens + tables_.genus->forward(genus_ids);
            if (present(family_ids)) tokens = tokens + tables_.family->forward(family_ids);
        }
        if (present(weights)) tokens = tokens + weight_proj_->forward(weights.unsqueeze(-1));
        if (present(masked_positions)) tokens.index_put_({masked_positions}, mask_embedding_);
        return tokens;
    }

    // The tokens through the self-attention layers, if any; `padding` is true
    // where a position is to be ignored. In float32 under autocast: the
    // key-padding fill of multi_head_attention_forward has no Half kernel.
    torch::Tensor self_attend(torch::Tensor tokens, const torch::Tensor& padding) {
        if (!self_attention_) return tokens;
        return run_fp32_under_autocast(
                   [this](const std::vector<torch::Tensor>& in) {
                       return self_attention_->forward(in[0], /*src_mask=*/{}, in[1]);
                   },
                   {tokens.transpose(0, 1), padding})
            .transpose(0, 1);
    }

    // Adds the plot's genus and family embeddings, each averaged with one
    // weight per species, to `pooled`.
    void add_taxonomy_summary(torch::Tensor& pooled, const PoolInputs& pool) {
        const auto w = pool.normalized(pool.mask.to(torch::kFloat32));
        if (present(pool.genus_ids)) {
            pooled = pooled + weighted_pool(pool.genus_ids, tables_.genus, w);
        }
        if (present(pool.family_ids)) {
            pooled = pooled + weighted_pool(pool.family_ids, tables_.family, w);
        }
    }

    SpeciesTables tables_;
    int64_t d_model_;
    TransformerPooling pooling_;
    AggregationMode aggregation_;
    float cover_dropout_;

    torch::nn::Linear weight_proj_{nullptr};
    torch::Tensor mask_embedding_;
    torch::nn::TransformerEncoder self_attention_{nullptr};
    torch::Tensor pool_query_;
    torch::nn::MultiheadAttention pool_attn_{nullptr};
    torch::nn::LayerNorm pool_norm_{nullptr};
    torch::Tensor cls_token_;
};

std::unique_ptr<SpeciesFeatures> make_transformer_features(torch::nn::Module& owner,
                                                           const SpeciesEncoderSpec& spec) {
    return std::make_unique<SpeciesTokenFeatures>(owner, spec);
}

PlotEncoderTransformerImpl::PlotEncoderTransformerImpl(const SpeciesEncoderSpec& spec) {
    auto features = std::make_unique<SpeciesTokenFeatures>(*this, spec);
    tokens_ = features.get();
    assemble(std::move(features), nullptr, spec.n_continuous, spec);
}

torch::Tensor PlotEncoderTransformerImpl::forward_tokens(
    torch::Tensor species_ids, torch::Tensor genus_ids, torch::Tensor family_ids,
    torch::Tensor weights, torch::Tensor mask, torch::Tensor masked_positions) {
    return tokens_->forward_tokens(species_ids, genus_ids, family_ids, weights, mask,
                                   masked_positions);
}

int PlotEncoderTransformerImpl::d_model() const noexcept { return tokens_->d_model(); }

}  // namespace resolve
