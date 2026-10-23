#include "resolve/species_encoder.hpp"

#include "resolve/enum_names.hpp"
#include "resolve/plot_encoder_embed.hpp"
#include "resolve/plot_encoder_hash.hpp"
#include "resolve/plot_encoder_rank_pool.hpp"
#include "resolve/plot_encoder_sparse.hpp"
#include "resolve/plot_encoder_transformer.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

namespace resolve {

// =============================================================================
// Registry
// =============================================================================

namespace {

template <typename Impl>
std::shared_ptr<SpeciesEncoderImpl> make_encoder(const SpeciesEncoderSpec& spec) {
    return std::make_shared<Impl>(spec);
}

// One row per encoding: {mode, hash_in_continuous, indexes_species_codes,
// pooled, taxonomy_slots, features, encoder}.
const SpeciesEncodingTraits kSpeciesEncodings[] = {
    {SpeciesEncodingMode::Hash, true, false, false, true,
     &make_hash_features, &make_encoder<PlotEncoderImpl>},
    {SpeciesEncodingMode::Embed, false, true, false, true,
     &make_embed_features, &make_encoder<PlotEncoderEmbedImpl>},
    {SpeciesEncodingMode::Sparse, false, true, false, true,
     &make_sparse_features, &make_encoder<PlotEncoderSparseImpl>},
    {SpeciesEncodingMode::RankPool, false, true, true, false,
     &make_rank_pool_features, &make_encoder<PlotEncoderRankPoolImpl>},
    {SpeciesEncodingMode::Transformer, false, true, true, false,
     &make_transformer_features, &make_encoder<PlotEncoderTransformerImpl>},
};

}  // namespace

const SpeciesEncodingTraits& species_encoding_traits(SpeciesEncodingMode mode) {
    const auto* row = std::find_if(std::begin(kSpeciesEncodings), std::end(kSpeciesEncodings),
                                   [mode](const SpeciesEncodingTraits& t) { return t.mode == mode; });
    if (row == std::end(kSpeciesEncodings)) {
        throw std::invalid_argument("species encoding " +
                                    std::to_string(static_cast<int>(mode)) +
                                    " has no registered encoder");
    }
    return *row;
}

SpeciesEncodingMode effective_species_encoding(const ModelConfig& config) {
    if (!config.uses_explicit_vector) return config.species_encoding;
    switch (config.species_encoding) {
        case SpeciesEncodingMode::Hash:
        case SpeciesEncodingMode::Sparse:
            return SpeciesEncodingMode::Sparse;
        case SpeciesEncodingMode::Embed:
        case SpeciesEncodingMode::RankPool:
        case SpeciesEncodingMode::Transformer:
            break;
    }
    throw std::invalid_argument(
        std::string("uses_explicit_vector=true makes the model read the plot's species "
                    "vector, which species_encoding=") +
        species_encoding_to_string(config.species_encoding) +
        " does not provide. Use species_encoding=sparse (or hash) with it, or set "
        "uses_explicit_vector=false.");
}

std::shared_ptr<SpeciesEncoderImpl> make_species_encoder(const SpeciesEncoderSpec& spec) {
    const SpeciesEncodingMode mode = effective_species_encoding(spec.config);
    const auto& traits = species_encoding_traits(mode);
    if (traits.indexes_species_codes && spec.n_species == 0) {
        throw std::runtime_error(std::string("species_encoding=") +
                                 species_encoding_to_string(mode) +
                                 " requires n_species_vocab > 0 in schema");
    }
    return traits.make_encoder(spec);
}

SpeciesEncoderSpec SpeciesEncoderSpec::from_schema(const ResolveSchema& schema,
                                                   const ModelConfig& config) {
    SpeciesEncoderSpec spec;
    spec.n_continuous = schema.continuous_width();
    spec.n_species = schema.n_species_vocab;
    spec.n_genera = schema.genus_table_size();
    spec.n_families = schema.family_table_size();
    spec.aggregation = schema.dataset_config.aggregation;
    spec.config = config;
    return spec;
}

// =============================================================================
// Fixed taxonomy slots
// =============================================================================

namespace {

class PerRankSlotTaxonomy final : public SlotTaxonomy {
public:
    PerRankSlotTaxonomy(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : n_slots_(spec.config.n_taxonomy_slots) {
        // padding_idx 0: the reserved code of an empty slot or unknown genus
        // stays the zero vector and learns nothing.
        auto table = [](int64_t rows, int dim) {
            return torch::nn::Embedding(torch::nn::EmbeddingOptions(rows, dim).padding_idx(0));
        };
        for (int k = 0; k < n_slots_; ++k) {
            genus_.push_back(owner.register_module(
                "genus_emb_" + std::to_string(k), table(spec.n_genera, spec.config.genus_emb_dim)));
            family_.push_back(owner.register_module(
                "family_emb_" + std::to_string(k),
                table(spec.n_families, spec.config.family_emb_dim)));
        }
        width_ = static_cast<int64_t>(n_slots_) *
                 (spec.config.genus_emb_dim + spec.config.family_emb_dim);
    }

    [[nodiscard]] int64_t width() const noexcept override { return width_; }

    void append(std::vector<torch::Tensor>& parts, const PlotInputs& inputs) override {
        check_concat_taxonomy(true, inputs.genus_ids, inputs.family_ids,
                              "taxonomy slots (one table per slot)");
        for (int k = 0; k < n_slots_; ++k) parts.push_back(genus_[k](inputs.genus_ids.select(1, k)));
        for (int k = 0; k < n_slots_; ++k) {
            parts.push_back(family_[k](inputs.family_ids.select(1, k)));
        }
    }

    [[nodiscard]] torch::Tensor genus_weights() const override { return slot_average(genus_); }
    [[nodiscard]] torch::Tensor family_weights() const override { return slot_average(family_); }

    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        for (const auto& table : genus_) out.push_back(table->weight);
        for (const auto& table : family_) out.push_back(table->weight);
    }

private:
    static torch::Tensor slot_average(const std::vector<torch::nn::Embedding>& tables) {
        std::vector<torch::Tensor> weights;
        weights.reserve(tables.size());
        for (const auto& table : tables) weights.push_back(table->weight.detach());
        return torch::stack(weights, 0).mean(0);
    }

    int n_slots_;
    int64_t width_ = 0;
    std::vector<torch::nn::Embedding> genus_;
    std::vector<torch::nn::Embedding> family_;
};

class FusedSlotTaxonomy final : public SlotTaxonomy {
public:
    FusedSlotTaxonomy(torch::nn::Module& owner, const SpeciesEncoderSpec& spec)
        : genus_(owner.register_module(
              "fused_genus", FusedPositionalEmbedding(spec.n_genera, spec.config.n_taxonomy_slots,
                                                      spec.config.genus_emb_dim))),
          family_(owner.register_module(
              "fused_family",
              FusedPositionalEmbedding(spec.n_families, spec.config.n_taxonomy_slots,
                                       spec.config.family_emb_dim))) {}

    [[nodiscard]] int64_t width() const noexcept override {
        return genus_->total_output_dim() + family_->total_output_dim();
    }

    void append(std::vector<torch::Tensor>& parts, const PlotInputs& inputs) override {
        check_concat_taxonomy(true, inputs.genus_ids, inputs.family_ids,
                              "taxonomy slots (fused tables)");
        parts.push_back(genus_->forward(inputs.genus_ids));
        parts.push_back(family_->forward(inputs.family_ids));
    }

    [[nodiscard]] torch::Tensor genus_weights() const override {
        return genus_->position_average();
    }
    [[nodiscard]] torch::Tensor family_weights() const override {
        return family_->position_average();
    }

    void append_composition_parameters(std::vector<torch::Tensor>& out) const override {
        out.push_back(genus_->embedding()->weight);
        out.push_back(family_->embedding()->weight);
    }

private:
    FusedPositionalEmbedding genus_;
    FusedPositionalEmbedding family_;
};

}  // namespace

std::unique_ptr<SlotTaxonomy> make_per_rank_slot_taxonomy(torch::nn::Module& owner,
                                                          const SpeciesEncoderSpec& spec) {
    if (!spec.has_taxonomy()) return nullptr;
    return std::make_unique<PerRankSlotTaxonomy>(owner, spec);
}

std::unique_ptr<SlotTaxonomy> make_fused_slot_taxonomy(torch::nn::Module& owner,
                                                       const SpeciesEncoderSpec& spec) {
    if (!spec.has_taxonomy()) return nullptr;
    return std::make_unique<FusedSlotTaxonomy>(owner, spec);
}

// =============================================================================
// SpeciesEncoderImpl
// =============================================================================

void SpeciesEncoderImpl::assemble(std::unique_ptr<SpeciesFeatures> features,
                                  std::unique_ptr<SlotTaxonomy> slots,
                                  int64_t n_continuous,
                                  const SpeciesEncoderSpec& spec) {
    features_ = std::move(features);
    slots_ = std::move(slots);
    has_taxonomy_ = spec.has_taxonomy();
    const int64_t input_dim = n_continuous + features_->width() + (slots_ ? slots_->width() : 0);
    build_encoder_tail(*this, tail_, input_dim, spec.config.hidden_dims,
                       MLPBlockConfig::from_model_config(spec.config), spec.config.tabm,
                       MoETailConfig::from_model_config(spec.config),
                       spec.config.parallel_layers);
}

TailOutput SpeciesEncoderImpl::encode(const PlotInputs& inputs,
                                      std::vector<torch::Tensor>* activations) {
    std::vector<torch::Tensor> parts{inputs.continuous};
    if (features_->width() > 0) parts.push_back(features_->forward(inputs, is_training()));
    if (slots_) slots_->append(parts, inputs);
    return forward_encoder_tail(tail_, torch::cat(parts, /*dim=*/1), activations);
}

torch::Tensor SpeciesEncoderImpl::species_weights() const { return features_->species_weights(); }

torch::Tensor SpeciesEncoderImpl::genus_weights() const {
    return slots_ ? slots_->genus_weights() : features_->genus_weights();
}

torch::Tensor SpeciesEncoderImpl::family_weights() const {
    return slots_ ? slots_->family_weights() : features_->family_weights();
}

std::vector<torch::Tensor> SpeciesEncoderImpl::composition_parameters() const {
    std::vector<torch::Tensor> out;
    features_->append_composition_parameters(out);
    if (slots_) slots_->append_composition_parameters(out);
    return out;
}

}  // namespace resolve
