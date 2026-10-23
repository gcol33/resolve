#pragma once

#include "resolve/species_encoder.hpp"

#include <memory>

namespace resolve {

class SpeciesTokenFeatures;

// species_encoding = transformer: one d_model token per species of the plot,
// the sum of its species embedding, its genus and family embeddings and a
// projection of its weight; n_attention_layers of self-attention over the
// tokens (none by default); then a pooled d_model vector, by a learned query
// attending over the tokens (TransformerPooling::Attention) or by a CLS token
// run through the self-attention with them (Cls, which needs at least one
// layer). The pooled vector and the plot's has_cover flag are the features;
// cover dropout (ModelConfig::cover_dropout) applies in training.
//
// Under AggregationMode::Count the genus and family embeddings leave the
// tokens: the attention reads species and weights, and the plot's genus and
// family embeddings, averaged with one weight per species, are added to the
// pooled vector, so each species counts once towards its taxon.
class PlotEncoderTransformerImpl : public SpeciesEncoderImpl {
public:
    explicit PlotEncoderTransformerImpl(const SpeciesEncoderSpec& spec);

    // The tokens after self-attention, before pooling: (batch, max_species,
    // d_model), for masked-species pretraining. `weights` may be undefined
    // (no weight term); the tokens at `masked_positions` (a bool mask of the
    // same shape as species_ids) are replaced by the learned mask embedding.
    // An omitted `mask` is the non-zero species codes.
    torch::Tensor forward_tokens(torch::Tensor species_ids, torch::Tensor genus_ids,
                                 torch::Tensor family_ids, torch::Tensor weights,
                                 torch::Tensor mask, torch::Tensor masked_positions = {});

    [[nodiscard]] int d_model() const noexcept;

private:
    SpeciesTokenFeatures* tokens_ = nullptr;
};

TORCH_MODULE(PlotEncoderTransformer);

// The transformer encoding's features: [pooled vector | has_cover].
[[nodiscard]] std::unique_ptr<SpeciesFeatures> make_transformer_features(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

}  // namespace resolve
