#pragma once

#include "resolve/species_encoder.hpp"

#include <memory>

namespace resolve {

// species_encoding = embed: the top_k_species ranked species of a plot, each
// slot looked up in its own region of one fused table ("fused_species", width
// top_k_species x species_embed_dim); with taxonomy, the fixed genus / family
// slots through fused tables of their own.
class PlotEncoderEmbedImpl : public SpeciesEncoderImpl {
public:
    explicit PlotEncoderEmbedImpl(const SpeciesEncoderSpec& spec);
};

TORCH_MODULE(PlotEncoderEmbed);

// The embed encoding's features: the fused species slots of species_ids.
[[nodiscard]] std::unique_ptr<SpeciesFeatures> make_embed_features(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

}  // namespace resolve
