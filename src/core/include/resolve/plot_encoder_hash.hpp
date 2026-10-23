#pragma once

#include "resolve/species_encoder.hpp"

#include <memory>

namespace resolve {

// species_encoding = hash: the species reach the model as the hash embedding,
// the tail of the continuous block (hash_dim columns), so the encoding adds no
// species features of its own; with taxonomy it reads the fixed genus / family
// slots through one table per slot.
class PlotEncoderImpl : public SpeciesEncoderImpl {
public:
    explicit PlotEncoderImpl(const SpeciesEncoderSpec& spec);
};

TORCH_MODULE(PlotEncoder);

// The hash encoding's features: none (width 0).
[[nodiscard]] std::unique_ptr<SpeciesFeatures> make_hash_features(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

}  // namespace resolve
