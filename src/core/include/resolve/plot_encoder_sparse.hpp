#pragma once

#include "resolve/species_encoder.hpp"

#include <memory>

namespace resolve {

// species_encoding = sparse (and hash with uses_explicit_vector): the plot's
// species vector -- sparse rows (sparse_rows.hpp) holding the value each
// species carries -- projected to species_embed_dim ("species_projection", the
// product a dense Linear over the matrix computes); with taxonomy, the fixed
// genus / family slots through one table per slot.
class PlotEncoderSparseImpl : public SpeciesEncoderImpl {
public:
    explicit PlotEncoderSparseImpl(const SpeciesEncoderSpec& spec);
};

TORCH_MODULE(PlotEncoderSparse);

// The sparse encoding's features: the projected species vector.
[[nodiscard]] std::unique_ptr<SpeciesFeatures> make_sparse_features(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

}  // namespace resolve
