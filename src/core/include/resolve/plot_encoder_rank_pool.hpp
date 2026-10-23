#pragma once

#include "resolve/species_encoder.hpp"

#include <memory>

namespace resolve {

// species_encoding = rank_pool: every species of the plot (species_ids and the
// pool fields) read through one species table and, with taxonomy, one genus
// and one family table. Each table is pooled as a weighted mean over the
// plot's species: the species table with the species weights normalized per
// plot, the genus and family tables with the same weights under
// AggregationMode::Abundance and with one per species under Count. The pooled
// vectors and the plot's has_cover flag are the features; cover dropout
// (ModelConfig::cover_dropout) applies in training.
//
// A species of unknown genus (code 0) pools the zero row, so unknown taxonomy
// scales a plot's genus pool towards zero by its share of the weight rather
// than being renormalized away.
class PlotEncoderRankPoolImpl : public SpeciesEncoderImpl {
public:
    explicit PlotEncoderRankPoolImpl(const SpeciesEncoderSpec& spec);
};

TORCH_MODULE(PlotEncoderRankPool);

// The rank_pool encoding's features: [species pool | genus pool | family pool
// | has_cover].
[[nodiscard]] std::unique_ptr<SpeciesFeatures> make_rank_pool_features(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

}  // namespace resolve
