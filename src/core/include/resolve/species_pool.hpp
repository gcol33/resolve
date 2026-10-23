#pragma once

// The parts the two encodings that read every species of a plot (rank_pool and
// transformer) share: their pool inputs with defaults filled in, their species /
// genus / family tables, cover dropout and the count weights of
// AggregationMode::Count.

#include "resolve/plot_inputs.hpp"
#include "resolve/species_encoder.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// A plot batch's pool fields with the omitted ones defaulted: mask = the
// non-zero species codes, weights = the mask (every species weighs the same),
// has_cover = 1 per plot.
struct PoolInputs {
    torch::Tensor species_ids;  // (batch, max_species) int64, 0 = padding
    torch::Tensor genus_ids;    // (batch, max_species) int64, or undefined
    torch::Tensor family_ids;   // (batch, max_species) int64, or undefined
    torch::Tensor weights;      // (batch, max_species) float
    torch::Tensor mask;         // (batch, max_species), true where a species sits
    torch::Tensor has_cover;    // (batch,) float

    [[nodiscard]] static PoolInputs from(const PlotInputs& inputs);

    // Under `training`, replaces the weights of a `cover_dropout` fraction of
    // the plots with their mask and clears their has_cover flag, so the model
    // learns to read a plot whose covers were not recorded.
    void apply_cover_dropout(bool training, float cover_dropout);

    // The mask with position 0 set on every row that holds no species, so an
    // attention softmax over the row has a key to attend to.
    [[nodiscard]] torch::Tensor mask_with_a_key() const;

    // `w` (batch, max_species) on the plots' species, divided by its sum over
    // each plot: weights summing to 1 over a plot's species. Padding carries
    // 0, and a plot with no species carries 0 throughout.
    [[nodiscard]] torch::Tensor normalized(const torch::Tensor& w) const;

    // The weights a pooled genus / family table is read with: the species'
    // normalized `weights` under AggregationMode::Abundance, the normalized
    // mask under Count, so each species counts once towards its taxon.
    [[nodiscard]] torch::Tensor taxonomy_weights(AggregationMode aggregation,
                                                 const torch::Tensor& species_weights) const;
};

// sum_j w[:, j] * table[ids[:, j]] for each plot, as one embedding_bag: no
// (batch, max_species, dim) gather is materialized. Code 0 (padding and
// unknown) is left out of the sum and its row frozen, as the tables'
// padding_idx says. (batch, dim).
[[nodiscard]] torch::Tensor weighted_pool(const torch::Tensor& ids,
                                          const torch::nn::Embedding& table,
                                          const torch::Tensor& w);

// The species table and, with taxonomy, the genus and family tables, each with
// the reserved row 0 zeroed and frozen (padding_idx), registered on `owner` as
// "species_embedding", "genus_embedding" and "family_embedding".
struct SpeciesTables {
    torch::nn::Embedding species{nullptr};
    torch::nn::Embedding genus{nullptr};
    torch::nn::Embedding family{nullptr};

    SpeciesTables(torch::nn::Module& owner, const SpeciesEncoderSpec& spec,
                  int species_dim, int genus_dim, int family_dim);

    [[nodiscard]] bool has_taxonomy() const noexcept { return static_cast<bool>(genus); }

    [[nodiscard]] torch::Tensor species_weights() const;
    [[nodiscard]] torch::Tensor genus_weights() const;
    [[nodiscard]] torch::Tensor family_weights() const;
    void append_composition_parameters(std::vector<torch::Tensor>& out) const;
};

}  // namespace resolve
