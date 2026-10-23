#pragma once

// The species encoders of the MLP architecture, and what each species encoding
// reads off a plot.
//
// A species encoding is one SpeciesFeatures: the features it computes from a
// plot's species (none for hash, whose features already sit in the continuous
// block; fused per-slot embeddings for embed; a projection of the species
// vector for sparse; pooled embeddings for rank_pool; attention-pooled tokens
// for transformer). The MLP species encoder (SpeciesEncoderImpl) concatenates
// the continuous block, those features and the fixed genus / family slots its
// encoding reads, and runs the EncoderTail (encoder.hpp). The tabular adapter
// appends the same features to its numerical inputs.
//
// The encodings are listed once, in the registry species_encoding_traits()
// reads (species_encoder.cpp): a new encoding is its own file -- its
// SpeciesFeatures and its PlotEncoder* class -- plus one row there.

#include "resolve/encoder.hpp"
#include "resolve/plot_inputs.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace resolve {

// What a species encoder is built from: the table sizes the schema fixes and
// the ModelConfig knobs. from_schema() is the one place a model derives it.
struct SpeciesEncoderSpec {
    // Width of the continuous block the encoder receives (categorical
    // embeddings included); the hash encoding adds its hash_dim columns.
    int64_t n_continuous = 0;
    // Species vocabulary size, the reserved code 0 included.
    int64_t n_species = 0;
    // Genus / family table heights, the reserved code 0 included; 0 without
    // taxonomy.
    int64_t n_genera = 0;
    int64_t n_families = 0;
    // How the encodings with per-species taxonomy (rank_pool, transformer)
    // weigh a plot's genera and families: by the species weights, or one per
    // species of the taxon (DatasetConfig::aggregation).
    AggregationMode aggregation = AggregationMode::Abundance;
    ModelConfig config;

    [[nodiscard]] static SpeciesEncoderSpec from_schema(const ResolveSchema& schema,
                                                        const ModelConfig& config);

    // Whether genus / family tables are built: a vocabulary of one entry holds
    // only the reserved code, so there is nothing to embed.
    [[nodiscard]] bool has_taxonomy() const noexcept { return n_genera > 1 || n_families > 1; }
};

// The features one species encoding computes from a plot's species. Its tables
// are registered on the module that owns it, under flat names, so an MLP
// encoder and the tabular adapter hold them under the same keys.
class SpeciesFeatures {
public:
    virtual ~SpeciesFeatures() = default;

    // Columns of forward()'s output.
    [[nodiscard]] virtual int64_t width() const noexcept = 0;

    // (batch, width()) features of the plots in `inputs`. `training` selects
    // the train-time behaviour (cover dropout).
    virtual torch::Tensor forward(const PlotInputs& inputs, bool training) = 0;

    // The species / genus / family embedding tables, one row per code
    // (averaged over slot positions where an encoding keeps one table per
    // slot); undefined when the encoding has none.
    [[nodiscard]] virtual torch::Tensor species_weights() const { return {}; }
    [[nodiscard]] virtual torch::Tensor genus_weights() const { return {}; }
    [[nodiscard]] virtual torch::Tensor family_weights() const { return {}; }

    // Appends the parameter tensors a plot's composition is read through.
    virtual void append_composition_parameters(std::vector<torch::Tensor>& out) const = 0;
};

// The fixed genus / family slots (PlotInputs::genus_ids / family_ids) the hash,
// embed and sparse encodings read: n_taxonomy_slots codes per rank, each
// embedded and concatenated, genus slots first.
class SlotTaxonomy {
public:
    virtual ~SlotTaxonomy() = default;
    [[nodiscard]] virtual int64_t width() const noexcept = 0;
    // Appends the slot embeddings of `inputs` to `parts`; both slot tensors
    // must be present (check_concat_taxonomy).
    virtual void append(std::vector<torch::Tensor>& parts, const PlotInputs& inputs) = 0;
    [[nodiscard]] virtual torch::Tensor genus_weights() const = 0;
    [[nodiscard]] virtual torch::Tensor family_weights() const = 0;
    virtual void append_composition_parameters(std::vector<torch::Tensor>& out) const = 0;
};

// One nn::Embedding per slot ("genus_emb_<k>", "family_emb_<k>").
[[nodiscard]] std::unique_ptr<SlotTaxonomy> make_per_rank_slot_taxonomy(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

// One FusedPositionalEmbedding per rank ("fused_genus", "fused_family").
[[nodiscard]] std::unique_ptr<SlotTaxonomy> make_fused_slot_taxonomy(
    torch::nn::Module& owner, const SpeciesEncoderSpec& spec);

// The species encoder of the MLP architecture: [continuous | species features
// | slot taxonomy] through the encoder tail. The PlotEncoder* classes are its
// encodings; each builds its parts in its constructor through assemble().
class SpeciesEncoderImpl : public torch::nn::Module {
public:
    // The tail output for `inputs`, whose `continuous` carries the categorical
    // embeddings (ResolveModel fuses them). With `activations`, the output of
    // every hidden layer of the tail is appended to it as a
    // (rows, neurons) tensor: each activation of a plain MLP, backbone or
    // parallel branch (a residual block's output), and each TabM layer with
    // its ensemble members as rows.
    TailOutput encode(const PlotInputs& inputs, std::vector<torch::Tensor>* activations = nullptr);

    // The latent alone.
    torch::Tensor forward(const PlotInputs& inputs) { return encode(inputs).latent; }

    [[nodiscard]] int64_t latent_dim() const noexcept { return tail_.latent_dim; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }

    // Embedding tables, one row per code; undefined where the encoding has
    // none (species for hash and sparse, taxonomy without taxonomy).
    [[nodiscard]] torch::Tensor species_weights() const;
    [[nodiscard]] torch::Tensor genus_weights() const;
    [[nodiscard]] torch::Tensor family_weights() const;

    // The species, genus and family tables this encoder reads a plot's
    // composition through, as the parameter tensors themselves.
    [[nodiscard]] std::vector<torch::Tensor> composition_parameters() const;

protected:
    // `features` were built on this module (registered first, so their keys
    // and parameter order lead), `slots` too unless the encoding has none
    // (nullptr); the tail is built over the concatenation, whose continuous
    // part is `n_continuous` wide.
    void assemble(std::unique_ptr<SpeciesFeatures> features,
                  std::unique_ptr<SlotTaxonomy> slots,
                  int64_t n_continuous,
                  const SpeciesEncoderSpec& spec);

    SpeciesFeatures& features() { return *features_; }

private:
    std::unique_ptr<SpeciesFeatures> features_;
    std::unique_ptr<SlotTaxonomy> slots_;
    EncoderTail tail_;
    bool has_taxonomy_ = false;
};

TORCH_MODULE(SpeciesEncoder);

// One species encoding: how it reaches the model, and how to build its parts.
struct SpeciesEncodingTraits {
    SpeciesEncodingMode mode;
    // The hash embedding is the tail of the continuous block
    // (ContinuousInputs::hash_embedding belongs to what the model reads).
    bool hash_in_continuous;
    // Species reach the model as vocabulary codes (embedding rows or vector
    // columns), so a dataset fitted to another vocabulary reads other species'
    // rows; the encoder needs a species vocabulary.
    bool indexes_species_codes;
    // Reads the pool fields: every species of the plot with its own genus,
    // family and weight (species_ids, pool_*).
    bool pooled;
    // Reads the fixed genus / family slots (genus_ids, family_ids).
    bool taxonomy_slots;
    // The encoding's features, registered on `owner`.
    std::unique_ptr<SpeciesFeatures> (*make_features)(torch::nn::Module& owner,
                                                      const SpeciesEncoderSpec& spec);
    // The MLP species encoder.
    std::shared_ptr<SpeciesEncoderImpl> (*make_encoder)(const SpeciesEncoderSpec& spec);
};

// The registry row of `mode`.
[[nodiscard]] const SpeciesEncodingTraits& species_encoding_traits(SpeciesEncodingMode mode);

// The encoding a model built from `config` runs: species_encoding, except that
// uses_explicit_vector turns hash into sparse (the species vector instead of
// the hash embedding). uses_explicit_vector with embed, rank_pool or
// transformer, which read no species vector, is refused with
// std::invalid_argument.
[[nodiscard]] SpeciesEncodingMode effective_species_encoding(const ModelConfig& config);

// The MLP species encoder for `spec`: the registry's encoder of the effective
// encoding. Refuses an encoding that indexes species codes when the spec has
// no species vocabulary.
[[nodiscard]] std::shared_ptr<SpeciesEncoderImpl> make_species_encoder(
    const SpeciesEncoderSpec& spec);

}  // namespace resolve
