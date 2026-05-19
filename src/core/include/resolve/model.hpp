#pragma once

#include "resolve/types.hpp"
#include "resolve/encoder.hpp"
#include "resolve/adapter.hpp"
#include "resolve/attention.hpp"
#include "resolve/categorical.hpp"
#include <torch/torch.h>
#include <variant>

namespace resolve {

// Forward result with optional MoE auxiliary loss
struct ModelForwardResult {
    std::unordered_map<std::string, torch::Tensor> outputs;
    torch::Tensor moe_aux_loss;  // Empty if MoE not used
};

// Full Resolve model: shared encoder + multiple task heads
// Supports three encoding modes:
// - Hash: Feature hashing for species (PlotEncoder)
// - Embed: Learnable embeddings for top-k species (PlotEncoderEmbed)
// - Sparse: Explicit species vector input (PlotEncoderSparse)
// Each mode can optionally use Mixture of Experts (MoE)
class ResolveModelImpl : public torch::nn::Module {
public:
    ResolveModelImpl(
        const ResolveSchema& schema,
        const ModelConfig& config = ModelConfig{}
    );

    // Forward pass for all targets
    // Use appropriate inputs based on encoding mode:
    // - Hash: continuous (includes hash embedding), genus_ids, family_ids
    // - Embed: continuous, species_ids, genus_ids, family_ids
    // - Sparse: continuous, species_vector, genus_ids, family_ids
    // - RankPool/Transformer: continuous, species_ids, pool_* fields
    //
    // categorical_ids is (batch, n_categoricals) int64 with codes produced
    // by CategoricalVocab (0 = UNK). It is embedded internally and
    // concatenated to `continuous` before the encoder runs — schemas
    // without categoricals pass an empty tensor and get the legacy behavior.
    std::unordered_map<std::string, torch::Tensor> forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {},
        // Pool-style encoder fields (rank_pool / transformer)
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        torch::Tensor categorical_ids = {}
    );

    // Forward pass returning outputs + MoE auxiliary loss (for training with MoE)
    ModelForwardResult forward_with_aux(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {},
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        torch::Tensor categorical_ids = {}
    );

    // Forward pass for single target
    torch::Tensor forward_single(
        const std::string& target,
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {},
        torch::Tensor categorical_ids = {}
    );

    // Get latent representation (without heads)
    torch::Tensor get_latent(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {},
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        torch::Tensor categorical_ids = {}
    );

    // Forward pass that returns intermediate activations (for diagnostics).
    // Only works with hash encoder for now. `categorical_ids` is fused into
    // `continuous` via fuse_categoricals_() so the encoder receives the
    // same input shape it was constructed for.
    std::pair<torch::Tensor, std::vector<torch::Tensor>> encode_with_activations(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor categorical_ids = {}
    );

    // Get MoE gating probabilities for analysis (only valid when MoE enabled)
    torch::Tensor get_gate_probs(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Accessors
    [[nodiscard]] const ResolveSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] int64_t latent_dim() const;
    [[nodiscard]] SpeciesEncodingMode species_encoding() const noexcept { return config_.species_encoding; }
    [[nodiscard]] bool uses_explicit_vector() const noexcept { return config_.uses_explicit_vector; }
    [[nodiscard]] bool uses_moe() const noexcept { return config_.moe_routing != MoERoutingType::None; }
    [[nodiscard]] int n_experts() const noexcept { return uses_moe() ? config_.n_experts : 0; }

    // Embedding weight extraction (delegates to active encoder)
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;
    [[nodiscard]] torch::Tensor get_species_weights() const;

    // Set species trait matrix (for TraitNet architecture)
    void set_traits(torch::Tensor traits);

    // Get task head by name
    [[nodiscard]] TaskHead& head(const std::string& name);
    [[nodiscard]] const TaskHead& head(const std::string& name) const;

private:
    // Dispatch taxonomy weight extraction to the active encoder
    torch::Tensor get_taxonomy_weights_(
        torch::Tensor (PlotEncoderMoEImpl::*moe_fn)() const,
        torch::Tensor (PlotEncoderImpl::*hash_fn)() const,
        torch::Tensor (PlotEncoderEmbedImpl::*embed_fn)() const,
        torch::Tensor (PlotEncoderSparseImpl::*sparse_fn)() const
    ) const;

    // Internal forward through encoder based on mode (returns latent only).
    // `continuous` here is the value AFTER categorical embeddings have been
    // concatenated; see fuse_categoricals_().
    torch::Tensor encode(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {}
    );

    // Internal forward with aux loss (for MoE). Same `continuous` convention
    // as encode() — caller pre-concatenates categorical embeddings.
    std::pair<torch::Tensor, torch::Tensor> encode_with_aux(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {}
    );

    // Single source of truth for categorical concat. Takes the user-supplied
    // continuous tensor and (possibly empty) categorical_ids; returns the
    // continuous tensor that matches the n_continuous the encoders were
    // constructed with. Behavior matches the Python POC
    // (src/resolve/model/resolve.py): if the model has no categoricals,
    // returns continuous unchanged; if it has categoricals but the caller
    // passed an empty/undefined cat_ids, pads with zeros to keep the encoder
    // shape valid.
    torch::Tensor fuse_categoricals_(torch::Tensor continuous,
                                     torch::Tensor categorical_ids);

    ResolveSchema schema_;
    ModelConfig config_;

    // Standard encoders (one will be used based on encoding mode)
    PlotEncoder encoder_hash_{nullptr};
    PlotEncoderEmbed encoder_embed_{nullptr};
    PlotEncoderSparse encoder_sparse_{nullptr};
    PlotEncoderRankPool encoder_rank_pool_{nullptr};
    PlotEncoderTransformer encoder_transformer_{nullptr};

    // TraitNet encoder (used when encoder_architecture == TraitNet)
    TraitNetEncoder trait_net_encoder_{nullptr};

    // MoE encoder (used when moe_routing != None AND hash encoding)
    PlotEncoderMoE encoder_moe_{nullptr};

    // Model-level MoE layer (used when moe_routing != None AND embed/sparse encoding)
    MixtureOfExperts post_moe_{nullptr};

    // Tabular adapter (used when encoder_architecture != MLP)
    TabularAdapter adapter_{nullptr};

    // Categorical embedder (one nn::Embedding table per categorical column).
    // Null when the schema has no categorical covariates.
    CategoricalEmbedder categorical_embedder_{nullptr};

    std::unordered_map<std::string, TaskHead> heads_;
};

TORCH_MODULE(ResolveModel);

// Alias for backwards compatibility
using SpaccModel = ResolveModel;
using SpaccModelImpl = ResolveModelImpl;

} // namespace resolve
