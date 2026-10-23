#pragma once

#include "resolve/types.hpp"
#include "resolve/encoder.hpp"
#include "resolve/adapter.hpp"
#include "resolve/attention.hpp"
#include "resolve/categorical.hpp"
#include "resolve/plot_inputs.hpp"
#include <torch/torch.h>
#include <variant>

namespace resolve {

// Forward result with optional MoE auxiliary loss
struct ModelForwardResult {
    std::unordered_map<std::string, torch::Tensor> outputs;
    torch::Tensor moe_aux_loss;  // Empty if MoE not used
};

// Full Resolve model: shared encoder + multiple task heads
// Species encodings:
// - Hash: Feature hashing for species (PlotEncoder)
// - Embed: Learnable embeddings for top-k species (PlotEncoderEmbed)
// - Sparse: Explicit species vector input (PlotEncoderSparse)
// - RankPool: Weighted pool over every species of the plot (PlotEncoderRankPool)
// - Transformer: Attention over the plot's species (PlotEncoderTransformer)
// Each can optionally use Mixture of Experts (MoE)
class ResolveModelImpl : public torch::nn::Module {
public:
    ResolveModelImpl(
        const ResolveSchema& schema,
        const ModelConfig& config = ModelConfig{}
    );

    // Every method below reads the fields of `inputs` its encoder needs:
    // - Hash: continuous (hash embedding at its tail), genus_ids, family_ids
    // - Embed: continuous, species_ids, genus_ids, family_ids
    // - Sparse: continuous, species_vector, genus_ids, family_ids
    // - RankPool/Transformer: continuous, species_ids, pool_* fields
    // categorical_ids is embedded and concatenated onto `continuous` before
    // the encoder runs; a model with categorical columns given none reads
    // every column as UNK.

    // Forward pass for all targets.
    std::unordered_map<std::string, torch::Tensor> forward(const PlotInputs& inputs);

    // Forward pass returning outputs + MoE auxiliary loss (for training with MoE)
    ModelForwardResult forward_with_aux(const PlotInputs& inputs);

    // Forward pass for one target's head.
    torch::Tensor forward_single(const std::string& target, const PlotInputs& inputs);

    // Latent representation (without heads).
    torch::Tensor get_latent(const PlotInputs& inputs);

    // The latent plus the hidden-layer activations of the encoder, for
    // diagnostics. Only the hash encoder reports activations; every other
    // encoder returns its latent with an empty list.
    std::pair<torch::Tensor, std::vector<torch::Tensor>> encode_with_activations(
        const PlotInputs& inputs);

    // MoE gating probabilities (batch, n_experts); undefined without MoE.
    torch::Tensor get_gate_probs(const PlotInputs& inputs);

    // Accessors
    [[nodiscard]] const ResolveSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] int64_t latent_dim() const;
    [[nodiscard]] SpeciesEncodingMode species_encoding() const noexcept { return config_.species_encoding; }
    [[nodiscard]] bool uses_explicit_vector() const noexcept { return config_.uses_explicit_vector; }
    // Whether the hash embedding is the tail of the continuous block, i.e.
    // whether ContinuousInputs::hash_embedding is part of what this model reads.
    [[nodiscard]] bool hash_in_continuous() const noexcept {
        return config_.species_encoding == SpeciesEncodingMode::Hash &&
               !config_.uses_explicit_vector;
    }
    [[nodiscard]] bool uses_moe() const noexcept { return config_.moe_routing != MoERoutingType::None; }
    [[nodiscard]] int n_experts() const noexcept { return uses_moe() ? config_.n_experts : 0; }

    // The coordinate-kNN GNN builds its spatial graph from the coordinates of the
    // plots in the current forward. Mini-batching would build the graph from an
    // arbitrary subset of plots, so each plot would attend to random batch-mates
    // rather than its true spatial neighbors. Training must therefore forward the
    // full node set in a single batch, so the graph is the global spatial
    // structure -- mirroring the single full-batch inference forward the Predictor
    // already forces for this architecture.
    [[nodiscard]] bool requires_full_batch_training() const noexcept {
        return config_.encoder_architecture == EncoderArchitecture::GNN;
    }

    // Embedding weight extraction (delegates to active encoder)
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;
    [[nodiscard]] torch::Tensor get_species_weights() const;

    // The species, genus and family tables the species encoder reads a plot's
    // composition through, as the parameters themselves (empty for an adapter
    // architecture or TraitNet). ModelConfig::freeze_composition stops their
    // gradient at construction.
    [[nodiscard]] std::vector<torch::Tensor> composition_parameters() const;

    // Set species trait matrix (for TraitNet architecture)
    void set_traits(torch::Tensor traits);

    // The species graph a HeterogeneousGNN passes messages on (species_graph.hpp).
    // It is built from the training data once and is part of the trained model,
    // so Trainer::prepare_data builds and sets it, Trainer::save writes it into
    // the checkpoint and a load restores it.
    void set_species_graph(torch::Tensor edge_index, torch::Tensor edge_type);
    [[nodiscard]] bool has_species_graph() const noexcept;
    [[nodiscard]] torch::Tensor species_graph_edge_index() const;
    [[nodiscard]] torch::Tensor species_graph_edge_type() const;

    // Whether this model needs one: the architecture that reads a species
    // graph, and the only one that accepts one.
    [[nodiscard]] bool requires_species_graph() const noexcept {
        return config_.encoder_architecture == EncoderArchitecture::HeterogeneousGNN;
    }

    // Get task head by name
    [[nodiscard]] TaskHead& head(const std::string& name);
    [[nodiscard]] const TaskHead& head(const std::string& name) const;

private:
    // Dispatch taxonomy weight extraction to the active encoder
    torch::Tensor get_taxonomy_weights_(
        torch::Tensor (PlotEncoderImpl::*hash_fn)() const,
        torch::Tensor (PlotEncoderEmbedImpl::*embed_fn)() const,
        torch::Tensor (PlotEncoderSparseImpl::*sparse_fn)() const
    ) const;

    // The one encoder dispatch. Runs whichever encoder was built and returns
    // its full tail output: the latent plus, when the mixture of experts sits
    // in the tail, the load-balancing loss and gate probabilities. encode(),
    // encode_with_aux() and get_gate_probs() are the three readings of this,
    // rather than three copies of the same if-else chain. Encoders with no tail
    // (the adapter architectures and TraitNet) report the latent alone.
    //
    // `inputs` here is the value fuse_categoricals_() returned: its
    // `continuous` already carries the categorical embeddings.
    TailOutput encode_all(const PlotInputs& inputs);

    // The latent alone, with the post-placement mixture applied. Same
    // `inputs` convention as encode_all().
    torch::Tensor encode(const PlotInputs& inputs);

    // The latent and the MoE auxiliary loss (undefined without MoE). Same
    // `inputs` convention as encode_all().
    std::pair<torch::Tensor, torch::Tensor> encode_with_aux(const PlotInputs& inputs);

    // The one place categorical codes reach the encoder: a copy of `inputs`
    // whose `continuous` has the categorical embeddings concatenated, so it
    // matches the width the encoders were constructed with. Unchanged when
    // the model has no categorical columns; a model that has them but is given
    // none embeds the all-UNK code.
    PlotInputs fuse_categoricals_(const PlotInputs& inputs);

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

    // Mixture of experts over the finished latent, built only for
    // moe_placement == Post. Under the default (Tail) the mixture lives inside
    // the encoder instead, as its final stage.
    MixtureOfExperts post_moe_{nullptr};

    // Tabular adapter (used when encoder_architecture != MLP)
    TabularAdapter adapter_{nullptr};

    // Categorical embedder (one nn::Embedding table per categorical column).
    // Null when the schema has no categorical covariates.
    CategoricalEmbedder categorical_embedder_{nullptr};

    std::unordered_map<std::string, TaskHead> heads_;
};

TORCH_MODULE(ResolveModel);

} // namespace resolve
