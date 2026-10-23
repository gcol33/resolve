#pragma once

#include "resolve/types.hpp"
#include "resolve/adapter.hpp"
#include "resolve/architectures.hpp"
#include "resolve/aux_loss.hpp"
#include "resolve/categorical.hpp"
#include "resolve/plot_encoder_transformer.hpp"
#include "resolve/plot_inputs.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/task_head.hpp"
#include "resolve/traitnet.hpp"
#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

// One forward's outputs per target and the auxiliary objective terms it adds
// to the training loss (aux_loss.hpp), each already weighted; a model with
// none returns an empty map.
struct ModelForwardResult {
    std::unordered_map<std::string, torch::Tensor> outputs;
    AuxLosses aux_losses;
};

// Full Resolve model: one encoder shared by the task heads of every target.
// Under encoder_architecture = MLP the encoder is the species encoder of the
// effective species encoding (species_encoder.hpp: hash, embed, sparse,
// rank_pool, transformer); the tabular architectures encode in the
// TabularAdapter, TraitNet through its trait matrix. A mixture of experts can
// stand in for the species encoder's MLP tail or run over the latent.
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

    // Forward pass returning the outputs and the auxiliary objective terms.
    ModelForwardResult forward_with_aux(const PlotInputs& inputs);

    // The names of the auxiliary terms forward_with_aux() returns for this
    // model, in AuxLosses order: kMoEAuxLoss with a mixture of experts,
    // kTabNetSparsityLoss for TabNet. Empty when it adds none.
    [[nodiscard]] std::vector<std::string> aux_loss_names() const;

    // Forward pass for one target's head.
    torch::Tensor forward_single(const std::string& target, const PlotInputs& inputs);

    // Latent representation (without heads).
    torch::Tensor get_latent(const PlotInputs& inputs);

    // The latent plus the hidden-layer activations of the species encoder's
    // tail (SpeciesEncoderImpl::encode), for diagnostics; the tabular
    // architectures and TraitNet return their latent with an empty list.
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
    // The encoding the model runs (effective_species_encoding): species_encoding
    // with uses_explicit_vector applied.
    [[nodiscard]] SpeciesEncodingMode effective_species_encoding() const noexcept {
        return effective_encoding_;
    }
    // Whether the hash embedding is the tail of the continuous block, i.e.
    // whether ContinuousInputs::hash_embedding is part of what this model reads.
    [[nodiscard]] bool hash_in_continuous() const {
        return species_encoding_traits(effective_encoding_).hash_in_continuous;
    }
    [[nodiscard]] bool uses_moe() const noexcept { return config_.moe_routing != MoERoutingType::None; }
    [[nodiscard]] int n_experts() const noexcept { return uses_moe() ? config_.n_experts : 0; }

    // An encoder that builds a plot's representation from the other plots of
    // the same forward (architectures.hpp: the plot GNN's kNN graph) would read
    // an arbitrary subset of plots under mini-batching, so it trains and
    // predicts on the whole set in one forward.
    [[nodiscard]] bool requires_full_batch_training() const noexcept {
        return is_batch_dependent(config_.encoder_architecture);
    }

    // The species encoder's embedding tables, one row per code (averaged over
    // slot positions where it keeps one table per slot); undefined for the
    // tabular architectures, TraitNet, and a table the encoding lacks.
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;
    [[nodiscard]] torch::Tensor get_species_weights() const;

    // The species, genus and family tables the species encoder reads a plot's
    // composition through, as the parameters themselves (empty for an adapter
    // architecture or TraitNet). ModelConfig::freeze_composition stops their
    // gradient at construction.
    [[nodiscard]] std::vector<torch::Tensor> composition_parameters() const;

    // Set species trait matrix (for TraitNet architecture). Stored in a
    // buffer, so it moves with the model and is saved in the checkpoint.
    void set_traits(torch::Tensor traits);

    // The species graph a HeterogeneousGNN passes messages on (species_graph.hpp).
    // It is built from the training data once and is part of the trained model:
    // Trainer::prepare_data builds and sets it, and it is held in model buffers,
    // so it is saved and loaded with the weights.
    void set_species_graph(torch::Tensor edge_index, torch::Tensor edge_type);
    [[nodiscard]] bool has_species_graph() const noexcept;
    [[nodiscard]] torch::Tensor species_graph_edge_index() const;
    [[nodiscard]] torch::Tensor species_graph_edge_type() const;

    // Whether this model needs one: the architecture that reads a species
    // graph, and the only one that accepts one.
    [[nodiscard]] bool requires_species_graph() const noexcept {
        return config_.encoder_architecture == EncoderArchitecture::HeterogeneousGNN;
    }

    // Buffers holding data the model was trained with rather than a quantity
    // of fixed shape -- the species graph, the TraitNet trait matrix, SAINT's
    // reference rows. A checkpoint load gives each the shape it was saved
    // with; every other tensor must match the model's shape exactly.
    [[nodiscard]] std::vector<torch::Tensor> data_buffers() const;

    // How many training rows SAINT's row attention keeps as the reference set
    // inference attends over (saint.hpp); 0 for every other architecture. The
    // reference rows are captured, as raw model inputs, from the training
    // batches: every training forward of at least as many rows as are held
    // refreshes them with its leading rows.
    [[nodiscard]] int64_t row_reference_rows() const noexcept { return row_reference_rows_; }

    // The transformer species encoder of a species_encoding == Transformer
    // model, which masked-species pretraining trains; throws
    // std::invalid_argument for any other model.
    [[nodiscard]] PlotEncoderTransformer transformer_encoder() const;

    // Get task head by name
    [[nodiscard]] TaskHead& head(const std::string& name);
    [[nodiscard]] const TaskHead& head(const std::string& name) const;

private:
    // What the encoder stage produces: the latent, the weighted auxiliary
    // terms, and the MoE gate probabilities (undefined without MoE).
    struct EncodedPlots {
        torch::Tensor latent;
        AuxLosses aux_losses;
        torch::Tensor gate_probs;
    };

    // The one encoder dispatch. Fuses the categorical embeddings into
    // `inputs`, runs whichever encoder was built and then the post-placement
    // mixture of experts if there is one. In training mode it also refreshes
    // SAINT's reference rows from `inputs`. encode(), forward_with_aux(),
    // encode_with_activations() and get_gate_probs() are readings of this
    // rather than copies of the chain. With `activations`, the species
    // encoder's tail appends its hidden-layer activations to it.
    EncodedPlots encode_all(const PlotInputs& inputs,
                            std::vector<torch::Tensor>* activations = nullptr);

    // The latent alone; same `inputs` convention as encode_all().
    torch::Tensor encode(const PlotInputs& inputs);

    // SAINT's reference rows, categorical embeddings fused, for an eval
    // forward; empty in training mode, for other architectures and before
    // any training forward.
    PlotInputs row_reference();

    // Keep the leading row_reference_rows_ rows of a training batch as the
    // reference set, unless the batch is smaller than the set already held.
    void capture_row_reference(const PlotInputs& inputs);

    // The one place categorical codes reach the encoder: a copy of `inputs`
    // whose `continuous` has the categorical embeddings concatenated, so it
    // matches the width the encoders were constructed with. Unchanged when
    // the model has no categorical columns; a model that has them but is given
    // none embeds the all-UNK code.
    PlotInputs fuse_categoricals_(const PlotInputs& inputs);

    ResolveSchema schema_;
    ModelConfig config_;
    SpeciesEncodingMode effective_encoding_;

    // The species encoder of the MLP architecture, registered as "encoder".
    SpeciesEncoder encoder_{nullptr};

    // TraitNet encoder (used when encoder_architecture == TraitNet)
    TraitNetEncoder trait_net_encoder_{nullptr};

    // Mixture of experts over the finished latent, built only for
    // moe_placement == Post. Under the default (Tail) the mixture lives inside
    // the encoder instead, as its final stage.
    MixtureOfExperts post_moe_{nullptr};

    // Tabular adapter (the architectures other than MLP and TraitNet)
    TabularAdapter adapter_{nullptr};

    // SAINT's reference rows as raw model inputs, one buffer per field
    // (empty until captured), and how many rows it holds at most.
    PlotInputs row_reference_;
    int64_t row_reference_rows_ = 0;

    // Categorical embedder (one nn::Embedding table per categorical column).
    // Null when the schema has no categorical covariates.
    CategoricalEmbedder categorical_embedder_{nullptr};

    std::unordered_map<std::string, TaskHead> heads_;
};

TORCH_MODULE(ResolveModel);

} // namespace resolve
