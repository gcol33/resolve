#include "resolve/model.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/plot_encoder_transformer.hpp"
#include <algorithm>
#include <stdexcept>

namespace resolve {

ResolveModelImpl::ResolveModelImpl(
    const ResolveSchema& schema,
    const ModelConfig& config
) : schema_(schema),
    config_(config),
    effective_encoding_(resolve::effective_species_encoding(config))
{
    // Synchronise the schema's categorical_embed_dim with the value the model
    // is actually going to use. The dataset loader leaves it at the default;
    // the ModelConfig is the source of truth at construction time. We record
    // the chosen value back on the schema so that (a) `schema()` reports the
    // truth, and (b) checkpoint save persists the right number for
    // Predictor.load to reconstruct the embedder.
    if (config.categorical_embed_dim <= 0) {
        throw std::runtime_error(
            "ResolveModel: ModelConfig.categorical_embed_dim must be > 0 "
            "(got " + std::to_string(config.categorical_embed_dim) + ")");
    }
    schema_.categorical_embed_dim = config.categorical_embed_dim;

    // Build the categorical embedder if the schema has any categorical
    // columns. Each column gets its own nn::Embedding(vocab_size, embed_dim)
    // table; CategoricalEmbedder concatenates the per-column lookups in
    // forward(). When there are no categoricals, the embedder is left as
    // nullptr and fuse_categoricals_() is a no-op.
    if (schema_.has_categoricals()) {
        categorical_embedder_ = register_module(
            "categorical_embedder",
            CategoricalEmbedder(schema_.categorical_vocab_sizes,
                                schema_.categorical_embed_dim));
    }

    // The width of the continuous block the encoder receives: the schema's
    // continuous columns plus the categorical embeddings fuse_categoricals_()
    // concatenates in forward.
    const int64_t n_continuous_base = schema_.continuous_width();

    // The tabular architectures encode inside the adapter, TraitNet through
    // its trait matrix; the MLP architecture runs a species encoder.
    const bool use_adapter = config.encoder_architecture != EncoderArchitecture::MLP &&
                             config.encoder_architecture != EncoderArchitecture::TraitNet;

    if (use_adapter) {
        adapter_ = register_module("adapter", TabularAdapter(schema_, config));
        row_reference_rows_ = adapter_->row_reference_rows();
        if (row_reference_rows_ > 0) {
            row_reference_.for_each([this](const PlotInputField& field, torch::Tensor& buffer) {
                buffer = register_buffer(std::string("row_reference_") + field.name,
                                         torch::empty({0}, field.dtype));
            });
        }
    }

    const bool use_moe = config.moe_routing != MoERoutingType::None;

    // MoEPlacement::Tail asks the mixture to stand in for the encoder's final
    // MLP stage. The adapter architectures and TraitNet have no such stage to
    // give up, so the request cannot be honoured there; name the placement that
    // can be, rather than dropping the knob or refusing MoE outright.
    const bool has_mlp_tail = config.encoder_architecture == EncoderArchitecture::MLP;
    if (use_moe && config.moe_placement == MoEPlacement::Tail && !has_mlp_tail) {
        throw std::invalid_argument(
            std::string("moe_placement=tail asks the mixture of experts to "
            "replace the encoder's final MLP stage, but encoder_architecture=") +
            encoder_architecture_to_string(config.encoder_architecture) +
            " has no MLP tail to replace. Set moe_placement=post to run the "
            "mixture over the finished latent instead, use "
            "encoder_architecture=mlp, or set moe_routing=none.");
    }

    if (config.encoder_architecture == EncoderArchitecture::TraitNet) {
        // TraitNet reads the environment block (coordinates, covariates,
        // unknown-species columns); the species enter through its trait
        // matrix, not through a species encoding.
        const auto& tc = config.trait_net;
        trait_net_encoder_ = register_module("trait_net_encoder", TraitNetEncoder(
            /*env_dim=*/n_continuous_base,
            /*trait_dim=*/tc.trait_dim,
            /*n_species=*/schema.n_species_vocab > 0 ? schema.n_species_vocab : schema.n_species,
            /*hidden_dim=*/tc.env_dim,
            /*n_layers=*/2,
            /*dropout=*/config.dropout,
            /*interaction_dim=*/tc.interaction_dim,
            /*interaction=*/tc.interaction,
            /*shared_trait_encoder=*/tc.shared_trait_encoder
        ));
    } else if (!use_adapter) {
        encoder_ = register_module(
            "encoder", SpeciesEncoder(make_species_encoder(
                           SpeciesEncoderSpec::from_schema(schema_, config))));
    }

    // MoEPlacement::Post: the encoder produced its latent already, and the
    // mixture maps that latent to one of the same width. This is the placement
    // an encoder with no MLP tail can still use.
    if (use_moe && config.moe_placement == MoEPlacement::Post) {
        int64_t encoder_out = latent_dim();
        post_moe_ = register_module("post_moe", MixtureOfExperts(
            encoder_out,
            config.expert_hidden_dims,
            encoder_out,  // Preserve latent dimension
            config.n_experts,
            config.moe_routing,
            config.moe_top_k,
            config.moe_noise_std,
            config.dropout
        ));
    }

    if (config.freeze_composition) {
        const auto tables = composition_parameters();
        if (tables.empty()) {
            throw std::invalid_argument(
                std::string("freeze_composition=true fixes the species, genus and "
                "family tables of a species encoder, but this model has none: "
                "encoder_architecture=") +
                encoder_architecture_to_string(config.encoder_architecture) +
                ", species_encoding=" + species_encoding_to_string(config.species_encoding) +
                (schema.has_taxonomy ? "" : ", and no taxonomy") +
                ". Use encoder_architecture=mlp with embed, sparse, rank_pool or "
                "transformer, or hash with taxonomy.");
        }
        // AdamW skips a parameter without a gradient, weight decay included,
        // so each table keeps its initial values through training.
        for (const auto& table : tables) table.requires_grad_(false);
    }

    for (const auto& target : schema.targets) {
        heads_.emplace(target.name, register_module(
            "head_" + target.name,
            TaskHead(latent_dim(), target.task, target.num_classes, target.transform,
                     config.head_hidden_dims, config.head_activation, config.head_dropout)));
    }
}

PlotInputs ResolveModelImpl::fuse_categoricals_(const PlotInputs& inputs) {
    // No-op when the model has no categorical columns.
    if (!categorical_embedder_) {
        return inputs;
    }
    const torch::Tensor& continuous = inputs.continuous;
    torch::Tensor categorical_ids = inputs.categorical_ids;
    if (!continuous.defined() || continuous.dim() != 2) {
        throw std::runtime_error(
            "ResolveModel: fuse_categoricals_ requires a 2-D continuous "
            "tensor (got " +
            std::string(continuous.defined() ? "dim=" +
                std::to_string(continuous.dim()) : "undefined") + ")");
    }

    const int64_t batch = continuous.size(0);
    const int64_t expected_cols = categorical_embedder_->n_columns();

    torch::Tensor cat_part;
    if (!categorical_ids.defined() || categorical_ids.numel() == 0) {
        // Caller didn't supply categoricals but the model has them. Feed an
        // all-UNK (code 0) id tensor through the embedder so the encoder sees
        // the trained UNK embedding, NOT a zero vector: the embedder has no
        // padding_idx, so row 0 is a learned parameter and forward(no ids) must
        // equal forward(all-UNK ids). A zeros pad would feed an input the model
        // never saw in training.
        auto unk_ids = torch::zeros({batch, expected_cols},
                                    torch::TensorOptions().dtype(torch::kLong)
                                        .device(continuous.device()));
        cat_part = categorical_embedder_->forward(unk_ids);
    } else {
        if (categorical_ids.dim() != 2) {
            throw std::runtime_error(
                "ResolveModel: categorical_ids must be 2-D (got dim=" +
                std::to_string(categorical_ids.dim()) + ")");
        }
        if (categorical_ids.size(0) != batch) {
            throw std::runtime_error(
                "ResolveModel: categorical_ids batch (" +
                std::to_string(categorical_ids.size(0)) +
                ") does not match continuous batch (" +
                std::to_string(batch) + ")");
        }
        if (categorical_ids.size(1) != expected_cols) {
            throw std::runtime_error(
                "ResolveModel: categorical_ids has " +
                std::to_string(categorical_ids.size(1)) +
                " columns but model was built for " +
                std::to_string(expected_cols) + " (schema mismatch)");
        }
        // Ensure tensor is on the same device as continuous before the
        // embedding lookup. CategoricalEmbedder also coerces dtype to int64.
        if (categorical_ids.device() != continuous.device()) {
            categorical_ids = categorical_ids.to(continuous.device());
        }
        cat_part = categorical_embedder_->forward(categorical_ids);
    }

    PlotInputs fused = inputs;
    fused.continuous = torch::cat({continuous, cat_part}, /*dim=*/1);
    return fused;
}

int64_t ResolveModelImpl::latent_dim() const {
    if (trait_net_encoder_) return trait_net_encoder_->output_dim();
    if (adapter_) return adapter_->latent_dim();
    return encoder_->latent_dim();
}

ResolveModelImpl::EncodedPlots ResolveModelImpl::encode_all(
    const PlotInputs& raw, std::vector<torch::Tensor>* activations) {
    if (is_training() && row_reference_rows_ > 0) {
        capture_row_reference(raw);
    }
    const PlotInputs inputs = fuse_categoricals_(raw);

    EncodedPlots encoded;
    // A tail mixture of experts reports its load-balancing loss with the
    // encoder output.
    auto take_tail = [&](TailOutput tail) {
        encoded.latent = tail.latent;
        encoded.gate_probs = tail.gate_probs;
        if (tail.aux_loss.defined()) {
            encoded.aux_losses[kMoEAuxLoss] = config_.moe_aux_loss_weight * tail.aux_loss;
        }
    };

    if (trait_net_encoder_) {
        // TraitNet reads the environment block; the species enter through the
        // trait matrix set_traits() stored.
        encoded.latent = trait_net_encoder_->forward(inputs.continuous);
    } else if (adapter_) {
        auto adapted = adapter_->forward(inputs, row_reference());
        encoded.latent = adapted.latent;
        encoded.aux_losses = std::move(adapted.aux_losses);
    } else {
        take_tail(encoder_->encode(inputs, activations));
    }

    // Only one MoE placement is ever built, so the post mixture never meets a
    // tail one.
    if (post_moe_) {
        auto moe = post_moe_->forward(encoded.latent);
        encoded.latent = moe.output;
        encoded.gate_probs = moe.gate_probs;
        if (moe.aux_loss.defined()) {
            encoded.aux_losses[kMoEAuxLoss] = config_.moe_aux_loss_weight * moe.aux_loss;
        }
    }
    return encoded;
}

torch::Tensor ResolveModelImpl::encode(const PlotInputs& inputs) {
    return encode_all(inputs).latent;
}

std::unordered_map<std::string, torch::Tensor> ResolveModelImpl::forward(
    const PlotInputs& inputs
) {
    return forward_with_aux(inputs).outputs;
}

ModelForwardResult ResolveModelImpl::forward_with_aux(const PlotInputs& inputs) {
    auto encoded = encode_all(inputs);

    std::unordered_map<std::string, torch::Tensor> outputs;
    for (auto& [name, head] : heads_) {
        outputs[name] = head->forward(encoded.latent);
    }
    return ModelForwardResult{std::move(outputs), std::move(encoded.aux_losses)};
}

std::vector<std::string> ResolveModelImpl::aux_loss_names() const {
    AuxLosses present;
    if (uses_moe()) present[kMoEAuxLoss];
    if (config_.encoder_architecture == EncoderArchitecture::TabNet) {
        present[kTabNetSparsityLoss];
    }
    std::vector<std::string> names;
    names.reserve(present.size());
    for (const auto& entry : present) names.push_back(entry.first);
    return names;
}

torch::Tensor ResolveModelImpl::forward_single(
    const std::string& target,
    const PlotInputs& inputs
) {
    return head(target)->forward(encode(inputs));
}

torch::Tensor ResolveModelImpl::get_latent(const PlotInputs& inputs) {
    return encode(inputs);
}

PlotInputs ResolveModelImpl::row_reference() {
    if (row_reference_rows_ == 0 || is_training()) return {};
    PlotInputs reference = row_reference_.without_empty().conformed();
    if (reference.n_plots() == 0) return {};
    return fuse_categoricals_(reference);
}

void ResolveModelImpl::capture_row_reference(const PlotInputs& inputs) {
    const int64_t n_rows = std::min(inputs.n_plots(), row_reference_rows_);
    if (n_rows == 0 || n_rows < row_reference_.n_plots()) return;

    std::vector<torch::Tensor> rows;
    inputs.slice(0, n_rows).for_each([&rows](const PlotInputField&, const torch::Tensor& t) {
        rows.push_back(t);
    });
    torch::NoGradGuard no_grad;
    std::size_t i = 0;
    row_reference_.for_each([&](const PlotInputField& field, torch::Tensor& buffer) {
        // A module buffer is dense, so a sparse-rows field is held as its
        // dense rows; row_reference() conforms it back.
        const torch::Tensor& source = rows[i++];
        buffer.set_data(source.defined()
            ? (is_sparse_rows(source) ? sparse_rows_to_dense(source) : source.detach().clone())
            : torch::empty({0}, buffer.options().dtype(field.dtype)));
    });
}

std::vector<torch::Tensor> ResolveModelImpl::data_buffers() const {
    std::vector<torch::Tensor> buffers;
    if (adapter_) buffers = adapter_->data_buffers();
    if (trait_net_encoder_) buffers.push_back(trait_net_encoder_->traits_buffer());
    if (row_reference_rows_ > 0) {
        row_reference_.for_each([&buffers](const PlotInputField&, const torch::Tensor& t) {
            buffers.push_back(t);
        });
    }
    return buffers;
}

PlotEncoderTransformer ResolveModelImpl::transformer_encoder() const {
    auto transformer = encoder_
        ? std::dynamic_pointer_cast<PlotEncoderTransformerImpl>(encoder_.ptr())
        : nullptr;
    if (!transformer) {
        throw std::invalid_argument(
            std::string("ResolveModel::transformer_encoder: the model has no "
                        "transformer species encoder (species_encoding = ") +
            species_encoding_to_string(config_.species_encoding) +
            ", encoder_architecture = " +
            encoder_architecture_to_string(config_.encoder_architecture) + ")");
    }
    return PlotEncoderTransformer(transformer);
}

TaskHead& ResolveModelImpl::head(const std::string& name) {
    auto it = heads_.find(name);
    if (it == heads_.end()) {
        throw std::runtime_error("Head not found: " + name);
    }
    return it->second;
}

const TaskHead& ResolveModelImpl::head(const std::string& name) const {
    auto it = heads_.find(name);
    if (it == heads_.end()) {
        throw std::runtime_error("Head not found: " + name);
    }
    return it->second;
}

std::pair<torch::Tensor, std::vector<torch::Tensor>> ResolveModelImpl::encode_with_activations(
    const PlotInputs& inputs
) {
    std::vector<torch::Tensor> activations;
    auto encoded = encode_all(inputs, &activations);
    return {encoded.latent, std::move(activations)};
}

torch::Tensor ResolveModelImpl::get_genus_weights() const {
    return encoder_ ? encoder_->genus_weights() : torch::Tensor();
}

torch::Tensor ResolveModelImpl::get_family_weights() const {
    return encoder_ ? encoder_->family_weights() : torch::Tensor();
}

torch::Tensor ResolveModelImpl::get_species_weights() const {
    return encoder_ ? encoder_->species_weights() : torch::Tensor();
}

std::vector<torch::Tensor> ResolveModelImpl::composition_parameters() const {
    return encoder_ ? encoder_->composition_parameters() : std::vector<torch::Tensor>{};
}

torch::Tensor ResolveModelImpl::get_gate_probs(const PlotInputs& inputs) {
    // Under MoEPlacement::Post the probabilities come from the mixture over
    // the latent; under Tail they come back with the encoder's own output.
    if (!uses_moe()) {
        return torch::Tensor();
    }
    return encode_all(inputs).gate_probs;
}

void ResolveModelImpl::set_traits(torch::Tensor traits) {
    if (!trait_net_encoder_) {
        throw std::runtime_error(
            "set_traits() is only valid when encoder_architecture is TraitNet");
    }
    trait_net_encoder_->set_traits(std::move(traits));
}

void ResolveModelImpl::set_species_graph(torch::Tensor edge_index,
                                        torch::Tensor edge_type) {
    if (!adapter_ || !requires_species_graph()) {
        throw std::invalid_argument(
            "set_species_graph() is only valid when encoder_architecture is "
            "heterogeneous_gnn; this model is built with " +
            std::string(enum_to_name(config_.encoder_architecture)) + ".");
    }
    adapter_->set_species_graph(edge_index, edge_type);
}

bool ResolveModelImpl::has_species_graph() const noexcept {
    return adapter_ && adapter_->has_species_graph();
}

torch::Tensor ResolveModelImpl::species_graph_edge_index() const {
    return adapter_ ? adapter_->species_graph_edge_index() : torch::Tensor();
}

torch::Tensor ResolveModelImpl::species_graph_edge_type() const {
    return adapter_ ? adapter_->species_graph_edge_type() : torch::Tensor();
}

} // namespace resolve
