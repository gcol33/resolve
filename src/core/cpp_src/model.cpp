#include "resolve/model.hpp"
#include "resolve/encoder.hpp"  // For MLPBlockConfig
#include <algorithm>
#include <stdexcept>

namespace resolve {

ResolveModelImpl::ResolveModelImpl(
    const ResolveSchema& schema,
    const ModelConfig& config
) : schema_(schema), config_(config)
{
    // Calculate number of continuous features based on mode
    int64_t n_coords = schema.has_coordinates ? 2 : 0;
    int64_t n_unknown_features = schema.track_unknown_fraction ? 1 : 0;
    if (schema.track_unknown_count) {
        n_unknown_features += 1;
    }

    int64_t n_continuous_base = n_coords + schema.covariate_names.size() + n_unknown_features;

    // Pre-compute vocab sizes for taxonomy (avoids repeating nested ternaries)
    auto genus_vocab_size = schema.has_taxonomy
        ? (schema.n_genera_vocab > 0 ? schema.n_genera_vocab : schema.n_genera + 1) : 0;
    auto family_vocab_size = schema.has_taxonomy
        ? (schema.n_families_vocab > 0 ? schema.n_families_vocab : schema.n_families + 1) : 0;

    // Check if using advanced architecture (non-MLP)
    bool use_adapter = (config.encoder_architecture != EncoderArchitecture::MLP &&
                        config.encoder_architecture != EncoderArchitecture::TraitNet);

    if (use_adapter) {
        // Use TabularAdapter for FT-Transformer, TabNet, SAINT, GNN
        adapter_ = register_module("adapter", TabularAdapter(schema, config));
    }

    // Check if MoE is enabled
    bool use_moe = (config.moe_routing != MoERoutingType::None);

    // Create MLP block config from model config
    MLPBlockConfig mlp_config = MLPBlockConfig::from_model_config(config);

    // Create appropriate encoder based on mode and MoE setting (only for MLP mode)
    if (use_adapter) {
        // Adapter handles encoding internally, no separate encoder needed
    }
    else if (config.species_encoding == SpeciesEncodingMode::Hash && !config.uses_explicit_vector) {
        // Hash mode: continuous includes hash_dim
        int64_t n_continuous = n_continuous_base + config.hash_dim;

        if (use_moe) {
            // MoE-enabled encoder with configurable architecture
            encoder_moe_ = register_module("encoder", PlotEncoderMoE(
                n_continuous,
                schema.has_taxonomy ? schema.n_genera + 1 : 0,
                schema.has_taxonomy ? schema.n_families + 1 : 0,
                config.genus_emb_dim,
                config.family_emb_dim,
                config.n_taxonomy_slots,
                config.hidden_dims,
                mlp_config,
                config.n_experts,
                config.expert_hidden_dims,
                config.moe_routing,
                config.moe_top_k,
                config.moe_noise_std
            ));
        } else {
            // Standard encoder with configurable architecture (+ optional TabM)
            encoder_hash_ = register_module("encoder", PlotEncoder(
                n_continuous,
                schema.has_taxonomy ? schema.n_genera + 1 : 0,
                schema.has_taxonomy ? schema.n_families + 1 : 0,
                config.genus_emb_dim,
                config.family_emb_dim,
                config.n_taxonomy_slots,
                config.hidden_dims,
                mlp_config,
                config.tabm
            ));
        }
    }
    else if (config.species_encoding == SpeciesEncodingMode::Embed) {
        // Embed mode: learnable species embeddings
        if (schema.n_species_vocab == 0) {
            throw std::runtime_error(
                "species_encoding=Embed requires n_species_vocab > 0 in schema"
            );
        }

        encoder_embed_ = register_module("encoder", PlotEncoderEmbed(
            n_continuous_base,
            schema.n_species_vocab,
            genus_vocab_size,
            family_vocab_size,
            config.species_embed_dim,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.top_k_species,
            config.n_taxonomy_slots,
            config.hidden_dims,
            mlp_config,
            config.tabm
        ));
    }
    else if (config.species_encoding == SpeciesEncodingMode::RankPool) {
        if (schema.n_species_vocab == 0) {
            throw std::runtime_error(
                "species_encoding=RankPool requires n_species_vocab > 0 in schema"
            );
        }
        encoder_rank_pool_ = register_module("encoder", PlotEncoderRankPool(
            n_continuous_base,
            schema.n_species_vocab,
            genus_vocab_size,
            family_vocab_size,
            config.species_embed_dim,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.hidden_dims,
            mlp_config,
            config.cover_dropout,
            config.tabm
        ));
    }
    else if (config.species_encoding == SpeciesEncodingMode::Transformer) {
        if (schema.n_species_vocab == 0) {
            throw std::runtime_error(
                "species_encoding=Transformer requires n_species_vocab > 0 in schema"
            );
        }
        encoder_transformer_ = register_module("encoder", PlotEncoderTransformer(
            n_continuous_base,
            schema.n_species_vocab,
            genus_vocab_size,
            family_vocab_size,
            config.d_model,
            config.n_heads,
            config.n_attention_layers,
            config.transformer_ff_dim,
            config.transformer_pooling,
            config.transformer_dropout,
            config.hidden_dims,
            mlp_config,
            config.cover_dropout,
            config.tabm
        ));
    }
    else {
        // Sparse mode (uses_explicit_vector=true): explicit species vector
        if (schema.n_species_vocab == 0) {
            throw std::runtime_error(
                "uses_explicit_vector=true requires n_species_vocab > 0 in schema"
            );
        }

        encoder_sparse_ = register_module("encoder", PlotEncoderSparse(
            n_continuous_base,
            schema.n_species_vocab,
            config.species_embed_dim,
            schema.has_taxonomy ? schema.n_genera + 1 : 0,
            schema.has_taxonomy ? schema.n_families + 1 : 0,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.n_taxonomy_slots,
            config.hidden_dims,
            mlp_config,
            config.tabm
        ));
    }

    // Create model-level MoE for embed/sparse modes (hash mode uses encoder_moe_ instead)
    if (use_moe && !encoder_moe_ && !use_adapter) {
        // Embed or sparse encoder + post-encoder MoE layer
        int64_t encoder_out = latent_dim();
        post_moe_ = register_module("post_moe", MixtureOfExperts(
            encoder_out,
            config.expert_hidden_dims,
            encoder_out,  // Preserve latent dimension
            config.n_experts,
            config.moe_routing,
            config.moe_top_k,
            config.moe_noise_std,
            mlp_config.dropout
        ));
    }

    // Create task heads with configurable architecture
    for (const auto& target : schema.targets) {
        TaskHead head = nullptr;  // Initialize as null
        if (config.head_hidden_dims.empty()) {
            // Legacy single-layer head
            head = register_module(
                "head_" + target.name,
                TaskHead(
                    latent_dim(),
                    target.task,
                    target.num_classes,
                    target.transform
                )
            );
        } else {
            // Multi-layer head with configurable architecture
            head = register_module(
                "head_" + target.name,
                TaskHead(
                    latent_dim(),
                    target.task,
                    target.num_classes,
                    target.transform,
                    config.head_hidden_dims,
                    config.head_activation,
                    config.head_dropout
                )
            );
        }
        heads_.emplace(target.name, std::move(head));
    }
}

int64_t ResolveModelImpl::latent_dim() const {
    if (adapter_) {
        return adapter_->latent_dim();
    } else if (encoder_moe_) {
        return encoder_moe_->latent_dim();
    } else if (encoder_rank_pool_) {
        return encoder_rank_pool_->latent_dim();
    } else if (encoder_transformer_) {
        return encoder_transformer_->latent_dim();
    } else if (encoder_hash_) {
        return encoder_hash_->latent_dim();
    } else if (encoder_embed_) {
        return encoder_embed_->latent_dim();
    } else {
        return encoder_sparse_->latent_dim();
    }
}

torch::Tensor ResolveModelImpl::encode(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover
) {
    torch::Tensor latent;
    if (adapter_) {
        latent = adapter_->forward(continuous, genus_ids, family_ids, species_ids, species_vector);
    } else if (encoder_moe_) {
        latent = encoder_moe_->forward_simple(continuous, genus_ids, family_ids);
    } else if (encoder_rank_pool_) {
        latent = encoder_rank_pool_->forward(
            continuous, species_ids, pool_genus_ids, pool_family_ids,
            pool_weights, pool_mask, pool_has_cover);
    } else if (encoder_transformer_) {
        latent = encoder_transformer_->forward(
            continuous, species_ids, pool_genus_ids, pool_family_ids,
            pool_weights, pool_mask, pool_has_cover);
    } else if (encoder_hash_) {
        latent = encoder_hash_->forward(continuous, genus_ids, family_ids);
    } else if (encoder_embed_) {
        latent = encoder_embed_->forward(continuous, species_ids, genus_ids, family_ids);
    } else {
        latent = encoder_sparse_->forward(continuous, species_vector, genus_ids, family_ids);
    }

    if (post_moe_) {
        latent = post_moe_->forward_simple(latent);
    }
    return latent;
}

std::pair<torch::Tensor, torch::Tensor> ResolveModelImpl::encode_with_aux(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover
) {
    if (adapter_) {
        auto latent = adapter_->forward(continuous, genus_ids, family_ids, species_ids, species_vector);
        if (post_moe_) {
            auto moe_result = post_moe_->forward(latent);
            return {moe_result.output, moe_result.aux_loss};
        }
        return {latent, torch::Tensor()};
    } else if (encoder_moe_) {
        return encoder_moe_->forward(continuous, genus_ids, family_ids);
    } else {
        torch::Tensor latent;
        if (encoder_rank_pool_) {
            latent = encoder_rank_pool_->forward(
                continuous, species_ids, pool_genus_ids, pool_family_ids,
                pool_weights, pool_mask, pool_has_cover);
        } else if (encoder_transformer_) {
            latent = encoder_transformer_->forward(
                continuous, species_ids, pool_genus_ids, pool_family_ids,
                pool_weights, pool_mask, pool_has_cover);
        } else if (encoder_hash_) {
            latent = encoder_hash_->forward(continuous, genus_ids, family_ids);
        } else if (encoder_embed_) {
            latent = encoder_embed_->forward(continuous, species_ids, genus_ids, family_ids);
        } else {
            latent = encoder_sparse_->forward(continuous, species_vector, genus_ids, family_ids);
        }
        if (post_moe_) {
            auto moe_result = post_moe_->forward(latent);
            return {moe_result.output, moe_result.aux_loss};
        }
        return {latent, torch::Tensor()};
    }
}

std::unordered_map<std::string, torch::Tensor> ResolveModelImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover
) {
    return forward_with_aux(continuous, genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover).outputs;
}

ModelForwardResult ResolveModelImpl::forward_with_aux(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover
) {
    auto [latent, aux_loss] = encode_with_aux(continuous, genus_ids, family_ids, species_ids, species_vector,
                                               pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover);

    std::unordered_map<std::string, torch::Tensor> outputs;
    for (auto& [name, head] : heads_) {
        outputs[name] = head->forward(latent);
    }

    return ModelForwardResult{outputs, aux_loss};
}

torch::Tensor ResolveModelImpl::forward_single(
    const std::string& target,
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    auto latent = encode(continuous, genus_ids, family_ids, species_ids, species_vector);
    return head(target)->forward(latent);
}

torch::Tensor ResolveModelImpl::get_latent(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover
) {
    return encode(continuous, genus_ids, family_ids, species_ids, species_vector,
                  pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover);
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
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    // Only implemented for hash encoder currently
    if (encoder_hash_) {
        return encoder_hash_->forward_with_activations(continuous, genus_ids, family_ids);
    }
    // For other encoders (embed, sparse, MoE), return empty tensor and activations
    // to signal that diagnostics aren't available. We don't call encode() here
    // because it would require species_ids/species_vector which aren't passed.
    return {torch::Tensor(), {}};
}

torch::Tensor ResolveModelImpl::get_taxonomy_weights_(
    torch::Tensor (PlotEncoderMoEImpl::*moe_fn)() const,
    torch::Tensor (PlotEncoderImpl::*hash_fn)() const,
    torch::Tensor (PlotEncoderEmbedImpl::*embed_fn)() const,
    torch::Tensor (PlotEncoderSparseImpl::*sparse_fn)() const
) const {
    if (encoder_moe_) return ((*encoder_moe_).*moe_fn)();
    if (encoder_hash_) return ((*encoder_hash_).*hash_fn)();
    if (encoder_embed_) return ((*encoder_embed_).*embed_fn)();
    if (encoder_sparse_) return ((*encoder_sparse_).*sparse_fn)();
    return torch::Tensor();
}

torch::Tensor ResolveModelImpl::get_genus_weights() const {
    if (encoder_rank_pool_) return encoder_rank_pool_->get_genus_weights();
    if (encoder_transformer_) return encoder_transformer_->get_genus_weights();
    return get_taxonomy_weights_(
        &PlotEncoderMoEImpl::get_genus_weights,
        &PlotEncoderImpl::get_genus_weights,
        &PlotEncoderEmbedImpl::get_genus_weights,
        &PlotEncoderSparseImpl::get_genus_weights
    );
}

torch::Tensor ResolveModelImpl::get_family_weights() const {
    if (encoder_rank_pool_) return encoder_rank_pool_->get_family_weights();
    if (encoder_transformer_) return encoder_transformer_->get_family_weights();
    return get_taxonomy_weights_(
        &PlotEncoderMoEImpl::get_family_weights,
        &PlotEncoderImpl::get_family_weights,
        &PlotEncoderEmbedImpl::get_family_weights,
        &PlotEncoderSparseImpl::get_family_weights
    );
}

torch::Tensor ResolveModelImpl::get_species_weights() const {
    if (encoder_rank_pool_) return encoder_rank_pool_->get_species_weights();
    if (encoder_transformer_) return encoder_transformer_->get_species_weights();
    if (encoder_embed_) return encoder_embed_->get_species_weights();
    return torch::Tensor();
}

torch::Tensor ResolveModelImpl::get_gate_probs(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    if (encoder_moe_) {
        return encoder_moe_->get_gate_probs(continuous, genus_ids, family_ids);
    }
    // Return empty tensor if MoE not enabled
    return torch::Tensor();
}

} // namespace resolve
