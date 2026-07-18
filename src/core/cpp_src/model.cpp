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
    int64_t n_categorical_embed = 0;
    if (schema_.has_categoricals()) {
        categorical_embedder_ = register_module(
            "categorical_embedder",
            CategoricalEmbedder(schema_.categorical_vocab_sizes,
                                schema_.categorical_embed_dim));
        n_categorical_embed = categorical_embedder_->output_dim();
    }

    // Calculate number of continuous features based on mode.
    // The encoder is built to accept (base continuous) + (categorical embed
    // dims) because fuse_categoricals_() will concatenate them in forward.
    int64_t n_coords = schema.has_coordinates ? 2 : 0;
    int64_t n_unknown_features = schema.track_unknown_fraction ? 1 : 0;
    if (schema.track_unknown_count) {
        n_unknown_features += 1;
    }

    int64_t n_continuous_base = n_coords + schema.covariate_names.size()
                              + n_unknown_features + n_categorical_embed;

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

    // MoE is only wired for the hash encoder (encoder_moe_) and the embed/sparse
    // encoders (post_moe_). The non-MLP adapter architectures build neither, so a
    // MoE request alongside an adapter arch would be silently discarded (issue
    // #83). Reject it loudly rather than train a plain adapter with inert MoE knobs.
    if (use_moe && use_adapter) {
        throw std::invalid_argument(
            "moe_routing is set but encoder_architecture is a non-MLP adapter "
            "architecture (FT-Transformer/TabNet/SAINT/GNN/ExcelFormer/"
            "HeterogeneousGNN); Mixture-of-Experts is not supported for these. "
            "Use encoder_architecture=MLP, or set moe_routing=None.");
    }

    // Create MLP block config from model config
    MLPBlockConfig mlp_config = MLPBlockConfig::from_model_config(config);

    // Create appropriate encoder based on mode and MoE setting (only for MLP mode)
    if (use_adapter) {
        // Adapter handles encoding internally, no separate encoder needed
    }
    else if (config.encoder_architecture == EncoderArchitecture::TraitNet) {
        // TraitNet: direct path, bypasses both adapter and species-encoding encoders.
        // env_dim = continuous base features (coords + covariates + unknown features).
        // TraitNet does NOT use hash embeddings or species IDs — it uses a trait matrix.
        const auto& tc = config.trait_net;
        trait_net_encoder_ = register_module("trait_net_encoder", TraitNetEncoder(
            /*env_dim=*/n_continuous_base,
            /*trait_dim=*/tc.trait_dim,
            /*n_species=*/schema.n_species_vocab > 0 ? schema.n_species_vocab : schema.n_species,
            /*hidden_dim=*/tc.env_dim,
            /*n_layers=*/2,
            /*dropout=*/config.dropout
        ));
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

torch::Tensor ResolveModelImpl::fuse_categoricals_(
    torch::Tensor continuous, torch::Tensor categorical_ids
) {
    // No-op when the model has no categorical columns.
    if (!categorical_embedder_) {
        return continuous;
    }
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

    return torch::cat({continuous, cat_part}, /*dim=*/1);
}

int64_t ResolveModelImpl::latent_dim() const {
    if (trait_net_encoder_) {
        return trait_net_encoder_->output_dim();
    } else if (adapter_) {
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
    if (trait_net_encoder_) {
        // TraitNet uses only env features (continuous without hash embedding).
        // Traits are pre-set via set_traits().
        latent = trait_net_encoder_->forward(continuous);
    } else if (adapter_) {
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
    if (trait_net_encoder_) {
        auto latent = trait_net_encoder_->forward(continuous);
        // Apply post_moe_ on the training path too. encode() (get_latent) always
        // runs it; omitting it here left post_moe_ untrained (no gradient) while
        // get_latent still applied it, so extracted latents diverged from what
        // the heads trained on.
        if (post_moe_) {
            auto moe_result = post_moe_->forward(latent);
            return {moe_result.output, moe_result.aux_loss};
        }
        return {latent, torch::Tensor()};
    } else if (adapter_) {
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
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids
) {
    return forward_with_aux(continuous, genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                            categorical_ids).outputs;
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
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids
) {
    // Fuse categorical embeddings into `continuous` before any encoder runs.
    // No-op when the model has no categorical columns.
    continuous = fuse_categoricals_(std::move(continuous),
                                    std::move(categorical_ids));

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
    torch::Tensor species_vector,
    torch::Tensor categorical_ids
) {
    // forward_single carries no per-species pool tensors, so a rank_pool /
    // transformer encoder would pool from empty inputs and return a wrong
    // latent (taxonomy skipped, weighting silently binary). Refuse loudly and
    // point callers at the full forward() path instead of failing silently.
    if (config_.species_encoding == SpeciesEncodingMode::RankPool ||
        config_.species_encoding == SpeciesEncodingMode::Transformer) {
        throw std::runtime_error(
            "forward_single does not support rank_pool/transformer encoders "
            "(no pool tensors). Use forward() / forward_with_aux() with the "
            "pool_* tensors instead.");
    }
    continuous = fuse_categoricals_(std::move(continuous),
                                    std::move(categorical_ids));
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
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids
) {
    continuous = fuse_categoricals_(std::move(continuous),
                                    std::move(categorical_ids));
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
    torch::Tensor family_ids,
    torch::Tensor categorical_ids
) {
    // Fuse categorical embeddings into continuous so the encoder sees the
    // same input shape it was constructed for. No-op for models without
    // categoricals.
    continuous = fuse_categoricals_(std::move(continuous),
                                    std::move(categorical_ids));
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

void ResolveModelImpl::set_traits(torch::Tensor traits) {
    if (!trait_net_encoder_) {
        throw std::runtime_error(
            "set_traits() is only valid when encoder_architecture is TraitNet");
    }
    trait_net_encoder_->set_traits(std::move(traits));
}

} // namespace resolve
