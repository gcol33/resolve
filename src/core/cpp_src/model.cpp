#include "resolve/model.hpp"
#include "resolve/encoder.hpp"  // For MLPBlockConfig
#include "resolve/enum_names.hpp"  // Architecture name in the MoE placement error
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

    // Pre-compute the taxonomy embedding-table sizes once (single source of
    // truth for every encoder family). TaxonomyVocab::fit reserves <UNK>=0, so
    // n_genera()/n_families() already count the UNK slot (max id = n_genera - 1)
    // and the correct table size is exactly n_genera, NOT n_genera + 1. The
    // hash/sparse/MoE/adapter paths previously passed n_genera + 1 directly and
    // over-allocated one unused row per table, disagreeing with the embed/pool/
    // transformer paths that used these locals (issue #99). n_*_vocab (set by the
    // dataset alongside n_*) is preferred; the bare-n_* fallback covers a
    // hand-built schema that left n_*_vocab at 0.
    auto genus_vocab_size = schema.has_taxonomy
        ? (schema.n_genera_vocab > 0 ? schema.n_genera_vocab : schema.n_genera) : 0;
    auto family_vocab_size = schema.has_taxonomy
        ? (schema.n_families_vocab > 0 ? schema.n_families_vocab : schema.n_families) : 0;

    // Check if using advanced architecture (non-MLP)
    bool use_adapter = (config.encoder_architecture != EncoderArchitecture::MLP &&
                        config.encoder_architecture != EncoderArchitecture::TraitNet);

    if (use_adapter) {
        // Use TabularAdapter for FT-Transformer, TabNet, SAINT, GNN
        adapter_ = register_module("adapter", TabularAdapter(schema, config));
    }

    // Check if MoE is enabled
    bool use_moe = (config.moe_routing != MoERoutingType::None);

    // MoEPlacement::Tail asks the mixture to stand in for the encoder's final
    // MLP stage. The adapter architectures and TraitNet have no such stage to
    // give up, so the request cannot be honoured there; name the placement that
    // can be, rather than dropping the knob or refusing MoE outright.
    const bool has_mlp_tail = !use_adapter &&
        config.encoder_architecture != EncoderArchitecture::TraitNet;
    if (use_moe && config.moe_placement == MoEPlacement::Tail && !has_mlp_tail) {
        throw std::invalid_argument(
            std::string("moe_placement=tail asks the mixture of experts to "
            "replace the encoder's final MLP stage, but encoder_architecture=") +
            encoder_architecture_to_string(config.encoder_architecture) +
            " has no MLP tail to replace. Set moe_placement=post to run the "
            "mixture over the finished latent instead, use "
            "encoder_architecture=mlp, or set moe_routing=none.");
    }

    // Create MLP block config from model config
    MLPBlockConfig mlp_config = MLPBlockConfig::from_model_config(config);

    // Reads the MoE knobs only under MoEPlacement::Tail; a Post run leaves this
    // inert and the mixture is built over the latent further down. Every
    // species encoder takes it, so the mixture reaches all five of them.
    MoETailConfig moe_tail = MoETailConfig::from_model_config(config);

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

        encoder_hash_ = register_module("encoder", PlotEncoder(
            n_continuous,
            genus_vocab_size,
            family_vocab_size,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.n_taxonomy_slots,
            config.hidden_dims,
            mlp_config,
            config.tabm,
            moe_tail
        ));
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
            config.tabm,
            moe_tail
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
            config.tabm,
            moe_tail
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
            config.tabm,
            moe_tail
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
            genus_vocab_size,
            family_vocab_size,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.n_taxonomy_slots,
            config.hidden_dims,
            mlp_config,
            config.tabm,
            moe_tail
        ));
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

TailOutput ResolveModelImpl::encode_all(
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
        // TraitNet uses only env features (continuous without hash embedding).
        // Traits are pre-set via set_traits().
        return {trait_net_encoder_->forward(std::move(continuous)), {}, {}};
    }
    if (adapter_) {
        return {adapter_->forward(std::move(continuous), std::move(genus_ids),
                                  std::move(family_ids), std::move(species_ids),
                                  std::move(species_vector)), {}, {}};
    }
    if (encoder_rank_pool_) {
        return encoder_rank_pool_->encode(
            std::move(continuous), std::move(species_ids),
            std::move(pool_genus_ids), std::move(pool_family_ids),
            std::move(pool_weights), std::move(pool_mask),
            std::move(pool_has_cover));
    }
    if (encoder_transformer_) {
        return encoder_transformer_->encode(
            std::move(continuous), std::move(species_ids),
            std::move(pool_genus_ids), std::move(pool_family_ids),
            std::move(pool_weights), std::move(pool_mask),
            std::move(pool_has_cover));
    }
    if (encoder_hash_) {
        return encoder_hash_->encode(std::move(continuous), std::move(genus_ids),
                                     std::move(family_ids));
    }
    if (encoder_embed_) {
        return encoder_embed_->encode(std::move(continuous), std::move(species_ids),
                                      std::move(genus_ids), std::move(family_ids));
    }
    return encoder_sparse_->encode(std::move(continuous), std::move(species_vector),
                                   std::move(genus_ids), std::move(family_ids));
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
    auto latent = encode_all(
        std::move(continuous), std::move(genus_ids), std::move(family_ids),
        std::move(species_ids), std::move(species_vector),
        std::move(pool_genus_ids), std::move(pool_family_ids),
        std::move(pool_weights), std::move(pool_mask),
        std::move(pool_has_cover)).latent;

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
    auto encoded = encode_all(
        std::move(continuous), std::move(genus_ids), std::move(family_ids),
        std::move(species_ids), std::move(species_vector),
        std::move(pool_genus_ids), std::move(pool_family_ids),
        std::move(pool_weights), std::move(pool_mask),
        std::move(pool_has_cover));

    // post_moe_ runs on the training path as well as in encode(). Omitting it
    // here once left it untrained -- no gradient reached it -- while get_latent
    // still applied it, so extracted latents diverged from what the heads
    // trained on. Only one placement is ever built, so the two auxiliary
    // losses never both exist.
    if (post_moe_) {
        auto moe_result = post_moe_->forward(encoded.latent);
        return {moe_result.output, moe_result.aux_loss};
    }
    return {encoded.latent, encoded.aux_loss};
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
    torch::Tensor (PlotEncoderImpl::*hash_fn)() const,
    torch::Tensor (PlotEncoderEmbedImpl::*embed_fn)() const,
    torch::Tensor (PlotEncoderSparseImpl::*sparse_fn)() const
) const {
    if (encoder_hash_) return ((*encoder_hash_).*hash_fn)();
    if (encoder_embed_) return ((*encoder_embed_).*embed_fn)();
    if (encoder_sparse_) return ((*encoder_sparse_).*sparse_fn)();
    return torch::Tensor();
}

torch::Tensor ResolveModelImpl::get_genus_weights() const {
    if (encoder_rank_pool_) return encoder_rank_pool_->get_genus_weights();
    if (encoder_transformer_) return encoder_transformer_->get_genus_weights();
    return get_taxonomy_weights_(
        &PlotEncoderImpl::get_genus_weights,
        &PlotEncoderEmbedImpl::get_genus_weights,
        &PlotEncoderSparseImpl::get_genus_weights
    );
}

torch::Tensor ResolveModelImpl::get_family_weights() const {
    if (encoder_rank_pool_) return encoder_rank_pool_->get_family_weights();
    if (encoder_transformer_) return encoder_transformer_->get_family_weights();
    return get_taxonomy_weights_(
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
    // A hash-mode model routes through here with the species signal already in
    // `continuous`, which is why this takes no species argument. Under
    // MoEPlacement::Post the probabilities come from the mixture over the
    // latent; under Tail they come back with the encoder's own output. An
    // encoder that needs species IDs or a species vector cannot be driven from
    // this signature, so it says so rather than reporting nothing -- the
    // undefined tensor this used to return for every non-hash model was
    // indistinguishable from "MoE is off".
    if (config_.moe_routing == MoERoutingType::None) {
        return torch::Tensor();
    }
    if (!encoder_hash_ && !trait_net_encoder_ && !adapter_) {
        throw std::invalid_argument(
            "get_gate_probs(continuous, genus_ids, family_ids) covers only the "
            "encoders whose species signal is already inside `continuous` "
            "(hash) or absent (TraitNet, adapter architectures). For an embed / "
            "sparse / rank_pool / transformer model, read the gate "
            "probabilities from forward_with_aux, which takes the species "
            "inputs those encoders need.");
    }

    continuous = fuse_categoricals_(std::move(continuous), torch::Tensor());
    auto encoded = encode_all(continuous, genus_ids, family_ids, {}, {});
    if (post_moe_) {
        return post_moe_->forward(encoded.latent).gate_probs;
    }
    return encoded.gate_probs;
}

void ResolveModelImpl::set_traits(torch::Tensor traits) {
    if (!trait_net_encoder_) {
        throw std::runtime_error(
            "set_traits() is only valid when encoder_architecture is TraitNet");
    }
    trait_net_encoder_->set_traits(std::move(traits));
}

} // namespace resolve
