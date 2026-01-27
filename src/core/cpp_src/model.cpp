#include "resolve/model.hpp"
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

    // Create appropriate encoder based on mode
    if (config.species_encoding == SpeciesEncodingMode::Hash && !config.uses_explicit_vector) {
        // Hash mode: continuous includes hash_dim
        int64_t n_continuous = n_continuous_base + config.hash_dim;

        encoder_hash_ = register_module("encoder", PlotEncoder(
            n_continuous,
            schema.has_taxonomy ? schema.n_genera + 1 : 0,
            schema.has_taxonomy ? schema.n_families + 1 : 0,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.n_taxonomy_slots,
            config.hidden_dims,
            config.dropout
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
            schema.has_taxonomy ? (schema.n_genera_vocab > 0 ? schema.n_genera_vocab : schema.n_genera + 1) : 0,
            schema.has_taxonomy ? (schema.n_families_vocab > 0 ? schema.n_families_vocab : schema.n_families + 1) : 0,
            config.species_embed_dim,
            config.genus_emb_dim,
            config.family_emb_dim,
            config.top_k_species,
            config.n_taxonomy_slots,
            config.hidden_dims,
            config.dropout
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
            config.dropout
        ));
    }

    // Create task heads
    for (const auto& target : schema.targets) {
        auto head = register_module(
            "head_" + target.name,
            TaskHead(
                latent_dim(),
                target.task,
                target.num_classes,
                target.transform
            )
        );
        heads_.emplace(target.name, head);
    }
}

int64_t ResolveModelImpl::latent_dim() const {
    if (encoder_hash_) {
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
    torch::Tensor species_vector
) {
    if (encoder_hash_) {
        return encoder_hash_->forward(continuous, genus_ids, family_ids);
    } else if (encoder_embed_) {
        return encoder_embed_->forward(continuous, species_ids, genus_ids, family_ids);
    } else {
        return encoder_sparse_->forward(continuous, species_vector, genus_ids, family_ids);
    }
}

std::unordered_map<std::string, torch::Tensor> ResolveModelImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    auto latent = encode(continuous, genus_ids, family_ids, species_ids, species_vector);

    std::unordered_map<std::string, torch::Tensor> outputs;
    for (auto& [name, head] : heads_) {
        outputs[name] = head->forward(latent);
    }
    return outputs;
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
    torch::Tensor species_vector
) {
    return encode(continuous, genus_ids, family_ids, species_ids, species_vector);
}

TaskHead& ResolveModelImpl::head(const std::string& name) {
    auto it = heads_.find(name);
    if (it == heads_.end()) {
        throw std::runtime_error("Head not found: " + name);
    }
    return it->second;
}

} // namespace resolve
