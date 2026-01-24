#include "resolve/encoder.hpp"
#include <cmath>

namespace resolve {

// ============================================================================
// PlotEncoderImpl implementation (hash mode)
// ============================================================================

PlotEncoderImpl::PlotEncoderImpl(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) : has_taxonomy_(n_genera > 0 && n_families > 0),
    top_k_(top_k)
{
    // Calculate input dimension
    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        // Create separate embedding for each rank position
        for (int k = 0; k < top_k_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(n_genera, genus_emb_dim)
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(n_families, family_emb_dim)
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_ * (genus_emb_dim + family_emb_dim);
    }

    // Build MLP
    torch::nn::Sequential mlp;
    int64_t prev_dim = input_dim;

    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
        mlp->push_back(torch::nn::BatchNorm1d(hidden_dims[i]));
        mlp->push_back(torch::nn::GELU());
        mlp->push_back(torch::nn::Dropout(dropout));
        prev_dim = hidden_dims[i];
    }

    mlp_ = register_module("mlp", mlp);
    latent_dim_ = hidden_dims.empty() ? input_dim : hidden_dims.back();
}

torch::Tensor PlotEncoderImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        // Get embeddings for each position
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    return mlp_->forward(x);
}


// ============================================================================
// PlotEncoderEmbedImpl implementation (embed mode)
// ============================================================================

PlotEncoderEmbedImpl::PlotEncoderEmbedImpl(
    int64_t n_continuous,
    int64_t n_species,
    int64_t n_genera,
    int64_t n_families,
    int species_embed_dim,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k_species,
    int top_k_taxonomy,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) : has_taxonomy_(n_genera > 0 && n_families > 0),
    top_k_species_(top_k_species),
    top_k_taxonomy_(top_k_taxonomy)
{
    // Calculate input dimension
    int64_t input_dim = n_continuous;

    // Species embeddings (one per position)
    for (int k = 0; k < top_k_species_; ++k) {
        auto species_emb = register_module(
            "species_emb_" + std::to_string(k),
            torch::nn::Embedding(
                torch::nn::EmbeddingOptions(n_species, species_embed_dim).padding_idx(0)
            )
        );
        species_embeddings_.push_back(species_emb);
    }
    input_dim += top_k_species_ * species_embed_dim;

    // Taxonomy embeddings
    if (has_taxonomy_) {
        for (int k = 0; k < top_k_taxonomy_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(n_genera, genus_emb_dim).padding_idx(0)
                )
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(
                    torch::nn::EmbeddingOptions(n_families, family_emb_dim).padding_idx(0)
                )
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_taxonomy_ * (genus_emb_dim + family_emb_dim);
    }

    // Build MLP
    torch::nn::Sequential mlp;
    int64_t prev_dim = input_dim;

    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
        mlp->push_back(torch::nn::BatchNorm1d(hidden_dims[i]));
        mlp->push_back(torch::nn::GELU());
        mlp->push_back(torch::nn::Dropout(dropout));
        prev_dim = hidden_dims[i];
    }

    mlp_ = register_module("mlp", mlp);
    latent_dim_ = hidden_dims.empty() ? input_dim : hidden_dims.back();
}

torch::Tensor PlotEncoderEmbedImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    // Embed species (one table per position)
    for (int k = 0; k < top_k_species_; ++k) {
        auto sp_emb = species_embeddings_[k](species_ids.select(1, k));
        parts.push_back(sp_emb);
    }

    // Embed taxonomy if available
    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        for (int k = 0; k < top_k_taxonomy_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_taxonomy_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    return mlp_->forward(x);
}


// ============================================================================
// PlotEncoderSparseImpl implementation (explicit species vector mode)
// ============================================================================

PlotEncoderSparseImpl::PlotEncoderSparseImpl(
    int64_t n_continuous,
    int64_t n_species,
    int species_embed_dim,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) : has_taxonomy_(n_genera > 0 && n_families > 0),
    n_species_(n_species),
    top_k_(top_k)
{
    // Calculate input dimension
    int64_t input_dim = n_continuous + species_embed_dim;

    // Linear projection from species abundances to embedding space
    species_projection_ = register_module(
        "species_projection",
        torch::nn::Linear(n_species, species_embed_dim)
    );

    // Taxonomy embeddings
    if (has_taxonomy_) {
        for (int k = 0; k < top_k_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(n_genera, genus_emb_dim)
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(n_families, family_emb_dim)
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_ * (genus_emb_dim + family_emb_dim);
    }

    // Build MLP
    torch::nn::Sequential mlp;
    int64_t prev_dim = input_dim;

    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
        mlp->push_back(torch::nn::BatchNorm1d(hidden_dims[i]));
        mlp->push_back(torch::nn::GELU());
        mlp->push_back(torch::nn::Dropout(dropout));
        prev_dim = hidden_dims[i];
    }

    mlp_ = register_module("mlp", mlp);
    latent_dim_ = hidden_dims.empty() ? input_dim : hidden_dims.back();
}

torch::Tensor PlotEncoderSparseImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    // Project species vector to embedding space
    auto species_emb = species_projection_->forward(species_vector);

    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    parts.push_back(species_emb);

    // Embed taxonomy if available
    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    return mlp_->forward(x);
}


// ============================================================================
// TaskHeadImpl implementation
// ============================================================================

TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform
) : task_(task), transform_(transform)
{
    int64_t out_features = (task == TaskType::Classification) ? num_classes : 1;
    head_ = register_module("head", torch::nn::Linear(latent_dim, out_features));
}

torch::Tensor TaskHeadImpl::forward(torch::Tensor latent) {
    return head_->forward(latent);
}

torch::Tensor TaskHeadImpl::predict(torch::Tensor latent) {
    auto output = forward(latent);

    if (task_ == TaskType::Classification) {
        return torch::argmax(output, /*dim=*/1);
    } else {
        output = output.squeeze(-1);
        return inverse_transform(output);
    }
}

torch::Tensor TaskHeadImpl::inverse_transform(torch::Tensor predictions) {
    if (transform_ == TransformType::Log1p) {
        return torch::expm1(torch::clamp(predictions, /*min=*/-88.0f, /*max=*/88.0f));
    }
    return predictions;
}

} // namespace resolve
