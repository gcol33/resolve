#pragma once

#include "types.hpp"
#include <torch/torch.h>
#include <vector>

namespace resolve {

/**
 * PlotEncoder: shared encoder for all tasks (hash mode)
 *
 * Architecture: learned taxonomy embeddings + MLP
 * Accepts continuous features (coords + covariates + hash embedding)
 */
class PlotEncoderImpl : public torch::nn::Module {
public:
    PlotEncoderImpl(
        int64_t n_continuous,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    /**
     * Forward pass.
     *
     * @param continuous (batch, n_continuous) - coords + covariates + hash embedding
     * @param genus_ids (batch, top_k) - optional
     * @param family_ids (batch, top_k) - optional
     */
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    int64_t latent_dim() const { return latent_dim_; }
    bool has_taxonomy() const { return has_taxonomy_; }

private:
    bool has_taxonomy_;
    int top_k_;
    int64_t latent_dim_;

    // Taxonomy embeddings (one per rank position)
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // MLP layers
    torch::nn::Sequential mlp_{nullptr};
};

TORCH_MODULE(PlotEncoder);


/**
 * PlotEncoderEmbed: learnable embeddings for top-k species
 *
 * Used when species_encoding="embed"
 */
class PlotEncoderEmbedImpl : public torch::nn::Module {
public:
    PlotEncoderEmbedImpl(
        int64_t n_continuous,
        int64_t n_species,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int species_embed_dim = 32,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k_species = 10,
        int top_k_taxonomy = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    /**
     * Forward pass.
     *
     * @param continuous (batch, n_continuous) - coords + covariates (NO hash embedding)
     * @param species_ids (batch, top_k_species) - integer IDs
     * @param genus_ids (batch, top_k_taxonomy) - optional
     * @param family_ids (batch, top_k_taxonomy) - optional
     */
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_ids,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    int64_t latent_dim() const { return latent_dim_; }
    bool has_taxonomy() const { return has_taxonomy_; }

private:
    bool has_taxonomy_;
    int top_k_species_;
    int top_k_taxonomy_;
    int64_t latent_dim_;

    // Species embeddings (one per position)
    std::vector<torch::nn::Embedding> species_embeddings_;

    // Taxonomy embeddings (one per rank position)
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // MLP layers
    torch::nn::Sequential mlp_{nullptr};
};

TORCH_MODULE(PlotEncoderEmbed);


/**
 * PlotEncoderSparse: explicit species abundance vectors
 *
 * Used for selection="all" or selection="presence_absence"
 */
class PlotEncoderSparseImpl : public torch::nn::Module {
public:
    PlotEncoderSparseImpl(
        int64_t n_continuous,
        int64_t n_species,
        int species_embed_dim = 64,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    /**
     * Forward pass.
     *
     * @param continuous (batch, n_continuous) - coords + covariates
     * @param species_vector (batch, n_species) - abundance or presence/absence vector
     * @param genus_ids (batch, top_k) - optional
     * @param family_ids (batch, top_k) - optional
     */
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_vector,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    int64_t latent_dim() const { return latent_dim_; }
    bool has_taxonomy() const { return has_taxonomy_; }
    int64_t n_species() const { return n_species_; }

private:
    bool has_taxonomy_;
    int64_t n_species_;
    int top_k_;
    int64_t latent_dim_;

    // Linear projection from species abundances to embedding
    torch::nn::Linear species_projection_{nullptr};

    // Taxonomy embeddings (one per rank position)
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // MLP layers
    torch::nn::Sequential mlp_{nullptr};
};

TORCH_MODULE(PlotEncoderSparse);


/**
 * Task head: prediction head for a single target
 */
class TaskHeadImpl : public torch::nn::Module {
public:
    TaskHeadImpl(
        int64_t latent_dim,
        TaskType task,
        int num_classes = 0,
        TransformType transform = TransformType::None
    );

    // Forward pass - returns raw output
    torch::Tensor forward(torch::Tensor latent);

    // Predict with inverse transform
    torch::Tensor predict(torch::Tensor latent);

    // Inverse transform for predictions
    torch::Tensor inverse_transform(torch::Tensor predictions);

    TaskType task() const { return task_; }
    TransformType transform() const { return transform_; }

private:
    TaskType task_;
    TransformType transform_;
    torch::nn::Linear head_{nullptr};
};

TORCH_MODULE(TaskHead);

} // namespace resolve
