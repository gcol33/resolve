#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include <torch/torch.h>

namespace resolve {

// Forward declaration
class ResolveDataset;

// Predictor for inference with trained models
class Predictor {
public:
    Predictor(
        ResolveModel model,
        Scalers scalers,
        torch::Device device = torch::kCPU
    );

    // Load from saved checkpoint
    static Predictor load(const std::string& path, torch::Device device = torch::kCPU);

    // Predict on a ResolveDataset (preferred API)
    ResolvePredictions predict(
        const ResolveDataset& dataset,
        bool return_latent = false
    );

    // Predict on new data (raw tensor API)
    // Returns predictions for all targets
    ResolvePredictions predict(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor unknown_fraction,
        torch::Tensor unknown_count,
        // Pool-style fields (rank_pool / transformer)
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        bool return_latent = false
    );

    // Get latent embeddings
    torch::Tensor get_embeddings(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor genus_ids,
        torch::Tensor family_ids
    );

    // Get learned embedding weights (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_embeddings() const;
    [[nodiscard]] torch::Tensor get_family_embeddings() const;
    [[nodiscard]] torch::Tensor get_species_embeddings() const;

    // Optimize model for inference (fuses BatchNorm into Linear layers)
    void optimize_for_inference();

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return model_; }
    [[nodiscard]] const ResolveModel& model() const noexcept { return model_; }
    [[nodiscard]] const Scalers& scalers() const noexcept { return scalers_; }
    [[nodiscard]] torch::Device device() const noexcept { return device_; }

private:
    ResolveModel model_;
    Scalers scalers_;
    torch::Device device_;
};

} // namespace resolve
