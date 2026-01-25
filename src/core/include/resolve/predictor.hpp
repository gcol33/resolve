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

    // Get learned genus embeddings
    torch::Tensor get_genus_embeddings() const;

    // Get learned family embeddings
    torch::Tensor get_family_embeddings() const;

    // Accessors
    ResolveModel& model() { return model_; }
    const Scalers& scalers() const { return scalers_; }

private:
    ResolveModel model_;
    Scalers scalers_;
    torch::Device device_;
};

} // namespace resolve
