#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/trainer.hpp"
#include <torch/torch.h>

namespace resolve {

// Predictor for inference with trained models
class Predictor {
public:
    Predictor(
        SpaccModel model,
        SpeciesEncoder encoder,
        Scalers scalers,
        torch::Device device = torch::kCPU
    );

    // Load from saved checkpoint
    static Predictor load(const std::string& path, torch::Device device = torch::kCPU);

    // Predict on new data
    // Returns predictions for all targets
    SpaccPredictions predict(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
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
    SpaccModel& model() { return model_; }
    const SpeciesEncoder& encoder() const { return encoder_; }
    const Scalers& scalers() const { return scalers_; }

private:
    SpaccModel model_;
    SpeciesEncoder encoder_;
    Scalers scalers_;
    torch::Device device_;
};

} // namespace resolve
