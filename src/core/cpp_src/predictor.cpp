#include "resolve/predictor.hpp"
#include "resolve/dataset.hpp"
#include <fstream>

namespace resolve {

Predictor::Predictor(
    ResolveModel model,
    Scalers scalers,
    torch::Device device
) : model_(model), scalers_(scalers), device_(device)
{
    model_->to(device_);
    model_->eval();
}

Predictor Predictor::load(const std::string& path, torch::Device device) {
    auto [model, scalers] = Trainer::load(path, device);
    return Predictor(model, scalers, device);
}

ResolvePredictions Predictor::predict(
    const ResolveDataset& dataset,
    bool return_latent
) {
    auto result = predict(
        dataset.coordinates(),
        dataset.covariates(),
        dataset.hash_embedding(),
        dataset.species_ids(),
        dataset.species_vector(),
        dataset.genus_ids(),
        dataset.family_ids(),
        return_latent
    );

    // Use actual plot IDs from dataset
    result.plot_ids = dataset.plot_ids();
    return result;
}

ResolvePredictions Predictor::predict(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor hash_embedding,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    bool return_latent
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Build continuous features based on encoding mode
    std::vector<torch::Tensor> continuous_parts;

    if (coordinates.defined() && coordinates.numel() > 0) {
        continuous_parts.push_back(coordinates);
    }
    if (covariates.defined() && covariates.numel() > 0) {
        continuous_parts.push_back(covariates);
    }

    // For hash mode, include hash embedding in continuous
    if (model_->species_encoding() == SpeciesEncodingMode::Hash &&
        !model_->uses_explicit_vector() &&
        hash_embedding.defined() && hash_embedding.numel() > 0) {
        continuous_parts.push_back(hash_embedding);
    }

    torch::Tensor continuous;
    if (!continuous_parts.empty()) {
        continuous = torch::cat(continuous_parts, /*dim=*/1);
    } else {
        int64_t n_samples = 0;
        if (hash_embedding.defined()) n_samples = hash_embedding.size(0);
        else if (species_ids.defined()) n_samples = species_ids.size(0);
        else if (species_vector.defined()) n_samples = species_vector.size(0);
        continuous = torch::zeros({n_samples, 0}, torch::kFloat32);
    }

    // Scale continuous features
    torch::Tensor scaled_continuous;
    if (scalers_.continuous_mean.defined() && continuous.size(1) > 0) {
        scaled_continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    } else {
        scaled_continuous = continuous;
    }
    scaled_continuous = scaled_continuous.to(device_);

    // Move tensors to device
    if (genus_ids.defined()) {
        genus_ids = genus_ids.to(device_);
    }
    if (family_ids.defined()) {
        family_ids = family_ids.to(device_);
    }
    if (species_ids.defined()) {
        species_ids = species_ids.to(device_);
    }
    if (species_vector.defined()) {
        species_vector = species_vector.to(device_);
    }

    // Get predictions using appropriate encoding mode
    auto outputs = model_->forward(scaled_continuous, genus_ids, family_ids, species_ids, species_vector);

    ResolvePredictions result;

    // Process each output
    for (const auto& cfg : model_->schema().targets) {
        auto out_it = outputs.find(cfg.name);
        if (out_it == outputs.end()) continue;

        auto pred = out_it->second;

        if (cfg.task == TaskType::Classification) {
            // Return class predictions
            result.predictions[cfg.name] = torch::argmax(pred, /*dim=*/1);
        } else {
            // Unscale and inverse transform
            pred = pred.squeeze(-1);

            auto scaler_it = scalers_.target_scalers.find(cfg.name);
            if (scaler_it != scalers_.target_scalers.end()) {
                pred = pred * scaler_it->second.second.to(device_) + scaler_it->second.first.to(device_);
            }

            if (cfg.transform == TransformType::Log1p) {
                pred = torch::expm1(torch::clamp(pred, /*min=*/-88.0f, /*max=*/88.0f));
            }

            result.predictions[cfg.name] = pred;
        }
    }

    // Optionally return latent
    if (return_latent) {
        result.latent = model_->get_latent(scaled_continuous, genus_ids, family_ids, species_ids, species_vector);
    }

    // Create plot indices as strings
    int64_t n_samples = scaled_continuous.size(0);
    for (int64_t i = 0; i < n_samples; ++i) {
        result.plot_ids.push_back(std::to_string(i));
    }

    return result;
}

torch::Tensor Predictor::get_embeddings(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor hash_embedding,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Concatenate continuous features
    std::vector<torch::Tensor> continuous_parts = {coordinates, hash_embedding};
    if (covariates.defined() && covariates.size(1) > 0) {
        continuous_parts.push_back(covariates);
    }
    auto continuous = torch::cat(continuous_parts, /*dim=*/1);

    // Scale continuous features
    auto scaled_continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    scaled_continuous = scaled_continuous.to(device_);

    if (genus_ids.defined()) {
        genus_ids = genus_ids.to(device_);
    }
    if (family_ids.defined()) {
        family_ids = family_ids.to(device_);
    }

    return model_->get_latent(scaled_continuous, genus_ids, family_ids);
}

torch::Tensor Predictor::get_genus_embeddings() const {
    // Placeholder - requires exposing embedding weights from PlotEncoder
    // TODO: Add accessor method to PlotEncoder for embedding weights
    return torch::Tensor();
}

torch::Tensor Predictor::get_family_embeddings() const {
    // Placeholder - requires exposing embedding weights from PlotEncoder
    return torch::Tensor();
}

} // namespace resolve
