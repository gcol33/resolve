#include "resolve/predictor.hpp"
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
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor hash_embedding,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    bool return_latent
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

    // Get predictions
    auto outputs = model_->forward(scaled_continuous, genus_ids, family_ids);

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
        result.latent = model_->get_latent(scaled_continuous, genus_ids, family_ids);
    }

    // Create plot indices as strings
    for (int64_t i = 0; i < coordinates.size(0); ++i) {
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
