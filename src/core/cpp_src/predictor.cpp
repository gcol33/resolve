#include "resolve/predictor.hpp"
#include "resolve/dataset.hpp"
#include "resolve/utils.hpp"
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

Predictor::Predictor(
    ResolveModel model,
    Scalers scalers,
    CategoricalVocab categorical_vocab,
    torch::Device device
) : model_(model),
    scalers_(scalers),
    categorical_vocab_(std::move(categorical_vocab)),
    device_(device)
{
    model_->to(device_);
    model_->eval();
}

Predictor Predictor::load(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    auto [model, scalers, vocab] = Trainer::load(path, device, vram_fraction);
    return Predictor(std::move(model), std::move(scalers), std::move(vocab), device);
}

ResolvePredictions Predictor::predict(
    const ResolveDataset& dataset,
    bool return_latent
) {
    // Thread the pool-* tensors through. Previously these were hardcoded to
    // empty, which crashed PlotEncoderRankPool / PlotEncoderTransformer
    // forward at the species_embedding lookup because species_ids was the
    // (n_plots, top_k_species) embed-mode tensor (or undefined) instead of
    // the rank-pool (n_plots, max_species) tensor.
    auto result = predict(
        dataset.coordinates(),
        dataset.covariates(),
        dataset.hash_embedding(),
        dataset.species_ids(),
        dataset.species_vector(),
        dataset.genus_ids(),
        dataset.family_ids(),
        dataset.unknown_fraction(),
        dataset.unknown_count(),
        dataset.pool_genus_ids(),
        dataset.pool_family_ids(),
        dataset.pool_weights(),
        dataset.pool_mask(),
        dataset.pool_has_cover(),
        dataset.categorical_ids(),
        return_latent
    );

    // Use actual plot IDs from dataset
    result.plot_ids = dataset.plot_ids();

    // Copy targets from dataset for residual analysis
    result.targets = dataset.targets();

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
    torch::Tensor unknown_fraction,
    torch::Tensor unknown_count,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids,
    bool return_latent
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Build continuous features based on encoding mode (must match trainer.cpp)
    std::vector<torch::Tensor> continuous_parts;
    push_if_defined(continuous_parts, coordinates);
    push_if_defined(continuous_parts, covariates);
    push_if_defined(continuous_parts, unknown_fraction, 1);
    if (unknown_count.defined() && unknown_count.numel() > 0) {
        continuous_parts.push_back(unknown_count.to(torch::kFloat32).unsqueeze(1));
    }

    // For hash mode, include hash embedding in continuous
    if (model_->species_encoding() == SpeciesEncodingMode::Hash &&
        !model_->uses_explicit_vector()) {
        push_if_defined(continuous_parts, hash_embedding);
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
    genus_ids = to_device_if_defined(genus_ids, device_);
    family_ids = to_device_if_defined(family_ids, device_);
    species_ids = to_device_if_defined(species_ids, device_);
    species_vector = to_device_if_defined(species_vector, device_);
    pool_genus_ids = to_device_if_defined(pool_genus_ids, device_);
    pool_family_ids = to_device_if_defined(pool_family_ids, device_);
    pool_weights = to_device_if_defined(pool_weights, device_);
    pool_mask = to_device_if_defined(pool_mask, device_);
    pool_has_cover = to_device_if_defined(pool_has_cover, device_);
    categorical_ids = to_device_if_defined(categorical_ids, device_);

    // Get predictions using appropriate encoding mode
    auto outputs = model_->forward(scaled_continuous, genus_ids, family_ids, species_ids, species_vector,
                                    pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                    categorical_ids);

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
                pred = torch::expm1(torch::clamp(pred, kExpClampMin, kExpClampMax));
            }

            result.predictions[cfg.name] = pred;
        }
    }

    // Optionally return latent
    if (return_latent) {
        result.latent = model_->get_latent(scaled_continuous, genus_ids, family_ids, species_ids, species_vector,
                                           pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                           categorical_ids);
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

    // Concatenate continuous features (hash_embedding may be empty for non-hash modes)
    std::vector<torch::Tensor> continuous_parts;
    if (coordinates.defined() && coordinates.numel() > 0) {
        continuous_parts.push_back(coordinates);
    }
    if (hash_embedding.defined() && hash_embedding.numel() > 0) {
        continuous_parts.push_back(hash_embedding);
    }
    if (covariates.defined() && covariates.size(1) > 0) {
        continuous_parts.push_back(covariates);
    }
    if (continuous_parts.empty()) {
        throw std::runtime_error("get_embeddings requires at least one non-empty input tensor");
    }
    auto continuous = torch::cat(continuous_parts, /*dim=*/1);

    // Scale continuous features
    auto scaled_continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    scaled_continuous = scaled_continuous.to(device_);

    genus_ids = to_device_if_defined(genus_ids, device_);
    family_ids = to_device_if_defined(family_ids, device_);

    return model_->get_latent(scaled_continuous, genus_ids, family_ids);
}

void Predictor::optimize_for_inference() {
    model_->eval();
    torch::NoGradGuard no_grad;

    // Fuse Linear+BatchNorm1d pairs in all Sequential modules
    for (auto& module : model_->modules(/*include_self=*/false)) {
        auto seq = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(module);
        if (!seq) continue;

        for (size_t i = 0; i + 1 < seq->size(); ++i) {
            auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>((*seq)[i]);
            auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>((*seq)[i + 1]);
            if (!linear || !bn) continue;

            // Fuse: W_new = bn.weight / sqrt(var + eps) * W_linear
            //        b_new = bn.weight / sqrt(var + eps) * b_linear + bn.bias - bn.weight * mean / sqrt(var + eps)
            auto std_val = torch::sqrt(bn->running_var + bn->options.eps());
            auto scale = bn->weight / std_val;

            linear->weight.mul_(scale.unsqueeze(1));
            if (linear->bias.defined()) {
                linear->bias.mul_(scale).add_(bn->bias - scale * bn->running_mean);
            }

            // Replace BN with Identity (identity module in Sequential)
            seq->replace_module(std::to_string(i + 1), torch::nn::Identity());
        }
    }
}

torch::Tensor Predictor::get_genus_embeddings() const {
    return model_->get_genus_weights();
}

torch::Tensor Predictor::get_family_embeddings() const {
    return model_->get_family_weights();
}

torch::Tensor Predictor::get_species_embeddings() const {
    return model_->get_species_weights();
}

} // namespace resolve
