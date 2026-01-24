#include "resolve/trainer.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_encoder.hpp"

#include <random>
#include <algorithm>
#include <numeric>

namespace resolve {

// ============================================================================
// DataLoader helper
// ============================================================================

/**
 * Simple data loader that yields batches of indices.
 */
class BatchSampler {
public:
    BatchSampler(int64_t dataset_size, int64_t batch_size, bool shuffle = true, uint64_t seed = 42)
        : dataset_size_(dataset_size), batch_size_(batch_size), shuffle_(shuffle), rng_(seed)
    {
        indices_.resize(dataset_size_);
        std::iota(indices_.begin(), indices_.end(), 0);
        reset();
    }

    void reset() {
        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
        current_idx_ = 0;
    }

    bool has_next() const {
        return current_idx_ < dataset_size_;
    }

    std::vector<int64_t> next_batch() {
        int64_t start = current_idx_;
        int64_t end = std::min(current_idx_ + batch_size_, dataset_size_);
        current_idx_ = end;

        std::vector<int64_t> batch(indices_.begin() + start, indices_.begin() + end);
        return batch;
    }

    int64_t n_batches() const {
        return (dataset_size_ + batch_size_ - 1) / batch_size_;
    }

private:
    int64_t dataset_size_;
    int64_t batch_size_;
    bool shuffle_;
    std::mt19937_64 rng_;
    std::vector<int64_t> indices_;
    int64_t current_idx_ = 0;
};


// ============================================================================
// PreparedData - holds all tensors for training
// ============================================================================

struct PreparedData {
    torch::Tensor continuous;       // (n, n_continuous)
    torch::Tensor genus_ids;        // (n, top_k)
    torch::Tensor family_ids;       // (n, top_k)
    torch::Tensor species_ids;      // (n, top_k_species) for embed mode
    torch::Tensor species_vector;   // (n, n_species) for sparse mode
    torch::Tensor hash_embedding;   // (n, hash_dim) for hash mode
    std::unordered_map<std::string, torch::Tensor> targets;
    std::unordered_map<std::string, torch::Tensor> target_masks;

    PreparedData to(torch::Device device) {
        PreparedData d;
        d.continuous = continuous.defined() ? continuous.to(device) : continuous;
        d.genus_ids = genus_ids.defined() ? genus_ids.to(device) : genus_ids;
        d.family_ids = family_ids.defined() ? family_ids.to(device) : family_ids;
        d.species_ids = species_ids.defined() ? species_ids.to(device) : species_ids;
        d.species_vector = species_vector.defined() ? species_vector.to(device) : species_vector;
        d.hash_embedding = hash_embedding.defined() ? hash_embedding.to(device) : hash_embedding;
        for (const auto& [name, tensor] : targets) {
            d.targets[name] = tensor.to(device);
        }
        for (const auto& [name, tensor] : target_masks) {
            d.target_masks[name] = tensor.to(device);
        }
        return d;
    }

    int64_t n_samples() const {
        return continuous.defined() ? continuous.size(0) : 0;
    }
};


// ============================================================================
// Trainer implementation helpers
// ============================================================================

// TODO: Migrate to PlotEncoder-based prepare_dataset
// The previous version used SpeciesEncoder which has been removed.
// This function needs to be reimplemented using PlotEncoder.

PreparedData prepare_dataset(
    const ResolveDataset& dataset,
    PlotEncoder& plot_encoder,
    StandardScaler& continuous_scaler,
    std::unordered_map<std::string, StandardScaler>& target_scalers,
    bool fit
) {
    PreparedData data;

    // Get plot IDs
    auto plot_ids = dataset.plot_ids();

    // TODO: Build PlotRecord and ObservationRecord from dataset
    // and use plot_encoder.fit_transform() or transform()

    // Process continuous features
    auto coords = dataset.get_coordinates();
    auto covariates = dataset.get_covariates();

    std::vector<torch::Tensor> continuous_parts;
    if (coords.defined() && coords.size(1) > 0) {
        continuous_parts.push_back(coords);
    }
    if (covariates.defined() && covariates.size(1) > 0) {
        continuous_parts.push_back(covariates);
    }

    if (!continuous_parts.empty()) {
        data.continuous = torch::cat(continuous_parts, 1);
    } else {
        data.continuous = torch::zeros({static_cast<int64_t>(plot_ids.size()), 0});
    }

    // Scale continuous features
    if (data.continuous.size(1) > 0) {
        if (fit) {
            data.continuous = continuous_scaler.fit_transform(data.continuous);
        } else {
            data.continuous = continuous_scaler.transform(data.continuous);
        }
    }

    // Process targets
    for (const auto& cfg : dataset.targets()) {
        auto target = dataset.get_target(cfg.name);
        auto mask = dataset.get_target_mask(cfg.name);

        if (cfg.task == TaskType::Regression) {
            if (fit) {
                auto& scaler = target_scalers[cfg.name];
                target = scaler.fit_transform(target.unsqueeze(1)).squeeze(1);
            } else {
                target = target_scalers[cfg.name].transform(target.unsqueeze(1)).squeeze(1);
            }
        }

        data.targets[cfg.name] = target;
        data.target_masks[cfg.name] = mask;
    }

    return data;
}


// ============================================================================
// Full Trainer training loop
// ============================================================================

float train_epoch_impl(
    ResolveModel& model,
    torch::optim::Optimizer& optimizer,
    MultiTaskLoss& loss_fn,
    OneCycleLR* scheduler,
    const PreparedData& data,
    int epoch,
    int batch_size,
    float max_grad_norm,
    SpeciesEncodingMode mode
) {
    model->train();

    BatchSampler sampler(data.n_samples(), batch_size, true);
    float total_loss = 0.0f;
    int n_batches = 0;

    while (sampler.has_next()) {
        auto indices = sampler.next_batch();
        auto idx_tensor = torch::tensor(indices, torch::kInt64);

        // Get batch data
        auto continuous = data.continuous.index_select(0, idx_tensor);
        auto genus_ids = data.genus_ids.defined() ? data.genus_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto family_ids = data.family_ids.defined() ? data.family_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto species_ids = data.species_ids.defined() ? data.species_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto species_vector = data.species_vector.defined() ? data.species_vector.index_select(0, idx_tensor) : torch::Tensor();

        // Get batch targets
        std::unordered_map<std::string, torch::Tensor> batch_targets;
        for (const auto& [name, tensor] : data.targets) {
            batch_targets[name] = tensor.index_select(0, idx_tensor);
        }

        // Forward pass
        optimizer.zero_grad();
        auto outputs = model->forward(continuous, genus_ids, family_ids, species_ids, species_vector);

        // Compute loss
        auto [loss, _] = loss_fn(outputs, batch_targets, epoch);

        // Backward pass
        loss.backward();

        // Gradient clipping
        if (max_grad_norm > 0) {
            torch::nn::utils::clip_grad_norm_(model->parameters(), max_grad_norm);
        }

        optimizer.step();

        if (scheduler) {
            scheduler->step();
        }

        total_loss += loss.item<float>();
        n_batches++;
    }

    return total_loss / n_batches;
}

std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
eval_epoch_impl(
    ResolveModel& model,
    MultiTaskLoss& loss_fn,
    const PreparedData& data,
    const std::vector<TargetConfig>& target_configs,
    const std::unordered_map<std::string, StandardScaler>& target_scalers,
    int epoch,
    int batch_size,
    SpeciesEncodingMode mode
) {
    model->eval();
    torch::NoGradGuard no_grad;

    // Collect all predictions
    std::unordered_map<std::string, std::vector<torch::Tensor>> all_preds;
    std::unordered_map<std::string, std::vector<torch::Tensor>> all_targets;

    BatchSampler sampler(data.n_samples(), batch_size, false);
    float total_loss = 0.0f;
    int n_batches = 0;

    while (sampler.has_next()) {
        auto indices = sampler.next_batch();
        auto idx_tensor = torch::tensor(indices, torch::kInt64);

        // Get batch data
        auto continuous = data.continuous.index_select(0, idx_tensor);
        auto genus_ids = data.genus_ids.defined() ? data.genus_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto family_ids = data.family_ids.defined() ? data.family_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto species_ids = data.species_ids.defined() ? data.species_ids.index_select(0, idx_tensor) : torch::Tensor();
        auto species_vector = data.species_vector.defined() ? data.species_vector.index_select(0, idx_tensor) : torch::Tensor();

        // Get batch targets
        std::unordered_map<std::string, torch::Tensor> batch_targets;
        for (const auto& [name, tensor] : data.targets) {
            batch_targets[name] = tensor.index_select(0, idx_tensor);
        }

        // Forward pass
        auto outputs = model->forward(continuous, genus_ids, family_ids, species_ids, species_vector);

        // Compute loss
        auto [loss, _] = loss_fn(outputs, batch_targets, epoch);
        total_loss += loss.item<float>();
        n_batches++;

        // Store predictions
        for (const auto& [name, pred] : outputs) {
            all_preds[name].push_back(pred);
            all_targets[name].push_back(batch_targets[name]);
        }
    }

    // Compute metrics
    std::unordered_map<std::string, std::unordered_map<std::string, float>> metrics;

    for (const auto& cfg : target_configs) {
        auto preds = torch::cat(all_preds[cfg.name], 0);
        auto targets = torch::cat(all_targets[cfg.name], 0);

        if (cfg.task == TaskType::Regression) {
            // Inverse transform to original scale
            preds = preds.squeeze(-1);

            auto scaler_it = target_scalers.find(cfg.name);
            if (scaler_it != target_scalers.end()) {
                preds = scaler_it->second.inverse_transform(preds);
                targets = scaler_it->second.inverse_transform(targets);
            }

            // Apply inverse transform if needed
            if (cfg.transform == TransformType::Log1p) {
                preds = torch::expm1(torch::clamp(preds, -87.0f, 87.0f));
                targets = torch::expm1(torch::clamp(targets, -87.0f, 87.0f));
            }

            metrics[cfg.name] = Metrics::compute_regression(preds, targets);
        } else {
            metrics[cfg.name] = Metrics::compute_classification(preds, targets);
        }
    }

    return {total_loss / n_batches, metrics};
}


// ============================================================================
// Trainer method implementations (to replace the inline ones in header)
// ============================================================================

// Note: The constructor and basic methods are already implemented inline in the header.
// Here we provide the full implementations that were marked as TODO.

// This file serves as additional implementation that links with the header.
// The actual TrainResult Trainer::fit() is already in the header but calls these helpers.

} // namespace resolve
