#pragma once

#include "types.hpp"
#include "dataset.hpp"
#include "loss.hpp"
#include "vocab.hpp"

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <functional>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace resolve {

namespace fs = std::filesystem;

// Forward declaration
class ResolveModel;

/**
 * Loss configuration presets.
 */
inline std::unordered_map<std::string, std::unordered_map<int, PhaseConfig>> LOSS_PRESETS = {
    {"mae", {{1, PhaseConfig::mae_only()}}},
    {"combined", {{1, PhaseConfig::combined(0.80f, 0.15f, 0.05f)}}},
    {"smape", {{1, PhaseConfig{0.5f, 0.0f, 0.0f, 0.5f, 0.0f}}}},
};

/**
 * Standard scaler for feature normalization.
 */
class StandardScaler {
public:
    StandardScaler() = default;

    /**
     * Fit scaler to data and transform.
     */
    torch::Tensor fit_transform(const torch::Tensor& data) {
        // Compute mean and std along dimension 0
        mean_ = data.mean(0);
        std_ = data.std(0);
        // Avoid division by zero
        std_ = torch::where(std_ < 1e-8f, torch::ones_like(std_), std_);
        fitted_ = true;
        return (data - mean_) / std_;
    }

    /**
     * Transform data using fitted parameters.
     */
    torch::Tensor transform(const torch::Tensor& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }
        return (data - mean_) / std_;
    }

    /**
     * Inverse transform data back to original scale.
     */
    torch::Tensor inverse_transform(const torch::Tensor& data) const {
        if (!fitted_) {
            throw std::runtime_error("Scaler not fitted");
        }
        return data * std_ + mean_;
    }

    torch::Tensor mean() const { return mean_; }
    torch::Tensor std() const { return std_; }
    bool fitted() const { return fitted_; }
    int64_t n_features() const { return mean_.defined() ? mean_.size(0) : 0; }

    // Serialization
    void save(const std::string& path) const {
        torch::save({mean_, std_}, path);
    }

    void load(const std::string& path) {
        std::vector<torch::Tensor> tensors;
        torch::load(tensors, path);
        mean_ = tensors[0];
        std_ = tensors[1];
        fitted_ = true;
    }

private:
    torch::Tensor mean_;
    torch::Tensor std_;
    bool fitted_ = false;
};


/**
 * OneCycleLR-style learning rate scheduler.
 *
 * Implements the 1cycle policy: warmup -> annealing.
 */
class OneCycleLR {
public:
    OneCycleLR(
        torch::optim::Optimizer& optimizer,
        float max_lr,
        int64_t total_steps,
        float pct_start = 0.3f,
        float div_factor = 25.0f,
        float final_div_factor = 1e4f
    ) : optimizer_(optimizer),
        max_lr_(max_lr),
        total_steps_(total_steps),
        pct_start_(pct_start),
        initial_lr_(max_lr / div_factor),
        final_lr_(max_lr / final_div_factor),
        step_count_(0)
    {
        warmup_steps_ = static_cast<int64_t>(total_steps * pct_start);
    }

    /**
     * Step the scheduler.
     */
    void step() {
        step_count_++;
        float lr = compute_lr();
        for (auto& group : optimizer_.param_groups()) {
            group.options().set_lr(lr);
        }
    }

    float get_last_lr() const {
        return compute_lr();
    }

    int64_t step_count() const { return step_count_; }

private:
    torch::optim::Optimizer& optimizer_;
    float max_lr_;
    int64_t total_steps_;
    float pct_start_;
    float initial_lr_;
    float final_lr_;
    int64_t warmup_steps_;
    int64_t step_count_;

    float compute_lr() const {
        if (step_count_ <= warmup_steps_) {
            // Warmup phase: linear increase
            float progress = static_cast<float>(step_count_) / warmup_steps_;
            return initial_lr_ + progress * (max_lr_ - initial_lr_);
        } else {
            // Annealing phase: cosine decay
            int64_t anneal_steps = total_steps_ - warmup_steps_;
            int64_t current_anneal = step_count_ - warmup_steps_;
            float progress = static_cast<float>(current_anneal) / anneal_steps;
            float cosine_decay = 0.5f * (1.0f + std::cos(M_PI * progress));
            return final_lr_ + cosine_decay * (max_lr_ - final_lr_);
        }
    }
};


/**
 * Checkpoint data for training resume.
 */
struct Checkpoint {
    int epoch = 0;
    int best_epoch = 0;
    float best_metric = -std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    std::vector<float> train_loss_history;
    std::vector<float> test_loss_history;

    // Model state
    std::unordered_map<std::string, torch::Tensor> model_state;
    std::unordered_map<std::string, torch::Tensor> best_state;

    // Optimizer state (serialized)
    std::vector<char> optimizer_state_buffer;

    // Scalers
    StandardScaler continuous_scaler;
    std::unordered_map<std::string, StandardScaler> target_scalers;

    // Configuration (for validation)
    int hash_dim = 0;
    std::vector<int64_t> hidden_dims;
    int max_epochs = 0;

    void save(const std::string& path) const;
    static std::optional<Checkpoint> load(const std::string& path);
};


/**
 * Progress callback type.
 *
 * Called after each epoch with (epoch, train_loss, test_loss, metrics).
 */
using ProgressCallback = std::function<void(
    int epoch,
    float train_loss,
    float test_loss,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
)>;


/**
 * Full-featured trainer for ResolveModel.
 *
 * Features:
 *   - Automatic data splitting and preprocessing
 *   - Species encoding (hash or embedding)
 *   - Standard scaling of features and targets
 *   - OneCycleLR learning rate scheduling
 *   - Gradient clipping
 *   - Early stopping with patience
 *   - Checkpointing and resume
 *   - Progress callbacks
 *   - Loss presets ("mae", "combined", "smape")
 */
class Trainer {
public:
    /**
     * Construct trainer with configuration.
     */
    Trainer(
        // Species encoding
        SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash,
        int hash_dim = 32,
        int top_k = 5,
        SelectionMode selection = SelectionMode::Top,
        NormalizationMode normalization = NormalizationMode::Norm,

        // Model architecture
        std::vector<int64_t> hidden_dims = {2048, 1024, 512, 256, 128, 64},
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        float dropout = 0.3f,

        // Training
        int batch_size = 4096,
        int max_epochs = 500,
        int patience = 50,
        float lr = 1e-3f,
        float weight_decay = 1e-4f,
        float max_grad_norm = 1.0f,

        // Loss configuration
        const std::string& loss_preset = "mae",  // "mae", "combined", "smape"
        std::optional<std::unordered_map<int, PhaseConfig>> custom_phases = std::nullopt,
        std::optional<std::vector<int>> phase_boundaries = std::nullopt,

        // Checkpointing
        const std::string& checkpoint_dir = "",
        int checkpoint_every = 50,
        bool resume = true,

        // Device
        torch::Device device = torch::kCPU
    );

    /**
     * Train the model on a dataset.
     *
     * @param dataset The dataset to train on (will be split 80/20)
     * @param progress_callback Optional callback for progress updates
     * @return Training results
     */
    TrainResult fit(
        const ResolveDataset& dataset,
        std::optional<ProgressCallback> progress_callback = std::nullopt
    );

    /**
     * Predict on a dataset.
     *
     * @param dataset Dataset to predict on
     * @param confidence_threshold Minimum confidence (0-1), predictions below are NaN
     * @return Map of target name to predictions tensor
     */
    std::unordered_map<std::string, torch::Tensor> predict(
        const ResolveDataset& dataset,
        float confidence_threshold = 0.0f
    );

    /**
     * Save trained model to file.
     */
    void save(const std::string& path) const;

    /**
     * Load model from file.
     */
    void load(const std::string& path);

    // Accessors
    torch::Device device() const { return device_; }
    int64_t n_params() const;
    bool is_fitted() const { return model_ != nullptr; }

private:
    // Configuration
    SpeciesEncodingMode species_encoding_;
    int hash_dim_;
    int top_k_;
    SelectionMode selection_;
    NormalizationMode normalization_;
    std::vector<int64_t> hidden_dims_;
    int genus_emb_dim_;
    int family_emb_dim_;
    float dropout_;
    int batch_size_;
    int max_epochs_;
    int patience_;
    float lr_;
    float weight_decay_;
    float max_grad_norm_;
    std::string checkpoint_dir_;
    int checkpoint_every_;
    bool resume_;
    torch::Device device_;

    // Loss configuration
    std::unordered_map<int, PhaseConfig> phases_;
    std::vector<int> phase_boundaries_;

    // Model and training state
    std::shared_ptr<ResolveModel> model_;
    std::unique_ptr<torch::optim::AdamW> optimizer_;
    std::unique_ptr<OneCycleLR> scheduler_;
    std::unique_ptr<MultiTaskLoss> loss_fn_;

    // Data preprocessing
    StandardScaler continuous_scaler_;
    std::unordered_map<std::string, StandardScaler> target_scalers_;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> target_scaler_params_;

    // Vocabularies
    std::unique_ptr<TaxonomyVocab> taxonomy_vocab_;
    std::unique_ptr<SpeciesVocab> species_vocab_;

    // Training state
    std::unordered_map<std::string, torch::Tensor> best_state_;
    ResolveSchema schema_;

    // Internal methods
    std::pair<torch::Tensor, torch::Tensor> prepare_data(
        const ResolveDataset& dataset,
        bool fit_scalers
    );

    float train_epoch(
        int epoch,
        const torch::Tensor& train_data,
        const std::vector<std::string>& target_names
    );

    std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
    eval_epoch(
        int epoch,
        const torch::Tensor& test_data,
        const std::vector<std::string>& target_names
    );

    void save_checkpoint(
        int epoch,
        int best_epoch,
        float best_metric,
        int epochs_without_improvement,
        const std::vector<float>& train_history,
        const std::vector<float>& test_history
    );

    std::optional<Checkpoint> load_checkpoint();

    fs::path checkpoint_path() const {
        if (checkpoint_dir_.empty()) return {};
        return fs::path(checkpoint_dir_) / "checkpoint.pt";
    }

    fs::path progress_path() const {
        if (checkpoint_dir_.empty()) return {};
        return fs::path(checkpoint_dir_) / "progress.json";
    }
};


// ============================================================================
// Implementation
// ============================================================================

inline Trainer::Trainer(
    SpeciesEncodingMode species_encoding,
    int hash_dim,
    int top_k,
    SelectionMode selection,
    NormalizationMode normalization,
    std::vector<int64_t> hidden_dims,
    int genus_emb_dim,
    int family_emb_dim,
    float dropout,
    int batch_size,
    int max_epochs,
    int patience,
    float lr,
    float weight_decay,
    float max_grad_norm,
    const std::string& loss_preset,
    std::optional<std::unordered_map<int, PhaseConfig>> custom_phases,
    std::optional<std::vector<int>> phase_boundaries,
    const std::string& checkpoint_dir,
    int checkpoint_every,
    bool resume,
    torch::Device device
) : species_encoding_(species_encoding),
    hash_dim_(hash_dim),
    top_k_(top_k),
    selection_(selection),
    normalization_(normalization),
    hidden_dims_(std::move(hidden_dims)),
    genus_emb_dim_(genus_emb_dim),
    family_emb_dim_(family_emb_dim),
    dropout_(dropout),
    batch_size_(batch_size),
    max_epochs_(max_epochs),
    patience_(patience),
    lr_(lr),
    weight_decay_(weight_decay),
    max_grad_norm_(max_grad_norm),
    checkpoint_dir_(checkpoint_dir),
    checkpoint_every_(checkpoint_every),
    resume_(resume),
    device_(device)
{
    // Resolve loss configuration
    if (custom_phases.has_value()) {
        phases_ = custom_phases.value();
    } else if (LOSS_PRESETS.count(loss_preset)) {
        phases_ = LOSS_PRESETS.at(loss_preset);
    } else {
        throw std::invalid_argument("Unknown loss preset: " + loss_preset);
    }

    if (phase_boundaries.has_value()) {
        phase_boundaries_ = phase_boundaries.value();
    }

    // Create checkpoint directory if needed
    if (!checkpoint_dir_.empty()) {
        fs::create_directories(checkpoint_dir_);
    }
}

inline TrainResult Trainer::fit(
    const ResolveDataset& dataset,
    std::optional<ProgressCallback> progress_callback
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Store schema
    schema_ = dataset.schema();

    // Check for existing checkpoint
    auto checkpoint = load_checkpoint();
    int start_epoch = 0;
    int best_epoch = 0;
    float best_metric = -std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    std::vector<float> train_history, test_history;

    // Split dataset
    auto [train_ds, test_ds] = dataset.split(0.2);

    // Prepare data
    bool fit_scalers = !checkpoint.has_value();
    auto [train_data, test_data] = prepare_data(train_ds, fit_scalers);

    // Build model (TODO: implement ResolveModel in C++)
    // For now, this is a placeholder
    // model_ = std::make_shared<ResolveModel>(...);

    // Move to device
    train_data = train_data.to(device_);
    test_data = test_data.to(device_);
    // model_->to(device_);

    // Setup optimizer
    // optimizer_ = std::make_unique<torch::optim::AdamW>(
    //     model_->parameters(),
    //     torch::optim::AdamWOptions(lr_).weight_decay(weight_decay_)
    // );

    // Setup scheduler
    int64_t steps_per_epoch = (train_data.size(0) + batch_size_ - 1) / batch_size_;
    int64_t total_steps = max_epochs_ * steps_per_epoch;
    // scheduler_ = std::make_unique<OneCycleLR>(*optimizer_, lr_, total_steps, 0.1f);

    // Setup loss
    loss_fn_ = std::make_unique<MultiTaskLoss>(
        schema_.targets,
        phases_,
        phase_boundaries_
    );

    // Restore from checkpoint if available
    if (checkpoint.has_value()) {
        start_epoch = checkpoint->epoch + 1;
        best_epoch = checkpoint->best_epoch;
        best_metric = checkpoint->best_metric;
        epochs_without_improvement = checkpoint->epochs_without_improvement;
        train_history = checkpoint->train_loss_history;
        test_history = checkpoint->test_loss_history;
        best_state_ = checkpoint->best_state;
        // Restore model state, optimizer state, etc.
        std::cout << "Resumed from epoch " << (start_epoch - 1)
                  << " (best=" << (best_metric * 100) << "% at epoch " << best_epoch << ")\n";
    }

    std::vector<std::string> target_names;
    for (const auto& cfg : schema_.targets) {
        target_names.push_back(cfg.name);
    }

    // Training loop
    TrainResult result;
    result.resumed_from_epoch = checkpoint.has_value() ? start_epoch - 1 : 0;

    for (int epoch = start_epoch; epoch < max_epochs_; ++epoch) {
        // Train
        float train_loss = train_epoch(epoch, train_data, target_names);
        train_history.push_back(train_loss);

        // Evaluate
        auto [test_loss, metrics] = eval_epoch(epoch, test_data, target_names);
        test_history.push_back(test_loss);

        // Track best by first target's band_25 or accuracy
        float current_metric = 0.0f;
        if (!target_names.empty()) {
            const auto& first_target = target_names[0];
            if (metrics.count(first_target)) {
                if (metrics[first_target].count("band_25")) {
                    current_metric = metrics[first_target]["band_25"];
                } else if (metrics[first_target].count("accuracy")) {
                    current_metric = metrics[first_target]["accuracy"];
                }
            }
        }

        if (current_metric > best_metric) {
            best_metric = current_metric;
            best_epoch = epoch;
            epochs_without_improvement = 0;
            // Save best state
            // best_state_ = model_->state_dict();
        } else {
            epochs_without_improvement++;
        }

        // Progress callback
        if (progress_callback.has_value()) {
            progress_callback.value()(epoch, train_loss, test_loss, metrics);
        }

        // Log progress
        int phase = loss_fn_ ? 1 : 1;  // TODO: get phase from loss_fn_
        std::cout << "Epoch " << epoch << " [P" << phase << "] | "
                  << "train=" << train_loss << " test=" << test_loss << " | "
                  << (current_metric * 100) << "%\n";

        // Save checkpoint
        if (!checkpoint_dir_.empty() && (epoch == 0 || (epoch + 1) % checkpoint_every_ == 0)) {
            save_checkpoint(epoch, best_epoch, best_metric, epochs_without_improvement,
                           train_history, test_history);
        }

        // Early stopping
        if (epochs_without_improvement >= patience_) {
            std::cout << "Early stopping at epoch " << epoch << "\n";
            result.early_stopped = true;
            break;
        }
    }

    // Restore best model
    // model_->load_state_dict(best_state_);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    result.best_epoch = best_epoch;
    result.best_loss = best_metric;
    result.train_loss_history = train_history;
    result.test_loss_history = test_history;
    result.train_time_seconds = static_cast<float>(duration.count());
    result.total_epochs = static_cast<int>(train_history.size());

    return result;
}

inline float Trainer::train_epoch(
    int epoch,
    const torch::Tensor& train_data,
    const std::vector<std::string>& target_names
) {
    // TODO: Implement training loop
    // This is a placeholder
    return 0.0f;
}

inline std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
Trainer::eval_epoch(
    int epoch,
    const torch::Tensor& test_data,
    const std::vector<std::string>& target_names
) {
    // TODO: Implement evaluation loop
    // This is a placeholder
    std::unordered_map<std::string, std::unordered_map<std::string, float>> metrics;
    return {0.0f, metrics};
}

inline std::pair<torch::Tensor, torch::Tensor> Trainer::prepare_data(
    const ResolveDataset& dataset,
    bool fit_scalers
) {
    // Get continuous features
    auto coords = dataset.get_coordinates();
    auto covariates = dataset.get_covariates();

    // Combine features
    std::vector<torch::Tensor> parts;
    if (coords.defined()) {
        parts.push_back(coords);
    }
    if (covariates.defined()) {
        parts.push_back(covariates);
    }

    torch::Tensor continuous;
    if (!parts.empty()) {
        continuous = torch::cat(parts, 1);
    } else {
        continuous = torch::zeros({static_cast<int64_t>(dataset.n_plots()), 0});
    }

    // Scale
    if (fit_scalers) {
        continuous = continuous_scaler_.fit_transform(continuous);
    } else {
        continuous = continuous_scaler_.transform(continuous);
    }

    // Process targets
    for (const auto& cfg : dataset.targets()) {
        auto target = dataset.get_target(cfg.name);

        if (cfg.task == TaskType::Regression) {
            if (fit_scalers) {
                auto& scaler = target_scalers_[cfg.name];
                target = scaler.fit_transform(target.unsqueeze(1)).squeeze(1);
                target_scaler_params_[cfg.name] = {
                    scaler.mean().to(device_),
                    scaler.std().to(device_)
                };
            } else {
                target = target_scalers_[cfg.name].transform(target.unsqueeze(1)).squeeze(1);
            }
        }
    }

    // TODO: Build complete data tensor with all features and targets
    return {continuous, continuous};  // Placeholder
}

inline void Trainer::save_checkpoint(
    int epoch,
    int best_epoch,
    float best_metric,
    int epochs_without_improvement,
    const std::vector<float>& train_history,
    const std::vector<float>& test_history
) {
    if (checkpoint_dir_.empty()) return;

    // Save binary checkpoint
    // TODO: Implement proper checkpoint saving

    // Save human-readable progress
    nlohmann::json progress;
    progress["epoch"] = epoch;
    progress["max_epochs"] = max_epochs_;
    progress["best_epoch"] = best_epoch;
    progress["best_metric"] = best_metric;
    progress["epochs_without_improvement"] = epochs_without_improvement;
    progress["patience"] = patience_;
    progress["progress_pct"] = 100.0 * epoch / max_epochs_;

    if (!test_history.empty()) {
        progress["latest_test_loss"] = test_history.back();
    }

    std::ofstream f(progress_path());
    f << progress.dump(2);

    std::cout << "  [Checkpoint saved: epoch " << epoch
              << ", best=" << (best_metric * 100) << "%]\n";
}

inline std::optional<Checkpoint> Trainer::load_checkpoint() {
    if (!resume_ || checkpoint_dir_.empty()) {
        return std::nullopt;
    }

    auto path = checkpoint_path();
    if (!fs::exists(path)) {
        return std::nullopt;
    }

    // TODO: Implement proper checkpoint loading
    std::cout << "Loading checkpoint from " << path << "\n";
    return std::nullopt;
}

inline void Trainer::save(const std::string& path) const {
    // TODO: Implement model saving
}

inline void Trainer::load(const std::string& path) {
    // TODO: Implement model loading
}

inline int64_t Trainer::n_params() const {
    if (!model_) return 0;
    // TODO: Count parameters
    return 0;
}

} // namespace resolve
