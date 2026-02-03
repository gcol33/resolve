#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/loss.hpp"
#include "resolve/dataset.hpp"
#include <torch/torch.h>
#include <chrono>

namespace resolve {

// Forward declaration
class ResolveDataset;

// Data scalers (mean, scale) per feature/target
struct Scalers {
    torch::Tensor continuous_mean;
    torch::Tensor continuous_scale;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> target_scalers;
};

// Trainer for ResolveModel
// Supports all three encoding modes (hash, embed, sparse)
class Trainer {
public:
    Trainer(
        ResolveModel model,
        const TrainConfig& config = TrainConfig{}
    );

    // Prepare data from a ResolveDataset (preferred API)
    void prepare_data(
        const ResolveDataset& dataset,
        float test_size = 0.2f,
        int seed = 42
    );

    // Prepare data for training (raw tensor API for backwards compatibility)
    // coordinates: (n_plots, 2) or empty if no coords
    // covariates: (n_plots, n_covariates) or empty
    // hash_embedding: (n_plots, hash_dim) for hash mode
    // species_ids: (n_plots, top_k_species) for embed mode
    // species_vector: (n_plots, n_species) for sparse mode
    // genus_ids: (n_plots, n_taxonomy_slots) or empty
    // family_ids: (n_plots, n_taxonomy_slots) or empty
    // unknown_fraction: (n_plots,) optional
    // unknown_count: (n_plots,) optional
    // targets: map of target_name -> (n_plots,) tensor
    void prepare_data(
        torch::Tensor coordinates,
        torch::Tensor covariates,
        torch::Tensor hash_embedding,
        torch::Tensor species_ids,
        torch::Tensor species_vector,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor unknown_fraction,
        torch::Tensor unknown_count,
        const std::unordered_map<std::string, torch::Tensor>& targets,
        float test_size = 0.2f,
        int seed = 42
    );

    // Train the model
    TrainResult fit();

    // Save model and state (optionally with run metadata for final checkpoint)
    void save(const std::string& path, const RunMetadata* metadata = nullptr) const;

    // Load model and state
    static std::tuple<ResolveModel, Scalers> load(
        const std::string& path,
        torch::Device device = torch::kCPU
    );

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return model_; }
    [[nodiscard]] const ResolveModel& model() const noexcept { return model_; }
    [[nodiscard]] const Scalers& scalers() const noexcept { return scalers_; }
    [[nodiscard]] const TrainConfig& config() const noexcept { return config_; }

    [[nodiscard]] NetworkDiagnostics compute_diagnostics();

    // Advanced evaluation methods

    // Compute calibration for a classification target
    // n_bins: number of probability bins (default 10)
    [[nodiscard]] CalibrationResult compute_calibration(
        const std::string& target_name,
        int n_bins = 10
    );

    // Compute residual analysis for a regression target
    [[nodiscard]] ResidualAnalysis compute_residuals(
        const std::string& target_name
    );

    // Perform k-fold cross-validation
    // Returns aggregated metrics across all folds
    [[nodiscard]] CrossValidationResult cross_validate(
        int n_folds = 5,
        int seed = 42
    );

private:
    // Train one epoch
    float train_epoch(int epoch);

    // Evaluate on test set
    std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
    eval_epoch(int epoch);

    // Create data loaders
    void create_loaders();

    // Compute learning rate for given epoch based on scheduler config
    float get_learning_rate(int epoch) const;

    // Update optimizer learning rate
    void update_learning_rate(float lr);

    // Pre-load all data to GPU for faster training
    void cache_data_to_gpu();

    ResolveModel model_;
    TrainConfig config_;
    Scalers scalers_;
    MultiTaskLoss loss_fn_;

    // Training data
    torch::Tensor train_continuous_;
    torch::Tensor train_genus_ids_;
    torch::Tensor train_family_ids_;
    torch::Tensor train_species_ids_;     // For embed mode
    torch::Tensor train_species_vector_;  // For sparse mode
    std::unordered_map<std::string, torch::Tensor> train_targets_;

    torch::Tensor test_continuous_;
    torch::Tensor test_genus_ids_;
    torch::Tensor test_family_ids_;
    torch::Tensor test_species_ids_;
    torch::Tensor test_species_vector_;
    std::unordered_map<std::string, torch::Tensor> test_targets_;

    // Best model state for restoring
    std::vector<char> best_model_state_;

    // Optimizer
    std::unique_ptr<torch::optim::AdamW> optimizer_;

    bool data_prepared_ = false;

    // Timestamp when training started (for run metadata)
    std::string created_at_;

    // GPU-cached training data (for fast epochs after first)
    bool gpu_data_cached_ = false;
    torch::Tensor gpu_continuous_;
    torch::Tensor gpu_genus_ids_;
    torch::Tensor gpu_family_ids_;
    torch::Tensor gpu_species_ids_;
    torch::Tensor gpu_species_vector_;
    std::unordered_map<std::string, torch::Tensor> gpu_targets_;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> gpu_scalers_;

    // GPU-cached test data (avoid repeated CPU->GPU transfer in eval)
    torch::Tensor gpu_test_continuous_;
    torch::Tensor gpu_test_genus_ids_;
    torch::Tensor gpu_test_family_ids_;
    torch::Tensor gpu_test_species_ids_;
    torch::Tensor gpu_test_species_vector_;
    std::unordered_map<std::string, torch::Tensor> gpu_test_targets_;

    // Shuffled training data (cached, reshuffled every N epochs)
    torch::Tensor shuffled_continuous_;
    torch::Tensor shuffled_genus_ids_;
    torch::Tensor shuffled_family_ids_;
    torch::Tensor shuffled_species_ids_;
    torch::Tensor shuffled_species_vector_;
    std::unordered_map<std::string, torch::Tensor> shuffled_targets_;

    // AMP (Automatic Mixed Precision) state
    bool amp_enabled_ = false;         // Whether AMP is actually enabled (CUDA only)
    float amp_scale_ = 65536.0f;       // Current gradient scale
    int amp_growth_tracker_ = 0;       // Steps since last overflow

    // CUDA hash computation: raw species data for on-the-fly batch hashing
    bool use_cuda_hash_ = false;       // Whether to use CUDA hash computation
    int32_t hash_dim_ = 0;             // Hash embedding dimension
    torch::Tensor raw_species_ids_;    // (n_records,) int64 - pre-hashed species IDs
    torch::Tensor raw_weights_;        // (n_records,) float32 - species weights
    torch::Tensor plot_offsets_;       // (n_plots+1,) int64 - CSR offsets for each plot
    torch::Tensor train_plot_offsets_; // Remapped offsets for training set
    torch::Tensor test_plot_offsets_;  // Remapped offsets for test set
    torch::Tensor train_raw_species_ids_;  // Species IDs for training plots
    torch::Tensor train_raw_weights_;      // Weights for training plots
    torch::Tensor test_raw_species_ids_;   // Species IDs for test plots
    torch::Tensor test_raw_weights_;       // Weights for test plots
    torch::Tensor gpu_train_raw_species_ids_;  // GPU-cached training species IDs
    torch::Tensor gpu_train_raw_weights_;      // GPU-cached training weights
    torch::Tensor gpu_train_plot_offsets_;     // GPU-cached training offsets

    // Original plot indices for train/test (needed for CUDA hash with CSR offsets)
    torch::Tensor train_indices_;             // Global plot indices for training set
    torch::Tensor test_indices_;              // Global plot indices for test set
    torch::Tensor gpu_test_indices_;          // GPU-cached test indices

    // Async prefetching for CUDA hash computation
    // Double-buffered hash embeddings: compute next batch while training on current
    torch::Tensor prefetch_hash_[2];          // Double-buffered hash embeddings
    torch::Tensor prefetch_batch_idx_;        // Batch indices for prefetched data
    int prefetch_buffer_idx_ = 0;             // Which buffer has prefetched data ready
    bool prefetch_valid_ = false;             // Whether prefetched data is valid
};

} // namespace resolve
