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

// =============================================================================
// Spatial Block Splitter for cross-validation
// =============================================================================

struct SpatialBlockConfig {
    float lat_size = 1.0f;     // Block size in degrees (latitude)
    float lon_size = 1.0f;     // Block size in degrees (longitude)
    bool balance = false;       // Greedy bin-packing (true) vs round-robin (false)
};

class SpatialBlockSplitter {
public:
    SpatialBlockSplitter(
        float lat_size = 1.0f,
        float lon_size = 1.0f,
        int n_splits = 5,
        int seed = 42,
        bool balance = false
    );

    // Split coordinates into spatial CV folds.
    // coords: (n_plots, 2) tensor [lat, lon]
    // Returns: vector of (train_indices, test_indices) pairs
    [[nodiscard]] std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
    split(torch::Tensor coords) const;

private:
    float lat_size_;
    float lon_size_;
    int n_splits_;
    int seed_;
    bool balance_;
};


// Trainer for ResolveModel
// Supports all encoding modes (hash, embed, sparse, rank_pool, transformer)
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
    // pool_genus_ids: (n_plots, max_species_per_plot) for rank_pool/transformer
    // pool_family_ids: (n_plots, max_species_per_plot) for rank_pool/transformer
    // pool_weights: (n_plots, max_species_per_plot) for rank_pool/transformer
    // pool_mask: (n_plots, max_species_per_plot) for rank_pool/transformer
    // pool_has_cover: (n_plots,) for rank_pool/transformer
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
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        // (n_plots, n_categoricals) int64 codes from CategoricalVocab.
        // Empty when the dataset has no categorical covariates.
        torch::Tensor categorical_ids = {},
        float test_size = 0.2f,
        int seed = 42
    );

    // Train the model
    TrainResult fit();

    // Save model and state (optionally with run metadata for final checkpoint)
    void save(const std::string& path, const RunMetadata* metadata = nullptr) const;

    // Load model and state. Returns model, scalers, and the categorical
    // vocabulary that was captured at prepare_data time. The vocab is empty
    // for checkpoints from datasets without categorical covariates (and for
    // older pre-categorical-port checkpoints, via back-compat in
    // CategoricalVocab::load).
    // vram_fraction caps the PyTorch CUDA caching allocator on the target
    // device before model weights are uploaded; matches TrainConfig's default
    // (1.0) so dedicated training/inference jobs on a solo GPU use the full
    // device. Pass an explicit lower value when sharing the GPU with a
    // desktop or other workloads. Ignored when device is CPU.
    static std::tuple<ResolveModel, Scalers, CategoricalVocab> load(
        const std::string& path,
        torch::Device device = torch::kCPU,
        float vram_fraction = 1.0f
    );

    // Recover the persisted training hyperparameters from a checkpoint without
    // loading the model. Returns a TrainConfig populated with the fields save()
    // wrote (batch_size, lr, weight_decay, phase_boundaries, loss_config,
    // lr_scheduler + params, band_thresholds, vram_fraction, batch_size_floor,
    // max_epochs, patience); fields not persisted (device, checkpoint_dir,
    // AMP/cuDNN flags, log callback) keep their TrainConfig defaults. Lets a
    // caller re-create a Trainer to resume or re-evaluate with the same recipe.
    static TrainConfig load_train_config(const std::string& path);

    // Recover the run metadata persisted in a checkpoint: timing, train/test
    // plot counts, best/total epochs, version + timestamps, and the per-target
    // final metric tree (e.g. final_metrics["area"]["rmse"]).
    static RunMetadata load_run_metadata(const std::string& path);

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return model_; }
    [[nodiscard]] const ResolveModel& model() const noexcept { return model_; }
    [[nodiscard]] const Scalers& scalers() const noexcept { return scalers_; }
    [[nodiscard]] const TrainConfig& config() const noexcept { return config_; }
    // Categorical vocabulary captured at prepare_data time. Empty when the
    // dataset had no categorical covariates. Used by save() to persist the
    // string -> code maps so Predictor.load() can decode new CSVs with the
    // same codes the model was trained against.
    [[nodiscard]] const CategoricalVocab& categorical_vocab() const noexcept {
        return categorical_vocab_;
    }

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

    // Per-plot predictions for a CLASSIFICATION target over the held-out
    // test fold. compute_residuals covers regression only (its predictions
    // are empty for classification); this is the classification counterpart,
    // exposing predicted class codes, the full softmax probability matrix,
    // and ground-truth codes per test plot so callers can compute per-class
    // F1, confusion matrices, and top-k. Returns a result with empty tensors
    // when the named target is not a classification target. Requires
    // prepare_data() first.
    [[nodiscard]] ClassificationPredictions compute_classification_predictions(
        const std::string& target_name
    );

    // Load model weights, scalers, and the categorical vocabulary from the
    // checkpoint at `path` INTO this trainer, in place. First-class
    // alternative to the static load() (whose
    // std::tuple<ResolveModel, Scalers, CategoricalVocab> return has no
    // nanobind/Rcpp converter and is therefore unusable from Python/R). The
    // trainer's model architecture must already match the checkpoint (same
    // schema/config) — typically the trainer was constructed as
    // Trainer(ResolveModel(ds.schema(), mc), tc). After this returns,
    // compute_residuals / compute_calibration /
    // compute_classification_predictions score the loaded weights against the
    // trainer's own test fold; call prepare_data() with the training seed
    // first to reconstruct that split. vram_fraction caps the CUDA allocator
    // before upload, matching fit() and the static load(); ignored on CPU.
    void load_state(
        const std::string& path,
        torch::Device device = torch::kCPU,
        float vram_fraction = 1.0f
    );

    // Global plot indices (into the dataset's original plot order) that
    // prepare_data assigned to the train / test split. int64, shapes
    // (n_train,) / (n_test,). Lets downstream code reconstruct exactly which
    // plots are in the held-out fold (combine with the dataset's plot_ids()).
    [[nodiscard]] torch::Tensor train_indices() const noexcept { return train_indices_; }
    [[nodiscard]] torch::Tensor test_indices() const noexcept { return test_indices_; }

    // Plot IDs for the train / test split, in fold order. Populated only when
    // prepare_data(const ResolveDataset&) was used (the raw-tensor overload
    // carries no plot IDs); empty otherwise.
    [[nodiscard]] std::vector<std::string> train_plot_ids() const;
    [[nodiscard]] std::vector<std::string> test_plot_ids() const;

    // Perform k-fold cross-validation
    // Returns aggregated metrics across all folds
    [[nodiscard]] CrossValidationResult cross_validate(
        int n_folds = 5,
        int seed = 42
    );

    // Spatial block cross-validation using coordinate-based splitting
    [[nodiscard]] CrossValidationResult cross_validate_spatial(
        const SpatialBlockConfig& spatial_config,
        int n_folds = 5,
        int seed = 42
    );

    // Predict on data (runs model in eval mode)
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> predict(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {},
        torch::Tensor pool_genus_ids = {},
        torch::Tensor pool_family_ids = {},
        torch::Tensor pool_weights = {},
        torch::Tensor pool_mask = {},
        torch::Tensor pool_has_cover = {},
        torch::Tensor categorical_ids = {}
    );

private:
    // Train one epoch
    float train_epoch(int epoch);

    // Evaluate on test set
    std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
    eval_epoch(int epoch);

    // Compute learning rate for given epoch based on scheduler config
    float get_learning_rate(int epoch) const;

    // Update optimizer learning rate
    void update_learning_rate(float lr);

    // Pre-load all data to GPU for faster training
    void cache_data_to_gpu();

    // Release per-attempt training state on the way out of an OOM retry.
    // Drops optimizer, AMP scaler, best-model snapshot, prefetch buffers,
    // and all GPU-cached tensors, then asks the CUDA caching allocator to
    // return free blocks to the device. After this returns the trainer is
    // back in the pre-fit configuration (data_prepared_ stays true) and
    // a fresh fit_attempt() can be made with a different batch_size.
    void release_training_state();

    // Helper: bundle of pool tensors to avoid repeating 5-line blocks
    struct PoolTensors {
        torch::Tensor genus_ids, family_ids, weights, mask, has_cover;
    };
    [[nodiscard]] PoolTensors get_test_pool_tensors() const;

    // Compute the CUDA hash embedding for the given global plot indices and
    // concatenate it onto `continuous`. Non-CUDA-hash models (and CPU builds)
    // return `continuous` unchanged, since their species representation is
    // already folded into it at prepare_data time. `use_cache` selects the
    // GPU-resident CSR buffers when the dataset is cached on device. Single
    // source of truth for the four test-fold forwards (issue #86).
    [[nodiscard]] torch::Tensor append_cuda_hash(
        torch::Tensor continuous, torch::Tensor plot_idx, bool use_cache) const;

    // Run the model in eval mode over the held-out test fold and return the
    // per-target prediction map (regression: scaled outputs; classification:
    // logits). Single source of truth for the test-fold forward pass shared
    // by compute_residuals / compute_calibration /
    // compute_classification_predictions.
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> forward_test_fold();

    // Map global plot indices to their plot-ID strings via plot_ids_. Returns
    // empty when plot_ids_ is empty (the raw-tensor prepare_data path).
    [[nodiscard]] std::vector<std::string> select_plot_ids(const torch::Tensor& indices) const;

    // Invert a prior standardization on continuous features and regression
    // targets in place, using `scalers`. Single source of truth shared by the
    // random and spatial cross-validation routines, which reassemble already-
    // standardized train/test tensors and must recover raw values before each
    // fold recomputes its own scalers (otherwise data is standardized twice).
    static void unscale_continuous_targets(
        torch::Tensor& continuous,
        std::unordered_map<std::string, torch::Tensor>& targets,
        const Scalers& scalers);

    // Copy params + buffers from a checkpoint archive into `model` under a
    // NoGradGuard (freshly-constructed leaf params require it). Single source
    // of truth for the static load() and the instance load_state().
    static void load_weights_into(torch::serialize::InputArchive& archive, ResolveModel& model);

    ResolveModel model_;
    TrainConfig config_;
    Scalers scalers_;
    MultiTaskLoss loss_fn_;

    // The batch size the caller requested at fit() entry, before the CUDA OOM
    // auto-halve retry may have shrunk config_.batch_size. Persisted so a fallback
    // run is detectable (train_effective_batch_size != train_batch_size) and
    // load_train_config restores the requested value (issue #86). 0 until fit runs.
    int requested_batch_size_ = 0;

    // Copy of the categorical vocabulary captured at prepare_data time so
    // it survives independently of the source ResolveDataset (which may go
    // out of scope before save).
    CategoricalVocab categorical_vocab_;

    // Plot IDs captured at prepare_data(const ResolveDataset&) time, in the
    // dataset's original plot order. Empty when the raw-tensor prepare_data
    // overload was used. Indexed by train_indices_ / test_indices_ to recover
    // the plot IDs of each fold (train_plot_ids() / test_plot_ids()).
    std::vector<std::string> plot_ids_;

    // Raw coordinates stored for spatial CV (before scaling/concatenation)
    torch::Tensor coordinates_;

    // Training data
    torch::Tensor train_continuous_;
    torch::Tensor train_genus_ids_;
    torch::Tensor train_family_ids_;
    torch::Tensor train_species_ids_;     // For embed mode
    torch::Tensor train_species_vector_;  // For sparse mode
    std::unordered_map<std::string, torch::Tensor> train_targets_;

    // Pool-style fields (rank_pool / transformer modes)
    torch::Tensor train_pool_genus_ids_;
    torch::Tensor train_pool_family_ids_;
    torch::Tensor train_pool_weights_;
    torch::Tensor train_pool_mask_;
    torch::Tensor train_pool_has_cover_;

    // Categorical covariate codes for the training split. Undefined when
    // the dataset has no categorical columns. Layout (n_train, n_categoricals)
    // int64.
    torch::Tensor train_categorical_ids_;

    torch::Tensor test_continuous_;
    torch::Tensor test_genus_ids_;
    torch::Tensor test_family_ids_;
    torch::Tensor test_species_ids_;
    torch::Tensor test_species_vector_;
    std::unordered_map<std::string, torch::Tensor> test_targets_;

    torch::Tensor test_pool_genus_ids_;
    torch::Tensor test_pool_family_ids_;
    torch::Tensor test_pool_weights_;
    torch::Tensor test_pool_mask_;
    torch::Tensor test_pool_has_cover_;
    torch::Tensor test_categorical_ids_;

    // Snapshot of every train/test split member plus the scalers and fold
    // indices. cross_validate / cross_validate_spatial overwrite these per fold;
    // capturing at entry and restoring at exit keeps the trainer's post-CV state
    // consistent so the checkpoint evaluators (compute_residuals /
    // compute_classification_predictions / test_plot_ids) run against the
    // original split rather than the last fold. Restore also invalidates the GPU
    // cache since the cached tensors no longer match the restored split.
    struct SplitState {
        torch::Tensor train_continuous, train_genus_ids, train_family_ids,
            train_species_ids, train_species_vector;
        torch::Tensor train_pool_genus_ids, train_pool_family_ids,
            train_pool_weights, train_pool_mask, train_pool_has_cover,
            train_categorical_ids;
        std::unordered_map<std::string, torch::Tensor> train_targets;
        torch::Tensor test_continuous, test_genus_ids, test_family_ids,
            test_species_ids, test_species_vector;
        torch::Tensor test_pool_genus_ids, test_pool_family_ids,
            test_pool_weights, test_pool_mask, test_pool_has_cover,
            test_categorical_ids;
        std::unordered_map<std::string, torch::Tensor> test_targets;
        torch::Tensor train_indices, test_indices;
        Scalers scalers;
    };
    SplitState capture_split_state() const;
    void restore_split_state(const SplitState& s);

    // Shared engine for cross_validate / cross_validate_spatial: given the fold
    // (train_idx, test_idx) index lists into the concatenated train++test rows,
    // runs every fold (per-fold model reset, split, scaler recompute, fit,
    // metric aggregation) and restores the pre-CV split. The two public entry
    // points differ only in how they generate `folds`.
    CrossValidationResult run_cross_validation(
        const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& folds);

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

    // GPU-cached pool fields (training)
    torch::Tensor gpu_pool_genus_ids_;
    torch::Tensor gpu_pool_family_ids_;
    torch::Tensor gpu_pool_weights_;
    torch::Tensor gpu_pool_mask_;
    torch::Tensor gpu_pool_has_cover_;
    torch::Tensor gpu_categorical_ids_;

    // GPU-cached test data (avoid repeated CPU->GPU transfer in eval)
    torch::Tensor gpu_test_continuous_;
    torch::Tensor gpu_test_genus_ids_;
    torch::Tensor gpu_test_family_ids_;
    torch::Tensor gpu_test_species_ids_;
    torch::Tensor gpu_test_species_vector_;
    std::unordered_map<std::string, torch::Tensor> gpu_test_targets_;

    // GPU-cached pool fields (test)
    torch::Tensor gpu_test_pool_genus_ids_;
    torch::Tensor gpu_test_pool_family_ids_;
    torch::Tensor gpu_test_pool_weights_;
    torch::Tensor gpu_test_pool_mask_;
    torch::Tensor gpu_test_pool_has_cover_;
    torch::Tensor gpu_test_categorical_ids_;

    // AMP (Automatic Mixed Precision) state
    bool amp_enabled_ = false;         // Whether AMP is actually enabled (CUDA only)
    float amp_scale_ = 65536.0f;       // Current gradient scale
    int amp_growth_tracker_ = 0;       // Steps since last overflow

    // Seed captured at prepare_data time. Drives the deterministic per-epoch
    // training shuffle (data_seed_ + epoch) via a dedicated generator, so a
    // fixed seed reproduces the run. Defaults to the prepare_data default.
    int data_seed_ = 42;

    // CUDA hash computation: raw species data for on-the-fly batch hashing
    bool use_cuda_hash_ = false;       // Whether to use CUDA hash computation
    int32_t hash_dim_ = 0;             // Hash embedding dimension
    torch::Tensor raw_species_ids_;    // (n_records,) int64 - pre-hashed species IDs
    torch::Tensor raw_weights_;        // (n_records,) float32 - species weights
    torch::Tensor plot_offsets_;       // (n_plots+1,) int64 - CSR offsets for each plot
    // The on-the-fly CUDA hash path uses the full-dataset CSR above and remaps
    // train-local shuffle positions to global plot indices per batch (see
    // train_epoch), so no train/test-local CSR copies are kept.
    torch::Tensor gpu_train_raw_species_ids_;  // GPU-cached training species IDs
    torch::Tensor gpu_train_raw_weights_;      // GPU-cached training weights
    torch::Tensor gpu_train_plot_offsets_;     // GPU-cached training offsets

    // Original plot indices for train/test (needed for CUDA hash with CSR offsets)
    torch::Tensor train_indices_;             // Global plot indices for training set
    torch::Tensor test_indices_;              // Global plot indices for test set
    torch::Tensor gpu_test_indices_;          // GPU-cached test indices
};

} // namespace resolve
