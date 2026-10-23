#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/loss.hpp"
#include "resolve/lr_schedule.hpp"
#include "resolve/dataset.hpp"
#include "resolve/scalers.hpp"
#include <torch/torch.h>
#include <chrono>
#include <limits>
#include <span>
#include <string>

namespace resolve {

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

    // Every prepare_data overload replaces the whole prepared state: the split,
    // the scalers, the plot IDs, the categorical vocabulary and the device
    // copies of an earlier call, so a later fit() trains on this split alone.

    // Prepare data from a ResolveDataset (preferred API). A seeded random
    // `test_size` fraction of the plots is held out for early stopping.
    void prepare_data(
        const ResolveDataset& dataset,
        float test_size = 0.2f,
        int seed = 42
    );

    // Prepare data from a ResolveDataset, holding out the plots the caller
    // names: a withheld spatial block or database, or the fold a published
    // protocol selects its epoch on. `stopping_indices` are positions in
    // dataset.plot_ids() (ResolveDataset::plot_indices maps plot IDs to them).
    // Early stopping, best_epoch, final_metrics, the baselines, a
    // reduce-on-plateau learning-rate schedule and test_indices() /
    // test_plot_ids() read exactly these plots, in this order; every other
    // plot is a fitting row, and the scalers are fitted on
    // those alone. `seed` orders the fitting rows and drives the per-epoch
    // shuffle as under the random split, so a fold equal to the random one
    // reproduces that split row for row. Throws std::invalid_argument for an
    // empty fold, an index outside [0, n_plots), a repeated index, a fold that
    // holds every plot, and a fold without a labelled plot for a masked target.
    //
    // Every overload throws std::invalid_argument for a learning-rate schedule
    // the split cannot drive (validate_lr_schedule): a reduce-on-plateau
    // schedule with nothing held out, or its knobs out of range.
    void prepare_data(
        const ResolveDataset& dataset,
        const std::vector<int64_t>& stopping_indices,
        int seed = 42
    );

    // Prepare data from tensors: the parts of the continuous block, which this
    // assembles and standardises as the dataset overloads do, and the other
    // model inputs (`inputs.continuous` stays empty; an empty field is an
    // input the data does not carry). `targets` maps each target name to its
    // (n_plots,) values. A seeded random `test_size` fraction is held out.
    // Tensors carry no plot IDs and no categorical vocabulary, so both are
    // empty afterwards.
    void prepare_data(
        const ContinuousInputs& parts,
        const PlotInputs& inputs,
        const TensorMap& targets,
        float test_size = 0.2f,
        int seed = 42
    );

    // Train the model
    TrainResult fit();

    // Continue a run from a checkpoint fit() wrote while it trained: best.pt
    // or checkpoint_<N>.pt under TrainConfig::checkpoint_dir. Those carry the
    // run's progress -- the epochs done, the loss histories, the best epoch's
    // loss, weights and metrics, the early-stopping patience, the learning
    // rates and a reduce-on-plateau schedule's state, the optimizer and AMP
    // state, the batch size the run had reached and the RNG streams -- and
    // resume() picks the run up at the epoch after the checkpoint's, as if it
    // had never stopped. The result covers the whole run;
    // TrainResult::resumed_from_epoch is the epoch it continued from.
    //
    // The trainer must hold the same model architecture and training recipe
    // (Trainer::load_train_config(path) recovers the recipe) and must already
    // be prepared on the interrupted run's data with its seed, so the split is
    // the one the run trained on. Throws std::invalid_argument when the
    // checkpoint carries no progress (a final checkpoint, or one save()
    // wrote), when the prepared split or its scalers differ from the run's,
    // and std::runtime_error when nothing is prepared.
    TrainResult resume(const std::string& checkpoint_path);

    // Save model and state (optionally with run metadata for final checkpoint)
    void save(const std::string& path, const RunMetadata* metadata = nullptr) const;

    // Load model and state. Returns model, scalers, and the categorical
    // vocabulary that was captured at prepare_data time. The vocab is empty
    // for checkpoints from datasets without categorical covariates.
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
    // The batch size the last fit() actually trained at: the requested
    // config().batch_size on a clean run, the post-halve value when the CUDA
    // auto-halve-on-OOM retry fired. 0 before the first fit().
    //
    // fit() restores config_.batch_size to the requested value on the way out
    // (so a later cross-validation fold does not inherit the shrink), which
    // makes config().batch_size unable to report a fallback run. This member is
    // the single source for the effective value: TrainResult::effective_batch_size
    // is a copy of it taken at fit() exit, and save() writes it to the
    // checkpoint's train_effective_batch_size key.
    [[nodiscard]] int effective_batch_size() const noexcept {
        return effective_batch_size_;
    }
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
    // first to reconstruct that split. A prepared split is re-expressed in the
    // loaded scalers, and the device copies and optimizer of an earlier fit()
    // are released. vram_fraction caps the CUDA allocator before upload,
    // matching fit() and the static load(); ignored on CPU.
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
    // a ResolveDataset overload of prepare_data was used (the raw-tensor
    // overload carries no plot IDs); empty otherwise.
    [[nodiscard]] std::vector<std::string> train_plot_ids() const;
    [[nodiscard]] std::vector<std::string> test_plot_ids() const;

    // The run as a checkpoint records it: the split prepare_data made (plot
    // counts, hold_out mode, test_size, split_seed) and, once fit() returns,
    // its timestamps, train time, best / total epochs and final metrics. fit()
    // writes it into checkpoint_dir/checkpoint.pt; pass it to save() to record
    // it in any other checkpoint.
    [[nodiscard]] const RunMetadata& run_metadata() const noexcept { return run_metadata_; }

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

    // Predict in eval mode. `inputs.continuous` is the raw (unstandardised)
    // block; it is standardised with the fitted scalers. Regression outputs
    // come back in original units, classification outputs as logits.
    [[nodiscard]] TensorMap predict(const PlotInputs& inputs);

    // Plots per chunk the held-out forward starts each fit() at, when the
    // training batch is smaller. The forward runs without gradients, so a
    // chunk this size costs far less memory than a training batch of it; an
    // out-of-memory error halves the chunk (decide_eval_chunk_retry).
    static constexpr int64_t kHeldOutChunk = 4096;

private:
    // Model inputs and targets of a set of plots, with the row operations
    // every split, batch, device mirror and cross-validation fold goes
    // through.
    struct LabelledPlots {
        PlotInputs inputs;
        TensorMap targets;

        [[nodiscard]] LabelledPlots index_select(const torch::Tensor& rows) const {
            return {inputs.index_select(rows), index_select_rows(targets, rows)};
        }
        [[nodiscard]] LabelledPlots to(const torch::Device& device) const {
            return {inputs.to(device), to_device(targets, device)};
        }
        [[nodiscard]] static LabelledPlots cat(const LabelledPlots& a, const LabelledPlots& b) {
            return {PlotInputs::cat({a.inputs, b.inputs}), cat_rows({a.targets, b.targets})};
        }
        [[nodiscard]] int64_t n_plots() const { return inputs.n_plots(); }
    };

    // The {mean, scale} of each regression target, keyed by target name.
    using TargetScalers = std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>;

    // The held-out fold a prepare_data call asks for: a `test_size` fraction
    // drawn by the seeded shuffle (Random), or the caller's plot indices
    // (Explicit), which must outlive the call.
    struct HoldOutRequest {
        HoldOutMode mode = HoldOutMode::Random;
        float test_size = 0.2f;
        std::span<const int64_t> stopping_indices;
    };

    // Forget everything a prepare_data call sets -- the split and its indices,
    // the scalers, coordinates, plot IDs, categorical vocabulary, the CUDA-hash
    // CSR and the run metadata -- and release the device copies and optimizer.
    // Both prepare paths start here, so no member of an earlier call survives.
    void reset_prepared_state();

    // The dataset half of both ResolveDataset overloads: captures what the
    // trainer keeps from the dataset, then hands its tensors to prepare_split.
    void prepare_from_dataset(const ResolveDataset& dataset,
                              const HoldOutRequest& hold_out, int seed);

    // The tensor half behind every prepare_data overload: validates the
    // inputs, picks the fold (hold_out_indices), assembles the continuous
    // block, transforms the regression targets, hands the plots to
    // apply_split and records the split in run_metadata_. Arguments as for
    // the tensor prepare_data.
    void prepare_split(
        const ContinuousInputs& parts,
        const PlotInputs& inputs,
        const TensorMap& targets,
        const HoldOutRequest& hold_out,
        int seed
    );

    // The (fitting rows, held-out fold) global plot indices `hold_out` asks
    // for among `n_plots` plots. Both modes start from the same seeded shuffle:
    // the random fold is its tail, and under an explicit fold the fitting rows
    // keep their shuffled order. `targets` (unscaled, keyed by target name) is
    // read to check an explicit fold against the masked targets.
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> hold_out_indices(
        const HoldOutRequest& hold_out, int64_t n_plots, int seed,
        const std::unordered_map<std::string, torch::Tensor>& targets) const;

    // The one split step behind prepare_data and every cross-validation fold.
    // `plots` holds the raw values: the continuous block unstandardised (a
    // missing cell NaN) and each regression target transformed but unscaled.
    // Rows `fit_rows` become the fitting rows and `held_rows` the held-out
    // fold. Refuses a split with no fitting row and a masked target with no
    // labelled fitting row before fitting anything, then fits the fill and
    // scalers on the fitting rows, standardises every row, replaces train_ /
    // test_ / scalers_ and releases the device copies. `global_index` maps a
    // row of `plots` to its plot index (undefined: row i is plot i), which is
    // what train_indices_ / test_indices_ record.
    void apply_split(const LabelledPlots& plots, const torch::Tensor& fit_rows,
                     const torch::Tensor& held_rows, const torch::Tensor& global_index);

    // `plots` standardised by `scalers`: the continuous block filled and
    // standardised, each regression target with a scaler standardised.
    [[nodiscard]] LabelledPlots standardized(LabelledPlots plots, const Scalers& scalers) const;

    // The inverse: `plots`, standardised by `scalers`, in raw values again,
    // with a continuous cell the schema flags as missing NaN once more, so a
    // refit fills it from its own fitting rows.
    [[nodiscard]] LabelledPlots unstandardized(LabelledPlots plots, const Scalers& scalers) const;

    // One epoch's loss: the value the total history records, and its split
    // into the term each target adds (loss_fn_.targets() order) plus the
    // auxiliary terms, which belong to no target (model_->aux_loss_names()
    // order; empty on the held-out fold, which never carries them).
    struct EpochLoss {
        float total = 0.0f;
        std::vector<float> by_target;
        std::vector<float> aux;
    };

    using MetricMap = std::unordered_map<std::string, std::unordered_map<std::string, float>>;

    // Train one epoch
    EpochLoss train_epoch(int epoch);

    // The held-out loss under the selection objective and the metrics of
    // every target.
    std::pair<EpochLoss, MetricMap> eval_epoch();

    // The metrics of target `cfg` over the rows of the held-out `target`
    // labelled for it, regression metrics in original units, plus the
    // labelled-row count of a masked target. eval_epoch and the baselines
    // read their model scores here, so the two agree.
    [[nodiscard]] std::unordered_map<std::string, float> held_out_metrics(
        const TargetConfig& cfg, const torch::Tensor& pred, const torch::Tensor& target,
        const TargetScalers& scalers) const;

    // The target scalers on the training device: the GPU mirror when cached.
    [[nodiscard]] TargetScalers device_target_scalers() const;

    // Update optimizer learning rate
    void update_learning_rate(double lr);

    // Pre-load all data to GPU for faster training
    void cache_data_to_gpu();

    // Release per-attempt training state: the optimizer, the AMP scaler and
    // every GPU-cached tensor, then ask the CUDA caching allocator to return
    // free blocks to the device. The host split stays, so a fresh attempt can
    // be made with a different batch size.
    void release_training_state();

    // Compute the CUDA hash embedding for the given global plot indices and
    // concatenate it onto `continuous`. Non-CUDA-hash models (and CPU builds)
    // return `continuous` unchanged, since their species representation is
    // already folded into it at prepare_data time. `use_cache` selects the
    // GPU-resident CSR buffers when the dataset is cached on device.
    [[nodiscard]] torch::Tensor append_cuda_hash(
        torch::Tensor continuous, torch::Tensor plot_idx, bool use_cache) const;

    // Rows [start, end) of the held-out fold's model inputs on the training
    // device, CUDA hash appended: sliced from the GPU mirror when `use_cache`
    // (cache_data_to_gpu ran), otherwise copied from the host split.
    [[nodiscard]] PlotInputs held_out_chunk(int64_t start, int64_t end, bool use_cache) const;

    // The model's outputs over the held-out fold, in fold order, on the
    // training device (regression: scaled outputs; classification: logits).
    // The one held-out forward behind eval_epoch, the baselines and the
    // test-fold evaluators: it runs in chunks of eval_chunk_ plots, or the
    // whole fold at once for a model that requires_full_batch_training(), and
    // an out-of-memory error shrinks the chunk (decide_eval_chunk_retry) or,
    // when it cannot, is thrown as std::runtime_error -- never as the
    // c10::Error fit() answers by halving the training batch.
    [[nodiscard]] TensorMap forward_held_out(bool use_cache);

    // forward_held_out in eval mode without gradients, for the evaluators.
    [[nodiscard]] TensorMap forward_test_fold();

    // The schema's config of target `name`, or nullptr when there is none.
    [[nodiscard]] const TargetConfig* find_target(const std::string& name) const;

    // Map global plot indices to their plot-ID strings via plot_ids_. Returns
    // empty when plot_ids_ is empty (the raw-tensor prepare_data path).
    [[nodiscard]] std::vector<std::string> select_plot_ids(const torch::Tensor& indices) const;

    // A run's state between epochs: everything the epoch loop reads and
    // writes, and what a checkpoint written during fit() carries so resume()
    // can continue the run.
    struct RunProgress {
        int next_epoch = 0;                 // first epoch still to run
        float best_loss = std::numeric_limits<float>::infinity();
        int patience_counter = 0;
        PlateauState plateau;               // the ReduceOnPlateau schedule's state
        TrainResult result;                 // histories, best_epoch, final_metrics so far
        std::string best_weights;           // snapshot_weights() of the best epoch; empty: none
        std::string optimizer_state;        // snapshot_optimizer() to start from; empty: fresh
        bool has_amp_state = false;
        float amp_scale = 0.0f;
        int amp_growth_tracker = 0;
        double elapsed_seconds = 0.0;       // training time spent so far
        std::string created_at;             // when the run started
        int requested_batch_size = 0;
        int effective_batch_size = 0;
        // Serialised states of the global CPU and CUDA generators (dropout,
        // cover dropout) to start from; undefined: leave them as they are.
        torch::Tensor cpu_rng_state;
        torch::Tensor cuda_rng_state;
    };

    // What one call to train() runs: the epoch range end, whether a held-out
    // fold exists, the stopping mode, whether checkpoints are written, and the
    // clock the progress' elapsed time is kept on.
    struct RunPlan {
        int n_epochs = 0;
        bool has_held_out = false;
        bool fixed_duration = false;
        bool use_checkpoints = false;
        double elapsed_before = 0.0;
        std::chrono::steady_clock::time_point started;
    };

    // The run behind fit() (from scratch) and resume() (from `resume_from`):
    // device setup, the auto-halve-on-OOM retry around run_epochs, the best
    // weights restored, the run metadata and final checkpoint, the baselines
    // and diagnostics.
    TrainResult train(const RunProgress* resume_from);

    // Epochs progress.next_epoch .. plan.n_epochs - 1 (or until early
    // stopping), advancing `progress`, writing best / periodic checkpoints and
    // the progress file.
    void run_epochs(RunProgress& progress, const RunPlan& plan);

    // The held-out fold scored against the naive baselines, per target. The
    // model's scores are held_out_metrics' (so model_mae equals
    // final_metrics[t]["mae"] for the weights fit() returns), and every
    // regression value is in original units.
    [[nodiscard]] std::unordered_map<std::string, BaselineMetrics> compute_baselines();

    // The model's parameters and buffers serialised, and their restore.
    [[nodiscard]] std::string snapshot_weights() const;
    void restore_weights(const std::string& snapshot);

    // The optimizer's state serialised, and its restore into optimizer_.
    [[nodiscard]] std::string snapshot_optimizer() const;
    void restore_optimizer(const std::string& snapshot);

    // Set the global CPU generator (and the training device's CUDA generator)
    // to the states `progress` carries; leaves a generator alone when its
    // state is undefined.
    void restore_rng_states(const RunProgress& progress) const;

    // save() with, when `progress` is given, the run's progress beside the
    // model (what resume() reads).
    void save_checkpoint(const std::string& path, const RunMetadata* metadata,
                         const RunProgress* progress) const;

    // The run progress a checkpoint carries; throws std::invalid_argument
    // naming `path` when it carries none.
    [[nodiscard]] RunProgress read_progress(torch::serialize::InputArchive& archive,
                                            const std::string& path) const;
    void write_progress(torch::serialize::OutputArchive& archive,
                        const RunProgress& progress) const;

    // A checkpoint read onto `device`, with the VRAM cap applied first on a
    // CUDA device and a transient read fault retried. The one way every load
    // path opens a checkpoint.
    static torch::serialize::InputArchive open_checkpoint(
        const std::string& path, torch::Device device, float vram_fraction,
        const LogCallback& log);

    // The trained model a checkpoint holds, put into `model` (whose
    // architecture must match): weights, the species graph of a
    // HeterogeneousGNN, then a move to `device`. Shared by the static load(),
    // load_state() and resume().
    static void restore_model_state(torch::serialize::InputArchive& archive,
                                    ResolveModel& model, torch::Device device);

    // Copy params + buffers from a checkpoint archive into `model` under a
    // NoGradGuard (freshly-constructed leaf params require it), refusing a
    // checkpoint whose tensors do not match the model's one for one.
    static void load_weights_into(torch::serialize::InputArchive& archive, ResolveModel& model);

    ResolveModel model_;
    TrainConfig config_;
    Scalers scalers_;
    MultiTaskLoss loss_fn_;

    // Serialized snapshot of the model's as-constructed (untrained) weights,
    // taken in the constructor. cross_validate resets each fold to this pristine
    // init rather than the trainer's current weights, so a cross_validate run
    // after fit() does not warm-start every fold from fully-trained weights whose
    // training data overlaps the fold's held-out test set.
    std::string pristine_model_state_;

    // The batch size the caller requested at fit() entry, before the CUDA OOM
    // auto-halve retry may have shrunk config_.batch_size. Persisted so a fallback
    // run is detectable (train_effective_batch_size != train_batch_size) and
    // load_train_config restores the requested value.
    //
    // -1 until fit() runs, which is save_train_config's "no separate request
    // known" sentinel: it then persists config_.batch_size, the size the trainer
    // was configured with. A 0 here would be read as a genuine request and a
    // save() before fit() would record a batch size of zero.
    int requested_batch_size_ = -1;

    // The batch size training is running (or last ran) at, tracked across the
    // OOM auto-halve retries so it survives fit()'s restore of
    // config_.batch_size. Read by effective_batch_size(), copied onto
    // TrainResult, and written to the checkpoint by save() -- which is called
    // both inside fit() (best / periodic checkpoints) and after it returns, so
    // the value cannot come from config_.batch_size. 0 until fit runs.
    int effective_batch_size_ = 0;

    // Copy of the categorical vocabulary captured at prepare_data time so
    // it survives independently of the source ResolveDataset (which may go
    // out of scope before save).
    CategoricalVocab categorical_vocab_;

    // Plot IDs captured by a ResolveDataset prepare_data overload, in the
    // dataset's original plot order. Empty when the raw-tensor prepare_data
    // overload was used. Indexed by train_indices_ / test_indices_ to recover
    // the plot IDs of each fold (train_plot_ids() / test_plot_ids()).
    std::vector<std::string> plot_ids_;

    // What run_metadata() reports: the split fields are set by prepare_split,
    // the rest by fit().
    RunMetadata run_metadata_;

    // Raw coordinates stored for spatial CV (before scaling/concatenation)
    torch::Tensor coordinates_;

    // The split prepare_data made, on the host: the fitting rows and the
    // held-out fold, continuous block standardised, regression targets
    // scaled. Fields the data does not carry stay undefined.
    LabelledPlots train_;
    LabelledPlots test_;

    // Snapshot of the split, its scalers and fold indices, and the run record
    // fit() leaves (run metadata, start time, requested and effective batch
    // size). cross_validate / cross_validate_spatial overwrite these per fold;
    // capturing at entry and restoring at exit keeps the trainer's post-CV state
    // consistent, so the checkpoint evaluators run against the original split
    // and a save() after CV records the trainer's own run rather than the last
    // fold's. Restore also releases the GPU cache, whose tensors no longer
    // match the restored split.
    struct SplitState {
        LabelledPlots train, test;
        torch::Tensor train_indices, test_indices;
        Scalers scalers;
        RunMetadata run_metadata;
        std::string created_at;
        int requested_batch_size = -1;
        int effective_batch_size = 0;
    };
    SplitState capture_split_state() const;
    void restore_split_state(const SplitState& s);

    // Shared engine for cross_validate / cross_validate_spatial: given the fold
    // (train_idx, test_idx) index lists into the concatenated train++test rows,
    // runs every fold (per-fold model reset, apply_split, fit, metric
    // aggregation) and restores the pre-CV split. The two public entry points
    // differ only in how they generate `folds`.
    CrossValidationResult run_cross_validation(
        const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& folds);

    // Optimizer
    std::unique_ptr<torch::optim::AdamW> optimizer_;

    bool data_prepared_ = false;

    // Timestamp when training started (for run metadata)
    std::string created_at_;

    // Device mirrors of train_ / test_ and the target scalers, uploaded once
    // by cache_data_to_gpu so an epoch only indexes on the device.
    bool gpu_data_cached_ = false;
    LabelledPlots gpu_train_;
    LabelledPlots gpu_test_;
    TargetScalers gpu_scalers_;

    // AMP (Automatic Mixed Precision) state
    bool amp_enabled_ = false;         // Whether AMP is actually enabled (CUDA only)
    float amp_scale_ = 65536.0f;       // Current gradient scale
    int amp_growth_tracker_ = 0;       // Steps since last overflow

    // Seed captured at prepare_data time. Drives the deterministic per-epoch
    // training shuffle (data_seed_ + epoch) via a dedicated generator, so a
    // fixed seed reproduces the run. Defaults to the prepare_data default.
    int data_seed_ = 42;

    // Plots per held-out forward chunk; see forward_held_out. Starts each
    // train() at max(batch_size, kHeldOutChunk) and keeps a shrink an
    // out-of-memory error forced for the rest of the run and the evaluators.
    int64_t eval_chunk_ = kHeldOutChunk;

    // CUDA hash computation: the full dataset's species records as a CSR over
    // plots, hashed per batch by global plot index.
    bool use_cuda_hash_ = false;       // Whether to use CUDA hash computation
    int32_t hash_dim_ = 0;             // Hash embedding dimension
    torch::Tensor raw_species_ids_;    // (n_records,) int64 - pre-hashed species IDs
    torch::Tensor raw_weights_;        // (n_records,) float32 - species weights
    torch::Tensor plot_offsets_;       // (n_plots+1,) int64 - CSR offsets for each plot
    torch::Tensor gpu_raw_species_ids_;  // GPU copies of the three above
    torch::Tensor gpu_raw_weights_;
    torch::Tensor gpu_plot_offsets_;

    // Original plot indices for train/test (needed for CUDA hash with CSR offsets)
    torch::Tensor train_indices_;             // Global plot indices for training set
    torch::Tensor test_indices_;              // Global plot indices for test set
    torch::Tensor gpu_test_indices_;          // GPU-cached test indices
};

} // namespace resolve
