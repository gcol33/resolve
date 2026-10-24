#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include "resolve/utils.hpp"
#include "resolve/checkpoint.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/gpu.hpp"
#include "resolve/grad_clip.hpp"
#include "resolve/selection_metric.hpp"
#include "resolve/species_graph.hpp"
#include "resolve/target_labels.hpp"
#include "resolve/target_transform.hpp"

#ifdef RESOLVE_HAS_CUDA
#include "resolve/cuda/feature_hash.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <c10/util/Exception.h>
#include <ATen/autocast_mode.h>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <cmath>
#include <ctime>
#include <tuple>

namespace resolve {

namespace {

// The held-out loss epoch `epoch` scored; +infinity when it scored none.
float held_out_loss_at(const TrainResult& result, int epoch) {
    if (epoch < 0 || static_cast<size_t>(epoch) >= result.test_loss_history.size()) {
        return std::numeric_limits<float>::infinity();
    }
    return result.test_loss_history[static_cast<size_t>(epoch)];
}

// ISO 8601 UTC timestamp ("YYYY-MM-DDTHH:MM:SSZ") for run metadata. Used only
// for informational created_at/completed_at fields, never for anything the run
// depends on, so the wall-clock read is fine.
std::string iso8601_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

// One-time structural validation of the CSR plot offsets that feed the CUDA
// hash kernels. A kernel skips a plot whose offset range is malformed, so a
// corrupt CSR would silently drop species records. The offsets are fixed for
// the whole run and live on the host at capture time, so they are validated
// once here (no per-batch sync) and a malformed CSR fails loudly.
void validate_csr_offsets(const torch::Tensor& offsets,
                          int64_t n_records, int64_t n_weights) {
    TORCH_CHECK(offsets.defined() && offsets.dim() == 1 && offsets.numel() >= 1,
                "CUDA hash: plot_offsets must be a non-empty 1-D tensor");
    TORCH_CHECK(n_records == n_weights,
                "CUDA hash: raw_species_ids length (", n_records,
                ") must match raw_weights length (", n_weights, ")");
    auto off = offsets.to(torch::kCPU).contiguous();
    auto a = off.accessor<int64_t, 1>();
    TORCH_CHECK(a[0] == 0, "CUDA hash: plot_offsets[0] must be 0, got ", a[0]);
    for (int64_t i = 1; i < off.numel(); ++i) {
        TORCH_CHECK(a[i] >= a[i - 1],
                    "CUDA hash: plot_offsets must be non-decreasing (offset[", i,
                    "]=", a[i], " < offset[", i - 1, "]=", a[i - 1], ")");
    }
    TORCH_CHECK(a[off.numel() - 1] == n_records,
                "CUDA hash: plot_offsets[-1] (", a[off.numel() - 1],
                ") must equal the record count (", n_records, ")");
}

// A masked target needs at least one labelled plot among the fitting rows:
// with none, its loss is zero on every batch and its head never trains, and a
// regression target has no values to fit its scaler to.
void require_labelled_fitting_rows(
    const std::vector<TargetConfig>& configs,
    const std::unordered_map<std::string, torch::Tensor>& train_targets
) {
    for (const auto& cfg : configs) {
        if (cfg.missing != TargetMissingPolicy::Mask) continue;
        auto it = train_targets.find(cfg.name);
        if (it == train_targets.end()) continue;
        if (labelled_rows(cfg, it->second).n_labelled == 0) {
            throw std::invalid_argument(
                "target '" + cfg.name + "' (missing = mask) is labelled on none "
                "of the " + std::to_string(it->second.size(0)) + " training "
                "plots of this split, so its head has nothing to fit to");
        }
    }
}

// The {mean, scale} a regression target is standardized by, fit on the
// training rows labelled for it: every training row of a Drop target, the
// labelled ones of a masked target. Standardizing leaves the sentinel (NaN) of
// an unlabelled row a NaN.
std::pair<torch::Tensor, torch::Tensor> fit_target_scaler(
    const TargetConfig& cfg, const torch::Tensor& train_target
) {
    const auto fitting = labelled_rows(cfg, train_target).select(train_target);
    return {fitting.mean(), standardization_scale(fitting)};
}

// RAII autocast scope covering a training step's forward and loss. The
// nesting counter and enabled flag unwind even if forward() or the loss throws
// -- the expected throw is the c10::OutOfMemoryError train() catches to halve
// the batch and retry -- so an OOM mid-forward neither leaks the incremented
// nesting (its cache is cleared only when nesting returns to 0) nor leaves
// autocast enabled across the retried epochs.
struct AutocastScope {
    bool active;
    explicit AutocastScope(bool a) : active(a) {
        if (active) {
            at::autocast::set_autocast_enabled(at::kCUDA, true);
            at::autocast::increment_nesting();
        }
    }
    ~AutocastScope() {
        if (active) {
            at::autocast::decrement_nesting();
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
    }
    AutocastScope(const AutocastScope&) = delete;
    AutocastScope& operator=(const AutocastScope&) = delete;
};

// Puts config.batch_size back to its value at construction on every way out
// of a run, so neither a halved retry nor a resumed run's batch size carries
// into the next fit().
struct BatchSizeRestorer {
    int& batch_size;
    int saved;
    explicit BatchSizeRestorer(int& b) : batch_size(b), saved(b) {}
    ~BatchSizeRestorer() { batch_size = saved; }
    BatchSizeRestorer(const BatchSizeRestorer&) = delete;
    BatchSizeRestorer& operator=(const BatchSizeRestorer&) = delete;
};

// Silences per-fold checkpoint writes during cross-validation: fit() otherwise
// overwrites the user's checkpoint.pt / best.pt with each fold's subset model,
// leaving the on-disk artifact describing the last fold rather than the
// full-data model the user trained before CV. Clears checkpoint_dir
// on construction and restores it on any exit path (exception-safe).
struct CheckpointDirSilencer {
    std::string& dir;
    std::string saved;
    explicit CheckpointDirSilencer(std::string& d) : dir(d), saved(d) { d.clear(); }
    ~CheckpointDirSilencer() { dir = saved; }
    CheckpointDirSilencer(const CheckpointDirSilencer&) = delete;
    CheckpointDirSilencer& operator=(const CheckpointDirSilencer&) = delete;
};

// The number of epochs fit() runs. Early stopping watches the held-out fold,
// so it needs one; a fixed duration does not, and has to lie inside the
// schedule it is cut from.
int epochs_to_run(const TrainConfig& config, bool has_held_out) {
    if (config.fixed_epochs < 0) {
        throw std::invalid_argument(
            "TrainConfig::fixed_epochs is " + std::to_string(config.fixed_epochs) +
            "; it counts epochs to run, so it is 0 (early stopping) or positive");
    }
    if (config.fixed_epochs > config.max_epochs) {
        throw std::invalid_argument(
            "TrainConfig::fixed_epochs = " + std::to_string(config.fixed_epochs) +
            " runs past max_epochs = " + std::to_string(config.max_epochs) +
            ", the length the learning-rate schedule is laid out over");
    }
    if (config.fixed_epochs > 0) {
        return config.fixed_epochs;
    }
    if (!has_held_out) {
        throw std::invalid_argument(
            "Trainer::fit: the data was prepared with no held-out fold "
            "(test_size = 0), so early stopping has nothing to watch. Set "
            "TrainConfig::fixed_epochs to train for a fixed number of epochs.");
    }
    return config.max_epochs;
}

// Appends one epoch's per-target terms (in `targets` order) to `history`,
// keyed by target name.
void append_by_target(std::unordered_map<std::string, std::vector<float>>& history,
                      const std::vector<TargetConfig>& targets,
                      const std::vector<float>& terms) {
    for (size_t i = 0; i < targets.size(); ++i) {
        history[targets[i].name].push_back(terms[i]);
    }
}

// The progress-line suffix that splits a multi-target epoch's loss by target:
// " | by target (train/test): a 0.12/0.15, b 0.40/0.52", followed by
// "; moe 0.001, tabnet_sparsity 0.0002" for the model's auxiliary terms.
// `test_terms` is null when nothing is held out, and each entry then shows its
// train term only.
std::string by_target_suffix(const std::vector<TargetConfig>& targets,
                             const std::vector<float>& train_terms,
                             const std::vector<float>* test_terms,
                             const std::vector<std::string>& aux_names,
                             const std::vector<float>& aux_terms) {
    std::ostringstream msg;
    msg << " | by target (" << (test_terms ? "train/test" : "train") << "): ";
    for (size_t i = 0; i < targets.size(); ++i) {
        if (i > 0) msg << ", ";
        msg << targets[i].name << " " << train_terms[i];
        if (test_terms) msg << "/" << (*test_terms)[i];
    }
    for (size_t i = 0; i < aux_names.size(); ++i) {
        msg << (i == 0 ? "; " : ", ") << aux_names[i] << " " << aux_terms[i];
    }
    return msg.str();
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

Trainer::Trainer(
    ResolveModel model,
    const TrainConfig& config
) : model_(model), config_(config),
    loss_fn_(model->schema().targets, config.phase_boundaries, config.loss_config,
             config.band_threshold, nca_term_of(config))
{
    model_->to(config_.device);

    // Snapshot the as-constructed weights so cross_validate can reset each fold
    // to a fresh init even when run after fit().
    pristine_model_state_ = snapshot_weights();
}

// =============================================================================
// Preparing the split
// =============================================================================

void Trainer::prepare_data(
    const ResolveDataset& dataset,
    float test_size,
    int seed
) {
    prepare_from_dataset(dataset, {HoldOutMode::Random, test_size, {}}, seed);
}

void Trainer::prepare_data(
    const ResolveDataset& dataset,
    const std::vector<int64_t>& stopping_indices,
    int seed
) {
    prepare_from_dataset(dataset, {HoldOutMode::Explicit, 0.0f, stopping_indices}, seed);
}

void Trainer::prepare_data(
    const ContinuousInputs& parts,
    const PlotInputs& inputs,
    const TensorMap& targets,
    float test_size,
    int seed
) {
    reset_prepared_state();
    prepare_split(parts, inputs, targets, {HoldOutMode::Random, test_size, {}}, seed);
}

void Trainer::reset_prepared_state() {
    release_training_state();
    data_prepared_ = false;
    train_ = {};
    test_ = {};
    train_indices_ = torch::Tensor();
    test_indices_ = torch::Tensor();
    scalers_ = {};
    coordinates_ = torch::Tensor();
    plot_ids_.clear();
    categorical_vocab_ = CategoricalVocab{};
    use_cuda_hash_ = false;
    hash_dim_ = 0;
    raw_species_ids_ = torch::Tensor();
    raw_weights_ = torch::Tensor();
    plot_offsets_ = torch::Tensor();
    run_metadata_ = RunMetadata{};
}

void Trainer::prepare_from_dataset(
    const ResolveDataset& dataset,
    const HoldOutRequest& hold_out,
    int seed
) {
    reset_prepared_state();

    // The dataset's species records as a CSR, when its encoding hashes them on
    // the device per batch.
    if (dataset.has_raw_species_data()) {
        use_cuda_hash_ = true;
        hash_dim_ = dataset.config().hash_dim;
        raw_species_ids_ = dataset.raw_species_ids();
        raw_weights_ = dataset.raw_weights();
        plot_offsets_ = dataset.plot_offsets();
        validate_csr_offsets(plot_offsets_, raw_species_ids_.numel(),
                             raw_weights_.numel());
    }

    // Capture the dataset's categorical vocab. The trainer outlives the
    // dataset (`save()` is called long after `prepare_data()`); without this
    // copy the vocab would be a dangling reference at save time.
    categorical_vocab_ = dataset.categorical_vocab();

    // Capture plot IDs so test_plot_ids() / train_plot_ids() can recover which
    // plots landed in each split. The raw-tensor prepare_data overload has no
    // plot IDs, so this is the only place plot_ids_ is populated.
    plot_ids_ = dataset.plot_ids();

    // A HeterogeneousGNN passes messages on a graph over the species, which is
    // built from the dataset's taxonomy and co-occurrence as its config asks
    // (species_graph.hpp). This is the one place with both the model and the
    // data, so it is where the graph is built; save() then writes it into the
    // checkpoint, because it is part of the trained model. A graph a caller
    // set explicitly is left alone.
    if (model_->requires_species_graph() && !model_->has_species_graph()) {
        auto graph = build_species_graph(dataset,
                                         model_->config().heterogeneous_gnn);
        config_.log("Species graph: " + std::to_string(graph.n_edges()) +
                    " edges over " + std::to_string(graph.n_species) +
                    " species");
        model_->set_species_graph(graph.edge_index, graph.edge_type);
    }

    prepare_split(dataset.continuous_inputs(), dataset.plot_inputs(), dataset.targets(),
                  hold_out, seed);
}

std::pair<torch::Tensor, torch::Tensor> Trainer::hold_out_indices(
    const HoldOutRequest& hold_out,
    int64_t n_plots,
    int seed,
    const std::unordered_map<std::string, torch::Tensor>& targets
) const {
    std::vector<int64_t> order(n_plots);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(order.begin(), order.end(), gen);

    if (hold_out.mode == HoldOutMode::Random) {
        const int64_t n_test = static_cast<int64_t>(n_plots * hold_out.test_size);
        const int64_t n_train = n_plots - n_test;

        // An empty test fold makes every held-out metric divide by zero
        // (accuracy, baseline, residuals) and returns NaN rather than an error.
        // Fail loudly so a too-small dataset / test_size is caught here, not as
        // silent NaNs downstream. test_size = 0 asks for no fold at all.
        if (hold_out.test_size > 0.0f && n_test <= 0) {
            throw std::runtime_error(
                "prepare_data: test_size=" + std::to_string(hold_out.test_size) +
                " yields an empty test fold for " + std::to_string(n_plots) +
                " plots; increase test_size or provide more data");
        }
        return {torch::tensor(std::vector<int64_t>(order.begin(), order.begin() + n_train),
                              torch::kInt64),
                torch::tensor(std::vector<int64_t>(order.begin() + n_train, order.end()),
                              torch::kInt64)};
    }

    // A plot index as a message names it: with its plot ID when the dataset
    // overload captured them.
    const auto describe = [&](int64_t index) {
        std::string out = "index " + std::to_string(index);
        if (index >= 0 && index < static_cast<int64_t>(plot_ids_.size()) &&
            static_cast<int64_t>(plot_ids_.size()) == n_plots) {
            out += " (plot '" + plot_ids_[static_cast<size_t>(index)] + "')";
        }
        return out;
    };

    const auto& fold = hold_out.stopping_indices;
    if (fold.empty()) {
        throw std::invalid_argument(
            "prepare_data: the stopping fold is empty; early stopping needs at "
            "least one held-out plot");
    }
    std::vector<char> held(static_cast<size_t>(n_plots), 0);
    for (const int64_t index : fold) {
        if (index < 0 || index >= n_plots) {
            throw std::invalid_argument(
                "prepare_data: stopping fold " + describe(index) +
                " is outside the " + std::to_string(n_plots) + " plots [0, " +
                std::to_string(n_plots) + ")");
        }
        auto& slot = held[static_cast<size_t>(index)];
        if (slot) {
            throw std::invalid_argument(
                "prepare_data: stopping fold lists " + describe(index) +
                " more than once");
        }
        slot = 1;
    }
    if (static_cast<int64_t>(fold.size()) == n_plots) {
        throw std::invalid_argument(
            "prepare_data: the stopping fold holds all " + std::to_string(n_plots) +
            " plots, which leaves none to fit");
    }

    auto test_idx = torch::tensor(std::vector<int64_t>(fold.begin(), fold.end()),
                                  torch::kInt64);

    // A masked target scored on a fold with none of its labels has no metric
    // and adds nothing to the stopping loss, so the fold could not select an
    // epoch for it. A random fold may draw that by chance; a named fold is
    // chosen, so naming one without the target's labels is refused.
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.missing != TargetMissingPolicy::Mask) continue;
        auto it = targets.find(cfg.name);
        if (it == targets.end()) continue;
        if (labelled_rows(cfg, it->second.index_select(0, test_idx)).n_labelled == 0) {
            throw std::invalid_argument(
                "prepare_data: target '" + cfg.name + "' (missing = mask) is "
                "labelled on none of the " + std::to_string(fold.size()) +
                " plots of the stopping fold, so the fold cannot select an "
                "epoch for it");
        }
    }

    std::vector<int64_t> train;
    train.reserve(static_cast<size_t>(n_plots) - fold.size());
    for (const int64_t index : order) {
        if (!held[static_cast<size_t>(index)]) train.push_back(index);
    }
    return {torch::tensor(train, torch::kInt64), test_idx};
}

void Trainer::prepare_split(
    const ContinuousInputs& parts,
    const PlotInputs& inputs,
    const TensorMap& targets,
    const HoldOutRequest& hold_out,
    int seed
) {
    // A fit needs something to fit to. Without this guard an empty target map
    // reaches the training loop, MultiTaskLoss accumulates over no terms, and
    // the failure surfaces at loss.backward() as "element 0 of tensors does not
    // require grad and does not have a grad_fn" -- an autograd message that
    // points nowhere near the target specification that caused it. Inference
    // is unaffected: Predictor::predict never prepares a split, so a
    // target-less dataset is still a legal thing to score.
    if (targets.empty()) {
        throw std::invalid_argument(
            "Trainer::prepare_data: the data carries no targets, so there is "
            "nothing to fit. Check the target specification the dataset was "
            "built with -- at least one target is required to train.");
    }

    if (inputs.continuous.defined()) {
        throw std::invalid_argument(
            "Trainer::prepare_data: PlotInputs::continuous must be empty; the "
            "continuous block is assembled from the ContinuousInputs parts");
    }
    PlotInputs all = inputs.without_empty();

    const int64_t n_plots = parts.n_rows() > 0 ? parts.n_rows() : all.n_plots();
    if (n_plots == 0) {
        throw std::runtime_error("No valid input tensors provided");
    }

    // DatasetConfig.hash_dim and ModelConfig.hash_dim are independent. If they
    // disagree, the model's first linear layer is sized for one value while the
    // block carries a hash embedding of the other width, and fit() fails later
    // with an opaque matmul shape error; the mismatch is named here instead.
    const torch::Tensor& hash_embedding = parts.hash_embedding;
    if (model_->hash_in_continuous() &&
        hash_embedding.defined() && hash_embedding.numel() > 0) {
        int64_t data_hash_dim = hash_embedding.size(1);
        int64_t model_hash_dim = model_->config().hash_dim;
        if (data_hash_dim != model_hash_dim) {
            throw std::runtime_error(
                "hash_dim mismatch: ModelConfig.hash_dim=" + std::to_string(model_hash_dim) +
                " but dataset hash_embedding has " + std::to_string(data_hash_dim) +
                " columns. Set DatasetConfig.hash_dim and ModelConfig.hash_dim to the same value."
            );
        }
    }

    // If the model was constructed for K categorical columns, the caller must
    // pass a (n_plots, K) tensor; a model without categorical columns takes
    // none.
    {
        const int64_t model_cat_cols = model_->schema().n_categoricals();
        const torch::Tensor& categorical_ids = all.categorical_ids;
        if (categorical_ids.defined()) {
            if (categorical_ids.dim() != 2) {
                throw std::runtime_error(
                    "Trainer::prepare_data: categorical_ids must be 2-D, got dim=" +
                    std::to_string(categorical_ids.dim()));
            }
            if (categorical_ids.size(1) != model_cat_cols) {
                throw std::runtime_error(
                    "Trainer::prepare_data: categorical_ids has " +
                    std::to_string(categorical_ids.size(1)) +
                    " columns but the model expects " +
                    std::to_string(model_cat_cols) +
                    " (schema.categorical_names)");
            }
            if (categorical_ids.size(0) != n_plots) {
                throw std::runtime_error(
                    "Trainer::prepare_data: categorical_ids has " +
                    std::to_string(categorical_ids.size(0)) +
                    " rows but other inputs have " +
                    std::to_string(n_plots) + " rows");
            }
        } else if (model_cat_cols > 0) {
            throw std::runtime_error(
                "Trainer::prepare_data: model was constructed for " +
                std::to_string(model_cat_cols) +
                " categorical columns but no categorical_ids tensor was "
                "provided. Pass dataset.categorical_ids() or use the "
                "ResolveDataset overload of prepare_data().");
        }
    }

    auto [fit_rows, held_rows] = hold_out_indices(hold_out, n_plots, seed, targets);
    validate_lr_schedule(config_, held_rows.numel() > 0);
    validate_selection_metric(config_, model_->schema().targets, held_rows.numel() > 0);

    // The continuous block (continuous_block.hpp), raw; in hash mode the hash
    // embedding rides along at its end.
    ContinuousInputs block_parts = parts;
    if (!model_->hash_in_continuous()) block_parts.hash_embedding = torch::Tensor();
    all.continuous = assemble_continuous(block_parts, model_->schema().dataset_config.missing_values, n_plots);

    // Each target the schema names, a regression target transformed.
    TensorMap raw_targets;
    for (const auto& cfg : model_->schema().targets) {
        auto target_it = targets.find(cfg.name);
        if (target_it == targets.end()) continue;
        auto target = target_it->second;
        if (cfg.task == TaskType::Regression && cfg.transform == TransformType::Log1p) {
            target = torch::log1p(target);
        }
        raw_targets[cfg.name] = target;
    }

    apply_split({std::move(all), std::move(raw_targets)}, fit_rows, held_rows,
                /*global_index=*/torch::Tensor());

    // Capture the seed so the per-epoch training shuffle is reproducible.
    data_seed_ = seed;

    // Raw coordinates in plot order, for spatial CV.
    if (parts.coordinates.defined() && parts.coordinates.numel() > 0) {
        coordinates_ = parts.coordinates.clone();
    }

    // How the plots were split, for run_metadata() and the checkpoint.
    run_metadata_ = RunMetadata{};
    run_metadata_.n_plots_train = train_indices_.numel();
    run_metadata_.n_plots_test = test_indices_.numel();
    run_metadata_.hold_out = hold_out.mode;
    run_metadata_.test_size =
        hold_out.mode == HoldOutMode::Random ? hold_out.test_size : 0.0f;
    run_metadata_.split_seed = seed;
}

void Trainer::apply_split(
    const LabelledPlots& plots,
    const torch::Tensor& fit_rows,
    const torch::Tensor& held_rows,
    const torch::Tensor& global_index
) {
    if (fit_rows.numel() == 0) {
        throw std::invalid_argument(
            "Trainer: the split leaves no fitting rows (" +
            std::to_string(held_rows.numel()) + " plots held out of " +
            std::to_string(fit_rows.numel() + held_rows.numel()) +
            "); hold out fewer plots");
    }

    // Validated before anything is fitted on the fitting rows.
    const LabelledPlots fitting = plots.index_select(fit_rows);
    require_labelled_fitting_rows(model_->schema().targets, fitting.targets);

    // The fill and scalers come from the fitting rows alone.
    Scalers scalers;
    fit_continuous_scalers(scalers, fitting.inputs.continuous);
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.task != TaskType::Regression) continue;
        auto it = fitting.targets.find(cfg.name);
        if (it == fitting.targets.end()) continue;
        scalers.target_scalers[cfg.name] = fit_target_scaler(cfg, it->second);
    }

    const LabelledPlots scaled = standardized(plots, scalers);
    release_training_state();
    scalers_ = std::move(scalers);
    train_ = scaled.index_select(fit_rows);
    test_ = scaled.index_select(held_rows);
    train_indices_ = global_index.defined() ? global_index.index_select(0, fit_rows)
                                            : fit_rows.clone();
    test_indices_ = global_index.defined() ? global_index.index_select(0, held_rows)
                                           : held_rows.clone();
    data_prepared_ = true;
}

Trainer::LabelledPlots Trainer::standardized(LabelledPlots plots, const Scalers& scalers) const {
    if (plots.inputs.continuous.defined()) {
        plots.inputs.continuous = standardize_continuous(plots.inputs.continuous, scalers);
    }
    for (const auto& [name, mean_scale] : scalers.target_scalers) {
        auto it = plots.targets.find(name);
        if (it == plots.targets.end() || !it->second.defined()) continue;
        it->second = (it->second - mean_scale.first) / mean_scale.second;
    }
    return plots;
}

Trainer::LabelledPlots Trainer::unstandardized(LabelledPlots plots, const Scalers& scalers) const {
    if (plots.inputs.continuous.defined()) {
        plots.inputs.continuous =
            unstandardize_continuous(plots.inputs.continuous, scalers, model_->schema());
    }
    for (const auto& [name, mean_scale] : scalers.target_scalers) {
        auto it = plots.targets.find(name);
        if (it == plots.targets.end() || !it->second.defined()) continue;
        it->second = it->second * mean_scale.second + mean_scale.first;
    }
    return plots;
}

// =============================================================================
// Epochs
// =============================================================================

Trainer::TargetScalers Trainer::device_target_scalers() const {
    if (gpu_data_cached_) return gpu_scalers_;
    TargetScalers out;
    for (const auto& [name, scaler] : scalers_.target_scalers) {
        out[name] = {scaler.first.to(config_.device), scaler.second.to(config_.device)};
    }
    return out;
}

int64_t Trainer::training_batch_size() const {
    // The coordinate-kNN GNN must see the whole train set in one forward so its
    // spatial graph spans all plots (their true neighbors), not an arbitrary
    // mini-batch. Force full-batch training for it, matching the single
    // full-batch inference forward the Predictor already uses.
    return model_->requires_full_batch_training() ? train_.n_plots()
                                                  : static_cast<int64_t>(config_.batch_size);
}

int64_t Trainer::training_steps_per_epoch() const {
    return steps_per_epoch(train_.n_plots(), training_batch_size());
}

Trainer::EpochLoss Trainer::train_epoch(int epoch, const PlateauState& plateau) {
    model_->train();

    // Use GPU-cached data if available, otherwise fallback to CPU tensors
    const LabelledPlots& src = gpu_data_cached_ ? gpu_train_ : train_;
    const TargetScalers scalers = device_target_scalers();

    int64_t n_train = src.n_plots();
    const int64_t batch_size = training_batch_size();

    // The run's step count at this epoch's first batch (lr_schedule.hpp).
    const int64_t epoch_steps = training_steps_per_epoch();
    const int64_t first_step = static_cast<int64_t>(epoch) * epoch_steps;

    // Shuffle training data with a dedicated, seeded generator so the run is
    // reproducible for a fixed seed. Generated on CPU (the permutation is tiny)
    // and moved to the data device; using a private generator keeps the global
    // RNG stream — and therefore dropout / cover-dropout — untouched.
    auto perm_gen = at::detail::createCPUGenerator(
        static_cast<uint64_t>(data_seed_) + static_cast<uint64_t>(epoch) + 1u);
    auto perm = torch::randperm(
        n_train, perm_gen, torch::TensorOptions().dtype(torch::kLong));
    if (src.inputs.continuous.device() != perm.device()) {
        perm = perm.to(src.inputs.continuous.device());
    }

    int n_batches = 0;

    // The auxiliary objective terms the model adds (already weighted).
    const std::vector<std::string> aux_names = model_->aux_loss_names();
    const auto n_aux = static_cast<int64_t>(aux_names.size());

    // Running sums over the counted batches, kept on the device and read back
    // once per epoch so the loop adds no host sync per batch: slot 0 the total
    // loss, then one slot per target in loss_fn_.targets() order, then one per
    // auxiliary term in aux_names order.
    const auto n_loss_targets = static_cast<int64_t>(loss_fn_.targets().size());
    torch::Tensor sums = torch::zeros(
        {1 + n_loss_targets + n_aux},
        torch::TensorOptions().dtype(torch::kFloat32).device(config_.device));

    // Async prefetching setup for CUDA hash computation
#ifdef RESOLVE_HAS_CUDA
    // RESOLVE_HAS_CUDA is a COMPILE-time guard, so a CUDA-enabled build reaches
    // this line on a CPU run too. The streams are acquired only when the
    // prefetch will run: acquiring them initialises the CUDA runtime, which
    // kills a device="cpu" run on a host with no usable driver -- a CPU queue
    // node, a CI runner, a laptop.
    const bool use_prefetch = use_hash_prefetch(
        use_cuda_hash_, config_.device.is_cuda(), n_train, batch_size);
    std::optional<c10::cuda::CUDAStream> prefetch_stream;
    std::optional<c10::cuda::CUDAStream> default_stream;
    if (use_prefetch) {
        prefetch_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
        default_stream = at::cuda::getCurrentCUDAStream();
    }
    torch::Tensor prefetched_hash;
    bool prefetch_ready = false;

    // Get source data pointers for hash computation
    const auto& hash_species_src = gpu_data_cached_ ? gpu_raw_species_ids_ : raw_species_ids_;
    const auto& hash_weights_src = gpu_data_cached_ ? gpu_raw_weights_ : raw_weights_;
    const auto& hash_offsets_src = gpu_data_cached_ ? gpu_plot_offsets_ : plot_offsets_;

    // Train-local shuffle positions map to GLOBAL plot indices before hashing:
    // train_ row k is plot train_indices_[k], and the CSR offsets are over the
    // full dataset, so a batch row's hash is looked up by its global index
    // (as the held-out forward does). Otherwise a row's species hash would
    // come from a different plot than its covariates and target.
    torch::Tensor hash_index_map;
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        hash_index_map = train_indices_.to(perm.device());
    }
#endif

    for (int64_t start = 0; start < n_train; start += batch_size) {
        int64_t end = std::min(start + batch_size, n_train);
        auto batch_idx = perm.slice(0, start, end);

        // The batch rows, already on the device when cached.
        LabelledPlots batch = src.index_select(batch_idx);
        if (!gpu_data_cached_) {
            batch = batch.to(config_.device);
        }

        // CUDA hash computation with async prefetching
#ifdef RESOLVE_HAS_CUDA
        if (use_cuda_hash_ && config_.device.is_cuda()) {
            torch::Tensor batch_hash;

            if (prefetch_ready) {
                // The previous iteration launched this batch's hash on the
                // prefetch stream; wait for it on the default stream.
                // prefetch_ready only ever becomes true under use_prefetch, so
                // the streams are engaged here.
                at::cuda::CUDAEvent prefetch_done;
                prefetch_done.record(*prefetch_stream);
                prefetch_done.block(*default_stream);

                batch_hash = prefetched_hash;
                prefetch_ready = false;
            } else {
                // First batch: compute synchronously. Hash by global plot index.
                batch_hash = cuda::compute_batch_hash_embedding_cuda(
                    hash_index_map.index_select(0, batch_idx),
                    hash_species_src,
                    hash_weights_src,
                    hash_offsets_src,
                    hash_dim_
                );
            }

            // Concatenate hash embedding with continuous features
            batch.inputs.continuous =
                torch::cat({batch.inputs.continuous, batch_hash}, /*dim=*/1);

            // Launch async prefetch for next batch if there is one
            int64_t next_start = start + batch_size;
            if (use_prefetch && next_start < n_train) {
                int64_t next_end = std::min(next_start + batch_size, n_train);
                auto next_batch_idx = perm.slice(0, next_start, next_end);

                // Record event on default stream, then wait for it on prefetch stream
                at::cuda::CUDAEvent compute_done;
                compute_done.record(*default_stream);
                compute_done.block(*prefetch_stream);

                // Compute next batch's hash on prefetch stream
                c10::cuda::CUDAStreamGuard guard(*prefetch_stream);
                prefetched_hash = cuda::compute_batch_hash_embedding_cuda(
                    hash_index_map.index_select(0, next_batch_idx),
                    hash_species_src,
                    hash_weights_src,
                    hash_offsets_src,
                    hash_dim_
                );
                prefetch_ready = true;
            }
        }
#endif

        // The rate of this batch's optimizer step: the epoch's schedule, the
        // per-step decay of LinearDecay and the warmup.
        set_learning_rate(*optimizer_,
                          step_learning_rate(config_, epoch, first_step + start / batch_size,
                                             epoch_steps, plateau));

        // Forward pass
        optimizer_->zero_grad();

        torch::Tensor loss;
        torch::Tensor batch_sums;

        {
            AutocastScope amp_scope(amp_enabled_);

            auto result = model_->forward_with_aux(batch.inputs);
            auto [loss_val, target_losses] =
                loss_fn_.compute(result.outputs, batch.targets, epoch, scalers);
            loss = loss_val;

            // The auxiliary terms join the objective as they come; a term the
            // model did not produce this batch counts 0.
            std::vector<torch::Tensor> aux_terms;
            aux_terms.reserve(aux_names.size());
            for (const auto& name : aux_names) {
                auto it = result.aux_losses.find(name);
                if (it != result.aux_losses.end() && it->second.defined()) {
                    loss = loss + it->second;
                    aux_terms.push_back(it->second.detach().to(sums.options()).reshape({1}));
                } else {
                    aux_terms.push_back(torch::zeros({1}, sums.options()));
                }
            }

            std::vector<torch::Tensor> parts{
                loss.detach().to(sums.options()).reshape({1}),
                loss_fn_.contributions(target_losses).to(sums.device())};
            parts.insert(parts.end(), aux_terms.begin(), aux_terms.end());
            batch_sums = torch::cat(parts);
        }  // autocast disabled here (also on exception unwind) before backward

        // A batch counts toward the epoch mean once its optimizer step runs.
        auto count_batch = [&]() {
            sums.add_(batch_sums);
            n_batches++;
        };

        // Backward pass with gradient scaling for AMP
        if (amp_enabled_) {
            // Scale loss before backward
            auto scaled_loss = loss * amp_scale_;
            scaled_loss.backward();

            // Collect defined gradients
            std::vector<torch::Tensor> grads;
            grads.reserve(model_->parameters().size());
            for (auto& param : model_->parameters()) {
                if (param.grad().defined()) {
                    grads.push_back(param.grad());
                }
            }

            // Fused inf/nan check + unscale in a single CUDA kernel, with one
            // host sync for the whole parameter set.
            auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(config_.device);
            auto found_inf_t = torch::zeros({1}, opts);
            auto inv_scale_t = torch::full({1}, 1.0f / amp_scale_, opts);
            at::_amp_foreach_non_finite_check_and_unscale_(grads, found_inf_t, inv_scale_t);
            bool found_inf = found_inf_t.item<float>() != 0.0f;

            if (found_inf) {
                // Skip this step, reduce scale
                amp_scale_ *= config_.amp_backoff_factor;
                amp_growth_tracker_ = 0;
                // Zero gradients since we're skipping this step
                optimizer_->zero_grad();
            } else {
                // Gradient clipping (on unscaled gradients)
                clip_grad_norm(model_->parameters(), kMaxGradNorm);

                // Optimizer step
                optimizer_->step();

                // Update scale
                amp_growth_tracker_++;
                if (amp_growth_tracker_ >= config_.amp_growth_interval) {
                    amp_scale_ *= config_.amp_growth_factor;
                    amp_growth_tracker_ = 0;
                }

                count_batch();
            }
        } else {
            // Standard backward pass without AMP
            loss.backward();

            // Gradient clipping
            clip_grad_norm(model_->parameters(), kMaxGradNorm);

            optimizer_->step();

            count_batch();
        }
    }

    // Every reported value is the mean over the counted batches, formed the
    // same way.
    auto epoch_mean = [n_batches](float sum) {
        return n_batches > 0 ? sum / n_batches : 0.0f;
    };
    const auto host_sums = sums.to(torch::kCPU);
    const auto s = host_sums.accessor<float, 1>();
    EpochLoss out;
    out.total = epoch_mean(s[0]);
    out.by_target.reserve(static_cast<size_t>(n_loss_targets));
    for (int64_t i = 0; i < n_loss_targets; ++i) {
        out.by_target.push_back(epoch_mean(s[i + 1]));
    }
    out.aux.reserve(static_cast<size_t>(n_aux));
    for (int64_t i = 0; i < n_aux; ++i) {
        out.aux.push_back(epoch_mean(s[1 + n_loss_targets + i]));
    }
    return out;
}

std::unordered_map<std::string, float> Trainer::held_out_metrics(
    const TargetConfig& cfg,
    const torch::Tensor& pred,
    const torch::Tensor& target,
    const TargetScalers& scalers
) const {
    // Regression metrics are reported in original units through the target's
    // scaler; a classification target has none, so its tensors stay undefined.
    torch::Tensor mean, scale;
    auto scaler_it = scalers.find(cfg.name);
    if (scaler_it != scalers.end()) {
        std::tie(mean, scale) = scaler_it->second;
    }
    // Scored on the held-out plots labelled for the target. A masked target
    // also reports how many those are, since it differs from target to
    // target; a fold with none of them reports only that.
    const auto rows = labelled_rows(cfg, target);
    std::unordered_map<std::string, float> metrics;
    if (rows.n_labelled > 0) {
        metrics = Metrics::compute(rows.select(pred), rows.select(target), cfg.task,
                                   cfg.transform, config_.band_thresholds,
                                   cfg.num_classes, mean, scale);
    }
    if (cfg.missing == TargetMissingPolicy::Mask) {
        metrics[kLabelledMetric] = static_cast<float>(rows.n_labelled);
    }
    return metrics;
}

std::pair<Trainer::EpochLoss, Trainer::MetricMap> Trainer::eval_epoch() {
    // eval_epoch runs inside the training loop, so it reads the GPU mirror
    // (and the GPU-resident CSR buffers for the hash) when the data is cached.
    const TensorMap predictions = forward_held_out(/*use_cache=*/gpu_data_cached_);
    const TensorMap test_targets =
        gpu_data_cached_ ? gpu_test_.targets : to_device(test_.targets, config_.device);
    const TargetScalers scalers = device_target_scalers();

    torch::NoGradGuard no_grad;

    // Evaluate the validation loss at a phase-INVARIANT reference epoch (the
    // last training epoch's phase) rather than the current epoch. The phased
    // loss adds strictly non-negative SMAPE/band terms as phases advance, so a
    // current-epoch eval loss jumps up at each phase boundary; comparing it
    // across epochs for best-model selection / early stopping would lock the
    // returned model to a phase-1 checkpoint and stop the curriculum early.
    // Scoring every epoch on the final-phase objective gives one consistent
    // scale, so best_loss is comparable across the whole run.
    const int sel_epoch = std::max(0, config_.max_epochs - 1);
    auto [loss, target_losses] = loss_fn_.compute(predictions, test_targets, sel_epoch, scalers);

    MetricMap all_metrics;
    for (const auto& cfg : model_->schema().targets) {
        auto pred_it = predictions.find(cfg.name);
        auto target_it = test_targets.find(cfg.name);
        if (pred_it == predictions.end() || target_it == test_targets.end()) continue;
        all_metrics[cfg.name] = held_out_metrics(cfg, pred_it->second, target_it->second, scalers);
    }

    // The held-out forward carries no MoE term, so the targets' terms make up
    // the whole selection loss.
    EpochLoss out;
    out.total = loss.item<float>();
    const auto terms = loss_fn_.contributions(target_losses).to(torch::kCPU);
    const auto t = terms.accessor<float, 1>();
    out.by_target.reserve(static_cast<size_t>(terms.size(0)));
    for (int64_t i = 0; i < terms.size(0); ++i) {
        out.by_target.push_back(t[i]);
    }
    return {std::move(out), std::move(all_metrics)};
}

void Trainer::cache_data_to_gpu() {
    if (gpu_data_cached_ || !config_.device.is_cuda()) {
        return;  // Already cached or not using CUDA
    }

    gpu_train_ = train_.to(config_.device);
    gpu_test_ = test_.to(config_.device);

    for (const auto& [name, scaler] : scalers_.target_scalers) {
        gpu_scalers_[name] = {
            scaler.first.to(config_.device),
            scaler.second.to(config_.device)
        };
    }

    if (use_cuda_hash_) {
        gpu_raw_species_ids_ = raw_species_ids_.to(config_.device);
        gpu_raw_weights_ = raw_weights_.to(config_.device);
        gpu_plot_offsets_ = plot_offsets_.to(config_.device);
        gpu_test_indices_ = test_indices_.to(config_.device);
    }

    gpu_data_cached_ = true;
}

void Trainer::release_training_state() {
    // Drop optimizer first so its parameter-state tensors (Adam m/v moments,
    // momentum buffers, etc.) are freed before we ask the allocator to
    // empty its cache.
    optimizer_.reset();

    // Reset AMP runtime state. The actual config knobs (init_scale,
    // growth_factor, etc.) live on config_ and are re-read on retry.
    amp_enabled_ = false;
    amp_scale_ = config_.amp_init_scale;
    amp_growth_tracker_ = 0;

    // Release every GPU-cached tensor; the host split stays, and the next
    // cache_data_to_gpu() call rebuilds the mirrors from it.
    gpu_data_cached_ = false;
    gpu_train_ = {};
    gpu_test_ = {};
    gpu_scalers_.clear();
    gpu_raw_species_ids_ = torch::Tensor();
    gpu_raw_weights_ = torch::Tensor();
    gpu_plot_offsets_ = torch::Tensor();
    gpu_test_indices_ = torch::Tensor();

    // Return free blocks to the device so the allocator's reserved pool
    // shrinks before the next attempt: without it the previous attempt's
    // reserved-but-unallocated blocks stay parked and a halved batch size
    // gains no headroom.
    release_cached_device_memory(config_.device);
}

// =============================================================================
// The run
// =============================================================================

TrainResult Trainer::fit() {
    return train(nullptr);
}

TrainResult Trainer::train(const RunProgress* resume_from) {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before training");
    }

    RunPlan plan;
    plan.started = std::chrono::steady_clock::now();
    plan.has_held_out = test_indices_.defined() && test_indices_.numel() > 0;
    plan.fixed_duration = config_.fixed_epochs > 0;
    plan.n_epochs = epochs_to_run(config_, plan.has_held_out);
    plan.use_checkpoints = !config_.checkpoint_dir.empty();

    // Apply CUDA performance optimizations and the VRAM cap BEFORE
    // cache_data_to_gpu(), so the allocator limit is in place before any
    // large device allocation. On Windows + WDDM, allocations beyond the
    // physical VRAM spill into shared system memory and hang the whole
    // desktop; the cap prevents that.
    if (config_.device.is_cuda()) {
        set_vram_fraction(
            static_cast<double>(config_.vram_fraction),
            config_.device.index(),
            config_.log
        );

        // cuDNN benchmark mode: auto-tunes algorithms for fixed input sizes
        // First batch is slower due to tuning, subsequent batches are faster
        at::globalContext().setBenchmarkCuDNN(config_.cudnn_benchmark);

        // TF32 (TensorFloat-32): faster matmuls on Ampere+ GPUs. Affects
        // float32 operations only, not float16/bfloat16.
        at::globalContext().setAllowTF32CuBLAS(config_.allow_tf32);
        at::globalContext().setAllowTF32CuDNN(config_.allow_tf32);
    }

    // The state the run starts from: nothing done yet, or the progress a
    // checkpoint carried.
    RunProgress start;
    if (resume_from != nullptr) {
        start = *resume_from;
    } else {
        start.created_at = iso8601_now();
        start.requested_batch_size = config_.batch_size;
        start.effective_batch_size = config_.batch_size;
        start.best_selected = selection_start(config_.selection_metric);
        start.plateau = plateau_start(config_);
    }
    plan.elapsed_before = start.elapsed_seconds;
    created_at_ = start.created_at;
    requested_batch_size_ = start.requested_batch_size;
    effective_batch_size_ = start.effective_batch_size;

    BatchSizeRestorer batch_size_restorer(config_.batch_size);
    config_.batch_size = effective_batch_size_;
    eval_chunk_ = std::max<int64_t>(config_.batch_size, kHeldOutChunk);

    if (plan.use_checkpoints) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }

    // If a CUDA OOM halves the batch size and forces a retry, training
    // restarts from `start` against these same weights -- restarting from a
    // partially-trained model under a different batch size mixes
    // optimizer-state assumptions and is fragile.
    const std::string start_weights = snapshot_weights();

    RunProgress progress;
    const auto attempt = [&](int n) {
        if (n > 0) {
            effective_batch_size_ = config_.batch_size;
            restore_weights(start_weights);
        }

        // Pre-cache all data on GPU for faster training (post-cap so it
        // respects the allocator fraction).
        cache_data_to_gpu();

        // Initialize AMP (only enabled on CUDA devices)
        amp_enabled_ = config_.use_amp && config_.device.is_cuda();
        amp_scale_ = start.has_amp_state ? start.amp_scale : config_.amp_init_scale;
        amp_growth_tracker_ = start.has_amp_state ? start.amp_growth_tracker : 0;
        if (amp_enabled_) {
            at::autocast::set_autocast_dtype(at::kCUDA, at::kHalf);
        }

        optimizer_ = std::make_unique<torch::optim::AdamW>(
            model_->parameters(),
            torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
        );
        if (!start.optimizer_state.empty()) {
            restore_optimizer(start.optimizer_state);
        }
        restore_rng_states(start);

        progress = start;
        run_epochs(progress, plan);
    };
    retry_halving_on_oom("Trainer::fit", config_.batch_size, config_.batch_size_floor,
                         requested_batch_size_, config_.log, attempt,
                         [this]() { release_training_state(); });

    TrainResult result = std::move(progress.result);
    result.resumed_from_epoch = start.next_epoch;

    // Restore the best epoch's weights. A fixed-duration run never snapshots
    // one, so it keeps the weights its last epoch left.
    if (!progress.best_weights.empty()) {
        restore_weights(progress.best_weights);
    }

    result.train_time_seconds =
        static_cast<float>(plan.elapsed_before + seconds_since(plan.started));

    // Complete the run metadata prepare_split started with the split, and save
    // the final checkpoint WITH it, so a saved checkpoint records its own
    // split, train time, best epoch, and reported metrics.
    run_metadata_.created_at = created_at_;
    run_metadata_.completed_at = iso8601_now();
    run_metadata_.train_time_seconds = result.train_time_seconds;
    run_metadata_.n_plots_train = train_indices_.defined() ? train_indices_.numel() : 0;
    run_metadata_.n_plots_test = test_indices_.defined() ? test_indices_.numel() : 0;
    run_metadata_.best_epoch = result.best_epoch;
    run_metadata_.total_epochs = static_cast<int>(result.train_loss_history.size());
    run_metadata_.final_metrics = result.final_metrics;
    if (plan.use_checkpoints) {
        save(config_.checkpoint_dir + "/checkpoint.pt", &run_metadata_);
    }

    // The batch size that actually trained the model, which config_ no longer
    // shows once batch_size_restorer has put the requested one back.
    result.effective_batch_size = effective_batch_size_;

    // Both the baseline comparisons and the network diagnostics score the
    // held-out fold; a fit prepared with nothing held out returns neither.
    if (!plan.has_held_out) {
        return result;
    }

    result.baselines = compute_baselines();
    result.diagnostics = compute_diagnostics();

    return result;
}

void Trainer::run_epochs(RunProgress& progress, const RunPlan& plan) {
    const auto& loss_targets = loss_fn_.targets();
    const std::vector<std::string> aux_names = model_->aux_loss_names();
    TrainResult& result = progress.result;

    // The phased loss activates later terms (SMAPE, band) only once their
    // phase begins, so early-stopping before the final phase would kill the
    // curriculum before those objectives ever train. Patience counts once
    // the objective is the one the last epoch trains on, which the loss
    // decides: from the start when its phases change nothing for these
    // targets, from the final phase otherwise.
    const int last_epoch = std::max(0, config_.max_epochs - 1);

    // A run resumed from the checkpoint of the epoch early stopping fired at
    // has nothing left to train.
    if (!plan.fixed_duration && progress.patience_counter > 0 &&
        progress.patience_counter >= config_.patience) {
        return;
    }

    // The step, cosine and linear schedules are laid out over max_epochs
    // whatever n_epochs is, so a fixed-duration run follows a full run's
    // trajectory.
    const int64_t epoch_steps = training_steps_per_epoch();
    const bool shows_lr = config_.lr_scheduler != LRSchedulerType::None ||
                          warmup_step_count(config_, epoch_steps) > 0;

    for (int epoch = progress.next_epoch; epoch < plan.n_epochs; ++epoch) {
        // The rate of the epoch's last step, the one lr_history records.
        const double current_lr =
            step_learning_rate(config_, epoch, (static_cast<int64_t>(epoch) + 1) * epoch_steps - 1,
                               epoch_steps, progress.plateau);

        const EpochLoss train = train_epoch(epoch, progress.plateau);
        const float train_loss = train.total;
        result.train_loss_history.push_back(train_loss);
        result.lr_history.push_back(static_cast<float>(current_lr));
        append_by_target(result.train_loss_history_by_target, loss_targets, train.by_target);
        for (size_t i = 0; i < aux_names.size(); ++i) {
            result.train_aux_loss_history[aux_names[i]].push_back(train.aux[i]);
        }

        float test_loss = std::numeric_limits<float>::quiet_NaN();
        EpochLoss test;
        MetricMap metrics;
        if (plan.has_held_out) {
            std::tie(test, metrics) = eval_epoch();
            test_loss = test.total;
            result.test_loss_history.push_back(test_loss);
            append_by_target(result.test_loss_history_by_target, loss_targets, test.by_target);
        }

        progress.next_epoch = epoch + 1;
        progress.elapsed_seconds = plan.elapsed_before + seconds_since(plan.started);

        // An epoch counts against patience and the plateau once the loss
        // objective has settled and the warmup is over.
        const bool settled = loss_fn_.objective_settled(epoch, last_epoch) &&
                             !epoch_in_warmup(config_, epoch, epoch_steps);
        bool stopping = false;

        // The value the epoch is selected by: the phase-invariant selection
        // loss (see eval_epoch), or the held-out accuracy eval_epoch scored,
        // either comparable across the whole run.
        const float selected =
            plan.has_held_out
                ? selected_value(config_.selection_metric, test_loss, metrics,
                                 model_->schema().targets)
                : std::numeric_limits<float>::quiet_NaN();
        if (plan.has_held_out) result.selection_history.push_back(selected);

        // A reduce-on-plateau schedule follows the value early stopping
        // reads, counting a bad epoch only where patience would, so the rate
        // the next epoch trains at is decided before any checkpoint of this
        // one.
        if (config_.lr_scheduler == LRSchedulerType::ReduceOnPlateau &&
            plateau_step(progress.plateau, config_, selected, settled)) {
            std::ostringstream msg;
            msg << "Epoch " << epoch << " - held-out " << enum_to_name(config_.selection_metric)
                << " on a plateau, learning rate " << progress.plateau.lr;
            config_.log(msg.str());
        }

        if (plan.fixed_duration) {
            // The returned weights are the last epoch's, whatever the
            // held-out value did, so best_epoch names that epoch and
            // final_metrics are its scores (none without a fold).
            result.best_epoch = epoch;
            result.final_metrics = std::move(metrics);
            if (plan.has_held_out) progress.best_selected = selected;
        } else if (improves_on(selected, progress.best_selected, /*threshold=*/0.0,
                               selection_maximizes(config_.selection_metric))) {
            // The returned model is the global best of the selected value.
            progress.best_selected = selected;
            result.best_epoch = epoch;
            result.final_metrics = std::move(metrics);
            progress.patience_counter = 0;
            progress.best_weights = snapshot_weights();

            if (plan.use_checkpoints) {
                save_checkpoint(config_.checkpoint_dir + "/best.pt", nullptr, &progress);
            }
        } else if (settled) {
            progress.patience_counter++;
            if (progress.patience_counter >= config_.patience) {
                config_.log("Early stopping at epoch " + std::to_string(epoch));
                stopping = true;
            }
        }

        if (plan.use_checkpoints && config_.checkpoint_every > 0 &&
            (epoch + 1) % config_.checkpoint_every == 0) {
            save_checkpoint(config_.checkpoint_dir + "/checkpoint_" +
                                std::to_string(epoch + 1) + ".pt",
                            nullptr, &progress);
        }

        if (plan.use_checkpoints) {
            write_progress_file(
                config_.checkpoint_dir, epoch, plan.n_epochs,
                result.best_epoch, held_out_loss_at(result, result.best_epoch),
                config_.selection_metric, progress.best_selected, progress.patience_counter,
                result.final_metrics
            );
        }

        // The stopping epoch still gets its periodic checkpoint and progress
        // file above, like every epoch that ran.
        if (stopping) break;

        if (epoch % 10 == 0) {
            std::ostringstream msg;
            msg << "Epoch " << epoch << " - Train: " << train_loss;
            if (plan.has_held_out) {
                msg << " Test: " << test_loss;
            }
            if (shows_lr) {
                msg << " LR: " << current_lr;
            }
            if (loss_targets.size() > 1) {
                msg << by_target_suffix(loss_targets, train.by_target,
                                        plan.has_held_out ? &test.by_target : nullptr,
                                        aux_names, train.aux);
            }
            config_.log(msg.str());
        }
    }
}

std::unordered_map<std::string, BaselineMetrics> Trainer::compute_baselines() {
    torch::NoGradGuard no_grad;

    const TensorMap predictions = forward_test_fold();
    const TargetScalers scalers = device_target_scalers();

    std::unordered_map<std::string, BaselineMetrics> out;
    for (const auto& cfg : model_->schema().targets) {
        auto pred_it = predictions.find(cfg.name);
        auto test_target_it = test_.targets.find(cfg.name);
        auto train_target_it = train_.targets.find(cfg.name);
        if (pred_it == predictions.end() || test_target_it == test_.targets.end() ||
            train_target_it == train_.targets.end()) {
            continue;
        }

        // reshape({-1}) keeps a single-plot fold one-dimensional.
        const torch::Device device = pred_it->second.device();
        const auto test_target = test_target_it->second.to(device).reshape({-1});
        const auto train_target = train_target_it->second.to(device).reshape({-1});

        // Both baselines and the model are scored on the test plots labelled
        // for the target, and the baseline is fit on the labelled training
        // plots. A target labelled on no test plot has no entry.
        const auto test_rows = labelled_rows(cfg, test_target);
        const auto train_rows = labelled_rows(cfg, train_target);
        if (test_rows.n_labelled == 0 || train_rows.n_labelled == 0) continue;

        const auto model_metrics =
            held_out_metrics(cfg, pred_it->second, test_target, scalers);
        const auto test_y = test_rows.select(test_target);
        const auto train_y = train_rows.select(train_target);

        BaselineMetrics baseline;
        if (cfg.task == TaskType::Regression) {
            torch::Tensor mean, scale;
            auto scaler_it = scalers.find(cfg.name);
            if (scaler_it != scalers.end()) {
                std::tie(mean, scale) = scaler_it->second;
            }
            const auto test_orig = to_original_units(test_y, cfg.transform, mean, scale);
            const auto train_orig = to_original_units(train_y, cfg.transform, mean, scale);
            auto pred = test_rows.select(pred_it->second).reshape({-1});
            const auto pred_orig = to_original_units(pred, cfg.transform, mean, scale);

            // The constant that minimises squared error in original units:
            // the training mean of the original values.
            const float train_mean = train_orig.mean().item<float>();
            const auto baseline_pred = torch::full_like(test_orig, train_mean);
            baseline.training_mean = train_mean;
            baseline.baseline_mse = Metrics::mse(baseline_pred, test_orig);
            baseline.baseline_mae = Metrics::mae(baseline_pred, test_orig);
            baseline.model_mse = Metrics::mse(pred_orig, test_orig);
            baseline.model_mae = model_metrics.at("mae");
            baseline.r_squared = model_metrics.at("r2");
            if (baseline.baseline_mse > kEpsilon) {
                baseline.skill_score = 1.0f - (baseline.model_mse / baseline.baseline_mse);
            }
        } else {
            const auto train_classes = train_y.to(torch::kLong);
            const int mode_class = std::get<0>(torch::mode(train_classes)).item<int>();
            baseline.training_mode = mode_class;
            baseline.baseline_accuracy =
                (test_y.to(torch::kLong) == mode_class).to(torch::kFloat32).mean().item<float>();
            baseline.model_accuracy = model_metrics.at(kAccuracyMetric);
            baseline.accuracy_lift = baseline.model_accuracy - baseline.baseline_accuracy;
        }

        out[cfg.name] = baseline;
    }
    return out;
}

// =============================================================================
// The held-out fold
// =============================================================================

std::vector<std::string> Trainer::select_plot_ids(const torch::Tensor& indices) const {
    std::vector<std::string> out;
    if (plot_ids_.empty() || !indices.defined() || indices.numel() == 0) {
        return out;
    }
    auto idx_cpu = indices.to(torch::kCPU).contiguous();
    auto acc = idx_cpu.accessor<int64_t, 1>();
    out.reserve(static_cast<size_t>(idx_cpu.size(0)));
    for (int64_t i = 0; i < idx_cpu.size(0); ++i) {
        const int64_t g = acc[i];
        if (g >= 0 && g < static_cast<int64_t>(plot_ids_.size())) {
            out.push_back(plot_ids_[static_cast<size_t>(g)]);
        }
    }
    return out;
}

std::vector<std::string> Trainer::train_plot_ids() const {
    return select_plot_ids(train_indices_);
}

std::vector<std::string> Trainer::test_plot_ids() const {
    return select_plot_ids(test_indices_);
}

torch::Tensor Trainer::append_cuda_hash(
    torch::Tensor continuous, torch::Tensor plot_idx, bool use_cache) const {
#ifdef RESOLVE_HAS_CUDA
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        torch::Tensor species_src = use_cache ? gpu_raw_species_ids_
                                              : raw_species_ids_.to(config_.device);
        torch::Tensor weights_src = use_cache ? gpu_raw_weights_
                                              : raw_weights_.to(config_.device);
        torch::Tensor offsets_src = use_cache ? gpu_plot_offsets_
                                              : plot_offsets_.to(config_.device);
        auto hash = cuda::compute_batch_hash_embedding_cuda(
            plot_idx.to(config_.device),
            species_src, weights_src, offsets_src, hash_dim_);
        return torch::cat({continuous, hash}, /*dim=*/1);
    }
#else
    (void)plot_idx;
    (void)use_cache;
#endif
    return continuous;
}

PlotInputs Trainer::held_out_chunk(int64_t start, int64_t end, bool use_cache) const {
    PlotInputs inputs = use_cache ? gpu_test_.inputs.slice(start, end)
                                  : test_.inputs.slice(start, end).to(config_.device);
    // CUDA hash mode computes the hash embedding on the fly; other encodings
    // already fold their species representation into the continuous block.
    if (use_cuda_hash_) {
        const torch::Tensor& indices = use_cache ? gpu_test_indices_ : test_indices_;
        inputs.continuous = append_cuda_hash(
            inputs.continuous, indices.slice(0, start, end), use_cache);
    }
    return inputs;
}

TensorMap Trainer::forward_held_out(bool use_cache) {
    model_->eval();
    torch::NoGradGuard no_grad;

    const int64_t n = test_.n_plots();
    const bool full_batch = model_->requires_full_batch_training();
    int64_t chunk = full_batch ? std::max<int64_t>(n, 1) : std::max<int64_t>(eval_chunk_, 1);

    std::unordered_map<std::string, std::vector<torch::Tensor>> parts;
    // At least one forward runs, so an empty fold still yields every output
    // with zero rows.
    int64_t start = 0;
    do {
        const int64_t end = std::min(start + chunk, n);
        try {
            const auto outputs = model_->forward(held_out_chunk(start, end, use_cache));
            for (const auto& [name, output] : outputs) parts[name].push_back(output);
            start = end;
        } catch (const c10::Error& error) {
            if (!is_out_of_memory(error)) {
                throw;
            }
            int64_t new_chunk = chunk;
            std::string log_msg;
            std::string err_msg;
            const bool retry = decide_eval_chunk_retry(
                chunk, full_batch, error.what(), new_chunk, log_msg, err_msg);
            release_cached_device_memory(config_.device);
            if (!retry) {
                throw std::runtime_error(err_msg);
            }
            config_.log(log_msg);
            chunk = new_chunk;
            eval_chunk_ = new_chunk;
        }
    } while (start < n);

    TensorMap out;
    for (auto& [name, pieces] : parts) {
        out[name] = pieces.size() == 1 ? pieces.front() : torch::cat(pieces, /*dim=*/0);
    }
    return out;
}

TensorMap Trainer::forward_test_fold() {
    return forward_held_out(/*use_cache=*/gpu_data_cached_);
}

const TargetConfig* Trainer::find_target(const std::string& name) const {
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.name == name) return &cfg;
    }
    return nullptr;
}

NetworkDiagnostics Trainer::compute_diagnostics() {
    NetworkDiagnostics diag;

    torch::NoGradGuard no_grad;
    model_->eval();

    // Use a subset of test data for diagnostics (max 10000 samples). Draw the
    // subset with a dedicated seeded generator so the diagnostics are
    // reproducible for a fixed seed and do not perturb the global RNG stream
    // mid-fit (matching the per-epoch shuffle).
    const int64_t n_test = test_.n_plots();
    int64_t n_samples = std::min(n_test, static_cast<int64_t>(10000));
    auto diag_gen = at::detail::createCPUGenerator(
        static_cast<uint64_t>(data_seed_) + 0x51ADu);
    auto sample_indices = torch::randperm(
        n_test, diag_gen,
        torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, n_samples);

    PlotInputs sample = test_.inputs.index_select(sample_indices).to(config_.device);
    // CUDA hash embedding for the sampled subset: map the test-local sample
    // indices to global plot indices, then reuse the shared hash helper.
    sample.continuous = append_cuda_hash(
        sample.continuous,
        test_indices_.index_select(0, sample_indices),
        /*use_cache=*/false);

    // The species encoder's tail reports each hidden layer's activations; the
    // tabular architectures and TraitNet report none.
    auto [latent, activations] = model_->encode_with_activations(sample);
    if (activations.empty()) {
        diag.summary = "Diagnostics not available for this encoder type.";
        return diag;
    }

    // Analyze each layer's activations
    std::stringstream issues;
    for (size_t i = 0; i < activations.size(); ++i) {
        LayerDiagnostics layer_diag;
        layer_diag.name = "hidden_" + std::to_string(i);

        auto act = activations[i].cpu();  // (batch, neurons)
        layer_diag.n_neurons = act.size(1);

        // Compute statistics per neuron (across batch)
        auto neuron_means = act.mean(0);  // (neurons,)
        auto neuron_stds = act.std(0);    // (neurons,)

        // Dead neurons: never activate (mean activation ~= 0)
        auto dead_mask = neuron_means.abs() < 1e-6f;
        layer_diag.n_dead = dead_mask.sum().item<int64_t>();
        layer_diag.dead_fraction = static_cast<float>(layer_diag.n_dead) / layer_diag.n_neurons;

        // Saturated neurons: always at high activation (mean > 5.0 for GELU)
        auto saturated_mask = neuron_means > 5.0f;
        layer_diag.n_saturated = saturated_mask.sum().item<int64_t>();
        layer_diag.saturated_fraction = static_cast<float>(layer_diag.n_saturated) / layer_diag.n_neurons;

        // Overall statistics
        layer_diag.mean_activation = neuron_means.mean().item<float>();
        layer_diag.std_activation = neuron_stds.mean().item<float>();

        // Sparsity: fraction of zero activations
        layer_diag.sparsity = (act < 1e-6f).sum().item<float>() / static_cast<float>(act.numel());

        // Accumulate totals
        diag.total_neurons += layer_diag.n_neurons;
        diag.total_dead += layer_diag.n_dead;
        diag.total_saturated += layer_diag.n_saturated;

        // Check for issues
        if (layer_diag.dead_fraction > 0.1f) {
            issues << "Layer " << layer_diag.name << ": "
                   << static_cast<int>(layer_diag.dead_fraction * 100) << "% dead neurons. ";
            diag.has_issues = true;
        }
        if (layer_diag.saturated_fraction > 0.1f) {
            issues << "Layer " << layer_diag.name << ": "
                   << static_cast<int>(layer_diag.saturated_fraction * 100) << "% saturated. ";
            diag.has_issues = true;
        }

        diag.layers.push_back(layer_diag);
    }

    // Compute overall fractions
    if (diag.total_neurons > 0) {
        diag.overall_dead_fraction = static_cast<float>(diag.total_dead) / diag.total_neurons;
        diag.overall_saturated_fraction = static_cast<float>(diag.total_saturated) / diag.total_neurons;
    }

    // Build summary
    if (diag.has_issues) {
        diag.summary = issues.str();
    } else {
        diag.summary = "Network health OK: no significant dead or saturated neurons detected.";
    }

    return diag;
}

CalibrationResult Trainer::compute_calibration(
    const std::string& target_name,
    int n_bins
) {
    if (!data_prepared_) {
        throw std::runtime_error(
            "compute_calibration requires prepare_data() first");
    }

    CalibrationResult result;
    result.target_name = target_name;

    const TargetConfig* target_cfg = find_target(target_name);
    if (!target_cfg || target_cfg->task != TaskType::Classification) {
        return result;
    }

    torch::NoGradGuard no_grad;
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_.targets.find(target_name);
    if (pred_it == predictions.end() || target_it == test_.targets.end()) {
        return result;
    }

    // Get predicted probabilities (softmax) on the test plots labelled for the
    // target.
    const auto rows = labelled_rows(*target_cfg, target_it->second);
    auto logits = rows.select(pred_it->second);  // (n_samples, n_classes)
    auto probs = torch::softmax(logits, /*dim=*/1).cpu();
    auto targets = rows.select(target_it->second).to(torch::kLong).cpu();  // (n_samples,)

    int64_t n_samples = probs.size(0);

    // Top-label calibration: each plot's highest class probability against
    // whether that class is the true one, binned by the probability.
    auto max_probs = std::get<0>(probs.max(1));  // (n_samples,)
    auto pred_classes = probs.argmax(1);         // (n_samples,)
    auto correct = (pred_classes == targets);    // (n_samples,)

    // Create bins
    result.bins.resize(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        result.bins[i].bin_start = static_cast<float>(i) / n_bins;
        result.bins[i].bin_end = static_cast<float>(i + 1) / n_bins;
    }

    // Assign samples to bins
    auto max_probs_a = max_probs.accessor<float, 1>();
    auto correct_a = correct.accessor<bool, 1>();

    for (int64_t s = 0; s < n_samples; ++s) {
        float p = max_probs_a[s];
        int bin_idx = std::min(static_cast<int>(p * n_bins), n_bins - 1);

        result.bins[bin_idx].count++;
        result.bins[bin_idx].mean_predicted_prob += p;
        if (correct_a[s]) {
            result.bins[bin_idx].actual_frequency += 1.0f;
        }
    }

    // Finalize bin statistics
    float ece = 0.0f;
    float mce = 0.0f;
    for (auto& bin : result.bins) {
        if (bin.count > 0) {
            bin.mean_predicted_prob /= bin.count;
            bin.actual_frequency /= bin.count;

            float bin_error = std::abs(bin.mean_predicted_prob - bin.actual_frequency);
            ece += bin_error * (static_cast<float>(bin.count) / n_samples);
            mce = std::max(mce, bin_error);
        }
    }

    result.expected_calibration_error = ece;
    result.max_calibration_error = mce;

    return result;
}

ResidualAnalysis Trainer::compute_residuals(
    const std::string& target_name
) {
    if (!data_prepared_) {
        throw std::runtime_error(
            "compute_residuals requires prepare_data() first");
    }

    ResidualAnalysis result;
    result.target_name = target_name;

    const TargetConfig* target_cfg = find_target(target_name);
    if (!target_cfg || target_cfg->task != TaskType::Regression) {
        return result;
    }

    torch::NoGradGuard no_grad;
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_.targets.find(target_name);
    if (pred_it == predictions.end() || target_it == test_.targets.end()) {
        return result;
    }

    // reshape({-1}), not squeeze(): a single-plot test fold has a (1,1) output
    // that squeeze() would collapse to a 0-D scalar. Only the test plots
    // labelled for the target have a residual.
    const auto rows = labelled_rows(*target_cfg, target_it->second);
    auto preds = rows.select(pred_it->second).reshape({-1}).cpu();
    auto targets = rows.select(target_it->second).reshape({-1}).cpu();
    {
        const auto positions = rows.positions();
        result.test_rows.assign(positions.data_ptr<int64_t>(),
                                positions.data_ptr<int64_t>() + positions.numel());
    }

    // Residuals in original units.
    torch::Tensor mean, scale;
    auto scaler_it = scalers_.target_scalers.find(target_name);
    if (scaler_it != scalers_.target_scalers.end()) {
        std::tie(mean, scale) = scaler_it->second;
    }
    preds = to_original_units(preds, target_cfg->transform, mean, scale);
    targets = to_original_units(targets, target_cfg->transform, mean, scale);

    // Compute residuals
    auto residuals = targets - preds;

    // Convert to vectors
    int64_t n = residuals.size(0);
    if (n == 0) return result;
    result.predictions.resize(n);
    result.actuals.resize(n);
    result.residuals.resize(n);

    auto preds_a = preds.accessor<float, 1>();
    auto targets_a = targets.accessor<float, 1>();
    auto residuals_a = residuals.accessor<float, 1>();

    for (int64_t i = 0; i < n; ++i) {
        result.predictions[i] = preds_a[i];
        result.actuals[i] = targets_a[i];
        result.residuals[i] = residuals_a[i];
    }

    // Compute summary statistics
    result.mean_residual = residuals.mean().item<float>();
    result.std_residual = residuals.std().item<float>();

    // Compute skewness and kurtosis
    auto centered = residuals - result.mean_residual;
    auto m2 = torch::mean(centered.pow(2)).item<float>();
    auto m3 = torch::mean(centered.pow(3)).item<float>();
    auto m4 = torch::mean(centered.pow(4)).item<float>();

    if (m2 > kEpsilon) {
        float std3 = std::pow(m2, 1.5f);
        float std4 = m2 * m2;
        result.skewness = m3 / std3;
        result.kurtosis = m4 / std4 - 3.0f;  // Excess kurtosis
    }

    // Compute quantiles
    auto sorted = std::get<0>(residuals.sort());
    auto get_quantile = [&](float q) {
        int64_t idx = static_cast<int64_t>(q * (n - 1));
        return sorted[idx].item<float>();
    };

    result.q05 = get_quantile(0.05f);
    result.q25 = get_quantile(0.25f);
    result.q50 = get_quantile(0.50f);
    result.q75 = get_quantile(0.75f);
    result.q95 = get_quantile(0.95f);

    return result;
}

ClassificationPredictions Trainer::compute_classification_predictions(
    const std::string& target_name
) {
    ClassificationPredictions result;
    result.target_name = target_name;

    if (!data_prepared_) {
        throw std::runtime_error(
            "compute_classification_predictions requires prepare_data() first");
    }

    // Non-matching or regression targets return empty tensors (parallels
    // compute_residuals returning empty for classification targets).
    const TargetConfig* target_cfg = find_target(target_name);
    if (!target_cfg || target_cfg->task != TaskType::Classification) {
        return result;
    }
    result.class_names = target_cfg->class_names;

    torch::NoGradGuard no_grad;
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_.targets.find(target_name);
    if (pred_it == predictions.end() || target_it == test_.targets.end()) {
        return result;
    }

    // The test plots labelled for the target, with their place in the fold.
    const auto rows = labelled_rows(*target_cfg, target_it->second);
    auto logits = rows.select(pred_it->second);    // (n_labelled, n_classes)
    auto probs = torch::softmax(logits, /*dim=*/1);
    auto predicted = probs.argmax(/*dim=*/1);      // (n_labelled,)

    result.probabilities = probs.detach().to(torch::kCPU).contiguous();
    result.predicted_classes =
        predicted.detach().to(torch::kCPU).contiguous().to(torch::kLong);
    result.actuals = rows.select(target_it->second)
        .detach().to(torch::kCPU).contiguous().to(torch::kLong);
    result.test_rows = rows.positions();

    return result;
}

// =============================================================================
// Cross-validation
// =============================================================================

Trainer::SplitState Trainer::capture_split_state() const {
    return {train_, test_, train_indices_, test_indices_, scalers_, run_metadata_,
            created_at_, requested_batch_size_, effective_batch_size_};
}

void Trainer::restore_split_state(const SplitState& s) {
    release_training_state();
    train_ = s.train;
    test_ = s.test;
    train_indices_ = s.train_indices;
    test_indices_ = s.test_indices;
    scalers_ = s.scalers;
    run_metadata_ = s.run_metadata;
    created_at_ = s.created_at;
    requested_batch_size_ = s.requested_batch_size;
    effective_batch_size_ = s.effective_batch_size;
}

CrossValidationResult Trainer::run_cross_validation(
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& folds
) {
    const auto start_time = std::chrono::steady_clock::now();

    CrossValidationResult cv_result;
    cv_result.n_folds = static_cast<int>(folds.size());

    // The trainer's own weights, split and run record, restored after the
    // folds.
    const std::string original_weights = snapshot_weights();
    const SplitState split_snapshot = capture_split_state();

    // Don't let per-fold fit() calls clobber the user's on-disk checkpoint.
    CheckpointDirSilencer ckpt_silencer(config_.checkpoint_dir);

    // The prepared rows in raw values, so each fold fits its own scalers on
    // its own fitting rows. Row i of `all` is plot all_global_idx[i], which
    // each fold records as its train / test indices.
    const LabelledPlots all = unstandardized(LabelledPlots::cat(train_, test_), scalers_);
    const torch::Tensor all_global_idx = torch::cat({train_indices_, test_indices_}, 0);

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> metric_values;

    for (size_t fold = 0; fold < folds.size(); ++fold) {
        config_.log("Cross-validation fold " + std::to_string(fold + 1) + "/" +
                    std::to_string(folds.size()));

        // Reset this fold to the as-constructed (untrained) weights, not the
        // trainer's current weights. If cross_validate is run after fit(), the
        // current weights are already trained on rows that fall in this fold's
        // held-out test set; warm-starting from them yields optimistically
        // biased CV metrics.
        restore_weights(pristine_model_state_);

        const auto& train_index = folds[fold].first;
        const auto& test_index = folds[fold].second;
        config_.log("  Train: " + std::to_string(train_index.size()) +
                    " plots, Test: " + std::to_string(test_index.size()) + " plots");

        apply_split(all, torch::tensor(train_index, torch::kInt64),
                    torch::tensor(test_index, torch::kInt64), all_global_idx);

        TrainResult fold_result = fit();
        cv_result.fold_results.push_back(fold_result);

        for (const auto& [target, metrics] : fold_result.final_metrics) {
            for (const auto& [metric_name, value] : metrics) {
                metric_values[target][metric_name].push_back(value);
            }
        }
    }

    // Mean and std of each metric across folds.
    for (const auto& [target, metrics] : metric_values) {
        for (const auto& [metric_name, values] : metrics) {
            float sum = 0.0f;
            for (float v : values) sum += v;
            float mean = sum / values.size();
            cv_result.mean_metrics[target][metric_name] = mean;

            float sq_sum = 0.0f;
            for (float v : values) sq_sum += (v - mean) * (v - mean);
            cv_result.std_metrics[target][metric_name] = std::sqrt(sq_sum / values.size());
        }
    }

    restore_weights(original_weights);
    restore_split_state(split_snapshot);

    cv_result.total_time_seconds = static_cast<float>(seconds_since(start_time));
    return cv_result;
}

CrossValidationResult Trainer::cross_validate(int n_folds, int seed) {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before cross-validation");
    }

    const int64_t n_total = train_.n_plots() + test_.n_plots();

    // Shuffled row indices into the concatenated train++test rows, partitioned
    // into n_folds contiguous test blocks (the remaining rows are the train set).
    std::vector<int64_t> indices(n_total);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    const int64_t fold_size = n_total / n_folds;
    const int64_t remainder = n_total % n_folds;

    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> folds;
    folds.reserve(n_folds);
    int64_t fold_start = 0;
    for (int fold = 0; fold < n_folds; ++fold) {
        const int64_t current = fold_size + (fold < remainder ? 1 : 0);
        const int64_t fold_end = fold_start + current;
        std::vector<int64_t> test_index(indices.begin() + fold_start, indices.begin() + fold_end);
        std::vector<int64_t> train_index;
        train_index.reserve(n_total - current);
        for (int64_t i = 0; i < fold_start; ++i) train_index.push_back(indices[i]);
        for (int64_t i = fold_end; i < n_total; ++i) train_index.push_back(indices[i]);
        folds.emplace_back(std::move(train_index), std::move(test_index));
        fold_start = fold_end;
    }

    auto cv_result = run_cross_validation(folds);

    std::ostringstream summary;
    summary << "Cross-validation complete (" << n_folds << " folds, "
            << cv_result.total_time_seconds << "s)\n";
    for (const auto& [target, metrics] : cv_result.mean_metrics) {
        summary << "  " << target << ": ";
        for (const auto& [name, mean] : metrics) {
            auto std_it = cv_result.std_metrics[target].find(name);
            float std = (std_it != cv_result.std_metrics[target].end()) ? std_it->second : 0.0f;
            summary << name << "=" << mean << "+/-" << std << " ";
        }
        summary << "\n";
    }
    config_.log(summary.str());

    return cv_result;
}

// =============================================================================
// Spatial Block Splitter
// =============================================================================

SpatialBlockSplitter::SpatialBlockSplitter(
    float lat_size, float lon_size, int n_splits, int seed, bool balance
) : lat_size_(lat_size), lon_size_(lon_size), n_splits_(n_splits),
    seed_(seed), balance_(balance)
{
    if (n_splits < 2) {
        throw std::invalid_argument("n_splits must be >= 2, got " + std::to_string(n_splits));
    }
    if (lat_size <= 0 || lon_size <= 0) {
        throw std::invalid_argument("Block sizes must be positive");
    }
}

std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
SpatialBlockSplitter::split(torch::Tensor coords) const {
    int64_t n = coords.size(0);
    auto coords_cpu = coords.cpu().to(torch::kFloat64);
    // A plot without coordinates cannot be placed in a block. It trains in
    // every fold and is never held out, so each test fold holds only plots
    // whose block is withheld.
    auto placed = (torch::isfinite(coords_cpu.select(1, 0)) &
                   torch::isfinite(coords_cpu.select(1, 1))).contiguous();
    auto placed_ptr = placed.data_ptr<bool>();
    coords_cpu = torch::nan_to_num(coords_cpu, 0.0, 0.0, 0.0);
    auto lat = coords_cpu.select(1, 0);
    auto lon = coords_cpu.select(1, 1);

    // Grid hash: block_row = floor(lat / lat_size), block_col = floor(lon / lon_size)
    auto block_row = torch::floor(lat / lat_size_).to(torch::kInt64);
    auto block_col = torch::floor(lon / lon_size_).to(torch::kInt64);

    // Linear block hash: label = row * 1e6 + col. Injective only while
    // |block_col| < 5e5, which holds for degree-scale grids (lon in [-180, 180]);
    // a sub-4e-4 deg lon_size could alias distinct blocks into one fold.
    auto block_labels = block_row * 1000000 + block_col;
    // An unplaced plot borrows the label of a placed one so it adds no block of
    // its own; it never counts toward a block's size or reaches a test fold.
    if (!placed.any().item<bool>()) {
        throw std::invalid_argument(
            "spatial CV needs coordinates, but no plot carries a finite pair");
    }
    block_labels = torch::where(placed, block_labels,
                                block_labels.masked_select(placed)[0]);

    // Find unique blocks
    auto unique_result = torch::_unique2(block_labels, /*sorted=*/true, /*return_inverse=*/true);
    auto unique_blocks = std::get<0>(unique_result);
    auto inverse_indices = std::get<1>(unique_result);
    int64_t n_blocks = unique_blocks.size(0);

    // Every fold needs at least one spatial block or its test set is empty, which
    // then divides by zero in the baseline metrics and NaNs the whole fold.
    // Round-robin and greedy bin-packing both give each fold >= 1 block once
    // n_blocks >= n_splits, so fail loudly below that rather than silently emit an
    // empty fold. Coarse lat_size/lon_size on geographically clustered data is the
    // usual cause (e.g. all plots in one grid cell -> 1 block).
    if (n_blocks < n_splits_) {
        throw std::invalid_argument(
            std::to_string(n_splits_) + "-fold spatial CV needs at least " +
            std::to_string(n_splits_) + " spatial blocks, but the grid produced only " +
            std::to_string(n_blocks) + " (lat_size=" + std::to_string(lat_size_) +
            ", lon_size=" + std::to_string(lon_size_) +
            "). Use a finer block size or fewer folds.");
    }

    // Count block sizes over the placed plots.
    auto inverse_cpu = inverse_indices.cpu();
    auto inv_ptr = inverse_cpu.data_ptr<int64_t>();
    std::vector<int64_t> block_sizes(n_blocks, 0);
    for (int64_t i = 0; i < n; ++i) {
        if (placed_ptr[i]) block_sizes[inv_ptr[i]]++;
    }

    // Shuffle block order
    std::vector<int64_t> block_order(n_blocks);
    std::iota(block_order.begin(), block_order.end(), 0);
    std::mt19937 rng(seed_);
    std::shuffle(block_order.begin(), block_order.end(), rng);

    // Assign blocks to folds
    std::vector<int> block_to_fold(n_blocks);
    if (balance_) {
        // Greedy bin-packing: sort by size (largest first), assign to fold with fewest plots
        std::vector<int64_t> sorted_order = block_order;
        std::sort(sorted_order.begin(), sorted_order.end(),
            [&](int64_t a, int64_t b) { return block_sizes[a] > block_sizes[b]; });

        std::vector<int64_t> fold_totals(n_splits_, 0);
        for (int64_t block_idx : sorted_order) {
            int best_fold = 0;
            for (int f = 1; f < n_splits_; ++f) {
                if (fold_totals[f] < fold_totals[best_fold]) best_fold = f;
            }
            block_to_fold[block_idx] = best_fold;
            fold_totals[best_fold] += block_sizes[block_idx];
        }
    } else {
        // Round-robin
        for (int64_t i = 0; i < n_blocks; ++i) {
            block_to_fold[block_order[i]] = static_cast<int>(i % n_splits_);
        }
    }

    // Build train/test index arrays per fold
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> folds(n_splits_);
    for (int f = 0; f < n_splits_; ++f) {
        std::vector<int64_t> train_idx, test_idx;
        for (int64_t i = 0; i < n; ++i) {
            if (placed_ptr[i] && block_to_fold[inv_ptr[i]] == f) {
                test_idx.push_back(i);
            } else {
                train_idx.push_back(i);
            }
        }
        folds[f] = {std::move(train_idx), std::move(test_idx)};
    }

    // No fold reaches the caller with an empty test or train set. n_blocks >=
    // n_splits above makes this hold for round-robin and bin-packing; it is
    // asserted regardless.
    for (int f = 0; f < n_splits_; ++f) {
        if (folds[f].second.empty() || folds[f].first.empty()) {
            throw std::runtime_error(
                "spatial CV produced an empty " +
                std::string(folds[f].second.empty() ? "test" : "train") +
                " fold (fold " + std::to_string(f) + " of " +
                std::to_string(n_splits_) + "); check block size vs fold count.");
        }
    }

    return folds;
}

CrossValidationResult Trainer::cross_validate_spatial(
    const SpatialBlockConfig& spatial_config,
    int n_folds,
    int seed
) {
    if (!data_prepared_) {
        throw std::runtime_error("Must call prepare_data() before cross_validate_spatial()");
    }
    if (!coordinates_.defined() || coordinates_.numel() == 0) {
        throw std::runtime_error("Spatial CV requires coordinates in the dataset");
    }

    SpatialBlockSplitter splitter(
        spatial_config.lat_size, spatial_config.lon_size,
        n_folds, seed, spatial_config.balance);

    config_.log("Spatial block CV: " + std::to_string(n_folds) + " folds, "
                "block_size=" + std::to_string(spatial_config.lat_size) + "x"
                + std::to_string(spatial_config.lon_size) + " deg");

    // coordinates_ is stored in ORIGINAL plot order (cloned in prepare_data
    // before the shuffle), but run_cross_validation applies the fold indices to
    // the concatenated train++test rows, which are in shuffled split order.
    // Reorder the coordinates into that same order so the geographic block for
    // fold index i lands on the plot whose coordinates produced it; otherwise
    // the block assignment is paired with a different plot's features and the
    // spatial split degenerates into a geographically-scrambled split.
    const auto all_global_idx = torch::cat({train_indices_, test_indices_}, 0);
    const auto split_order_coords =
        coordinates_.index_select(0, all_global_idx.to(coordinates_.device()));

    auto folds = splitter.split(split_order_coords);

    auto cv_result = run_cross_validation(folds);

    config_.log("Spatial CV complete (" + std::to_string(n_folds) + " folds, "
                + std::to_string(cv_result.total_time_seconds) + "s)");

    return cv_result;
}

// =============================================================================
// Predict (eval-mode inference)
// =============================================================================

TensorMap Trainer::predict(const PlotInputs& inputs) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Standardize the continuous block with the fitted scalers, exactly as
    // Predictor::predict does. Callers pass raw features; the model was trained
    // on standardized inputs, so skipping this silently biases every prediction.
    PlotInputs scaled = inputs;
    if (scaled.continuous.defined() && scaled.continuous.size(1) > 0) {
        scaled.continuous = standardize_continuous(scaled.continuous, scalers_);
    }

    auto outputs = model_->forward(scaled.to(config_.device));

    // Regression outputs in original units, as Predictor::predict returns
    // them. Classification outputs stay raw logits (this map is the low-level
    // surface; argmax / softmax is the caller's choice).
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.task != TaskType::Regression) continue;
        auto it = outputs.find(cfg.name);
        if (it == outputs.end()) continue;

        torch::Tensor mean, scale;
        auto scaler_it = scalers_.target_scalers.find(cfg.name);
        if (scaler_it != scalers_.target_scalers.end()) {
            std::tie(mean, scale) = scaler_it->second;
        }
        it->second = to_original_units(it->second.squeeze(-1), cfg.transform, mean, scale);
    }

    return outputs;
}

} // namespace resolve
