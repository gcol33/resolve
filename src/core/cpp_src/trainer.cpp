// Define _USE_MATH_DEFINES before cmath for M_PI on Windows
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

// Fallback definition of M_PI if still not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "resolve/trainer.hpp"
#include "resolve/dataset.hpp"
#include "resolve/utils.hpp"
#include "resolve/checkpoint.hpp"
#include "resolve/gpu.hpp"
#include "resolve/io_retry.hpp"

#ifdef RESOLVE_HAS_CUDA
#include "resolve/cuda/feature_hash.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <c10/util/Exception.h>
#include <ATen/autocast_mode.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <random>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <ctime>

namespace resolve {

namespace {

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
// hash kernels. The kernels (cuda/kernels.cu) guard against malformed ranges
// per thread but only bump a device-side error flag that no host code reads,
// so a corrupt CSR would silently drop species records. The offsets are fixed
// for the whole run and live on the host at capture time, so validate them once
// here (no per-batch sync) and fail loudly on a malformed CSR instead.
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

}  // namespace

Trainer::Trainer(
    ResolveModel model,
    const TrainConfig& config
) : model_(model), config_(config), loss_fn_(model->schema().targets, config.phase_boundaries, config.loss_config)
{
    model_->to(config_.device);

    // Snapshot the as-constructed weights so cross_validate can reset each fold
    // to a fresh init even when run after fit() (issue #97).
    std::ostringstream pristine_stream;
    {
        torch::serialize::OutputArchive archive;
        model_->save(archive);
        archive.save_to(pristine_stream);
    }
    pristine_model_state_ = pristine_stream.str();
}

void Trainer::prepare_data(
    const ResolveDataset& dataset,
    float test_size,
    int seed
) {
    // Check if dataset has raw species data for CUDA hash computation
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
    // plots landed in each split. The raw-tensor prepare_data overload below
    // has no plot IDs, so this is the only place plot_ids_ is populated.
    plot_ids_ = dataset.plot_ids();

    // Delegate to the raw tensor API using data from the dataset
    prepare_data(
        dataset.coordinates(),
        dataset.covariates(),
        dataset.hash_embedding(),
        dataset.species_ids(),
        dataset.species_vector(),
        dataset.genus_ids(),
        dataset.family_ids(),
        dataset.unknown_fraction(),
        dataset.unknown_count(),
        dataset.targets(),
        dataset.pool_genus_ids(),
        dataset.pool_family_ids(),
        dataset.pool_weights(),
        dataset.pool_mask(),
        dataset.pool_has_cover(),
        dataset.categorical_ids(),
        test_size,
        seed
    );
}

void Trainer::prepare_data(
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
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids,
    float test_size,
    int seed
) {
    // Capture the seed so the per-epoch training shuffle is reproducible
    // (the dataset overload forwards its seed here, so both paths set it).
    data_seed_ = seed;

    // Store raw coordinates for spatial CV (before scaling/concatenation)
    if (coordinates.defined() && coordinates.numel() > 0) {
        coordinates_ = coordinates.clone();
    }

    // Determine n_plots from first defined tensor
    int64_t n_plots = 0;
    if (coordinates.defined() && coordinates.numel() > 0) {
        n_plots = coordinates.size(0);
    } else if (hash_embedding.defined() && hash_embedding.numel() > 0) {
        n_plots = hash_embedding.size(0);
    } else if (species_ids.defined() && species_ids.numel() > 0) {
        n_plots = species_ids.size(0);
    } else if (species_vector.defined() && species_vector.numel() > 0) {
        n_plots = species_vector.size(0);
    } else {
        throw std::runtime_error("No valid input tensors provided");
    }

    // Create indices and shuffle
    std::vector<int64_t> indices(n_plots);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Split indices
    int64_t n_test = static_cast<int64_t>(n_plots * test_size);
    int64_t n_train = n_plots - n_test;

    // An empty test fold makes every held-out metric divide by zero (accuracy,
    // baseline, residuals) and returns NaN rather than an error. Fail loudly so
    // a too-small dataset / test_size is caught here, not as silent NaNs
    // downstream (issue #74).
    if (test_size > 0.0f && n_test <= 0) {
        throw std::runtime_error(
            "prepare_data: test_size=" + std::to_string(test_size) +
            " yields an empty test fold for " + std::to_string(n_plots) +
            " plots; increase test_size or provide more data");
    }

    auto train_idx = torch::tensor(std::vector<int64_t>(indices.begin(), indices.begin() + n_train));
    auto test_idx = torch::tensor(std::vector<int64_t>(indices.begin() + n_train, indices.end()));

    // Store indices for CUDA hash computation
    train_indices_ = train_idx.clone();
    test_indices_ = test_idx.clone();

    // Build continuous features based on encoding mode
    std::vector<torch::Tensor> continuous_parts;
    push_if_defined(continuous_parts, coordinates);
    push_if_defined(continuous_parts, covariates);
    push_if_defined(continuous_parts, unknown_fraction, 1);
    if (unknown_count.defined() && unknown_count.numel() > 0) {
        continuous_parts.push_back(unknown_count.to(torch::kFloat32).unsqueeze(1));
    }

    // For hash mode, include hash embedding in continuous.
    // Why: DatasetConfig.hash_dim and ModelConfig.hash_dim are independent.
    // If they disagree, the model's first linear layer is sized for one
    // value while the trainer concatenates a hash embedding of the other
    // width, and fit() blows up later with an opaque matmul shape error.
    // Validate at prepare_data so the failure points at the actual mismatch.
    if (model_->species_encoding() == SpeciesEncodingMode::Hash &&
        !model_->uses_explicit_vector()) {
        if (hash_embedding.defined() && hash_embedding.numel() > 0) {
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
        push_if_defined(continuous_parts, hash_embedding);
    }

    torch::Tensor continuous;
    if (!continuous_parts.empty()) {
        continuous = torch::cat(continuous_parts, /*dim=*/1);
    } else {
        continuous = torch::zeros({n_plots, 0}, torch::kFloat32);
    }

    // Compute scalers on training data
    auto train_continuous = continuous.index_select(0, train_idx);
    if (train_continuous.size(1) > 0) {
        scalers_.continuous_mean = train_continuous.mean(0);
        scalers_.continuous_scale = train_continuous.std(0) + 1e-8f;

        // Scale continuous features
        continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    }

    // Split data
    train_continuous_ = continuous.index_select(0, train_idx);
    test_continuous_ = continuous.index_select(0, test_idx);

    if (genus_ids.defined() && genus_ids.numel() > 0) {
        train_genus_ids_ = genus_ids.index_select(0, train_idx);
        test_genus_ids_ = genus_ids.index_select(0, test_idx);
    }
    if (family_ids.defined() && family_ids.numel() > 0) {
        train_family_ids_ = family_ids.index_select(0, train_idx);
        test_family_ids_ = family_ids.index_select(0, test_idx);
    }
    if (species_ids.defined() && species_ids.numel() > 0) {
        train_species_ids_ = species_ids.index_select(0, train_idx);
        test_species_ids_ = species_ids.index_select(0, test_idx);
    }
    if (species_vector.defined() && species_vector.numel() > 0) {
        train_species_vector_ = species_vector.index_select(0, train_idx);
        test_species_vector_ = species_vector.index_select(0, test_idx);
    }

    // Split pool fields (for rank_pool / transformer modes)
    auto split_pool = [&](const torch::Tensor& t, torch::Tensor& train_dst, torch::Tensor& test_dst) {
        if (t.defined() && t.numel() > 0) {
            train_dst = t.index_select(0, train_idx);
            test_dst = t.index_select(0, test_idx);
        }
    };
    split_pool(pool_genus_ids, train_pool_genus_ids_, test_pool_genus_ids_);
    split_pool(pool_family_ids, train_pool_family_ids_, test_pool_family_ids_);
    split_pool(pool_weights, train_pool_weights_, test_pool_weights_);
    split_pool(pool_mask, train_pool_mask_, test_pool_mask_);
    split_pool(pool_has_cover, train_pool_has_cover_, test_pool_has_cover_);

    // ---- Categorical covariates ----
    // Validate caller intent: if the model was constructed for K categorical
    // columns, the caller must pass a (n_plots, K) tensor. If the model has
    // no categorical columns, the caller must pass an empty tensor.
    {
        const int64_t model_cat_cols = model_->schema().n_categoricals();
        if (categorical_ids.defined() && categorical_ids.numel() > 0) {
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
            train_categorical_ids_ = categorical_ids.index_select(0, train_idx);
            test_categorical_ids_ = categorical_ids.index_select(0, test_idx);
        } else if (model_cat_cols > 0) {
            throw std::runtime_error(
                "Trainer::prepare_data: model was constructed for " +
                std::to_string(model_cat_cols) +
                " categorical columns but no categorical_ids tensor was "
                "provided. Pass dataset.categorical_ids() or use the "
                "ResolveDataset overload of prepare_data().");
        }
    }

    // Scale and split targets
    for (const auto& cfg : model_->schema().targets) {
        auto target_it = targets.find(cfg.name);
        if (target_it == targets.end()) continue;

        auto target = target_it->second.clone();

        // Only apply transforms and scaling for regression targets
        if (cfg.task == TaskType::Regression) {
            // Apply transform if needed
            if (cfg.transform == TransformType::Log1p) {
                target = torch::log1p(target);
            }

            // Compute scaler on training data
            auto train_target = target.index_select(0, train_idx);
            auto target_mean = train_target.mean();
            auto target_scale = train_target.std() + 1e-8f;

            scalers_.target_scalers[cfg.name] = {target_mean, target_scale};

            // Apply scaling
            target = (target - target_mean) / target_scale;
        }

        train_targets_[cfg.name] = target.index_select(0, train_idx);
        test_targets_[cfg.name] = target.index_select(0, test_idx);
    }

    data_prepared_ = true;
}

float Trainer::train_epoch(int epoch) {
    model_->train();

    // Use GPU-cached data if available, otherwise fallback to CPU tensors
    const auto& continuous_src = gpu_data_cached_ ? gpu_continuous_ : train_continuous_;
    const auto& genus_src = gpu_data_cached_ ? gpu_genus_ids_ : train_genus_ids_;
    const auto& family_src = gpu_data_cached_ ? gpu_family_ids_ : train_family_ids_;
    const auto& species_ids_src = gpu_data_cached_ ? gpu_species_ids_ : train_species_ids_;
    const auto& species_vec_src = gpu_data_cached_ ? gpu_species_vector_ : train_species_vector_;
    const auto& targets_src = gpu_data_cached_ ? gpu_targets_ : train_targets_;
    const auto& scalers_src = gpu_data_cached_ ? gpu_scalers_ : std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>>{};

    // Pool-style data sources (rank_pool / transformer modes)
    const auto& pool_genus_src = gpu_data_cached_ ? gpu_pool_genus_ids_ : train_pool_genus_ids_;
    const auto& pool_family_src = gpu_data_cached_ ? gpu_pool_family_ids_ : train_pool_family_ids_;
    const auto& pool_weights_src = gpu_data_cached_ ? gpu_pool_weights_ : train_pool_weights_;
    const auto& pool_mask_src = gpu_data_cached_ ? gpu_pool_mask_ : train_pool_mask_;
    const auto& pool_has_cover_src = gpu_data_cached_ ? gpu_pool_has_cover_ : train_pool_has_cover_;

    // Categorical covariate source (per-plot int64 codes). Empty when the
    // dataset has no categorical columns; the model handles that gracefully.
    const auto& categorical_src = gpu_data_cached_ ? gpu_categorical_ids_ : train_categorical_ids_;

    int64_t n_train = continuous_src.size(0);
    int batch_size = config_.batch_size;

    // Shuffle training data with a dedicated, seeded generator so the run is
    // reproducible for a fixed seed. Generated on CPU (the permutation is tiny)
    // and moved to the data device; using a private generator keeps the global
    // RNG stream — and therefore dropout / cover-dropout — untouched.
    auto perm_gen = at::detail::createCPUGenerator(
        static_cast<uint64_t>(data_seed_) + static_cast<uint64_t>(epoch) + 1u);
    auto perm = torch::randperm(
        n_train, perm_gen, torch::TensorOptions().dtype(torch::kLong));
    if (continuous_src.device() != perm.device()) {
        perm = perm.to(continuous_src.device());
    }

    float total_loss = 0.0f;
    int n_batches = 0;

    // Check if MoE is enabled for aux loss handling
    bool use_moe = model_->uses_moe();
    float moe_aux_weight = model_->config().moe_aux_loss_weight;

    // Async prefetching setup for CUDA hash computation
#ifdef RESOLVE_HAS_CUDA
    c10::cuda::CUDAStream prefetch_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
    c10::cuda::CUDAStream default_stream = at::cuda::getCurrentCUDAStream();
    bool use_prefetch = use_cuda_hash_ && config_.device.is_cuda() && n_train > batch_size;
    torch::Tensor prefetched_hash;
    torch::Tensor prefetched_continuous;
    bool prefetch_ready = false;

    // Get source data pointers for hash computation
    const auto& hash_species_src = gpu_data_cached_ ? gpu_train_raw_species_ids_ : raw_species_ids_;
    const auto& hash_weights_src = gpu_data_cached_ ? gpu_train_raw_weights_ : raw_weights_;
    const auto& hash_offsets_src = gpu_data_cached_ ? gpu_train_plot_offsets_ : plot_offsets_;

    // Map train-local shuffle positions to GLOBAL plot indices before hashing.
    // train_continuous_ row k is global plot train_indices_[k], but the CSR
    // offsets used for the on-the-fly hash are full-dataset, so the hash must be
    // looked up by global plot index (matching eval_epoch / the baseline paths).
    // Without this remap, a batch row's species-hash comes from a different plot
    // than its covariates and target.
    torch::Tensor hash_index_map;
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        hash_index_map = train_indices_.to(perm.device());
    }
#endif

    for (int64_t start = 0; start < n_train; start += batch_size) {
        int64_t end = std::min(start + batch_size, n_train);
        auto batch_idx = perm.slice(0, start, end);

        // Get batch data - already on GPU if cached, just index_select
        auto batch_continuous = continuous_src.index_select(0, batch_idx);
        auto batch_genus_ids = genus_src.defined() ? genus_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_family_ids = family_src.defined() ? family_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_species_ids = species_ids_src.defined() ? species_ids_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_species_vector = species_vec_src.defined() ? species_vec_src.index_select(0, batch_idx) : torch::Tensor();

        // Pool-style batch slicing
        auto batch_pool_genus = pool_genus_src.defined() ? pool_genus_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_pool_family = pool_family_src.defined() ? pool_family_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_pool_weights = pool_weights_src.defined() ? pool_weights_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_pool_mask = pool_mask_src.defined() ? pool_mask_src.index_select(0, batch_idx) : torch::Tensor();
        auto batch_pool_has_cover = pool_has_cover_src.defined() ? pool_has_cover_src.index_select(0, batch_idx) : torch::Tensor();

        // Categorical batch slicing
        auto batch_categorical_ids = categorical_src.defined() && categorical_src.numel() > 0
            ? categorical_src.index_select(0, batch_idx)
            : torch::Tensor();

        // If not GPU cached, move to device
        if (!gpu_data_cached_) {
            batch_continuous = batch_continuous.to(config_.device);
            if (batch_genus_ids.defined()) batch_genus_ids = batch_genus_ids.to(config_.device);
            if (batch_family_ids.defined()) batch_family_ids = batch_family_ids.to(config_.device);
            if (batch_species_ids.defined()) batch_species_ids = batch_species_ids.to(config_.device);
            if (batch_species_vector.defined()) batch_species_vector = batch_species_vector.to(config_.device);
            if (batch_pool_genus.defined()) batch_pool_genus = batch_pool_genus.to(config_.device);
            if (batch_pool_family.defined()) batch_pool_family = batch_pool_family.to(config_.device);
            if (batch_pool_weights.defined()) batch_pool_weights = batch_pool_weights.to(config_.device);
            if (batch_pool_mask.defined()) batch_pool_mask = batch_pool_mask.to(config_.device);
            if (batch_pool_has_cover.defined()) batch_pool_has_cover = batch_pool_has_cover.to(config_.device);
            if (batch_categorical_ids.defined()) batch_categorical_ids = batch_categorical_ids.to(config_.device);
        }

        // CUDA hash computation with async prefetching
#ifdef RESOLVE_HAS_CUDA
        if (use_cuda_hash_ && config_.device.is_cuda()) {
            torch::Tensor batch_hash;

            if (prefetch_ready) {
                // Use prefetched hash from previous iteration
                // Wait for prefetch stream to complete on default stream
                at::cuda::CUDAEvent prefetch_done;
                prefetch_done.record(prefetch_stream);
                prefetch_done.block(default_stream);

                batch_hash = prefetched_hash;
                prefetch_ready = false;
            } else {
                // First batch: compute synchronously. Hash by global plot index.
                batch_hash = cuda::compute_batch_hash_embedding_cuda(
                    hash_index_map.index_select(0, batch_idx),
                    torch::Tensor(),
                    hash_species_src,
                    hash_weights_src,
                    hash_offsets_src,
                    hash_dim_
                );
            }

            // Concatenate hash embedding with continuous features
            batch_continuous = torch::cat({batch_continuous, batch_hash}, /*dim=*/1);

            // Launch async prefetch for next batch if there is one
            int64_t next_start = start + batch_size;
            if (use_prefetch && next_start < n_train) {
                int64_t next_end = std::min(next_start + batch_size, n_train);
                auto next_batch_idx = perm.slice(0, next_start, next_end);

                // Record event on default stream, then wait for it on prefetch stream
                at::cuda::CUDAEvent compute_done;
                compute_done.record(default_stream);
                compute_done.block(prefetch_stream);

                // Compute next batch's hash on prefetch stream
                c10::cuda::CUDAStreamGuard guard(prefetch_stream);
                prefetched_hash = cuda::compute_batch_hash_embedding_cuda(
                    hash_index_map.index_select(0, next_batch_idx),
                    torch::Tensor(),
                    hash_species_src,
                    hash_weights_src,
                    hash_offsets_src,
                    hash_dim_
                );
                prefetch_ready = true;
            }
        }
#endif

        std::unordered_map<std::string, torch::Tensor> batch_targets;
        for (const auto& [name, tensor] : targets_src) {
            batch_targets[name] = tensor.index_select(0, batch_idx);
            if (!gpu_data_cached_) {
                batch_targets[name] = batch_targets[name].to(config_.device);
            }
        }

        // Forward pass
        optimizer_->zero_grad();

        std::unordered_map<std::string, torch::Tensor> predictions;
        torch::Tensor moe_aux_loss;
        torch::Tensor loss;

        // RAII autocast scope covering the forward and loss. The nesting
        // counter and enabled flag must unwind even if forward()/compute()
        // throw — the expected throw is the c10::OutOfMemoryError that fit()
        // catches to halve the batch and retry. Without RAII, an OOM mid-
        // forward would leak the incremented nesting (its cache is cleared
        // only when nesting returns to 0) and leave autocast enabled across
        // every retried epoch.
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

        {
            AutocastScope amp_scope(amp_enabled_);

            if (use_moe) {
                // Use forward_with_aux to get MoE auxiliary loss
                auto result = model_->forward_with_aux(
                    batch_continuous, batch_genus_ids, batch_family_ids,
                    batch_species_ids, batch_species_vector,
                    batch_pool_genus, batch_pool_family,
                    batch_pool_weights, batch_pool_mask, batch_pool_has_cover,
                    batch_categorical_ids
                );
                predictions = std::move(result.outputs);
                moe_aux_loss = result.moe_aux_loss;
            } else {
                predictions = model_->forward(
                    batch_continuous, batch_genus_ids, batch_family_ids,
                    batch_species_ids, batch_species_vector,
                    batch_pool_genus, batch_pool_family,
                    batch_pool_weights, batch_pool_mask, batch_pool_has_cover,
                    batch_categorical_ids
                );
            }

            // Compute loss - use cached scalers if available
            std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> batch_scalers;
            if (gpu_data_cached_) {
                batch_scalers = scalers_src;
            } else {
                for (const auto& [name, scaler] : scalers_.target_scalers) {
                    batch_scalers[name] = {
                        scaler.first.to(config_.device),
                        scaler.second.to(config_.device)
                    };
                }
            }

            auto [loss_val, _] = loss_fn_.compute(predictions, batch_targets, epoch, batch_scalers);
            loss = loss_val;

            // Add MoE auxiliary loss for load balancing
            if (use_moe && moe_aux_loss.defined()) {
                loss = loss + moe_aux_weight * moe_aux_loss;
            }
        }  // autocast disabled here (also on exception unwind) before backward

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

            // Fused inf/nan check + unscale in a single CUDA kernel.
            // Replaces a per-parameter isinf/isnan loop and a separate
            // per-parameter div_ loop, both of which forced ~10-20 host syncs
            // per batch via .item<bool>(). See gcol33/resolve#1.
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
                torch::nn::utils::clip_grad_norm_(model_->parameters(), kMaxGradNorm);

                // Optimizer step
                optimizer_->step();

                // Update scale
                amp_growth_tracker_++;
                if (amp_growth_tracker_ >= config_.amp_growth_interval) {
                    amp_scale_ *= config_.amp_growth_factor;
                    amp_growth_tracker_ = 0;
                }

                total_loss += loss.item<float>();
                n_batches++;
            }
        } else {
            // Standard backward pass without AMP
            loss.backward();

            // Gradient clipping
            torch::nn::utils::clip_grad_norm_(model_->parameters(), kMaxGradNorm);

            optimizer_->step();

            total_loss += loss.item<float>();
            n_batches++;
        }
    }

    return n_batches > 0 ? total_loss / n_batches : 0.0f;
}

std::pair<float, std::unordered_map<std::string, std::unordered_map<std::string, float>>>
Trainer::eval_epoch(int epoch) {
    model_->eval();
    torch::NoGradGuard no_grad;

    // Use GPU-cached test data if available
    torch::Tensor test_continuous, test_genus_ids, test_family_ids, test_species_ids, test_species_vector;
    torch::Tensor test_categorical_ids;
    PoolTensors test_pool;
    std::unordered_map<std::string, torch::Tensor> test_targets;
    std::unordered_map<std::string, std::pair<torch::Tensor, torch::Tensor>> batch_scalers;

    if (gpu_data_cached_) {
        test_continuous = gpu_test_continuous_;
        test_genus_ids = gpu_test_genus_ids_;
        test_family_ids = gpu_test_family_ids_;
        test_species_ids = gpu_test_species_ids_;
        test_species_vector = gpu_test_species_vector_;
        test_pool = {gpu_test_pool_genus_ids_, gpu_test_pool_family_ids_,
                     gpu_test_pool_weights_, gpu_test_pool_mask_, gpu_test_pool_has_cover_};
        test_categorical_ids = gpu_test_categorical_ids_;
        test_targets = gpu_test_targets_;
        batch_scalers = gpu_scalers_;
    } else {
        test_continuous = test_continuous_.to(config_.device);
        test_genus_ids = to_device_if_defined(test_genus_ids_, config_.device);
        test_family_ids = to_device_if_defined(test_family_ids_, config_.device);
        test_species_ids = to_device_if_defined(test_species_ids_, config_.device);
        test_species_vector = to_device_if_defined(test_species_vector_, config_.device);
        test_pool = {to_device_if_defined(test_pool_genus_ids_, config_.device),
                     to_device_if_defined(test_pool_family_ids_, config_.device),
                     to_device_if_defined(test_pool_weights_, config_.device),
                     to_device_if_defined(test_pool_mask_, config_.device),
                     to_device_if_defined(test_pool_has_cover_, config_.device)};
        test_categorical_ids = to_device_if_defined(test_categorical_ids_, config_.device);
        for (const auto& [name, tensor] : test_targets_) {
            test_targets[name] = tensor.to(config_.device);
        }
        for (const auto& [name, scaler] : scalers_.target_scalers) {
            batch_scalers[name] = {
                scaler.first.to(config_.device),
                scaler.second.to(config_.device)
            };
        }
    }

    // CUDA hash computation for the full test set. eval_epoch runs inside the
    // training loop, so it uses the GPU-resident CSR buffers + test indices when
    // the dataset is cached on device.
    test_continuous = append_cuda_hash(
        test_continuous,
        gpu_data_cached_ ? gpu_test_indices_ : test_indices_,
        /*use_cache=*/gpu_data_cached_);

    // Forward pass
    auto predictions = model_->forward(
        test_continuous, test_genus_ids, test_family_ids,
        test_species_ids, test_species_vector,
        test_pool.genus_ids, test_pool.family_ids,
        test_pool.weights, test_pool.mask, test_pool.has_cover,
        test_categorical_ids
    );

    // Evaluate the validation loss at a phase-INVARIANT reference epoch (the
    // last training epoch's phase) rather than the current epoch. The phased
    // loss adds strictly non-negative SMAPE/band terms as phases advance, so a
    // current-epoch eval loss jumps up at each phase boundary; comparing it
    // across epochs for best-model selection / early stopping would lock the
    // returned model to a phase-1 checkpoint and stop the curriculum early.
    // Scoring every epoch on the final-phase objective gives one consistent
    // scale, so best_loss is comparable across the whole run.
    const int sel_epoch = std::max(0, config_.max_epochs - 1);
    auto [loss, _] = loss_fn_.compute(predictions, test_targets, sel_epoch, batch_scalers);

    // AMP regression diagnostic (gcol33/resolve#21). Gated by RESOLVE_AMP_DEBUG.
    // Compares the eval-mode test loss (BatchNorm running stats) against the
    // train-mode test loss on the SAME inputs (BatchNorm batch stats), with
    // running-stat save/restore so the probe does not pollute eval state, and
    // scans BatchNorm running statistics for divergence. If trainmode_test is
    // low while eval_test rises, the running statistics are the culprit.
    static const bool amp_dbg = [] {
        const char* v = std::getenv("RESOLVE_AMP_DEBUG");
        return v != nullptr && std::strcmp(v, "0") != 0;
    }();
    if (amp_dbg) {
        double max_mean = 0.0, max_var = 0.0, min_var = 1e30;
        bool nonfinite = false;
        std::vector<std::shared_ptr<torch::nn::BatchNorm1dImpl>> bns;
        std::vector<std::pair<torch::Tensor, torch::Tensor>> saved;
        std::vector<torch::Tensor> saved_nbt;  // num_batches_tracked
        for (auto& m : model_->modules(/*include_self=*/false)) {
            if (auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(m)) {
                if (bn->running_mean.defined() && bn->running_var.defined()) {
                    max_mean = std::max(max_mean, bn->running_mean.abs().max().item<double>());
                    max_var = std::max(max_var, bn->running_var.max().item<double>());
                    min_var = std::min(min_var, bn->running_var.min().item<double>());
                    if (!torch::isfinite(bn->running_mean).all().item<bool>() ||
                        !torch::isfinite(bn->running_var).all().item<bool>()) {
                        nonfinite = true;
                    }
                    bns.push_back(bn);
                    saved.emplace_back(bn->running_mean.clone(), bn->running_var.clone());
                    saved_nbt.push_back(bn->num_batches_tracked.defined()
                        ? bn->num_batches_tracked.clone() : torch::Tensor());
                }
            }
        }
        model_->train();
        auto tm_preds = model_->forward(
            test_continuous, test_genus_ids, test_family_ids,
            test_species_ids, test_species_vector,
            test_pool.genus_ids, test_pool.family_ids,
            test_pool.weights, test_pool.mask, test_pool.has_cover,
            test_categorical_ids);
        auto [tm_loss, tm_ignored] = loss_fn_.compute(tm_preds, test_targets, sel_epoch, batch_scalers);
        for (size_t i = 0; i < bns.size(); ++i) {
            bns[i]->running_mean.copy_(saved[i].first);
            bns[i]->running_var.copy_(saved[i].second);
            if (saved_nbt[i].defined() && bns[i]->num_batches_tracked.defined()) {
                bns[i]->num_batches_tracked.copy_(saved_nbt[i]);
            }
        }
        model_->eval();
        std::ostringstream dbg;
        dbg << "  [amp_dbg ep" << epoch << "] eval_test=" << loss.item<float>()
            << " trainmode_test=" << tm_loss.item<float>()
            << " | BN n=" << bns.size() << " max|mean|=" << max_mean
            << " max_var=" << max_var << " min_var=" << min_var
            << " nonfinite=" << (nonfinite ? "Y" : "N");
        config_.log(dbg.str());
    }

    // Compute metrics per target
    std::unordered_map<std::string, std::unordered_map<std::string, float>> all_metrics;

    for (const auto& cfg : model_->schema().targets) {
        auto pred_it = predictions.find(cfg.name);
        auto target_it = test_targets.find(cfg.name);

        if (pred_it != predictions.end() && target_it != test_targets.end()) {
            // Pass the target scalers so regression metrics are reported in
            // original units (the model and test_targets are both in
            // standardized(-log) space here). Classification targets have no
            // scaler entry -> undefined tensors -> Metrics::compute skips the
            // unscale, as intended.
            torch::Tensor sc_mean, sc_scale;
            auto scaler_it = batch_scalers.find(cfg.name);
            if (scaler_it != batch_scalers.end()) {
                sc_mean = scaler_it->second.first;
                sc_scale = scaler_it->second.second;
            }
            all_metrics[cfg.name] = Metrics::compute(
                pred_it->second, target_it->second, cfg.task, cfg.transform,
                config_.band_thresholds, cfg.num_classes, sc_mean, sc_scale
            );
        }
    }

    return {loss.item<float>(), all_metrics};
}

float Trainer::get_learning_rate(int epoch) const {
    switch (config_.lr_scheduler) {
        case LRSchedulerType::StepLR: {
            // Step decay: multiply LR by gamma every lr_step_size epochs
            int n_decays = config_.lr_step_size > 0 ? epoch / config_.lr_step_size : 0;
            return config_.lr * std::pow(config_.lr_gamma, static_cast<float>(n_decays));
        }
        case LRSchedulerType::CosineAnnealing: {
            // Cosine annealing from lr (epoch 0) to lr_min (final epoch). Divide
            // by (max_epochs - 1) so progress reaches 1.0 on the last epoch and
            // the schedule actually lands on lr_min; dividing by max_epochs
            // leaves the endpoint a step short of lr_min (issue #74).
            float progress = config_.max_epochs > 1
                ? static_cast<float>(epoch) / (config_.max_epochs - 1) : 1.0f;
            progress = std::min(progress, 1.0f);
            float cosine = 0.5f * (1.0f + std::cos(M_PI * progress));
            return config_.lr_min + (config_.lr - config_.lr_min) * cosine;
        }
        case LRSchedulerType::None:
        default:
            return config_.lr;
    }
}

void Trainer::update_learning_rate(float lr) {
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
    }
}

void Trainer::cache_data_to_gpu() {
    if (gpu_data_cached_ || !config_.device.is_cuda()) {
        return;  // Already cached or not using CUDA
    }

    // cuDNN benchmark policy is set once from config_.cudnn_benchmark in fit()
    // (issue #92). Do not force-enable it here: that silently overrode
    // cudnn_benchmark=false and made the determinism knob a no-op.

    // Cache training data on GPU
    gpu_continuous_ = train_continuous_.to(config_.device);
    if (train_genus_ids_.defined()) {
        gpu_genus_ids_ = train_genus_ids_.to(config_.device);
    }
    if (train_family_ids_.defined()) {
        gpu_family_ids_ = train_family_ids_.to(config_.device);
    }
    if (train_species_ids_.defined()) {
        gpu_species_ids_ = train_species_ids_.to(config_.device);
    }
    if (train_species_vector_.defined()) {
        gpu_species_vector_ = train_species_vector_.to(config_.device);
    }

    // Cache pool fields on GPU (training)
    auto cache_if_defined = [&](const torch::Tensor& src, torch::Tensor& dst) {
        if (src.defined()) dst = src.to(config_.device);
    };
    cache_if_defined(train_pool_genus_ids_, gpu_pool_genus_ids_);
    cache_if_defined(train_pool_family_ids_, gpu_pool_family_ids_);
    cache_if_defined(train_pool_weights_, gpu_pool_weights_);
    cache_if_defined(train_pool_mask_, gpu_pool_mask_);
    cache_if_defined(train_pool_has_cover_, gpu_pool_has_cover_);

    // Cache categorical IDs on GPU (training)
    cache_if_defined(train_categorical_ids_, gpu_categorical_ids_);

    for (const auto& [name, tensor] : train_targets_) {
        gpu_targets_[name] = tensor.to(config_.device);
    }

    // Cache scalers on GPU
    for (const auto& [name, scaler] : scalers_.target_scalers) {
        gpu_scalers_[name] = {
            scaler.first.to(config_.device),
            scaler.second.to(config_.device)
        };
    }

    // Cache test data on GPU
    gpu_test_continuous_ = test_continuous_.to(config_.device);
    if (test_genus_ids_.defined()) {
        gpu_test_genus_ids_ = test_genus_ids_.to(config_.device);
    }
    if (test_family_ids_.defined()) {
        gpu_test_family_ids_ = test_family_ids_.to(config_.device);
    }
    if (test_species_ids_.defined()) {
        gpu_test_species_ids_ = test_species_ids_.to(config_.device);
    }
    if (test_species_vector_.defined()) {
        gpu_test_species_vector_ = test_species_vector_.to(config_.device);
    }

    // Cache pool fields on GPU (test)
    cache_if_defined(test_pool_genus_ids_, gpu_test_pool_genus_ids_);
    cache_if_defined(test_pool_family_ids_, gpu_test_pool_family_ids_);
    cache_if_defined(test_pool_weights_, gpu_test_pool_weights_);
    cache_if_defined(test_pool_mask_, gpu_test_pool_mask_);
    cache_if_defined(test_pool_has_cover_, gpu_test_pool_has_cover_);

    // Cache categorical IDs on GPU (test)
    cache_if_defined(test_categorical_ids_, gpu_test_categorical_ids_);

    for (const auto& [name, tensor] : test_targets_) {
        gpu_test_targets_[name] = tensor.to(config_.device);
    }

    // Cache raw species data for CUDA hash computation
    if (use_cuda_hash_) {
        if (raw_species_ids_.defined()) {
            gpu_train_raw_species_ids_ = raw_species_ids_.to(config_.device);
        }
        if (raw_weights_.defined()) {
            gpu_train_raw_weights_ = raw_weights_.to(config_.device);
        }
        if (plot_offsets_.defined()) {
            gpu_train_plot_offsets_ = plot_offsets_.to(config_.device);
        }
        // Cache test indices for computing test hash embedding
        if (test_indices_.defined()) {
            gpu_test_indices_ = test_indices_.to(config_.device);
        }
    }

    gpu_data_cached_ = true;
}

void Trainer::release_training_state() {
    // Drop optimizer first so its parameter-state tensors (Adam m/v moments,
    // momentum buffers, etc.) are freed before we ask the allocator to
    // empty its cache.
    optimizer_.reset();

    // Forget best-model snapshot; the retry will compute a new one.
    best_model_state_.clear();
    best_model_state_.shrink_to_fit();

    // Reset AMP runtime state. The actual config knobs (init_scale,
    // growth_factor, etc.) live on config_ and are re-read on retry.
    amp_enabled_ = false;
    amp_scale_ = config_.amp_init_scale;
    amp_growth_tracker_ = 0;

    // Release every GPU-cached tensor. We don't drop the host-side train_*
    // / test_* tensors because prepare_data_ semantics promise to keep them
    // alive across fit() calls; only the GPU mirrors are torn down so the
    // next cache_data_to_gpu() call can rebuild them.
    gpu_data_cached_ = false;
    gpu_continuous_ = torch::Tensor();
    gpu_genus_ids_ = torch::Tensor();
    gpu_family_ids_ = torch::Tensor();
    gpu_species_ids_ = torch::Tensor();
    gpu_species_vector_ = torch::Tensor();
    gpu_targets_.clear();
    gpu_scalers_.clear();
    gpu_pool_genus_ids_ = torch::Tensor();
    gpu_pool_family_ids_ = torch::Tensor();
    gpu_pool_weights_ = torch::Tensor();
    gpu_pool_mask_ = torch::Tensor();
    gpu_pool_has_cover_ = torch::Tensor();
    gpu_categorical_ids_ = torch::Tensor();
    gpu_test_continuous_ = torch::Tensor();
    gpu_test_genus_ids_ = torch::Tensor();
    gpu_test_family_ids_ = torch::Tensor();
    gpu_test_species_ids_ = torch::Tensor();
    gpu_test_species_vector_ = torch::Tensor();
    gpu_test_targets_.clear();
    gpu_test_pool_genus_ids_ = torch::Tensor();
    gpu_test_pool_family_ids_ = torch::Tensor();
    gpu_test_pool_weights_ = torch::Tensor();
    gpu_test_pool_mask_ = torch::Tensor();
    gpu_test_pool_has_cover_ = torch::Tensor();
    gpu_test_categorical_ids_ = torch::Tensor();
    gpu_train_raw_species_ids_ = torch::Tensor();
    gpu_train_raw_weights_ = torch::Tensor();
    gpu_train_plot_offsets_ = torch::Tensor();
    gpu_test_indices_ = torch::Tensor();

#ifdef RESOLVE_HAS_CUDA
    // Return free blocks to the device so the allocator's reserved pool
    // shrinks before the next attempt. This is the bridge that makes the
    // halved batch_size actually have more headroom: without it the
    // previous attempt's reserved-but-unallocated blocks stay parked.
    if (config_.device.is_cuda()) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
#endif
}

TrainResult Trainer::fit() {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before training");
    }

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

        // TF32 (TensorFloat-32): faster matmuls on Ampere+ GPUs (RTX 30xx, A100, etc.)
        // Trades ~0.1% precision for ~3x speedup on tensor cores
        // Only affects float32 operations, not float16/bfloat16
        at::globalContext().setAllowTF32CuBLAS(config_.allow_tf32);
        at::globalContext().setAllowTF32CuDNN(config_.allow_tf32);
    }

    // Snapshot the freshly-initialized model weights once. If a CUDA OOM
    // halves the batch size and forces a retry, training restarts from
    // epoch 0 against this same starting point — restarting from a
    // partially-trained model under a different batch size mixes
    // optimizer-state assumptions and is fragile.
    std::vector<char> initial_model_state;
    {
        torch::serialize::OutputArchive snap;
        model_->save(snap);
        std::ostringstream oss;
        snap.save_to(oss);
        auto str = oss.str();
        initial_model_state.assign(str.begin(), str.end());
    }

    const int batch_size_at_entry = config_.batch_size;
    requested_batch_size_ = batch_size_at_entry;  // persisted distinct from effective (#86)
    auto start_time = std::chrono::high_resolution_clock::now();
    created_at_ = iso8601_now();

    // Create checkpoint directory if specified
    bool use_checkpoints = !config_.checkpoint_dir.empty();
    if (use_checkpoints) {
        std::filesystem::create_directories(config_.checkpoint_dir);
    }

    TrainResult result;
    float best_loss = std::numeric_limits<float>::infinity();
    int patience_counter = 0;
    bool training_complete = false;

    while (!training_complete) {
        // Pre-cache all data on GPU for faster training (post-cap so it
        // respects the allocator fraction). May be a no-op on a fresh attempt
        // (cleared by release_training_state on retry).
        cache_data_to_gpu();

        // Initialize AMP (only enabled on CUDA devices)
        amp_enabled_ = config_.use_amp && config_.device.is_cuda();
        if (amp_enabled_) {
            amp_scale_ = config_.amp_init_scale;
            amp_growth_tracker_ = 0;
            // Set autocast dtype to float16 for CUDA
            at::autocast::set_autocast_dtype(at::kCUDA, at::kHalf);
        }

        // Create optimizer
        optimizer_ = std::make_unique<torch::optim::AdamW>(
            model_->parameters(),
            torch::optim::AdamWOptions(config_.lr).weight_decay(config_.weight_decay)
        );

        // Reset per-attempt accumulators. If a retry kicks in below, these
        // are wiped along with the optimizer.
        result = TrainResult{};
        best_loss = std::numeric_limits<float>::infinity();
        patience_counter = 0;

        // The phased loss activates later terms (SMAPE, band) only once their
        // phase begins, so early-stopping before the final phase would kill the
        // curriculum before those objectives ever train. Only count patience
        // once we are in the phase the last epoch is in; ask the loss for the
        // phase so MAE/SMAPE mode's remapped boundaries are respected.
        const int final_phase = loss_fn_.phase_for(std::max(0, config_.max_epochs - 1));

        try {
            for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
                // Update learning rate based on scheduler
                float current_lr = get_learning_rate(epoch);
                update_learning_rate(current_lr);

                float train_loss = train_epoch(epoch);
                auto [test_loss, metrics] = eval_epoch(epoch);

                result.train_loss_history.push_back(train_loss);
                result.test_loss_history.push_back(test_loss);

                const bool in_final_phase = loss_fn_.phase_for(epoch) == final_phase;

                // Check for improvement. test_loss is the phase-invariant
                // selection loss (see eval_epoch), so best_loss is comparable
                // across the whole run and the returned model is the global best.
                if (test_loss < best_loss) {
                    best_loss = test_loss;
                    result.best_epoch = epoch;
                    result.final_metrics = metrics;
                    patience_counter = 0;

                    // Save best model state to memory
                    std::ostringstream oss;
                    torch::serialize::OutputArchive archive;
                    model_->save(archive);
                    archive.save_to(oss);
                    auto str = oss.str();
                    best_model_state_.assign(str.begin(), str.end());

                    // Save best checkpoint
                    if (use_checkpoints) {
                        save(config_.checkpoint_dir + "/best.pt");
                    }
                } else if (in_final_phase) {
                    patience_counter++;
                    if (patience_counter >= config_.patience) {
                        config_.log("Early stopping at epoch " + std::to_string(epoch));
                        break;
                    }
                }

                // Periodic checkpoint saving
                if (use_checkpoints && config_.checkpoint_every > 0 && (epoch + 1) % config_.checkpoint_every == 0) {
                    save(config_.checkpoint_dir + "/checkpoint_" + std::to_string(epoch + 1) + ".pt");
                }

                // Write progress file
                if (use_checkpoints) {
                    write_progress_file(
                        config_.checkpoint_dir, epoch, config_.max_epochs,
                        result.best_epoch, best_loss, patience_counter, result.final_metrics
                    );
                }

                // Print progress
                // Print progress using log callback
                if (epoch % 10 == 0) {
                    std::ostringstream msg;
                    msg << "Epoch " << epoch << " - Train: " << train_loss << " Test: " << test_loss;
                    if (config_.lr_scheduler != LRSchedulerType::None) {
                        msg << " LR: " << current_lr;
                    }
                    config_.log(msg.str());
                }
            }
            training_complete = true;
        } catch (const c10::Error& cerr) {
            // Two CUDA out-of-memory paths to catch:
            //   - c10::OutOfMemoryError raised by the caching allocator
            //     when its reserved cap is hit but cudaMalloc would succeed.
            //   - c10::AcceleratorError carrying cudaErrorMemoryAllocation
            //     when the allocator already released free blocks back to
            //     the driver and a fresh cudaMalloc fails outright. This is
            //     the terminal failure mode on near-cap workloads: the
            //     cascade descends through several OutOfMemoryError catches
            //     and eventually trips an AcceleratorError when even the
            //     bare malloc can't be satisfied.
            // We treat both as "retry with a halved batch", and let any
            // other c10::Error propagate unchanged.
            const std::string what_str = cerr.what();
            const bool is_oom_err =
                dynamic_cast<const c10::OutOfMemoryError*>(&cerr) != nullptr;
            const bool is_cuda_oom =
                what_str.find("out of memory") != std::string::npos
                || what_str.find("cudaErrorMemoryAllocation") != std::string::npos;
            if (!is_oom_err && !is_cuda_oom) {
                throw;
            }

            const int prev_bs = config_.batch_size;
            int new_bs = prev_bs;
            std::string log_msg;
            std::string err_msg;
            const bool retry = decide_oom_retry(
                prev_bs,
                config_.batch_size_floor,
                batch_size_at_entry,
                cerr.what(),
                new_bs,
                log_msg,
                err_msg
            );

            // Tear state down regardless: the trainer must be left in a
            // consistent shape whether we retry or rethrow.
            release_training_state();

            if (!retry) {
                throw std::runtime_error(err_msg);
            }

            config_.log(log_msg);
            config_.batch_size = new_bs;

            // Restore the original model state so the retry sees the same
            // initial conditions, not a half-trained model.
            {
                std::istringstream iss(std::string(initial_model_state.begin(),
                                                   initial_model_state.end()));
                torch::serialize::InputArchive archive;
                archive.load_from(iss);
                model_->load(archive);
            }
            // Fall through and loop.
        }
    }

    // Restore best model state
    if (!best_model_state_.empty()) {
        std::istringstream iss(std::string(best_model_state_.begin(), best_model_state_.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model_->load(archive);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    result.train_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

    // Save final checkpoint WITH run metadata so a saved checkpoint records its
    // own train time, best epoch, and reported metrics (load_run_metadata /
    // the JSON sidecar were otherwise write-only -- fit() never populated them).
    if (use_checkpoints) {
        RunMetadata meta;
        meta.created_at = created_at_;
        meta.completed_at = iso8601_now();
        meta.train_time_seconds = result.train_time_seconds;
        meta.n_plots_train = train_indices_.defined() ? train_indices_.numel() : 0;
        meta.n_plots_test = test_indices_.defined() ? test_indices_.numel() : 0;
        meta.best_epoch = result.best_epoch;
        meta.total_epochs = static_cast<int>(result.train_loss_history.size());
        meta.final_metrics = result.final_metrics;
        save(config_.checkpoint_dir + "/checkpoint.pt", &meta);
    }

    // Restore the requested batch size. fit() may have shrunk config_.batch_size
    // in place via the OOM auto-halve loop; the checkpoint above intentionally
    // recorded the effective (shrunk) value, but the in-memory config must not
    // carry the shrink into a subsequent fit() — e.g. later cross-validation
    // folds would otherwise silently train at the reduced batch size.
    config_.batch_size = batch_size_at_entry;

    // Compute baseline comparisons for each target
    {
        torch::NoGradGuard no_grad;
        model_->eval();

        // Move test data to GPU for prediction (use CPU-stored tensors)
        auto test_cont_gpu = test_continuous_.to(config_.device);
        auto test_genus_gpu = to_device_if_defined(test_genus_ids_, config_.device);
        auto test_family_gpu = to_device_if_defined(test_family_ids_, config_.device);
        auto test_species_gpu = to_device_if_defined(test_species_ids_, config_.device);
        auto test_vector_gpu = to_device_if_defined(test_species_vector_, config_.device);
        auto test_cat_gpu = to_device_if_defined(test_categorical_ids_, config_.device);
        auto baseline_pool = get_test_pool_tensors();

        // CUDA hash embedding for the test set in baseline eval.
        test_cont_gpu = append_cuda_hash(test_cont_gpu, test_indices_, /*use_cache=*/false);

        // Get final predictions on test set
        auto predictions = model_->forward(
            test_cont_gpu, test_genus_gpu, test_family_gpu,
            test_species_gpu, test_vector_gpu,
            baseline_pool.genus_ids, baseline_pool.family_ids,
            baseline_pool.weights, baseline_pool.mask, baseline_pool.has_cover,
            test_cat_gpu
        );

        for (const auto& cfg : model_->schema().targets) {
            BaselineMetrics baseline;

            auto pred_it = predictions.find(cfg.name);
            auto test_target_it = test_targets_.find(cfg.name);
            auto train_target_it = train_targets_.find(cfg.name);

            if (pred_it == predictions.end() || test_target_it == test_targets_.end() ||
                train_target_it == train_targets_.end()) {
                continue;
            }

            // reshape({-1}) instead of squeeze() so a single-sample test fold
            // (n_test == 1) keeps a 1-D batch dimension instead of collapsing to
            // a 0-D scalar (which would break .size(0) / argmax below).
            auto test_target = test_target_it->second.to(config_.device).reshape({-1});
            auto train_target = train_target_it->second.to(config_.device).reshape({-1});

            if (cfg.task == TaskType::Regression) {
                auto pred = pred_it->second.reshape({-1});
                // Training mean (in scaled space for regression)
                float train_mean = train_target.mean().item<float>();
                baseline.training_mean = train_mean;

                // Create baseline predictions (all equal to training mean)
                auto baseline_pred = torch::full_like(test_target, train_mean);

                // Compute MSE for baseline and model (both should be 1D)
                baseline.baseline_mse = torch::mse_loss(baseline_pred, test_target).item<float>();
                baseline.model_mse = torch::mse_loss(pred, test_target).item<float>();

                // Compute MAE for baseline and model
                baseline.baseline_mae = torch::mean(torch::abs(baseline_pred - test_target)).item<float>();
                baseline.model_mae = torch::mean(torch::abs(pred - test_target)).item<float>();

                // Skill score: 1 - (model_mse / baseline_mse)
                if (baseline.baseline_mse > kEpsilon) {
                    baseline.skill_score = 1.0f - (baseline.model_mse / baseline.baseline_mse);
                }

                // R-squared: 1 - (SS_res / SS_tot)
                auto test_mean = test_target.mean();
                auto ss_tot = torch::sum(torch::pow(test_target - test_mean, 2));
                auto ss_res = torch::sum(torch::pow(test_target - pred, 2));
                if (ss_tot.item<float>() > kEpsilon) {
                    baseline.r_squared = 1.0f - (ss_res.item<float>() / ss_tot.item<float>());
                }
            } else {
                // Classification: mode baseline
                auto mode_result = torch::mode(train_target.to(torch::kLong));
                int mode_class = std::get<0>(mode_result).item<int>();
                baseline.training_mode = mode_class;

                auto baseline_correct = (test_target.to(torch::kLong) == mode_class).sum();
                baseline.baseline_accuracy = baseline_correct.item<float>() / test_target.size(0);

                // Keep the (n_test, n_classes) shape so argmax over classes works
                // even when n_test == 1 (a squeezed (n_classes,) would misfire).
                auto pred_logits = pred_it->second;
                if (pred_logits.dim() == 1) pred_logits = pred_logits.unsqueeze(0);
                auto pred_classes = pred_logits.argmax(1);
                auto model_correct = (pred_classes == test_target.to(torch::kLong)).sum();
                baseline.model_accuracy = model_correct.item<float>() / test_target.size(0);
                baseline.accuracy_lift = baseline.model_accuracy - baseline.baseline_accuracy;
            }

            result.baselines[cfg.name] = baseline;
        }
    }

    // Compute network diagnostics
    result.diagnostics = compute_diagnostics();

    return result;
}

void Trainer::save(const std::string& path, const RunMetadata* metadata) const {
    torch::serialize::OutputArchive archive;

    // Save model parameters manually (NOT using model_->save() which does TorchScript
    // export and ignores our custom metadata). Use prefixed keys to avoid collisions.
    for (const auto& pair : model_->named_parameters()) {
        archive.write("param_" + pair.key(), pair.value());
    }
    for (const auto& pair : model_->named_buffers()) {
        archive.write("buffer_" + pair.key(), pair.value());
    }

    // Save config, schema, and scalers using checkpoint utilities
    save_model_config(archive, model_->config());
    save_schema(archive, model_->schema());
    save_scalers(archive, scalers_);

    // Save categorical vocabulary (string -> code maps for each categorical
    // column). Empty vocab writes a count of zero — load() handles that as
    // a no-op for back-compat with pre-categorical-port checkpoints.
    categorical_vocab_.save(archive, "trainer_categorical_");

    // Save training configuration for reproducibility. Pass the requested batch
    // size so an OOM fallback (effective < requested) is recoverable (issue #86).
    save_train_config(archive, config_, requested_batch_size_);

    // Save run metadata if provided (final checkpoint only)
    if (metadata != nullptr) {
        save_run_metadata(archive, *metadata);
    }

    // Retry a transient write fault on flaky storage (issue #20); a small
    // checkpoint re-write is cheap, so any failure is retried.
    io::with_retry([&] { archive.save_to(path); }, "checkpoint save");

    // Write human-readable JSON metadata alongside checkpoint
    if (metadata != nullptr) {
        write_metadata_json(path, model_->config(), config_, *metadata, model_->schema(),
                            requested_batch_size_);
    }
}

std::tuple<ResolveModel, Scalers, CategoricalVocab> Trainer::load(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    // Apply VRAM cap BEFORE any device allocation so model weights and
    // buffers respect the limit on first upload. Matches the cap applied
    // in Trainer::fit; no-ops on CPU or fraction >= 1.0.
    if (device.is_cuda()) {
        set_vram_fraction(
            static_cast<double>(vram_fraction),
            device.index()
        );
    }

    torch::serialize::InputArchive archive;
    // Retry a transient read fault on flaky storage (issue #20). Note: a
    // mmap-backed load can still fault as an OS structured exception, which is
    // #19's domain (fail fast); this catches the throwing-read failures.
    io::with_retry([&] { archive.load_from(path); }, "checkpoint load");

    // Load config, schema, and scalers using checkpoint utilities
    ModelConfig config = load_model_config(archive);
    ResolveSchema schema = load_schema(archive);
    Scalers scalers = load_scalers(archive);

    // Load the categorical vocabulary. Back-compat for pre-categorical-port
    // checkpoints is handled inside CategoricalVocab::load (returns empty).
    CategoricalVocab vocab = CategoricalVocab::load(archive, "trainer_categorical_");

    // Create model with loaded schema
    ResolveModel model(schema, config);

    // Load model weights into the freshly-constructed model (matching the
    // prefixed save format).
    load_weights_into(archive, model);

    model->to(device);

    return {model, scalers, vocab};
}

TrainConfig Trainer::load_train_config(const std::string& path) {
    torch::serialize::InputArchive archive;
    io::with_retry([&] { archive.load_from(path); }, "checkpoint load (train config)");
    // Qualify to the free function (checkpoint.cpp); the unqualified name would
    // resolve to this static member and recurse.
    return resolve::load_train_config(archive);
}

RunMetadata Trainer::load_run_metadata(const std::string& path) {
    torch::serialize::InputArchive archive;
    io::with_retry([&] { archive.load_from(path); }, "checkpoint load (run metadata)");
    return resolve::load_run_metadata(archive);
}

void Trainer::load_weights_into(torch::serialize::InputArchive& archive, ResolveModel& model) {
    // Freshly-constructed model parameters are leaf tensors with
    // requires_grad=true. Calling .copy_() on them directly trips autograd's
    // check_inplace ("a leaf Variable that requires grad is being used in an
    // in-place operation"). Mirror PyTorch's copy-inside-torch.no_grad().
    torch::NoGradGuard no_grad;
    int64_t n_expected = 0, n_loaded = 0;
    std::string first_missing;
    auto note_missing = [&](const std::string& key) {
        if (first_missing.empty()) first_missing = key;
    };
    for (const auto& pair : model->named_parameters()) {
        ++n_expected;
        torch::Tensor t;
        if (archive.try_read("param_" + pair.key(), t)) {
            pair.value().copy_(t);
            ++n_loaded;
        } else {
            note_missing("param_" + pair.key());
        }
    }
    for (const auto& pair : model->named_buffers()) {
        ++n_expected;
        torch::Tensor t;
        if (archive.try_read("buffer_" + pair.key(), t)) {
            pair.value().copy_(t);
            ++n_loaded;
        } else {
            note_missing("buffer_" + pair.key());
        }
    }
    // A silent skip leaves those tensors at fresh random init, so a mismatched
    // architecture would load as a partly-untrained model. Fail loudly instead
    // (this is what makes an encoder_architecture / sub-config drift visible).
    if (n_loaded != n_expected) {
        std::ostringstream msg;
        msg << "load_weights_into: checkpoint is missing " << (n_expected - n_loaded)
            << " of " << n_expected << " model tensors, so the loaded model would "
               "keep randomly-initialized weights. The model architecture likely "
               "does not match the checkpoint (e.g. a different encoder_architecture "
               "or architecture sub-config). First missing tensor: " << first_missing;
        throw std::runtime_error(msg.str());
    }
}

void Trainer::load_state(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    // Apply the VRAM cap before any device allocation, matching fit() and the
    // static load(); no-op on CPU or fraction >= 1.0.
    if (device.is_cuda()) {
        set_vram_fraction(static_cast<double>(vram_fraction), device.index());
    }

    torch::serialize::InputArchive archive;
    io::with_retry([&] { archive.load_from(path); }, "checkpoint load (state)");

    // Restore weights into the existing model_ (its architecture must already
    // match the checkpoint), the fitted scalers, and the categorical vocab.
    load_weights_into(archive, model_);
    model_->to(device);

    scalers_ = load_scalers(archive);
    categorical_vocab_ = CategoricalVocab::load(archive, "trainer_categorical_");

    // Keep config_.device coherent with where the weights now live so the
    // eval helpers move test tensors to the matching device.
    config_.device = device;
}

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
        torch::Tensor species_src = use_cache ? gpu_train_raw_species_ids_
                                              : raw_species_ids_.to(config_.device);
        torch::Tensor weights_src = use_cache ? gpu_train_raw_weights_
                                              : raw_weights_.to(config_.device);
        torch::Tensor offsets_src = use_cache ? gpu_train_plot_offsets_
                                              : plot_offsets_.to(config_.device);
        auto hash = cuda::compute_batch_hash_embedding_cuda(
            plot_idx.to(config_.device),
            torch::Tensor(),  // raw_plot_indices not needed with CSR offsets
            species_src, weights_src, offsets_src, hash_dim_);
        return torch::cat({continuous, hash}, /*dim=*/1);
    }
#else
    (void)plot_idx;
    (void)use_cache;
#endif
    return continuous;
}

std::unordered_map<std::string, torch::Tensor> Trainer::forward_test_fold() {
    model_->eval();
    torch::NoGradGuard no_grad;

    auto test_continuous = test_continuous_.to(config_.device);
    auto test_genus_ids = to_device_if_defined(test_genus_ids_, config_.device);
    auto test_family_ids = to_device_if_defined(test_family_ids_, config_.device);
    auto test_species_ids = to_device_if_defined(test_species_ids_, config_.device);
    auto test_species_vector = to_device_if_defined(test_species_vector_, config_.device);
    auto test_cat_ids = to_device_if_defined(test_categorical_ids_, config_.device);
    auto pool = get_test_pool_tensors();

    // CUDA hash mode computes the hash embedding on the fly; other encodings
    // already fold their species representation into test_continuous_.
    test_continuous = append_cuda_hash(test_continuous, test_indices_, /*use_cache=*/false);

    return model_->forward(
        test_continuous, test_genus_ids, test_family_ids,
        test_species_ids, test_species_vector,
        pool.genus_ids, pool.family_ids, pool.weights, pool.mask, pool.has_cover,
        test_cat_ids
    );
}

NetworkDiagnostics Trainer::compute_diagnostics() {
    NetworkDiagnostics diag;

    torch::NoGradGuard no_grad;
    model_->eval();

    // Use a subset of test data for diagnostics (max 10000 samples). Draw the
    // subset with a dedicated seeded generator so the diagnostics are
    // reproducible for a fixed seed and do not perturb the global RNG stream
    // mid-fit (matching the per-epoch shuffle; cf. #15).
    int64_t n_samples = std::min(test_continuous_.size(0), static_cast<int64_t>(10000));
    auto diag_gen = at::detail::createCPUGenerator(
        static_cast<uint64_t>(data_seed_) + 0x51ADu);
    auto sample_indices = torch::randperm(
        test_continuous_.size(0), diag_gen,
        torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, n_samples);

    auto sample_continuous = test_continuous_.index_select(0, sample_indices).to(config_.device);
    auto sample_genus = to_device_if_defined(
        test_genus_ids_.defined() ? test_genus_ids_.index_select(0, sample_indices) : torch::Tensor(),
        config_.device);
    auto sample_family = to_device_if_defined(
        test_family_ids_.defined() ? test_family_ids_.index_select(0, sample_indices) : torch::Tensor(),
        config_.device);
    // Slice categorical_ids for the sampled subset so encode_with_activations
    // can fuse them into the continuous tensor and match the encoder's
    // constructed n_continuous.
    auto sample_cat_ids = (test_categorical_ids_.defined()
                           && test_categorical_ids_.numel() > 0)
        ? test_categorical_ids_.index_select(0, sample_indices).to(config_.device)
        : torch::Tensor();

    // CUDA hash embedding for the sampled subset: map the test-local sample
    // indices to global plot indices, then reuse the shared hash helper.
    sample_continuous = append_cuda_hash(
        sample_continuous,
        test_indices_.index_select(0, sample_indices),
        /*use_cache=*/false);

    // Use encode_with_activations to get intermediate layer outputs
    auto [latent, activations] = model_->encode_with_activations(
        sample_continuous, sample_genus, sample_family, sample_cat_ids
    );

    // If no activations available (non-hash encoder), return empty diagnostics
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

    torch::NoGradGuard no_grad;
    model_->eval();

    // Find target config
    const TargetConfig* target_cfg = nullptr;
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.name == target_name) {
            target_cfg = &cfg;
            break;
        }
    }

    if (!target_cfg || target_cfg->task != TaskType::Classification) {
        result.bins.clear();
        return result;
    }

    // Get test predictions (shared single-source test-fold forward).
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_targets_.find(target_name);
    if (pred_it == predictions.end() || target_it == test_targets_.end()) {
        return result;
    }

    // Get predicted probabilities (softmax)
    auto logits = pred_it->second;  // (n_samples, n_classes)
    auto probs = torch::softmax(logits, /*dim=*/1).cpu();
    auto targets = target_it->second.to(torch::kLong).cpu();  // (n_samples,)

    int64_t n_samples = probs.size(0);
    int64_t n_classes = probs.size(1);

    // For each class, compute calibration (here we do class 0 as example for binary,
    // or the most common class for multiclass)
    // Using the max probability and whether prediction was correct
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

    torch::NoGradGuard no_grad;
    model_->eval();

    // Find target config
    const TargetConfig* target_cfg = nullptr;
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.name == target_name) {
            target_cfg = &cfg;
            break;
        }
    }

    if (!target_cfg || target_cfg->task != TaskType::Regression) {
        return result;
    }

    // Get test predictions (shared single-source test-fold forward).
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_targets_.find(target_name);
    if (pred_it == predictions.end() || target_it == test_targets_.end()) {
        return result;
    }

    // Get predictions and targets. reshape({-1}), not squeeze(): a single-plot
    // test fold has a (1,1) output that squeeze() would collapse to a 0-D
    // scalar, breaking size(0)/accessor and the element-wise subtraction below.
    auto preds = pred_it->second.reshape({-1}).cpu();
    auto targets = target_it->second.reshape({-1}).cpu();

    // Unscale if needed (predictions are in scaled space)
    auto scaler_it = scalers_.target_scalers.find(target_name);
    if (scaler_it != scalers_.target_scalers.end()) {
        auto [mean, scale] = scaler_it->second;
        preds = preds * scale + mean;
        targets = targets * scale + mean;
    }

    // Apply inverse transform if needed
    if (target_cfg->transform == TransformType::Log1p) {
        preds = torch::expm1(preds);
        targets = torch::expm1(targets);
    }

    // Compute residuals
    auto residuals = targets - preds;

    // Convert to vectors
    int64_t n = residuals.size(0);
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

    // Locate the target and confirm it is a classification head. Non-matching
    // or regression targets return empty tensors (parallels compute_residuals
    // returning empty for classification targets).
    const TargetConfig* target_cfg = nullptr;
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.name == target_name) {
            target_cfg = &cfg;
            break;
        }
    }
    if (!target_cfg || target_cfg->task != TaskType::Classification) {
        return result;
    }
    result.class_names = target_cfg->class_names;

    // Shared single-source test-fold forward.
    auto predictions = forward_test_fold();

    auto pred_it = predictions.find(target_name);
    auto target_it = test_targets_.find(target_name);
    if (pred_it == predictions.end() || target_it == test_targets_.end()) {
        return result;
    }

    auto logits = pred_it->second;                 // (n_test, n_classes)
    auto probs = torch::softmax(logits, /*dim=*/1);
    auto predicted = probs.argmax(/*dim=*/1);      // (n_test,)

    result.probabilities = probs.detach().to(torch::kCPU).contiguous();
    result.predicted_classes =
        predicted.detach().to(torch::kCPU).contiguous().to(torch::kLong);
    result.actuals =
        target_it->second.detach().to(torch::kCPU).contiguous().to(torch::kLong);

    return result;
}

void Trainer::unscale_continuous_targets(
    torch::Tensor& continuous,
    std::unordered_map<std::string, torch::Tensor>& targets,
    const Scalers& scalers) {
    // continuous: x_scaled -> x_raw = x_scaled * scale + mean (per feature).
    if (continuous.defined() && continuous.size(1) > 0 &&
        scalers.continuous_mean.defined() && scalers.continuous_scale.defined()) {
        continuous = continuous * scalers.continuous_scale + scalers.continuous_mean;
    }
    // Regression targets: only those present in target_scalers were scaled;
    // {mean, scale} stored as {first, second} at fit time.
    for (const auto& [name, ms] : scalers.target_scalers) {
        auto it = targets.find(name);
        if (it != targets.end() && it->second.defined()) {
            it->second = it->second * ms.second + ms.first;
        }
    }
}

namespace {
// Silences per-fold checkpoint writes during cross-validation: fit() otherwise
// overwrites the user's checkpoint.pt / best.pt with each fold's subset model,
// leaving the on-disk artifact describing the last fold rather than the
// full-data model the user trained before CV (issue #75). Clears checkpoint_dir
// on construction and restores it on any exit path (exception-safe).
struct CheckpointDirSilencer {
    std::string& dir;
    std::string saved;
    explicit CheckpointDirSilencer(std::string& d) : dir(d), saved(d) { d.clear(); }
    ~CheckpointDirSilencer() { dir = saved; }
    CheckpointDirSilencer(const CheckpointDirSilencer&) = delete;
    CheckpointDirSilencer& operator=(const CheckpointDirSilencer&) = delete;
};
}  // namespace

Trainer::SplitState Trainer::capture_split_state() const {
    SplitState s;
    s.train_continuous = train_continuous_;
    s.train_genus_ids = train_genus_ids_;
    s.train_family_ids = train_family_ids_;
    s.train_species_ids = train_species_ids_;
    s.train_species_vector = train_species_vector_;
    s.train_pool_genus_ids = train_pool_genus_ids_;
    s.train_pool_family_ids = train_pool_family_ids_;
    s.train_pool_weights = train_pool_weights_;
    s.train_pool_mask = train_pool_mask_;
    s.train_pool_has_cover = train_pool_has_cover_;
    s.train_categorical_ids = train_categorical_ids_;
    s.train_targets = train_targets_;
    s.test_continuous = test_continuous_;
    s.test_genus_ids = test_genus_ids_;
    s.test_family_ids = test_family_ids_;
    s.test_species_ids = test_species_ids_;
    s.test_species_vector = test_species_vector_;
    s.test_pool_genus_ids = test_pool_genus_ids_;
    s.test_pool_family_ids = test_pool_family_ids_;
    s.test_pool_weights = test_pool_weights_;
    s.test_pool_mask = test_pool_mask_;
    s.test_pool_has_cover = test_pool_has_cover_;
    s.test_categorical_ids = test_categorical_ids_;
    s.test_targets = test_targets_;
    s.train_indices = train_indices_;
    s.test_indices = test_indices_;
    s.scalers = scalers_;
    return s;
}

void Trainer::restore_split_state(const SplitState& s) {
    train_continuous_ = s.train_continuous;
    train_genus_ids_ = s.train_genus_ids;
    train_family_ids_ = s.train_family_ids;
    train_species_ids_ = s.train_species_ids;
    train_species_vector_ = s.train_species_vector;
    train_pool_genus_ids_ = s.train_pool_genus_ids;
    train_pool_family_ids_ = s.train_pool_family_ids;
    train_pool_weights_ = s.train_pool_weights;
    train_pool_mask_ = s.train_pool_mask;
    train_pool_has_cover_ = s.train_pool_has_cover;
    train_categorical_ids_ = s.train_categorical_ids;
    train_targets_ = s.train_targets;
    test_continuous_ = s.test_continuous;
    test_genus_ids_ = s.test_genus_ids;
    test_family_ids_ = s.test_family_ids;
    test_species_ids_ = s.test_species_ids;
    test_species_vector_ = s.test_species_vector;
    test_pool_genus_ids_ = s.test_pool_genus_ids;
    test_pool_family_ids_ = s.test_pool_family_ids;
    test_pool_weights_ = s.test_pool_weights;
    test_pool_mask_ = s.test_pool_mask;
    test_pool_has_cover_ = s.test_pool_has_cover;
    test_categorical_ids_ = s.test_categorical_ids;
    test_targets_ = s.test_targets;
    train_indices_ = s.train_indices;
    test_indices_ = s.test_indices;
    scalers_ = s.scalers;
    // The per-fold loop cached fold data on the GPU; those tensors no longer
    // match the restored split, so force a re-cache on the next fit().
    gpu_data_cached_ = false;
}

CrossValidationResult Trainer::run_cross_validation(
    const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>& folds
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    CrossValidationResult cv_result;
    cv_result.n_folds = static_cast<int>(folds.size());

    // Concatenate the prepared train/test tensors so each fold re-splits them.
    auto all_continuous = torch::cat({train_continuous_, test_continuous_}, 0);
    auto cat_if_both_defined = [](const torch::Tensor& a, const torch::Tensor& b) -> torch::Tensor {
        if (a.defined() && b.defined()) return torch::cat({a, b}, 0);
        return {};
    };
    auto all_genus_ids = cat_if_both_defined(train_genus_ids_, test_genus_ids_);
    auto all_family_ids = cat_if_both_defined(train_family_ids_, test_family_ids_);
    auto all_species_ids = cat_if_both_defined(train_species_ids_, test_species_ids_);
    auto all_species_vector = cat_if_both_defined(train_species_vector_, test_species_vector_);
    auto all_pool_genus_ids = cat_if_both_defined(train_pool_genus_ids_, test_pool_genus_ids_);
    auto all_pool_family_ids = cat_if_both_defined(train_pool_family_ids_, test_pool_family_ids_);
    auto all_pool_weights = cat_if_both_defined(train_pool_weights_, test_pool_weights_);
    auto all_pool_mask = cat_if_both_defined(train_pool_mask_, test_pool_mask_);
    auto all_pool_has_cover = cat_if_both_defined(train_pool_has_cover_, test_pool_has_cover_);
    auto all_categorical_ids = cat_if_both_defined(train_categorical_ids_, test_categorical_ids_);

    std::unordered_map<std::string, torch::Tensor> all_targets;
    for (const auto& [name, train_tensor] : train_targets_) {
        auto test_it = test_targets_.find(name);
        if (test_it != test_targets_.end()) {
            all_targets[name] = torch::cat({train_tensor, test_it->second}, 0);
        }
    }

    // Snapshot original model weights + scalers + the full split for restoration.
    std::ostringstream original_state_stream;
    {
        torch::serialize::OutputArchive archive;
        model_->save(archive);
        archive.save_to(original_state_stream);
    }
    std::string original_state = original_state_stream.str();
    Scalers original_scalers = scalers_;
    SplitState split_snapshot = capture_split_state();

    // Don't let per-fold fit() calls clobber the user's on-disk checkpoint.
    CheckpointDirSilencer ckpt_silencer(config_.checkpoint_dir);

    // Invert prepare_data's standardization once so each fold recomputes its own
    // scalers from raw values exactly once (see the note in cross-validation).
    unscale_continuous_targets(all_continuous, all_targets, original_scalers);

    // Global plot index per concatenated row, for the CUDA-hash path. Row i of
    // all_continuous is global plot all_global_idx[i]; each fold reconstructs
    // train_indices_ / test_indices_ from it so on-the-fly GPU hashing indexes
    // the correct plots.
    torch::Tensor all_global_idx;
    if (train_indices_.defined() && test_indices_.defined()) {
        all_global_idx = torch::cat({train_indices_, test_indices_}, 0);
    }

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> metric_values;

    for (size_t fold = 0; fold < folds.size(); ++fold) {
        config_.log("Cross-validation fold " + std::to_string(fold + 1) + "/" +
                    std::to_string(folds.size()));

        // Reset this fold to the as-constructed (untrained) weights, not the
        // trainer's current weights. If cross_validate is run after fit(), the
        // current weights are already trained on rows that fall in this fold's
        // held-out test set; warm-starting from them yields optimistically
        // biased CV metrics (issue #97).
        {
            std::istringstream iss(pristine_model_state_);
            torch::serialize::InputArchive archive;
            archive.load_from(iss);
            model_->load(archive);
        }

        const auto& train_index = folds[fold].first;
        const auto& test_index = folds[fold].second;
        auto train_idx = torch::tensor(train_index, torch::kInt64);
        auto test_idx = torch::tensor(test_index, torch::kInt64);

        config_.log("  Train: " + std::to_string(train_index.size()) +
                    " plots, Test: " + std::to_string(test_index.size()) + " plots");

        // Reconstruct this fold's global plot indices for the CUDA-hash path and
        // invalidate the GPU cache so fit() re-uploads THIS fold's split
        // (cache_data_to_gpu early-returns while gpu_data_cached_ is set, so
        // without this every fold after the first would reuse fold 0's data).
        if (all_global_idx.defined()) {
            train_indices_ = all_global_idx.index_select(0, train_idx);
            test_indices_ = all_global_idx.index_select(0, test_idx);
        }
        gpu_data_cached_ = false;

        train_continuous_ = all_continuous.index_select(0, train_idx);
        test_continuous_ = all_continuous.index_select(0, test_idx);

        auto split_if_defined = [&](const torch::Tensor& all, torch::Tensor& train_dst, torch::Tensor& test_dst) {
            if (all.defined()) {
                train_dst = all.index_select(0, train_idx);
                test_dst = all.index_select(0, test_idx);
            }
        };
        split_if_defined(all_genus_ids, train_genus_ids_, test_genus_ids_);
        split_if_defined(all_family_ids, train_family_ids_, test_family_ids_);
        split_if_defined(all_species_ids, train_species_ids_, test_species_ids_);
        split_if_defined(all_species_vector, train_species_vector_, test_species_vector_);
        split_if_defined(all_pool_genus_ids, train_pool_genus_ids_, test_pool_genus_ids_);
        split_if_defined(all_pool_family_ids, train_pool_family_ids_, test_pool_family_ids_);
        split_if_defined(all_pool_weights, train_pool_weights_, test_pool_weights_);
        split_if_defined(all_pool_mask, train_pool_mask_, test_pool_mask_);
        split_if_defined(all_pool_has_cover, train_pool_has_cover_, test_pool_has_cover_);
        split_if_defined(all_categorical_ids, train_categorical_ids_, test_categorical_ids_);

        for (const auto& [name, tensor] : all_targets) {
            train_targets_[name] = tensor.index_select(0, train_idx);
            test_targets_[name] = tensor.index_select(0, test_idx);
        }

        // Recompute scalers for this fold's training data.
        if (train_continuous_.size(1) > 0) {
            scalers_.continuous_mean = train_continuous_.mean(0);
            scalers_.continuous_scale = train_continuous_.std(0) + 1e-8f;
            train_continuous_ = (train_continuous_ - scalers_.continuous_mean) / scalers_.continuous_scale;
            test_continuous_ = (test_continuous_ - scalers_.continuous_mean) / scalers_.continuous_scale;
        }

        for (const auto& cfg : model_->schema().targets) {
            if (cfg.task != TaskType::Regression) continue;
            auto train_it = train_targets_.find(cfg.name);
            if (train_it == train_targets_.end()) continue;
            auto target_mean = train_it->second.mean();
            auto target_scale = train_it->second.std() + 1e-8f;
            scalers_.target_scalers[cfg.name] = {target_mean, target_scale};
            train_targets_[cfg.name] = (train_targets_[cfg.name] - target_mean) / target_scale;
            test_targets_[cfg.name] = (test_targets_[cfg.name] - target_mean) / target_scale;
        }

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

    // Restore the original model weights and the pre-CV split (tensors, indices,
    // scalers) so post-CV evaluators run against the original split.
    {
        std::istringstream iss(original_state);
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model_->load(archive);
    }
    restore_split_state(split_snapshot);

    auto end_time = std::chrono::high_resolution_clock::now();
    cv_result.total_time_seconds = std::chrono::duration<float>(end_time - start_time).count();
    return cv_result;
}

CrossValidationResult Trainer::cross_validate(int n_folds, int seed) {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before cross-validation");
    }

    const int64_t n_total = train_continuous_.size(0) + test_continuous_.size(0);

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
    auto lat = coords_cpu.select(1, 0);
    auto lon = coords_cpu.select(1, 1);

    // Grid hash: block_row = floor(lat / lat_size), block_col = floor(lon / lon_size)
    auto block_row = torch::floor(lat / lat_size_).to(torch::kInt64);
    auto block_col = torch::floor(lon / lon_size_).to(torch::kInt64);

    // Linear block hash: label = row * 1e6 + col. Injective only while
    // |block_col| < 5e5, which holds for degree-scale grids (lon in [-180, 180]);
    // a sub-4e-4 deg lon_size could alias distinct blocks into one fold.
    auto block_labels = block_row * 1000000 + block_col;

    // Find unique blocks
    auto unique_result = torch::_unique2(block_labels, /*sorted=*/true, /*return_inverse=*/true);
    auto unique_blocks = std::get<0>(unique_result);
    auto inverse_indices = std::get<1>(unique_result);
    int64_t n_blocks = unique_blocks.size(0);

    // Every fold needs at least one spatial block or its test set is empty, which
    // then divides by zero in the baseline metrics and NaNs the whole fold (issue
    // #82). Round-robin and greedy bin-packing both give each fold >= 1 block once
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

    // Count block sizes
    auto inverse_cpu = inverse_indices.cpu();
    auto inv_ptr = inverse_cpu.data_ptr<int64_t>();
    std::vector<int64_t> block_sizes(n_blocks, 0);
    for (int64_t i = 0; i < n; ++i) {
        block_sizes[inv_ptr[i]]++;
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
            if (block_to_fold[inv_ptr[i]] == f) {
                test_idx.push_back(i);
            } else {
                train_idx.push_back(i);
            }
        }
        folds[f] = {std::move(train_idx), std::move(test_idx)};
    }

    // Defensive: guarantee no empty test/train fold reached the caller even if a
    // future assignment scheme changed (issue #82). n_blocks >= n_splits above
    // makes this hold for round-robin and bin-packing; assert it regardless.
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
    // spatial split degenerates into a geographically-scrambled split (issue #70).
    torch::Tensor split_order_coords = coordinates_;
    if (train_indices_.defined() && test_indices_.defined()) {
        auto all_global_idx = torch::cat({train_indices_, test_indices_}, 0);
        split_order_coords = coordinates_.index_select(
            0, all_global_idx.to(coordinates_.device()));
    }

    auto folds = splitter.split(split_order_coords);

    auto cv_result = run_cross_validation(folds);

    config_.log("Spatial CV complete (" + std::to_string(n_folds) + " folds, "
                + std::to_string(cv_result.total_time_seconds) + "s)");

    return cv_result;
}

// =============================================================================
// Pool tensor helpers
// =============================================================================

Trainer::PoolTensors Trainer::get_test_pool_tensors() const {
    if (gpu_data_cached_) {
        return {gpu_test_pool_genus_ids_, gpu_test_pool_family_ids_,
                gpu_test_pool_weights_, gpu_test_pool_mask_, gpu_test_pool_has_cover_};
    }
    return {to_device_if_defined(test_pool_genus_ids_, config_.device),
            to_device_if_defined(test_pool_family_ids_, config_.device),
            to_device_if_defined(test_pool_weights_, config_.device),
            to_device_if_defined(test_pool_mask_, config_.device),
            to_device_if_defined(test_pool_has_cover_, config_.device)};
}

// =============================================================================
// Predict (eval-mode inference)
// =============================================================================

std::unordered_map<std::string, torch::Tensor> Trainer::predict(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    auto to_dev = [&](torch::Tensor t) -> torch::Tensor {
        return t.defined() ? t.to(config_.device) : t;
    };

    // Standardize the continuous block with the fitted scalers, exactly as
    // Predictor::predict does. Callers pass raw features; the model was trained
    // on standardized inputs, so skipping this silently biases every prediction.
    torch::Tensor scaled_continuous = continuous;
    if (scalers_.continuous_mean.defined() && continuous.defined() &&
        continuous.size(1) > 0) {
        scaled_continuous =
            (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    }
    scaled_continuous = scaled_continuous.to(config_.device);

    auto outputs = model_->forward(
        scaled_continuous, to_dev(genus_ids), to_dev(family_ids),
        to_dev(species_ids), to_dev(species_vector),
        to_dev(pool_genus_ids), to_dev(pool_family_ids),
        to_dev(pool_weights), to_dev(pool_mask), to_dev(pool_has_cover),
        to_dev(categorical_ids)
    );

    // Inverse-transform regression outputs back to original units (un-scale +
    // log1p inverse), mirroring Predictor::predict. Classification outputs are
    // returned as raw logits (this map is the low-level surface; argmax /
    // softmax is the caller's choice).
    for (const auto& cfg : model_->schema().targets) {
        if (cfg.task != TaskType::Regression) continue;
        auto it = outputs.find(cfg.name);
        if (it == outputs.end()) continue;

        auto pred = it->second.squeeze(-1);
        auto scaler_it = scalers_.target_scalers.find(cfg.name);
        if (scaler_it != scalers_.target_scalers.end()) {
            pred = pred * scaler_it->second.second.to(pred.device()) +
                   scaler_it->second.first.to(pred.device());
        }
        if (cfg.transform == TransformType::Log1p) {
            pred = torch::expm1(torch::clamp(pred, kExpClampMin, kExpClampMax));
        }
        it->second = pred;
    }

    return outputs;
}

} // namespace resolve
