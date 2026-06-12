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
#include <random>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>

namespace resolve {

namespace {

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

void Trainer::create_loaders() {
    // Note: For simplicity, we handle batching manually in train_epoch/eval_epoch
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
                // First batch: compute synchronously
                batch_hash = cuda::compute_batch_hash_embedding_cuda(
                    batch_idx,
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
                    next_batch_idx,
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
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);

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
            torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);

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

    // CUDA hash computation: compute hash embedding for entire test set
#ifdef RESOLVE_HAS_CUDA
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        // Get the source data (GPU cached or original)
        const auto& species_src = gpu_data_cached_ ? gpu_train_raw_species_ids_ : raw_species_ids_;
        const auto& weights_src = gpu_data_cached_ ? gpu_train_raw_weights_ : raw_weights_;
        const auto& offsets_src = gpu_data_cached_ ? gpu_train_plot_offsets_ : plot_offsets_;
        const auto& test_idx = gpu_data_cached_ ? gpu_test_indices_ : test_indices_.to(config_.device);

        // Compute hash embedding for test set
        auto test_hash = cuda::compute_batch_hash_embedding_cuda(
            test_idx,
            torch::Tensor(),  // raw_plot_indices not needed with CSR offsets
            species_src,
            weights_src,
            offsets_src,
            hash_dim_
        );

        // Concatenate hash embedding with continuous features
        test_continuous = torch::cat({test_continuous, test_hash}, /*dim=*/1);
    }
#endif

    // Forward pass
    auto predictions = model_->forward(
        test_continuous, test_genus_ids, test_family_ids,
        test_species_ids, test_species_vector,
        test_pool.genus_ids, test_pool.family_ids,
        test_pool.weights, test_pool.mask, test_pool.has_cover,
        test_categorical_ids
    );

    auto [loss, _] = loss_fn_.compute(predictions, test_targets, epoch, batch_scalers);

    // Compute metrics per target
    std::unordered_map<std::string, std::unordered_map<std::string, float>> all_metrics;

    for (const auto& cfg : model_->schema().targets) {
        auto pred_it = predictions.find(cfg.name);
        auto target_it = test_targets.find(cfg.name);

        if (pred_it != predictions.end() && target_it != test_targets.end()) {
            all_metrics[cfg.name] = Metrics::compute(
                pred_it->second, target_it->second, cfg.task, cfg.transform,
                config_.band_thresholds, cfg.num_classes
            );
        }
    }

    return {loss.item<float>(), all_metrics};
}

float Trainer::get_learning_rate(int epoch) const {
    switch (config_.lr_scheduler) {
        case LRSchedulerType::StepLR: {
            // Step decay: multiply LR by gamma every lr_step_size epochs
            int n_decays = epoch / config_.lr_step_size;
            return config_.lr * std::pow(config_.lr_gamma, static_cast<float>(n_decays));
        }
        case LRSchedulerType::CosineAnnealing: {
            // Cosine annealing from lr to lr_min
            float progress = static_cast<float>(epoch) / config_.max_epochs;
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

    // Enable cuDNN auto-tuner for faster convolutions
    torch::globalContext().setBenchmarkCuDNN(true);

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

    // Drop prefetched hash buffers (CUDA-only path).
    prefetch_hash_[0] = torch::Tensor();
    prefetch_hash_[1] = torch::Tensor();
    prefetch_batch_idx_ = torch::Tensor();
    prefetch_buffer_idx_ = 0;
    prefetch_valid_ = false;

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

    // Shuffled mirrors (rarely used today but cleared for completeness).
    shuffled_continuous_ = torch::Tensor();
    shuffled_genus_ids_ = torch::Tensor();
    shuffled_family_ids_ = torch::Tensor();
    shuffled_species_ids_ = torch::Tensor();
    shuffled_species_vector_ = torch::Tensor();
    shuffled_targets_.clear();

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
    auto start_time = std::chrono::high_resolution_clock::now();

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

        try {
            for (int epoch = 0; epoch < config_.max_epochs; ++epoch) {
                // Update learning rate based on scheduler
                float current_lr = get_learning_rate(epoch);
                update_learning_rate(current_lr);

                float train_loss = train_epoch(epoch);
                auto [test_loss, metrics] = eval_epoch(epoch);

                result.train_loss_history.push_back(train_loss);
                result.test_loss_history.push_back(test_loss);

                // Check for improvement
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
                } else {
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

    // Save final checkpoint
    if (use_checkpoints) {
        save(config_.checkpoint_dir + "/checkpoint.pt");
    }

    // Restore the requested batch size. fit() may have shrunk config_.batch_size
    // in place via the OOM auto-halve loop; the checkpoint above intentionally
    // recorded the effective (shrunk) value, but the in-memory config must not
    // carry the shrink into a subsequent fit() — e.g. later cross-validation
    // folds would otherwise silently train at the reduced batch size.
    config_.batch_size = batch_size_at_entry;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.train_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

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

        // CUDA hash computation: compute hash embedding for test set in baseline eval
#ifdef RESOLVE_HAS_CUDA
        if (use_cuda_hash_ && config_.device.is_cuda()) {
            auto test_idx = test_indices_.to(config_.device);
            auto test_hash = cuda::compute_batch_hash_embedding_cuda(
                test_idx,
                torch::Tensor(),  // raw_plot_indices not needed with CSR offsets
                raw_species_ids_.to(config_.device),
                raw_weights_.to(config_.device),
                plot_offsets_.to(config_.device),
                hash_dim_
            );
            test_cont_gpu = torch::cat({test_cont_gpu, test_hash}, /*dim=*/1);
        }
#endif

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

            auto pred = pred_it->second.squeeze();  // Squeeze to 1D for regression
            auto test_target = test_target_it->second.to(config_.device).squeeze();
            auto train_target = train_target_it->second.to(config_.device).squeeze();

            if (cfg.task == TaskType::Regression) {
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

                auto pred_classes = pred.argmax(1);
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

    // Save training configuration for reproducibility
    save_train_config(archive, config_);

    // Save run metadata if provided (final checkpoint only)
    if (metadata != nullptr) {
        save_run_metadata(archive, *metadata);
    }

    archive.save_to(path);

    // Write human-readable JSON metadata alongside checkpoint
    if (metadata != nullptr) {
        write_metadata_json(path, model_->config(), config_, *metadata, model_->schema());
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
    archive.load_from(path);

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
    archive.load_from(path);
    // Qualify to the free function (checkpoint.cpp); the unqualified name would
    // resolve to this static member and recurse.
    return resolve::load_train_config(archive);
}

RunMetadata Trainer::load_run_metadata(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    return resolve::load_run_metadata(archive);
}

void Trainer::load_weights_into(torch::serialize::InputArchive& archive, ResolveModel& model) {
    // Freshly-constructed model parameters are leaf tensors with
    // requires_grad=true. Calling .copy_() on them directly trips autograd's
    // check_inplace ("a leaf Variable that requires grad is being used in an
    // in-place operation"). Mirror PyTorch's copy-inside-torch.no_grad().
    torch::NoGradGuard no_grad;
    for (const auto& pair : model->named_parameters()) {
        torch::Tensor t;
        if (archive.try_read("param_" + pair.key(), t)) {
            pair.value().copy_(t);
        }
    }
    for (const auto& pair : model->named_buffers()) {
        torch::Tensor t;
        if (archive.try_read("buffer_" + pair.key(), t)) {
            pair.value().copy_(t);
        }
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
    archive.load_from(path);

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

    // CUDA hash mode stores raw species and computes the hash embedding on the
    // fly; the precomputed-hash and other encodings already fold their species
    // representation into test_continuous_ at prepare_data time.
#ifdef RESOLVE_HAS_CUDA
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        auto test_idx = test_indices_.to(config_.device);
        auto test_hash = cuda::compute_batch_hash_embedding_cuda(
            test_idx,
            torch::Tensor(),
            raw_species_ids_.to(config_.device),
            raw_weights_.to(config_.device),
            plot_offsets_.to(config_.device),
            hash_dim_
        );
        test_continuous = torch::cat({test_continuous, test_hash}, /*dim=*/1);
    }
#endif

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

    // Use a subset of test data for diagnostics (max 10000 samples)
    int64_t n_samples = std::min(test_continuous_.size(0), static_cast<int64_t>(10000));
    auto sample_indices = torch::randperm(test_continuous_.size(0)).slice(0, 0, n_samples);

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

    // CUDA hash computation: compute hash embedding for sampled test data
#ifdef RESOLVE_HAS_CUDA
    if (use_cuda_hash_ && config_.device.is_cuda()) {
        // Map sample_indices (test-local) to global plot indices
        auto global_indices = test_indices_.index_select(0, sample_indices).to(config_.device);
        auto sample_hash = cuda::compute_batch_hash_embedding_cuda(
            global_indices,
            torch::Tensor(),
            raw_species_ids_.to(config_.device),
            raw_weights_.to(config_.device),
            plot_offsets_.to(config_.device),
            hash_dim_
        );
        sample_continuous = torch::cat({sample_continuous, sample_hash}, /*dim=*/1);
    }
#endif

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

    // Get predictions and targets
    auto preds = pred_it->second.squeeze().cpu();
    auto targets = target_it->second.cpu();

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

CrossValidationResult Trainer::cross_validate(int n_folds, int seed) {
    if (!data_prepared_) {
        throw std::runtime_error("Data must be prepared before cross-validation");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    CrossValidationResult cv_result;
    cv_result.n_folds = n_folds;

    // Combine train and test data for CV splitting
    int64_t n_train = train_continuous_.size(0);
    int64_t n_test = test_continuous_.size(0);
    int64_t n_total = n_train + n_test;

    // Concatenate all data
    auto all_continuous = torch::cat({train_continuous_, test_continuous_}, 0);
    auto cat_if_both_defined = [](const torch::Tensor& a, const torch::Tensor& b) -> torch::Tensor {
        if (a.defined() && b.defined()) return torch::cat({a, b}, 0);
        return {};
    };
    auto all_genus_ids = cat_if_both_defined(train_genus_ids_, test_genus_ids_);
    auto all_family_ids = cat_if_both_defined(train_family_ids_, test_family_ids_);
    auto all_species_ids = cat_if_both_defined(train_species_ids_, test_species_ids_);
    auto all_species_vector = cat_if_both_defined(train_species_vector_, test_species_vector_);

    // Concatenate pool fields
    auto all_pool_genus_ids = cat_if_both_defined(train_pool_genus_ids_, test_pool_genus_ids_);
    auto all_pool_family_ids = cat_if_both_defined(train_pool_family_ids_, test_pool_family_ids_);
    auto all_pool_weights = cat_if_both_defined(train_pool_weights_, test_pool_weights_);
    auto all_pool_mask = cat_if_both_defined(train_pool_mask_, test_pool_mask_);
    auto all_pool_has_cover = cat_if_both_defined(train_pool_has_cover_, test_pool_has_cover_);

    // Concatenate categorical IDs across train+test for per-fold re-splitting
    auto all_categorical_ids = cat_if_both_defined(train_categorical_ids_, test_categorical_ids_);

    std::unordered_map<std::string, torch::Tensor> all_targets;
    for (const auto& [name, train_tensor] : train_targets_) {
        auto test_it = test_targets_.find(name);
        if (test_it != test_targets_.end()) {
            all_targets[name] = torch::cat({train_tensor, test_it->second}, 0);
        }
    }

    // Create shuffled indices
    std::vector<int64_t> indices(n_total);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Compute fold sizes
    int64_t fold_size = n_total / n_folds;
    int64_t remainder = n_total % n_folds;

    // Store original model state for restoration after each fold
    std::ostringstream original_state_stream;
    {
        torch::serialize::OutputArchive archive;
        model_->save(archive);
        archive.save_to(original_state_stream);
    }
    std::string original_state = original_state_stream.str();

    // Store original scalers
    Scalers original_scalers = scalers_;

    // Metric accumulators
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> metric_values;

    // Run each fold
    int64_t fold_start = 0;
    for (int fold = 0; fold < n_folds; ++fold) {
        config_.log("Cross-validation fold " + std::to_string(fold + 1) + "/" + std::to_string(n_folds));

        // Restore original model weights
        {
            std::istringstream iss(original_state);
            torch::serialize::InputArchive archive;
            archive.load_from(iss);
            model_->load(archive);
        }

        // Determine fold boundaries
        int64_t current_fold_size = fold_size + (fold < remainder ? 1 : 0);
        int64_t fold_end = fold_start + current_fold_size;

        // Create train/test indices for this fold
        std::vector<int64_t> test_indices(indices.begin() + fold_start, indices.begin() + fold_end);
        std::vector<int64_t> train_indices;
        train_indices.reserve(n_total - current_fold_size);
        for (int64_t i = 0; i < fold_start; ++i) {
            train_indices.push_back(indices[i]);
        }
        for (int64_t i = fold_end; i < n_total; ++i) {
            train_indices.push_back(indices[i]);
        }

        auto train_idx = torch::tensor(train_indices);
        auto test_idx = torch::tensor(test_indices);

        // Split data for this fold
        train_continuous_ = all_continuous.index_select(0, train_idx);
        test_continuous_ = all_continuous.index_select(0, test_idx);

        if (all_genus_ids.defined()) {
            train_genus_ids_ = all_genus_ids.index_select(0, train_idx);
            test_genus_ids_ = all_genus_ids.index_select(0, test_idx);
        }
        if (all_family_ids.defined()) {
            train_family_ids_ = all_family_ids.index_select(0, train_idx);
            test_family_ids_ = all_family_ids.index_select(0, test_idx);
        }
        if (all_species_ids.defined()) {
            train_species_ids_ = all_species_ids.index_select(0, train_idx);
            test_species_ids_ = all_species_ids.index_select(0, test_idx);
        }
        if (all_species_vector.defined()) {
            train_species_vector_ = all_species_vector.index_select(0, train_idx);
            test_species_vector_ = all_species_vector.index_select(0, test_idx);
        }

        // Split pool fields for this fold
        auto split_pool_fold = [&](const torch::Tensor& all, torch::Tensor& train_dst, torch::Tensor& test_dst) {
            if (all.defined()) {
                train_dst = all.index_select(0, train_idx);
                test_dst = all.index_select(0, test_idx);
            }
        };
        split_pool_fold(all_pool_genus_ids, train_pool_genus_ids_, test_pool_genus_ids_);
        split_pool_fold(all_pool_family_ids, train_pool_family_ids_, test_pool_family_ids_);
        split_pool_fold(all_pool_weights, train_pool_weights_, test_pool_weights_);
        split_pool_fold(all_pool_mask, train_pool_mask_, test_pool_mask_);
        split_pool_fold(all_pool_has_cover, train_pool_has_cover_, test_pool_has_cover_);
        split_pool_fold(all_categorical_ids, train_categorical_ids_, test_categorical_ids_);

        for (const auto& [name, tensor] : all_targets) {
            train_targets_[name] = tensor.index_select(0, train_idx);
            test_targets_[name] = tensor.index_select(0, test_idx);
        }

        // Recompute scalers for this fold's training data
        if (train_continuous_.size(1) > 0) {
            scalers_.continuous_mean = train_continuous_.mean(0);
            scalers_.continuous_scale = train_continuous_.std(0) + 1e-8f;

            // Apply scaling
            train_continuous_ = (train_continuous_ - scalers_.continuous_mean) / scalers_.continuous_scale;
            test_continuous_ = (test_continuous_ - scalers_.continuous_mean) / scalers_.continuous_scale;
        }

        // Recompute target scalers (only for regression targets)
        for (const auto& cfg : model_->schema().targets) {
            if (cfg.task != TaskType::Regression) continue;

            auto train_it = train_targets_.find(cfg.name);
            if (train_it == train_targets_.end()) continue;

            auto target_mean = train_it->second.mean();
            auto target_scale = train_it->second.std() + 1e-8f;
            scalers_.target_scalers[cfg.name] = {target_mean, target_scale};

            // Apply scaling
            train_targets_[cfg.name] = (train_targets_[cfg.name] - target_mean) / target_scale;
            test_targets_[cfg.name] = (test_targets_[cfg.name] - target_mean) / target_scale;
        }

        // Train this fold
        TrainResult fold_result = fit();
        cv_result.fold_results.push_back(fold_result);

        // Accumulate metrics
        for (const auto& [target, metrics] : fold_result.final_metrics) {
            for (const auto& [metric_name, value] : metrics) {
                metric_values[target][metric_name].push_back(value);
            }
        }

        fold_start = fold_end;
    }

    // Compute mean and std of metrics across folds
    for (const auto& [target, metrics] : metric_values) {
        for (const auto& [metric_name, values] : metrics) {
            float sum = 0.0f;
            for (float v : values) sum += v;
            float mean = sum / values.size();
            cv_result.mean_metrics[target][metric_name] = mean;

            float sq_sum = 0.0f;
            for (float v : values) sq_sum += (v - mean) * (v - mean);
            float std = std::sqrt(sq_sum / values.size());
            cv_result.std_metrics[target][metric_name] = std;
        }
    }

    // Restore original state
    {
        std::istringstream iss(original_state);
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model_->load(archive);
    }
    scalers_ = original_scalers;

    auto end_time = std::chrono::high_resolution_clock::now();
    cv_result.total_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

    // Log summary
    std::ostringstream summary;
    summary << "Cross-validation complete (" << n_folds << " folds, "
            << cv_result.total_time_seconds << "s)\n";
    for (const auto& [target, metrics] : cv_result.mean_metrics) {
        summary << "  " << target << ": ";
        for (const auto& [name, mean] : metrics) {
            auto std_it = cv_result.std_metrics[target].find(name);
            float std = (std_it != cv_result.std_metrics[target].end()) ? std_it->second : 0.0f;
            summary << name << "=" << mean << "±" << std << " ";
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

    // Cantor pairing: label = row * 1e6 + col
    auto block_labels = block_row * 1000000 + block_col;

    // Find unique blocks
    auto unique_result = torch::_unique2(block_labels, /*sorted=*/true, /*return_inverse=*/true);
    auto unique_blocks = std::get<0>(unique_result);
    auto inverse_indices = std::get<1>(unique_result);
    int64_t n_blocks = unique_blocks.size(0);

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

    auto folds = splitter.split(coordinates_);

    config_.log("Spatial block CV: " + std::to_string(n_folds) + " folds, "
                "block_size=" + std::to_string(spatial_config.lat_size) + "x"
                + std::to_string(spatial_config.lon_size) + " deg");

    // Concatenate all data for fold splitting
    auto all_continuous = torch::cat({train_continuous_, test_continuous_}, 0);
    auto cat_if_both = [](const torch::Tensor& a, const torch::Tensor& b) -> torch::Tensor {
        if (a.defined() && b.defined()) return torch::cat({a, b}, 0);
        return {};
    };
    auto all_genus_ids = cat_if_both(train_genus_ids_, test_genus_ids_);
    auto all_family_ids = cat_if_both(train_family_ids_, test_family_ids_);
    auto all_species_ids = cat_if_both(train_species_ids_, test_species_ids_);
    auto all_species_vector = cat_if_both(train_species_vector_, test_species_vector_);
    auto all_pool_genus_ids = cat_if_both(train_pool_genus_ids_, test_pool_genus_ids_);
    auto all_pool_family_ids = cat_if_both(train_pool_family_ids_, test_pool_family_ids_);
    auto all_pool_weights = cat_if_both(train_pool_weights_, test_pool_weights_);
    auto all_pool_mask = cat_if_both(train_pool_mask_, test_pool_mask_);
    auto all_pool_has_cover = cat_if_both(train_pool_has_cover_, test_pool_has_cover_);
    auto all_categorical_ids = cat_if_both(train_categorical_ids_, test_categorical_ids_);

    std::unordered_map<std::string, torch::Tensor> all_targets;
    for (const auto& [name, train_tensor] : train_targets_) {
        auto test_it = test_targets_.find(name);
        if (test_it != test_targets_.end()) {
            all_targets[name] = torch::cat({train_tensor, test_it->second}, 0);
        }
    }

    // Save original model state for restoration after each fold
    std::ostringstream original_state_stream;
    {
        torch::serialize::OutputArchive archive;
        model_->save(archive);
        archive.save_to(original_state_stream);
    }
    std::string original_state = original_state_stream.str();
    Scalers original_scalers = scalers_;

    CrossValidationResult cv_result;
    cv_result.n_folds = n_folds;

    auto cv_start = std::chrono::high_resolution_clock::now();

    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> all_metrics_values;

    for (int fold = 0; fold < n_folds; ++fold) {
        config_.log("\n--- Fold " + std::to_string(fold + 1) + "/" + std::to_string(n_folds) + " ---");

        // Restore model to initial state
        {
            std::istringstream iss(original_state);
            torch::serialize::InputArchive archive;
            archive.load_from(iss);
            model_->load(archive);
        }

        auto& [train_idx, test_idx] = folds[fold];
        auto train_tensor = torch::tensor(train_idx, torch::kInt64);
        auto test_tensor = torch::tensor(test_idx, torch::kInt64);

        config_.log("  Train: " + std::to_string(train_idx.size()) +
                    " plots, Test: " + std::to_string(test_idx.size()) + " plots");

        // Split data for this fold
        train_continuous_ = all_continuous.index_select(0, train_tensor);
        test_continuous_ = all_continuous.index_select(0, test_tensor);

        auto split_if_defined = [&](const torch::Tensor& all, torch::Tensor& train_dst, torch::Tensor& test_dst) {
            if (all.defined()) {
                train_dst = all.index_select(0, train_tensor);
                test_dst = all.index_select(0, test_tensor);
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
            train_targets_[name] = tensor.index_select(0, train_tensor);
            test_targets_[name] = tensor.index_select(0, test_tensor);
        }

        // Recompute scalers for this fold's training data
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

        // Train this fold
        TrainResult fold_result = fit();
        cv_result.fold_results.push_back(fold_result);

        for (const auto& [target, metrics] : fold_result.final_metrics) {
            for (const auto& [metric_name, value] : metrics) {
                all_metrics_values[target][metric_name].push_back(value);
            }
        }
    }

    // Compute mean and std of metrics across folds
    for (const auto& [target, metrics] : all_metrics_values) {
        for (const auto& [metric_name, values] : metrics) {
            float sum = 0.0f;
            for (float v : values) sum += v;
            float mean = sum / values.size();
            cv_result.mean_metrics[target][metric_name] = mean;

            float sq_sum = 0.0f;
            for (float v : values) sq_sum += (v - mean) * (v - mean);
            float std_val = std::sqrt(sq_sum / values.size());
            cv_result.std_metrics[target][metric_name] = std_val;
        }
    }

    // Restore original state
    {
        std::istringstream iss(original_state);
        torch::serialize::InputArchive archive;
        archive.load_from(iss);
        model_->load(archive);
    }
    scalers_ = original_scalers;

    auto cv_end = std::chrono::high_resolution_clock::now();
    cv_result.total_time_seconds = std::chrono::duration<float>(cv_end - cv_start).count();

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

Trainer::PoolTensors Trainer::get_train_pool_tensors() const {
    if (gpu_data_cached_) {
        return {gpu_pool_genus_ids_, gpu_pool_family_ids_,
                gpu_pool_weights_, gpu_pool_mask_, gpu_pool_has_cover_};
    }
    return {to_device_if_defined(train_pool_genus_ids_, config_.device),
            to_device_if_defined(train_pool_family_ids_, config_.device),
            to_device_if_defined(train_pool_weights_, config_.device),
            to_device_if_defined(train_pool_mask_, config_.device),
            to_device_if_defined(train_pool_has_cover_, config_.device)};
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

    continuous = continuous.to(config_.device);
    auto to_dev = [&](torch::Tensor t) -> torch::Tensor {
        return t.defined() ? t.to(config_.device) : t;
    };

    return model_->forward(
        continuous, to_dev(genus_ids), to_dev(family_ids),
        to_dev(species_ids), to_dev(species_vector),
        to_dev(pool_genus_ids), to_dev(pool_family_ids),
        to_dev(pool_weights), to_dev(pool_mask), to_dev(pool_has_cover),
        to_dev(categorical_ids)
    );
}

} // namespace resolve
