// Define _USE_MATH_DEFINES before cmath for M_PI on Windows
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "resolve/pretraining.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <random>

namespace resolve {

// =============================================================================
// Config validation
// =============================================================================

void PretrainConfig::validate() const {
    if (batch_size < 1) {
        throw std::invalid_argument(
            "PretrainConfig.batch_size must be >= 1 (got " +
            std::to_string(batch_size) + "); a batch_size of 0 divides by zero "
            "when computing the step count.");
    }
    if (!(mask_ratio > 0.0f && mask_ratio < 1.0f)) {
        throw std::invalid_argument(
            "PretrainConfig.mask_ratio must be in (0, 1) (got " +
            std::to_string(mask_ratio) + "); mask_ratio >= 1 makes the Block "
            "strategy's block size span all features and its randint bound <= 0.");
    }
    if (!(corruption_rate >= 0.0f && corruption_rate <= 1.0f)) {
        throw std::invalid_argument(
            "PretrainConfig.corruption_rate must be in [0, 1] (got " +
            std::to_string(corruption_rate) + ").");
    }
}

void MLMPretrainConfig::validate() const {
    if (batch_size < 1) {
        throw std::invalid_argument(
            "MLMPretrainConfig.batch_size must be >= 1 (got " +
            std::to_string(batch_size) + ").");
    }
    if (!(mask_prob > 0.0f && mask_prob < 1.0f)) {
        throw std::invalid_argument(
            "MLMPretrainConfig.mask_prob must be in (0, 1) (got " +
            std::to_string(mask_prob) + ").");
    }
}

// =============================================================================
// FeatureMasker
// =============================================================================

FeatureMaskerImpl::FeatureMaskerImpl(
    int64_t n_features,
    float mask_ratio,
    MaskStrategy strategy
) : n_features_(n_features),
    mask_ratio_(mask_ratio),
    strategy_(strategy)
{
    // Learnable mask token: one value per feature dimension
    mask_token_ = register_parameter("mask_token",
        torch::zeros({1, n_features}));
    torch::nn::init::normal_(mask_token_, 0.0, 0.02);
}

torch::Tensor FeatureMaskerImpl::create_mask(int64_t batch_size) const {
    // Generate random mask: 1 = keep, 0 = mask
    auto mask = torch::rand({batch_size, n_features_}, mask_token_.options());

    switch (strategy_) {
        case MaskStrategy::Random:
            // Simple random masking
            mask = (mask > mask_ratio_).to(torch::kFloat32);
            break;

        case MaskStrategy::Block: {
            // Block masking: mask contiguous blocks of features
            // Determine block size from mask_ratio
            int64_t block_size = std::max(static_cast<int64_t>(1),
                static_cast<int64_t>(n_features_ * mask_ratio_));
            mask = torch::ones({batch_size, n_features_}, mask_token_.options());

            // Random start position for each sample
            auto starts = torch::randint(0, n_features_ - block_size + 1, {batch_size},
                torch::TensorOptions().dtype(torch::kLong).device(mask_token_.device()));
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t start = starts[i].item<int64_t>();
                mask[i].slice(0, start, start + block_size).fill_(0.0f);
            }
            break;
        }

        case MaskStrategy::Structured: {
            // Structured masking: mask contiguous feature groups as whole blocks.
            // Groups approximate RESOLVE's feature layout: coords (2), species embedding,
            // covariates. Each sample randomly selects groups to mask.
            mask.fill_(1.0f);  // start with all visible
            // Target ~4 features per group (at least one group).
            constexpr int64_t group_target_size = 4;
            int64_t n_groups = std::max(int64_t(1), n_features_ / group_target_size);
            int64_t group_size = n_features_ / n_groups;
            int64_t n_mask = std::max(int64_t(1), static_cast<int64_t>(n_groups * mask_ratio_));
            for (int64_t i = 0; i < batch_size; ++i) {
                // Random permutation of group indices, mask first n_mask groups
                auto perm = torch::randperm(n_groups, torch::kLong);
                for (int64_t g = 0; g < n_mask; ++g) {
                    int64_t gid = perm[g].item<int64_t>();
                    int64_t start = gid * group_size;
                    int64_t end = (gid == n_groups - 1) ? n_features_ : start + group_size;
                    mask[i].slice(0, start, end).fill_(0.0f);
                }
            }
            break;
        }
    }

    return mask;
}

torch::Tensor FeatureMaskerImpl::apply_mask(torch::Tensor features, torch::Tensor mask) const {
    // Replace masked positions with learnable mask token
    // mask: 1 = keep original, 0 = replace with mask_token
    auto expanded_token = mask_token_.expand_as(features);
    return features * mask + expanded_token * (1.0f - mask);
}

// =============================================================================
// JEPAPredictor
// =============================================================================

JEPAPredictorImpl::JEPAPredictorImpl(
    int64_t latent_dim,
    int hidden_dim,
    int n_layers,
    float dropout
) {
    torch::nn::Sequential layers;

    int64_t in_dim = latent_dim;
    for (int i = 0; i < n_layers; ++i) {
        layers->push_back(torch::nn::Linear(in_dim, hidden_dim));
        layers->push_back(torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_dim})));
        layers->push_back(torch::nn::GELU());
        if (dropout > 0.0f) {
            layers->push_back(torch::nn::Dropout(dropout));
        }
        in_dim = hidden_dim;
    }

    // Final projection back to latent_dim
    layers->push_back(torch::nn::Linear(hidden_dim, latent_dim));

    mlp_ = register_module("mlp", layers);
}

torch::Tensor JEPAPredictorImpl::forward(torch::Tensor context_repr) {
    return mlp_->forward(context_repr);
}

// =============================================================================
// SCARFCorruptor
// =============================================================================

SCARFCorruptorImpl::SCARFCorruptorImpl(
    int64_t n_features,
    float corruption_rate
) : n_features_(n_features),
    corruption_rate_(corruption_rate)
{
}

torch::Tensor SCARFCorruptorImpl::corrupt(torch::Tensor features) const {
    auto batch_size = features.size(0);

    // Create corruption mask: 1 = corrupt, 0 = keep
    auto corruption_mask = (torch::rand({batch_size, n_features_}, features.options()) < corruption_rate_)
        .to(torch::kFloat32);

    // Generate replacement values by shuffling within each feature column
    // For each feature, draw values from other samples in the batch
    auto shuffled = torch::empty_like(features);
    for (int64_t j = 0; j < n_features_; ++j) {
        auto perm = torch::randperm(batch_size,
            torch::TensorOptions().dtype(torch::kLong).device(features.device()));
        shuffled.select(1, j) = features.select(1, j).index_select(0, perm);
    }

    // Apply corruption: replace selected positions with shuffled values
    return features * (1.0f - corruption_mask) + shuffled * corruption_mask;
}

// =============================================================================
// ProjectionHead
// =============================================================================

ProjectionHeadImpl::ProjectionHeadImpl(
    int64_t input_dim,
    int64_t projection_dim
) {
    torch::nn::Sequential layers;
    layers->push_back(torch::nn::Linear(input_dim, input_dim));
    layers->push_back(torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({input_dim})));
    layers->push_back(torch::nn::GELU());
    layers->push_back(torch::nn::Linear(input_dim, projection_dim));

    mlp_ = register_module("mlp", layers);
}

torch::Tensor ProjectionHeadImpl::forward(torch::Tensor x) {
    return mlp_->forward(x);
}

// =============================================================================
// JEPAPretrainer
// =============================================================================

JEPAPretrainer::JEPAPretrainer(
    ResolveModel model,
    const PretrainConfig& config
) : context_encoder_(model),
    config_(config)
{
    config_.validate();
    int64_t latent = context_encoder_->latent_dim();

    // Create predictor
    predictor_ = JEPAPredictor(
        latent,
        config.predictor_hidden_dim,
        config.predictor_n_layers,
        config.predictor_dropout
    );

    // masker_ is (re)built in pretrain() from the actual runtime feature count
    // (continuous.size(1)); a schema-derived construction here would be a dead,
    // and for hash mode inconsistent, first build.

    // Create target encoder as a deep copy (EMA copy)
    // We copy weights from context encoder after the first forward pass
    target_encoder_ = ResolveModel(context_encoder_->schema(), context_encoder_->config());

    // Copy weights from context to target
    {
        torch::NoGradGuard no_grad;
        auto context_params = context_encoder_->named_parameters();
        auto target_params = target_encoder_->named_parameters();
        for (const auto& pair : context_params) {
            for (const auto& t_pair : target_params) {
                if (pair.key() == t_pair.key()) {
                    t_pair.value().copy_(pair.value());
                    break;
                }
            }
        }
        auto context_bufs = context_encoder_->named_buffers();
        auto target_bufs = target_encoder_->named_buffers();
        for (const auto& pair : context_bufs) {
            for (const auto& t_pair : target_bufs) {
                if (pair.key() == t_pair.key()) {
                    t_pair.value().copy_(pair.value());
                    break;
                }
            }
        }
    }

    // Target encoder does not require gradients
    for (auto& p : target_encoder_->parameters()) {
        p.set_requires_grad(false);
    }
}

void JEPAPretrainer::update_target_encoder(float decay) {
    torch::NoGradGuard no_grad;
    auto context_params = context_encoder_->named_parameters();
    auto target_params = target_encoder_->named_parameters();

    for (const auto& pair : context_params) {
        for (const auto& t_pair : target_params) {
            if (pair.key() == t_pair.key()) {
                t_pair.value().mul_(decay).add_(pair.value(), 1.0f - decay);
                break;
            }
        }
    }

    // Buffers (BatchNorm running_mean/running_var/num_batches_tracked) are NOT
    // gradient-tracked and are not updated by the parameter EMA above. Without
    // this the target encoder — always run in eval() — would normalize with the
    // construction-time init stats (mean 0, var 1) for the entire run, breaking
    // the "target is a slow copy of the online encoder" invariant (issue #81).
    // BYOL/JEPA copy buffers from the online encoder rather than EMA-ing them.
    auto context_bufs = context_encoder_->named_buffers();
    auto target_bufs = target_encoder_->named_buffers();
    for (const auto& pair : context_bufs) {
        for (const auto& t_pair : target_bufs) {
            if (pair.key() == t_pair.key()) {
                t_pair.value().copy_(pair.value());
                break;
            }
        }
    }
}

float JEPAPretrainer::get_ema_decay(int step, int total_steps) const {
    // Cosine schedule from ema_decay to ema_decay_end
    float progress = static_cast<float>(step) / std::max(1, total_steps);
    float cosine = 0.5f * (1.0f + std::cos(M_PI * progress));
    return config_.ema_decay_end - (config_.ema_decay_end - config_.ema_decay) * cosine;
}

namespace {

// Mask the species side of a JEPA context view. FeatureMasker only masks the
// continuous block; for hash-mode encoders the species signal lives there and
// is masked, but embed / rank / sparse encoders read composition from the
// species-ID / explicit-vector tensors, which would otherwise pass through the
// context encoder unmasked and let the pretext task read the answer. Dropping a
// `ratio` fraction of species tokens (id -> 0 padding, and the same rows on
// genus/family) and of explicit-vector entries restores a non-trivial task.
struct MaskedSpeciesView {
    torch::Tensor genus, family, species, vector;
};

MaskedSpeciesView mask_species_view(
    const torch::Tensor& genus_ids,
    const torch::Tensor& family_ids,
    const torch::Tensor& species_ids,
    const torch::Tensor& species_vector,
    float ratio
) {
    MaskedSpeciesView v{genus_ids, family_ids, species_ids, species_vector};

    // Drop a `ratio` fraction of valid (non-padding) id tokens to 0. Each id
    // tensor is masked from its OWN shape rather than a single mask derived from
    // species_ids: in embed mode species_ids is [batch, top_k_species] while
    // genus/family are [batch, n_taxonomy_slots], so those column counts need
    // not match and a shared mask would shape-error when indexing genus/family.
    // Never flip existing padding (id == 0) into a "kept" state.
    auto mask_ids = [ratio](const torch::Tensor& ids) -> torch::Tensor {
        if (!ids.defined() || ids.numel() == 0) return ids;
        auto keep = (torch::rand_like(ids.to(torch::kFloat32)) >= ratio);
        auto valid = (ids != 0);
        auto drop = valid & ~keep;
        auto out = ids.clone();
        out.index_put_({drop}, 0);
        return out;
    };

    v.species = mask_ids(species_ids);
    v.genus = mask_ids(genus_ids);
    v.family = mask_ids(family_ids);

    if (species_vector.defined() && species_vector.numel() > 0) {
        auto keep = (torch::rand_like(species_vector) >= ratio).to(species_vector.dtype());
        v.vector = species_vector * keep;
    }

    return v;
}

}  // namespace

PretrainResult JEPAPretrainer::pretrain(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    PretrainResult result;

    int64_t n_samples = continuous.size(0);
    int64_t n_features = continuous.size(1);

    // Recreate masker with correct feature dimension
    masker_ = FeatureMasker(n_features, config_.mask_ratio, config_.mask_strategy);

    // Move everything to device
    context_encoder_->to(config_.device);
    target_encoder_->to(config_.device);
    predictor_->to(config_.device);
    masker_->to(config_.device);

    continuous = continuous.to(config_.device);
    if (genus_ids.defined()) genus_ids = genus_ids.to(config_.device);
    if (family_ids.defined()) family_ids = family_ids.to(config_.device);
    if (species_ids.defined()) species_ids = species_ids.to(config_.device);
    if (species_vector.defined()) species_vector = species_vector.to(config_.device);

    // Optimizer for context encoder + predictor + masker parameters
    std::vector<torch::Tensor> params;
    for (auto& p : context_encoder_->parameters()) params.push_back(p);
    for (auto& p : predictor_->parameters()) params.push_back(p);
    for (auto& p : masker_->parameters()) params.push_back(p);

    auto optimizer = torch::optim::AdamW(
        params,
        torch::optim::AdamWOptions(config_.pretrain_lr)
            .weight_decay(config_.pretrain_weight_decay)
    );

    int total_steps = config_.pretrain_epochs *
        ((n_samples + config_.batch_size - 1) / config_.batch_size);
    int global_step = 0;

    for (int epoch = 0; epoch < config_.pretrain_epochs; ++epoch) {
        context_encoder_->train();
        float epoch_loss = 0.0f;
        int n_batches = 0;

        auto perm = torch::randperm(n_samples,
            torch::TensorOptions().dtype(torch::kLong).device(config_.device));

        for (int64_t start = 0; start < n_samples; start += config_.batch_size) {
            int64_t end = std::min(start + static_cast<int64_t>(config_.batch_size), n_samples);
            auto idx = perm.slice(0, start, end);
            int64_t batch_size = end - start;

            auto batch_cont = continuous.index_select(0, idx);
            auto batch_genus = genus_ids.defined() ? genus_ids.index_select(0, idx) : torch::Tensor();
            auto batch_family = family_ids.defined() ? family_ids.index_select(0, idx) : torch::Tensor();
            auto batch_species = species_ids.defined() ? species_ids.index_select(0, idx) : torch::Tensor();
            auto batch_vector = species_vector.defined() ? species_vector.index_select(0, idx) : torch::Tensor();

            // Create mask for this batch
            auto mask = masker_->create_mask(batch_size);

            // Masked view -> context encoder. Mask the continuous block AND the
            // species-ID / explicit-vector inputs so embed/rank/sparse encoders
            // cannot read the composition straight from an unmasked input.
            auto masked_cont = masker_->apply_mask(batch_cont, mask);
            auto sv = mask_species_view(
                batch_genus, batch_family, batch_species, batch_vector,
                config_.mask_ratio);
            auto context_repr = context_encoder_->get_latent(
                masked_cont, sv.genus, sv.family, sv.species, sv.vector);

            // Predict target representation from context
            auto predicted_repr = predictor_->forward(context_repr);

            // Target representation (unmasked) -> target encoder (no grad)
            torch::Tensor target_repr;
            {
                torch::NoGradGuard no_grad;
                target_encoder_->eval();
                target_repr = target_encoder_->get_latent(
                    batch_cont, batch_genus, batch_family, batch_species, batch_vector);
            }

            // L2 loss between predicted and target representations
            // Normalize representations before computing loss (cosine similarity variant)
            auto pred_norm = torch::nn::functional::normalize(predicted_repr,
                torch::nn::functional::NormalizeFuncOptions().dim(1));
            auto target_norm = torch::nn::functional::normalize(target_repr,
                torch::nn::functional::NormalizeFuncOptions().dim(1));

            auto loss = torch::mse_loss(pred_norm, target_norm);

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(params, 1.0);
            optimizer.step();

            // Update target encoder via EMA
            float decay = get_ema_decay(global_step, total_steps);
            update_target_encoder(decay);

            epoch_loss += loss.item<float>();
            n_batches++;
            global_step++;
        }

        float avg_loss = (n_batches > 0) ? epoch_loss / n_batches : 0.0f;
        result.loss_history.push_back(avg_loss);

        if (epoch % 10 == 0) {
            std::ostringstream msg;
            msg << "Pretrain epoch " << epoch << " - JEPA loss: " << avg_loss;
            config_.log(msg.str());
        }
    }

    result.epochs_completed = config_.pretrain_epochs;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

    return result;
}

// =============================================================================
// SCARFPretrainer
// =============================================================================

SCARFPretrainer::SCARFPretrainer(
    ResolveModel model,
    const PretrainConfig& config
) : model_(model),
    config_(config)
{
    config_.validate();
    int64_t latent = model_->latent_dim();

    // Corruptor will be recreated with correct n_features in pretrain()
    corruptor_ = SCARFCorruptor(1, config.corruption_rate);

    // Projection head: maps latent -> contrastive space
    projection_head_ = ProjectionHead(latent, config.projection_dim);
}

PretrainResult SCARFPretrainer::pretrain(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    PretrainResult result;

    int64_t n_samples = continuous.size(0);
    int64_t n_features = continuous.size(1);

    // Recreate corruptor with correct feature dimension
    corruptor_ = SCARFCorruptor(n_features, config_.corruption_rate);

    // Move to device
    model_->to(config_.device);
    projection_head_->to(config_.device);

    continuous = continuous.to(config_.device);
    if (genus_ids.defined()) genus_ids = genus_ids.to(config_.device);
    if (family_ids.defined()) family_ids = family_ids.to(config_.device);
    if (species_ids.defined()) species_ids = species_ids.to(config_.device);
    if (species_vector.defined()) species_vector = species_vector.to(config_.device);

    // Optimizer for model + projection head
    std::vector<torch::Tensor> params;
    for (auto& p : model_->parameters()) params.push_back(p);
    for (auto& p : projection_head_->parameters()) params.push_back(p);

    auto optimizer = torch::optim::AdamW(
        params,
        torch::optim::AdamWOptions(config_.pretrain_lr)
            .weight_decay(config_.pretrain_weight_decay)
    );

    for (int epoch = 0; epoch < config_.pretrain_epochs; ++epoch) {
        model_->train();
        float epoch_loss = 0.0f;
        int n_batches = 0;

        auto perm = torch::randperm(n_samples,
            torch::TensorOptions().dtype(torch::kLong).device(config_.device));

        for (int64_t start = 0; start < n_samples; start += config_.batch_size) {
            int64_t end = std::min(start + static_cast<int64_t>(config_.batch_size), n_samples);
            auto idx = perm.slice(0, start, end);
            int64_t batch_size = end - start;

            auto batch_cont = continuous.index_select(0, idx);
            auto batch_genus = genus_ids.defined() ? genus_ids.index_select(0, idx) : torch::Tensor();
            auto batch_family = family_ids.defined() ? family_ids.index_select(0, idx) : torch::Tensor();
            auto batch_species = species_ids.defined() ? species_ids.index_select(0, idx) : torch::Tensor();
            auto batch_vector = species_vector.defined() ? species_vector.index_select(0, idx) : torch::Tensor();

            // View 1: original features -> encoder -> projection
            auto repr_1 = model_->get_latent(
                batch_cont, batch_genus, batch_family, batch_species, batch_vector);
            auto z_1 = projection_head_->forward(repr_1);

            // View 2: corrupted features -> encoder -> projection. The species
            // side is masked too (issue #93): for non-hash encoders the species
            // composition lives in the ID / explicit-vector tensors, not in
            // `continuous`, so feeding both views identical species tensors
            // makes the InfoNCE positive pair matchable by species identity
            // alone and the encoder degenerates to passing species through.
            // Mirrors the JEPA mask_species_view fix. (For hash mode the species
            // hash is inside `continuous` and already corrupted; the species-ID
            // tensors are empty there, so masking is a no-op.)
            auto corrupted_cont = corruptor_->corrupt(batch_cont);
            auto masked = mask_species_view(
                batch_genus, batch_family, batch_species, batch_vector,
                config_.corruption_rate);
            auto repr_2 = model_->get_latent(
                corrupted_cont, masked.genus, masked.family, masked.species, masked.vector);
            auto z_2 = projection_head_->forward(repr_2);

            // Normalize projections
            z_1 = torch::nn::functional::normalize(z_1,
                torch::nn::functional::NormalizeFuncOptions().dim(1));
            z_2 = torch::nn::functional::normalize(z_2,
                torch::nn::functional::NormalizeFuncOptions().dim(1));

            // InfoNCE loss
            // Similarity matrix: (batch, batch)
            auto sim = torch::mm(z_1, z_2.t()) / config_.temperature;

            // Labels: diagonal (positive pairs)
            auto labels = torch::arange(batch_size,
                torch::TensorOptions().dtype(torch::kLong).device(config_.device));

            // Symmetric loss: both directions
            auto loss_12 = torch::nn::functional::cross_entropy(sim, labels);
            auto loss_21 = torch::nn::functional::cross_entropy(sim.t(), labels);
            auto loss = (loss_12 + loss_21) * 0.5f;

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(params, 1.0);
            optimizer.step();

            epoch_loss += loss.item<float>();
            n_batches++;
        }

        float avg_loss = (n_batches > 0) ? epoch_loss / n_batches : 0.0f;
        result.loss_history.push_back(avg_loss);

        if (epoch % 10 == 0) {
            std::ostringstream msg;
            msg << "Pretrain epoch " << epoch << " - SCARF loss: " << avg_loss;
            config_.log(msg.str());
        }
    }

    result.epochs_completed = config_.pretrain_epochs;

    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_seconds = std::chrono::duration<float>(end_time - start_time).count();

    return result;
}

// =============================================================================
// Masked Species Pretraining (BERT-style MLM)
// =============================================================================

MaskedSpeciesHeadImpl::MaskedSpeciesHeadImpl(int64_t d_model, int64_t n_species) {
    proj_ = register_module("proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, n_species).bias(false)));
}

torch::Tensor MaskedSpeciesHeadImpl::forward(torch::Tensor token_embeddings) {
    return proj_->forward(token_embeddings);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mask_species_batch(
    torch::Tensor species_ids,
    torch::Tensor valid_mask,
    int64_t n_species,
    float mask_prob
) {
    auto device = species_ids.device();
    auto masked_ids = species_ids.clone();

    // Select positions to mask: valid positions with probability mask_prob
    auto rand_vals = torch::rand_like(valid_mask.to(torch::kFloat32));
    auto mlm_mask = valid_mask & (rand_vals < mask_prob);

    // Save targets before masking
    auto mlm_targets = species_ids.index({mlm_mask}).clone();

    // The 80% subset whose token embedding the encoder swaps for the learned
    // mask embedding. Kept separate from mlm_mask so the 10%-random and
    // 10%-keep ids in masked_ids survive to the encoder (passing the full
    // mlm_mask here would overwrite every masked position with the mask
    // embedding, nullifying the 80/10/10 split entirely).
    auto mask_token_positions = torch::zeros_like(mlm_mask);

    // BERT-style 80/10/10 split, fully vectorized (issue #90). The prior
    // per-token loop called .item() twice per masked position, forcing a
    // GPU->CPU sync thousands of times per batch and serializing the MLM hot
    // loop on CUDA. Here we draw one action value per masked slot, scatter the
    // 80%/10% decisions back onto the (B, S) grid in mlm_mask's row-major order,
    // and apply them with tensor ops (no host sync inside the split).
    auto n_masked = mlm_mask.sum().item<int64_t>();  // single sync (empty-case guard)
    if (n_masked > 0) {
        auto action_rand = torch::rand({n_masked}, torch::TensorOptions().device(device));

        // 80% -> mask token; next 10% -> random species id; last 10% -> keep.
        auto mask_replace = action_rand < 0.8f;                          // (n_masked,)
        auto mask_random = (action_rand >= 0.8f) & (action_rand < 0.9f); // (n_masked,)

        // index_put_ with a boolean mask writes the 1-D values in the row-major
        // order of the mask's true positions — the same order action_rand was
        // drawn in — so these land on exactly the intended grid cells.
        auto replace_grid = torch::zeros_like(mlm_mask);
        auto random_grid = torch::zeros_like(mlm_mask);
        replace_grid.index_put_({mlm_mask}, mask_replace);
        random_grid.index_put_({mlm_mask}, mask_random);

        mask_token_positions = replace_grid;
        masked_ids = masked_ids.masked_fill(replace_grid, 0);

        // Random-id replacement: draw a full-grid random-id tensor and select it
        // only at the random positions. Needs at least two species ids (1..n-1).
        if (n_species > 1) {
            auto rand_ids = torch::randint(
                1, n_species, mlm_mask.sizes(),
                torch::TensorOptions().dtype(torch::kInt64).device(device));
            masked_ids = torch::where(random_grid, rand_ids, masked_ids);
        }
    }

    return {masked_ids, mlm_mask, mlm_targets, mask_token_positions};
}

MaskedSpeciesPretrainer::MaskedSpeciesPretrainer(
    PlotEncoderTransformer encoder,
    int64_t n_species,
    const MLMPretrainConfig& config
) : encoder_(std::move(encoder)),
    n_species_(n_species),
    config_(config)
{
    config_.validate();
    mlm_head_ = MaskedSpeciesHead(encoder_->d_model(), n_species);
}

std::vector<float> MaskedSpeciesPretrainer::pretrain(
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor weights,
    torch::Tensor valid_mask
) {
    auto device = config_.device;
    species_ids = species_ids.to(device);
    if (genus_ids.defined()) genus_ids = genus_ids.to(device);
    if (family_ids.defined()) family_ids = family_ids.to(device);
    if (weights.defined()) weights = weights.to(device);
    valid_mask = valid_mask.to(device);

    encoder_->to(device);
    mlm_head_->to(device);
    encoder_->train();
    mlm_head_->train();

    // Collect all trainable parameters
    std::vector<torch::Tensor> all_params;
    for (auto& p : encoder_->parameters()) all_params.push_back(p);
    for (auto& p : mlm_head_->parameters()) all_params.push_back(p);

    auto optimizer = torch::optim::AdamW(
        all_params,
        torch::optim::AdamWOptions(config_.pretrain_lr)
            .weight_decay(config_.pretrain_weight_decay));

    int64_t n_samples = species_ids.size(0);
    std::vector<float> loss_history;

    for (int epoch = 0; epoch < config_.pretrain_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int n_batches = 0;

        // Shuffle
        auto perm = torch::randperm(n_samples, torch::TensorOptions().dtype(torch::kInt64));

        for (int64_t start = 0; start < n_samples; start += config_.batch_size) {
            int64_t end = std::min(start + static_cast<int64_t>(config_.batch_size), n_samples);
            auto idx = perm.slice(0, start, end).to(device);

            auto batch_sp = species_ids.index_select(0, idx);
            auto batch_mask = valid_mask.index_select(0, idx);
            auto batch_g = (genus_ids.defined() && genus_ids.numel() > 0)
                ? genus_ids.index_select(0, idx) : torch::Tensor();
            auto batch_f = (family_ids.defined() && family_ids.numel() > 0)
                ? family_ids.index_select(0, idx) : torch::Tensor();
            auto batch_w = (weights.defined() && weights.numel() > 0)
                ? weights.index_select(0, idx) : torch::Tensor();

            // Apply BERT masking
            auto [masked_ids, mlm_mask, mlm_targets, mask_token_positions] =
                mask_species_batch(batch_sp, batch_mask, n_species_, config_.mask_prob);

            if (mlm_targets.numel() == 0) continue;

            // Get pre-pooling token embeddings. Only the 80% mask-token subset
            // is replaced with the mask embedding; the 10%-random and 10%-keep
            // ids reach the encoder through masked_ids.
            auto tokens = encoder_->forward_tokens(
                masked_ids, batch_g, batch_f, batch_w, batch_mask, mask_token_positions);

            // Extract masked positions and project to species logits
            auto masked_tokens = tokens.index({mlm_mask});  // (N_masked, d_model)
            auto logits = mlm_head_->forward(masked_tokens);  // (N_masked, n_species)

            // Cross-entropy loss (ignore padding index 0)
            auto loss = torch::nn::functional::cross_entropy(
                logits, mlm_targets,
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0));

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
            n_batches++;
        }

        float avg_loss = (n_batches > 0) ? epoch_loss / n_batches : 0.0f;
        loss_history.push_back(avg_loss);

        if (config_.log) {
            config_.log("MLM pretrain epoch " + std::to_string(epoch + 1) +
                       "/" + std::to_string(config_.pretrain_epochs) +
                       " loss=" + std::to_string(avg_loss));
        }
    }

    return loss_history;
}

} // namespace resolve
