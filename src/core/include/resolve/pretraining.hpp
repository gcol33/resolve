#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include <torch/torch.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace resolve {

// =============================================================================
// Seeded RNG for the pretraining path
// =============================================================================

// Every stochastic quantity a pretext task draws -- the epoch shuffle, the
// feature mask, the species-view mask, the SCARF corruption, the BERT 80/10/10
// split, the VAE reparameterization noise -- goes through this handle. Seeded
// mode owns a dedicated CPU generator, so a fixed PretrainConfig::seed
// reproduces the run AND the global RNG stream is left untouched, matching what
// Trainer::train_epoch does for the training shuffle.
//
// Draws are taken on CPU and moved to the requested device: a CPU generator
// cannot seed a CUDA factory, and drawing host-side makes the stream identical
// on CPU and CUDA runs. The permutations and masks are small next to the
// forward/backward they feed, so the host-to-device copy is not a hot cost.
//
// Copying a PretrainRng shares the underlying generator state (at::Generator is
// a refcounted handle), which is what lets the masker / corruptor / masking
// helpers take it by value with a default and still advance the caller's stream.
class PretrainRng {
public:
    // Global-stream mode: draws come from ATen's default CPU generator, the one
    // torch::manual_seed sets. This is what a direct call to the masker,
    // corruptor or mask_species_batch made outside a pretraining loop gets.
    PretrainRng() = default;

    // Seeded mode: a dedicated CPU generator seeded from `seed`.
    explicit PretrainRng(int seed);

    // Reseed to (seed + epoch + 1) so an epoch's draws depend only on
    // (seed, epoch) and not on how many batches earlier epochs consumed -- the
    // same expression Trainer::train_epoch uses for its per-epoch shuffle
    // generator. No-op in global-stream mode.
    void seed_epoch(int epoch);

    [[nodiscard]] bool is_seeded() const noexcept { return gen_.has_value(); }

    // The dedicated generator, or nullopt in global-stream mode. Hand this to
    // ATen factories that already accept a `Generator?` argument.
    [[nodiscard]] const std::optional<at::Generator>& generator() const noexcept {
        return gen_;
    }

    // Uniform [0, 1) of the given shape. `options` selects dtype (must be
    // floating) and destination device.
    [[nodiscard]] torch::Tensor rand(
        at::IntArrayRef size, const torch::TensorOptions& options);

    // Shape/dtype/device taken from `other`. ATen's rand_like / randn_like have
    // no generator overload, so these route through the sized factories -- the
    // single place that translation is spelled out.
    [[nodiscard]] torch::Tensor rand_like(const torch::Tensor& other);
    [[nodiscard]] torch::Tensor randn_like(const torch::Tensor& other);

    // Uniform integers in [low, high).
    [[nodiscard]] torch::Tensor randint(
        int64_t low, int64_t high, at::IntArrayRef size,
        const torch::TensorOptions& options);

    // int64 permutation of [0, n), materialized on `device`.
    [[nodiscard]] torch::Tensor randperm(int64_t n, torch::Device device);

private:
    // Move a CPU-side draw onto `device` (no-op when it is already there).
    [[nodiscard]] static torch::Tensor to_device(
        torch::Tensor drawn, torch::Device device);

    uint64_t base_seed_ = 0;
    std::optional<at::Generator> gen_;  // nullopt == global stream
};

// =============================================================================
// Configuration for self-supervised pretraining
// =============================================================================

// Masking strategy for feature corruption. All three operate on the continuous
// feature vector; none is species/taxonomy-aware (that masking is done
// separately by mask_species_view for the ID/vector inputs).
enum class MaskStrategy {
    Random,     // Mask random individual features uniformly
    Block,      // Mask one contiguous feature range per sample
    Structured  // Mask whole contiguous feature groups (n_features/4 groups)
};

// Pretraining configuration
struct PretrainConfig {
    // Common settings
    float mask_ratio = 0.3f;                  // Fraction of features to mask
    MaskStrategy mask_strategy = MaskStrategy::Random;
    int pretrain_epochs = 100;
    float pretrain_lr = 1e-4f;
    float pretrain_weight_decay = 1e-4f;
    int batch_size = 4096;
    torch::Device device = torch::kCPU;
    LogCallback log = default_log;

    // Seed for the dedicated pretraining RNG. Every stochastic draw the pretext
    // task makes is taken from a private generator seeded from this value, so a
    // fixed seed reproduces the run and the global RNG stream is left untouched
    // (issue #107). Matches the Trainer / prepare_data seed default. Module
    // dropout is the one draw outside this generator -- torch::nn::Dropout has
    // no generator argument -- so a pretrainer configured with dropout > 0 still
    // advances the global stream through its dropout masks.
    int seed = 42;

    // T-JEPA specific
    float ema_decay = 0.996f;                 // EMA decay for target encoder
    float ema_decay_end = 1.0f;               // Final EMA decay (cosine schedule)
    int predictor_hidden_dim = 256;           // Predictor MLP hidden dimension
    int predictor_n_layers = 2;               // Predictor MLP depth
    float predictor_dropout = 0.1f;

    // SCARF specific
    float corruption_rate = 0.6f;             // Fraction of features to corrupt
    float temperature = 0.1f;                 // InfoNCE temperature
    int projection_dim = 128;                 // Projection head output dimension

    // Throws std::invalid_argument if batch_size < 1, mask_ratio not in (0, 1),
    // or corruption_rate not in [0, 1]. Only reachable via the C-API/Python
    // bindings (the CLI does not expose pretraining), so guard here rather than
    // at a CLI parse site: batch_size == 0 divides by zero when computing steps,
    // and mask_ratio >= 1 makes the Block strategy's randint bound <= 0 (throws).
    // `seed` is unconstrained: every int, negative ones included, is a usable
    // generator seed once converted to uint64.
    void validate() const;
};

// =============================================================================
// FeatureMasker: creates masked views of input data
// =============================================================================

class FeatureMaskerImpl : public torch::nn::Module {
public:
    // `rng` seeds the learnable mask token's initialization. JEPA rebuilds its
    // masker inside pretrain() at the runtime feature count, so an unseeded
    // init here would draw from the global RNG stream in the middle of a
    // pretraining run.
    FeatureMaskerImpl(
        int64_t n_features,
        float mask_ratio = 0.3f,
        MaskStrategy strategy = MaskStrategy::Random,
        PretrainRng rng = PretrainRng{}
    );

    // Create a binary mask (1.0 = keep, 0.0 = mask).
    // Returns: (batch_size, n_features) kFloat32 tensor (not bool — apply_mask
    // multiplies by it, `features * mask + token * (1 - mask)`).
    [[nodiscard]] torch::Tensor create_mask(
        int64_t batch_size, PretrainRng rng = PretrainRng{}) const;

    // Apply mask to features: masked positions replaced with learnable mask token
    // Returns: masked features tensor
    [[nodiscard]] torch::Tensor apply_mask(torch::Tensor features, torch::Tensor mask) const;

    // Accessors
    [[nodiscard]] int64_t n_features() const noexcept { return n_features_; }
    [[nodiscard]] float mask_ratio() const noexcept { return mask_ratio_; }

private:
    int64_t n_features_;
    float mask_ratio_;
    MaskStrategy strategy_;
    torch::Tensor mask_token_;  // Learnable mask token (1, n_features)
};

TORCH_MODULE(FeatureMasker);

// =============================================================================
// JEPAPredictor: predicts target representations from context representations
// =============================================================================

class JEPAPredictorImpl : public torch::nn::Module {
public:
    JEPAPredictorImpl(
        int64_t latent_dim,
        int hidden_dim = 256,
        int n_layers = 2,
        float dropout = 0.1f
    );

    // Predict target representation from context representation
    // context_repr: (batch, latent_dim) from context encoder
    // Returns: (batch, latent_dim) predicted target representation
    [[nodiscard]] torch::Tensor forward(torch::Tensor context_repr);

private:
    torch::nn::Sequential mlp_{nullptr};
};

TORCH_MODULE(JEPAPredictor);

// =============================================================================
// SCARFCorruptor: creates contrastive views by feature corruption
// =============================================================================

class SCARFCorruptorImpl : public torch::nn::Module {
public:
    SCARFCorruptorImpl(
        int64_t n_features,
        float corruption_rate = 0.6f
    );

    // Create corrupted view by replacing random features with values from other samples
    // features: (batch, n_features)
    // Returns: corrupted features (batch, n_features)
    [[nodiscard]] torch::Tensor corrupt(
        torch::Tensor features, PretrainRng rng = PretrainRng{}) const;

    // Accessors
    [[nodiscard]] int64_t n_features() const noexcept { return n_features_; }
    [[nodiscard]] float corruption_rate() const noexcept { return corruption_rate_; }

private:
    int64_t n_features_;
    float corruption_rate_;
};

TORCH_MODULE(SCARFCorruptor);

// =============================================================================
// ProjectionHead: maps latent representations to contrastive space
// =============================================================================

class ProjectionHeadImpl : public torch::nn::Module {
public:
    ProjectionHeadImpl(
        int64_t input_dim,
        int64_t projection_dim = 128
    );

    [[nodiscard]] torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential mlp_{nullptr};
};

TORCH_MODULE(ProjectionHead);

// =============================================================================
// Pretraining result
// =============================================================================

struct PretrainResult {
    std::vector<float> loss_history;
    float total_time_seconds = 0.0f;
    int epochs_completed = 0;
};

// =============================================================================
// Shared pretraining loop
// =============================================================================

// Row-sliced views of the tensors handed to run_pretrain_loop, in the same
// order and count; entries that were undefined (or empty) stay undefined.
using PretrainBatch = std::vector<torch::Tensor>;

// The one thing a pretext task supplies: its per-batch objective. Returns the
// scalar loss to backpropagate, or an UNDEFINED tensor to skip the batch
// entirely (the MLM task skips a batch in which no position was selected for
// masking). Every stochastic quantity the objective needs must be drawn from
// `rng` so the run stays reproducible and the global RNG stream untouched.
using PretrainBatchFn =
    std::function<torch::Tensor(const PretrainBatch&, PretrainRng&)>;

// Optional per-task extension points on the shared loop.
struct PretrainLoopHooks {
    // Start of an epoch, before the shuffle. The VAE advances its KL-annealing
    // weight and resets its per-epoch component accumulators here.
    std::function<void(int epoch)> on_epoch_begin;

    // After optimizer.step() for every non-skipped batch. JEPA runs its EMA
    // target-encoder update here.
    std::function<void()> on_step_end;

    // After the epoch's mean loss has been appended to loss_history and before
    // it is logged. The VAE pushes its reconstruction / KL histories here.
    std::function<void(int epoch, float mean_loss)> on_epoch_end;

    // Task-specific tail appended to the shared per-epoch progress line, so all
    // four tasks log from one site (the VAE reports its recon / KL split).
    std::function<std::string(int epoch)> epoch_detail;
};

// Everything the shared loop needs that is not the objective itself.
struct PretrainLoopSpec {
    // Moved to `device` once, then put in train() mode at the top of every
    // epoch. A frozen side module (JEPA's EMA target encoder, always run in
    // eval()) is deliberately NOT listed here; its owner moves it itself.
    std::vector<std::shared_ptr<torch::nn::Module>> modules;

    // Handed to AdamW and to clip_grad_norm_. Collected by the task because
    // only the task knows which of its modules are trainable. Safe to collect
    // before the loop's device move: Module::to() rebinds parameter storage via
    // set_data(), so these handles follow their module onto the device.
    std::vector<torch::Tensor> params;

    // Row-aligned inputs. inputs[0] is required and supplies the sample count;
    // undefined / empty entries are forwarded to batch_fn undefined rather than
    // sliced.
    std::vector<torch::Tensor> inputs;

    int epochs = 1;
    int batch_size = 4096;
    float lr = 1e-4f;
    float weight_decay = 1e-4f;
    // Max global gradient norm. <= 0 disables clipping, which is what the MLM
    // task asks for; the other three pretext tasks clip at 1.0.
    float grad_clip_norm = 1.0f;
    torch::Device device = torch::kCPU;
    // Seeds the loop's PretrainRng, reseeded per epoch to (seed + epoch + 1).
    int seed = 42;
    LogCallback log = default_log;
    std::string task_name = "pretrain";  // names the task in the progress line
    int log_every = 10;                  // <= 0 silences per-epoch logging
};

// The single epoch/shuffle/optimizer scaffold behind every pretext task: device
// move, AdamW over `params`, per-epoch reseed and shuffle, batch slicing of
// every input, zero_grad / backward / clip / step, loss accumulation, progress
// logging, and the timing + epoch count on the returned result. JEPA, SCARF,
// MLM and the VAE differ only in their batch_fn (and, for JEPA and the VAE, a
// hook), so adding a fifth pretext task is one loss lambda.
PretrainResult run_pretrain_loop(
    PretrainLoopSpec spec,
    const PretrainBatchFn& batch_fn,
    const PretrainLoopHooks& hooks = PretrainLoopHooks{}
);

// =============================================================================
// JEPAPretrainer: T-JEPA self-supervised pretraining orchestrator (C++)
// =============================================================================

class JEPAPretrainer {
public:
    JEPAPretrainer(
        ResolveModel model,
        const PretrainConfig& config = PretrainConfig{}
    );

    // Run pretraining on continuous features (self-supervised, no labels needed)
    // continuous: (n_samples, n_features) - all available data
    // genus_ids, family_ids, species_ids, species_vector: optional auxiliary inputs
    PretrainResult pretrain(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return context_encoder_; }
    // The slow EMA target encoder (no gradients). Exposed so callers can inspect
    // or export it, and so the buffer-sync invariant is testable (issue #81).
    [[nodiscard]] const ResolveModel& target_encoder() const noexcept { return target_encoder_; }
    [[nodiscard]] const PretrainConfig& config() const noexcept { return config_; }

private:
    // Update target encoder via EMA
    void update_target_encoder(float decay);

    // Compute EMA decay with cosine schedule
    [[nodiscard]] float get_ema_decay(int step, int total_steps) const;

    ResolveModel context_encoder_{nullptr};    // Online encoder (gets gradients)
    ResolveModel target_encoder_{nullptr};     // EMA copy (no gradients)
    JEPAPredictor predictor_{nullptr};
    FeatureMasker masker_{nullptr};
    PretrainConfig config_;
};

// =============================================================================
// SCARFPretrainer: SCARF contrastive pretraining orchestrator (C++)
// =============================================================================

class SCARFPretrainer {
public:
    SCARFPretrainer(
        ResolveModel model,
        const PretrainConfig& config = PretrainConfig{}
    );

    // Run contrastive pretraining
    PretrainResult pretrain(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Accessors
    [[nodiscard]] ResolveModel& model() noexcept { return model_; }
    [[nodiscard]] const PretrainConfig& config() const noexcept { return config_; }

private:
    ResolveModel model_{nullptr};
    SCARFCorruptor corruptor_{nullptr};
    ProjectionHead projection_head_{nullptr};
    PretrainConfig config_;
};

// =============================================================================
// Masked Species Pretraining (BERT-style MLM for Transformer encoder)
// =============================================================================

// Configuration for MLM pretraining
struct MLMPretrainConfig {
    float mask_prob = 0.15f;
    int pretrain_epochs = 50;
    float pretrain_lr = 1e-4f;
    float pretrain_weight_decay = 1e-4f;
    int batch_size = 4096;
    torch::Device device = torch::kCPU;
    LogCallback log = default_log;

    // Seed for the dedicated pretraining RNG; see PretrainConfig::seed for the
    // contract it buys (reproducible run, untouched global RNG stream).
    int seed = 42;

    // Throws std::invalid_argument if batch_size < 1 or mask_prob not in (0, 1).
    // `seed` is unconstrained.
    void validate() const;
};

// Linear head projecting token embeddings to species logits
class MaskedSpeciesHeadImpl : public torch::nn::Module {
public:
    MaskedSpeciesHeadImpl(int64_t d_model, int64_t n_species);

    // (N_masked, d_model) → (N_masked, n_species)
    torch::Tensor forward(torch::Tensor token_embeddings);

private:
    torch::nn::Linear proj_{nullptr};
};

TORCH_MODULE(MaskedSpeciesHead);

// Apply BERT-style masking: 15% of valid positions; of those 80% mask, 10% random, 10% keep.
// Returns: (masked_ids, mlm_mask [bool: all masked positions, used for targets/loss],
//           mlm_targets [original IDs at masked positions],
//           mask_token_positions [bool: the 80% subset the encoder replaces with
//           the learned mask embedding]). The 80% subset is kept distinct from
// mlm_mask so the 10%-random and 10%-keep ids in masked_ids actually reach the
// encoder instead of every masked position collapsing to the mask embedding.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mask_species_batch(
    torch::Tensor species_ids,
    torch::Tensor valid_mask,
    int64_t n_species,
    float mask_prob = 0.15f,
    PretrainRng rng = PretrainRng{}
);

// MLM pretrainer: trains a PlotEncoderTransformer via masked species prediction.
// After pretraining, the MLM head is discarded; encoder weights are retained.
class MaskedSpeciesPretrainer {
public:
    MaskedSpeciesPretrainer(
        PlotEncoderTransformer encoder,
        int64_t n_species,
        const MLMPretrainConfig& config = MLMPretrainConfig{}
    );

    // Run MLM pretraining. Returns per-epoch loss history.
    std::vector<float> pretrain(
        torch::Tensor species_ids,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor weights,
        torch::Tensor valid_mask
    );

    // Access encoder after pretraining
    PlotEncoderTransformer& encoder() { return encoder_; }
    [[nodiscard]] const MLMPretrainConfig& config() const noexcept { return config_; }

private:
    PlotEncoderTransformer encoder_{nullptr};
    MaskedSpeciesHead mlm_head_{nullptr};
    MLMPretrainConfig config_;
    int64_t n_species_;
};

} // namespace resolve
