#pragma once

#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include <torch/torch.h>

namespace resolve {

// =============================================================================
// Configuration for self-supervised pretraining
// =============================================================================

// Masking strategy for feature corruption
enum class MaskStrategy {
    Random,     // Mask random features uniformly
    Block,      // Mask contiguous blocks of species features
    Structured  // Mask by taxonomic group (all species in a genus)
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
};

// =============================================================================
// FeatureMasker: creates masked views of input data
// =============================================================================

class FeatureMaskerImpl : public torch::nn::Module {
public:
    FeatureMaskerImpl(
        int64_t n_features,
        float mask_ratio = 0.3f,
        MaskStrategy strategy = MaskStrategy::Random
    );

    // Create a binary mask (1 = keep, 0 = mask)
    // Returns: (batch_size, n_features) boolean tensor
    [[nodiscard]] torch::Tensor create_mask(int64_t batch_size) const;

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
    [[nodiscard]] torch::Tensor corrupt(torch::Tensor features) const;

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
    float mask_prob = 0.15f
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

private:
    PlotEncoderTransformer encoder_{nullptr};
    MaskedSpeciesHead mlm_head_{nullptr};
    MLMPretrainConfig config_;
    int64_t n_species_;
};

} // namespace resolve
