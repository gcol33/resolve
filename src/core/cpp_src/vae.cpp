#include "resolve/vae.hpp"
#include <chrono>
#include <sstream>
#include <utility>

namespace resolve {

// =============================================================================
// SpeciesVAE
// =============================================================================

SpeciesVAEImpl::SpeciesVAEImpl(
    int64_t input_dim,
    const VAEConfig& config
) : input_dim_(input_dim),
    latent_dim_(config.latent_dim)
{
    // Build encoder
    torch::nn::Sequential enc;
    int64_t in_dim = input_dim;
    for (auto hidden : config.encoder_dims) {
        enc->push_back(torch::nn::Linear(in_dim, hidden));
        enc->push_back(torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden})));
        enc->push_back(torch::nn::GELU());
        if (config.dropout > 0.0f) {
            enc->push_back(torch::nn::Dropout(config.dropout));
        }
        in_dim = hidden;
    }
    encoder_ = register_module("encoder", enc);

    // Latent projections
    mu_layer_ = register_module("mu", torch::nn::Linear(in_dim, config.latent_dim));
    logvar_layer_ = register_module("logvar", torch::nn::Linear(in_dim, config.latent_dim));

    // Build decoder (mirror of encoder)
    torch::nn::Sequential dec;
    in_dim = config.latent_dim;
    for (auto hidden : config.decoder_dims) {
        dec->push_back(torch::nn::Linear(in_dim, hidden));
        dec->push_back(torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden})));
        dec->push_back(torch::nn::GELU());
        if (config.dropout > 0.0f) {
            dec->push_back(torch::nn::Dropout(config.dropout));
        }
        in_dim = hidden;
    }
    // Final output layer (no activation, raw reconstruction)
    dec->push_back(torch::nn::Linear(in_dim, input_dim));
    decoder_ = register_module("decoder", dec);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SpeciesVAEImpl::forward(
    torch::Tensor species_vector,
    PretrainRng rng
) {
    auto [mu, log_var] = encode(species_vector);
    auto z = reparameterize(mu, log_var, rng);
    auto reconstruction = decode(z);
    return {reconstruction, mu, log_var};
}

std::pair<torch::Tensor, torch::Tensor> SpeciesVAEImpl::encode(torch::Tensor species_vector) {
    auto h = encoder_->forward(species_vector);
    auto mu = mu_layer_->forward(h);
    auto log_var = logvar_layer_->forward(h);
    return {mu, log_var};
}

torch::Tensor SpeciesVAEImpl::decode(torch::Tensor z) {
    return decoder_->forward(z);
}

torch::Tensor SpeciesVAEImpl::reparameterize(
    torch::Tensor mu, torch::Tensor log_var, PretrainRng rng
) {
    auto std = torch::exp(0.5f * log_var);
    auto eps = rng.randn_like(std);
    return mu + eps * std;
}

torch::Tensor SpeciesVAEImpl::kl_divergence(torch::Tensor mu, torch::Tensor log_var) {
    // -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) over the latent dim, meaned
    // over the batch. Summing over latents (not meaning) matches the textbook
    // ELBO so kl_weight = 1 is beta = 1 (issue #96).
    return (-0.5f * (1.0f + log_var - mu.pow(2) - log_var.exp()).sum(/*dim=*/1)).mean();
}

torch::Tensor SpeciesVAEImpl::vae_loss(
    torch::Tensor reconstruction,
    torch::Tensor input,
    torch::Tensor mu,
    torch::Tensor log_var,
    float kl_weight
) {
    // Reconstruction loss: MSE for continuous abundance vectors
    auto recon_loss = torch::mse_loss(reconstruction, input, torch::Reduction::Mean);

    return recon_loss + kl_weight * kl_divergence(mu, log_var);
}

torch::Tensor SpeciesVAEImpl::get_projection_weights() const {
    // First encoder linear weight, shape (encoder_dims[0], input_dim) in the
    // nn::Linear (out_features, in_features) convention. Warm-starts a
    // Linear(n_species, species_embed_dim) species projection when
    // species_embed_dim == encoder_dims[0] (see header contract, issue #85).
    for (const auto& module : encoder_->children()) {
        auto linear = module->as<torch::nn::LinearImpl>();
        if (linear) {
            return linear->weight.detach().clone();  // (encoder_dims[0], input_dim)
        }
    }
    throw std::runtime_error("No linear layer found in VAE encoder");
}

// =============================================================================
// VAE Pretrainer
// =============================================================================

VAEPretrainer::VAEPretrainer(
    int64_t n_species,
    const VAEConfig& config
) : config_(config)
{
    vae_ = SpeciesVAE(n_species, config);
}

VAEPretrainResult VAEPretrainer::pretrain(torch::Tensor species_vectors) {
    VAEPretrainResult result;

    // Per-epoch accumulators for the ELBO's two components, which the shared
    // loop does not track (it only knows the scalar it backpropagates).
    float epoch_recon = 0.0f;
    float epoch_kl = 0.0f;
    int epoch_batches = 0;
    float kl_w = config_.kl_weight;

    PretrainLoopSpec spec;
    spec.modules = {vae_.ptr()};
    spec.params = vae_->parameters();
    spec.inputs = {species_vectors};
    spec.epochs = config_.pretrain_epochs;
    spec.batch_size = config_.batch_size;
    spec.lr = config_.pretrain_lr;
    spec.weight_decay = config_.pretrain_weight_decay;
    spec.device = config_.device;
    spec.seed = config_.seed;
    spec.log = config_.log;
    spec.task_name = "VAE";

    auto batch_fn = [&](const PretrainBatch& batch, PretrainRng& rng) {
        const auto& x = batch[0];
        auto [recon, mu, log_var] = vae_->forward(x, rng);

        // Track the components separately for the per-epoch histories.
        auto recon_loss = torch::mse_loss(recon, x, torch::Reduction::Mean);
        auto kl_loss = SpeciesVAEImpl::kl_divergence(mu, log_var);

        epoch_recon += recon_loss.item<float>();
        epoch_kl += kl_loss.item<float>();
        ++epoch_batches;

        return recon_loss + kl_w * kl_loss;
    };

    PretrainLoopHooks hooks;
    hooks.on_epoch_begin = [&](int epoch) {
        epoch_recon = 0.0f;
        epoch_kl = 0.0f;
        epoch_batches = 0;

        // KL annealing: linearly increase from 0 to kl_weight
        kl_w = (config_.kl_anneal_epochs > 0 && epoch < config_.kl_anneal_epochs)
            ? config_.kl_weight * static_cast<float>(epoch) / config_.kl_anneal_epochs
            : config_.kl_weight;
    };
    hooks.on_epoch_end = [&](int /*epoch*/, float /*mean_loss*/) {
        // Guard against an empty pretraining set (no batches) so the loss
        // histories record 0 rather than NaN from a divide-by-zero.
        const float inv_nb = epoch_batches > 0
            ? 1.0f / static_cast<float>(epoch_batches) : 0.0f;
        result.recon_loss_history.push_back(epoch_recon * inv_nb);
        result.kl_loss_history.push_back(epoch_kl * inv_nb);
    };
    hooks.epoch_detail = [&](int /*epoch*/) {
        std::ostringstream detail;
        detail << " recon: " << result.recon_loss_history.back()
               << " kl: " << result.kl_loss_history.back();
        return detail.str();
    };

    auto loop = run_pretrain_loop(std::move(spec), batch_fn, hooks);
    result.loss_history = std::move(loop.loss_history);
    result.epochs_completed = loop.epochs_completed;
    result.total_time_seconds = loop.total_time_seconds;

    return result;
}

} // namespace resolve
