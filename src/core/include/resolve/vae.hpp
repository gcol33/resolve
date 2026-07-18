#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>

namespace resolve {

// =============================================================================
// VAE Configuration
// =============================================================================

struct VAEConfig {
    int64_t latent_dim = 64;           // VAE latent space dimension
    std::vector<int64_t> encoder_dims = {512, 256, 128};  // Encoder hidden layers
    std::vector<int64_t> decoder_dims = {128, 256, 512};  // Decoder hidden layers
    float dropout = 0.1f;
    float kl_weight = 1.0f;            // Beta-VAE weight on KL term
    float kl_anneal_epochs = 20;       // Epochs to linearly anneal KL weight from 0 to kl_weight
    int pretrain_epochs = 100;
    float pretrain_lr = 1e-3f;
    int batch_size = 4096;
    torch::Device device = torch::kCPU;
    LogCallback log = default_log;
};

// =============================================================================
// SpeciesVAE: Variational autoencoder on species abundance vectors
// =============================================================================

// Pretrain on all available species vectors (unlabeled data).
// After training, the encoder can be used to initialize the species projection
// layer in PlotEncoderSparse.

class SpeciesVAEImpl : public torch::nn::Module {
public:
    SpeciesVAEImpl(
        int64_t input_dim,            // Number of species (vocabulary size)
        const VAEConfig& config = VAEConfig{}
    );

    // Full forward pass: encode -> reparameterize -> decode
    // Returns: (reconstruction, mu, log_var)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor species_vector
    );

    // Encode species vector to latent representation
    // Returns: (mu, log_var)
    std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor species_vector);

    // Decode latent representation back to species vector
    torch::Tensor decode(torch::Tensor z);

    // Reparameterization trick: z = mu + eps * exp(0.5 * log_var)
    [[nodiscard]] static torch::Tensor reparameterize(
        torch::Tensor mu, torch::Tensor log_var);

    // Gaussian KL divergence D_KL(q(z|x) || N(0, I)) for a diagonal posterior:
    //   -0.5 * sum_j (1 + log_var_j - mu_j^2 - exp(log_var_j))
    // summed over the latent dimension (dim=1) and meaned over the batch, giving
    // the true per-sample ELBO KL. Single source of truth for vae_loss and the
    // pretraining loop so the two cannot drift (issue #96).
    [[nodiscard]] static torch::Tensor kl_divergence(
        torch::Tensor mu, torch::Tensor log_var);

    // Compute VAE loss: reconstruction + beta * KL divergence
    [[nodiscard]] static torch::Tensor vae_loss(
        torch::Tensor reconstruction,
        torch::Tensor input,
        torch::Tensor mu,
        torch::Tensor log_var,
        float kl_weight = 1.0f
    );

    // Encoder weights for warm-starting the species projection in
    // PlotEncoderSparse / TabularAdapter. Returns the VAE encoder's FIRST linear
    // layer weight, shape (encoder_dims[0], input_dim) == (first_hidden_dim,
    // n_species) in nn::Linear (out, in) convention. This directly initializes a
    // Linear(n_species, species_embed_dim) species projection when
    // species_embed_dim == encoder_dims[0]; the caller is responsible for that
    // shape match (issue #85 — the prior doc claimed a transposed (input_dim,
    // latent_dim) shape the implementation never returned).
    [[nodiscard]] torch::Tensor get_projection_weights() const;

    // Accessors
    [[nodiscard]] int64_t input_dim() const noexcept { return input_dim_; }
    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }

private:
    int64_t input_dim_;
    int64_t latent_dim_;

    torch::nn::Sequential encoder_{nullptr};
    torch::nn::Linear mu_layer_{nullptr};
    torch::nn::Linear logvar_layer_{nullptr};
    torch::nn::Sequential decoder_{nullptr};
};

TORCH_MODULE(SpeciesVAE);

// =============================================================================
// VAE Pretrainer
// =============================================================================

struct VAEPretrainResult {
    std::vector<float> loss_history;
    std::vector<float> recon_loss_history;
    std::vector<float> kl_loss_history;
    float total_time_seconds = 0.0f;
    int epochs_completed = 0;
};

class VAEPretrainer {
public:
    VAEPretrainer(
        int64_t n_species,
        const VAEConfig& config = VAEConfig{}
    );

    // Train VAE on species abundance vectors (unsupervised)
    VAEPretrainResult pretrain(torch::Tensor species_vectors);

    // Get the trained VAE
    [[nodiscard]] SpeciesVAE& vae() noexcept { return vae_; }
    [[nodiscard]] const VAEConfig& config() const noexcept { return config_; }

private:
    SpeciesVAE vae_{nullptr};
    VAEConfig config_;
};

} // namespace resolve
