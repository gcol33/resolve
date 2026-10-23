// SpeciesVAE (encode / decode / reparameterise / loss terms / projection
// weights) and the VAEPretrainer that fits it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/vae.hpp"

#include <torch/torch.h>

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// SpeciesVAE
// ============================================================================

TEST_CASE("SpeciesVAE forward returns triple", "[vae]") {
    VAEConfig config;
    config.latent_dim = 32;
    config.encoder_dims = {64, 32};
    config.decoder_dims = {32, 64};

    SpeciesVAE vae(100, config);
    auto species = torch::randn({8, 100});
    auto [recon, mu, logvar] = vae->forward(species);

    REQUIRE(recon.size(0) == 8);
    REQUIRE(recon.size(1) == 100);   // Reconstructed species vector
    REQUIRE(mu.size(0) == 8);
    REQUIRE(mu.size(1) == 32);       // Latent dim
    REQUIRE(logvar.size(0) == 8);
    REQUIRE(logvar.size(1) == 32);
}

TEST_CASE("SpeciesVAE encode", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};

    SpeciesVAE vae(50, config);
    auto species = torch::randn({8, 50});
    auto [mu, logvar] = vae->encode(species);

    REQUIRE(mu.size(0) == 8);
    REQUIRE(mu.size(1) == 16);
    REQUIRE(logvar.size(0) == 8);
    REQUIRE(logvar.size(1) == 16);
}

TEST_CASE("SpeciesVAE decode", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};

    SpeciesVAE vae(50, config);
    auto z = torch::randn({8, 16});
    auto decoded = vae->decode(z);

    REQUIRE(decoded.size(0) == 8);
    REQUIRE(decoded.size(1) == 50);
}

TEST_CASE("SpeciesVAE reparameterize", "[vae]") {
    auto mu = torch::zeros({8, 16});
    auto logvar = torch::zeros({8, 16});  // var = 1, so z ~ N(0, 1)

    auto z = SpeciesVAEImpl::reparameterize(mu, logvar);
    REQUIRE(z.size(0) == 8);
    REQUIRE(z.size(1) == 16);

    // Mean should be ~0 for many samples
    auto mean = z.mean().item<float>();
    REQUIRE_THAT(mean, WithinAbs(0.0f, 0.5f));
}

TEST_CASE("SpeciesVAE vae_loss", "[vae]") {
    auto recon = torch::randn({8, 50});
    auto input = torch::randn({8, 50});
    auto mu = torch::randn({8, 16});
    auto logvar = torch::randn({8, 16});

    auto loss = SpeciesVAEImpl::vae_loss(recon, input, mu, logvar, 1.0f);
    REQUIRE(loss.dim() == 0);  // Scalar
    REQUIRE(loss.item<float>() > 0.0f);
}

TEST_CASE("SpeciesVAE kl_divergence sums over latent dim", "[vae]") {
    // mu = 1, log_var = 0 over a 16-dim latent, batch 8:
    //   per element: 1 + 0 - 1^2 - exp(0) = -1
    //   -0.5 * sum_over_16(-1) = 8, meaned over batch = 8.
    // A mean over the latent dimension would give 0.5.
    int64_t batch = 8, latent = 16;
    auto mu = torch::ones({batch, latent});
    auto logvar = torch::zeros({batch, latent});

    auto kl = SpeciesVAEImpl::kl_divergence(mu, logvar);
    REQUIRE(kl.dim() == 0);
    REQUIRE_THAT(kl.item<float>(), WithinAbs(8.0f, 1e-4f));

    // Standard normal posterior (mu = 0, log_var = 0) has zero KL.
    auto kl_zero = SpeciesVAEImpl::kl_divergence(
        torch::zeros({batch, latent}), torch::zeros({batch, latent}));
    REQUIRE_THAT(kl_zero.item<float>(), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("SpeciesVAE projection weights are (encoder_dims[0], input_dim)", "[vae]") {
    const int64_t input_dim = 50;
    VAEConfig cfg;  // encoder_dims default {512, 256, 128}, latent_dim 64
    SpeciesVAE vae(input_dim, cfg);
    auto w = vae->get_projection_weights();
    REQUIRE(w.dim() == 2);
    REQUIRE(w.size(0) == cfg.encoder_dims.front());  // 512, NOT latent_dim
    REQUIRE(w.size(1) == input_dim);                 // 50
}

TEST_CASE("SpeciesVAE accessors", "[vae]") {
    VAEConfig config;
    config.latent_dim = 32;
    config.encoder_dims = {64};
    config.decoder_dims = {64};

    SpeciesVAE vae(100, config);
    REQUIRE(vae->input_dim() == 100);
    REQUIRE(vae->latent_dim() == 32);
}

// ============================================================================
// VAEPretrainer
// ============================================================================

TEST_CASE("VAEPretrainer construction", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};
    config.pretrain_epochs = 2;
    config.batch_size = 16;

    VAEPretrainer pretrainer(50, config);

    auto& vae = pretrainer.vae();
    REQUIRE(vae->input_dim() == 50);
    REQUIRE(vae->latent_dim() == 16);
}

TEST_CASE("VAEPretrainer pretrain runs", "[vae]") {
    VAEConfig config;
    config.latent_dim = 8;
    config.encoder_dims = {16};
    config.decoder_dims = {16};
    config.pretrain_epochs = 2;
    config.batch_size = 8;

    VAEPretrainer pretrainer(20, config);
    auto species = torch::randn({32, 20}).abs();

    auto result = pretrainer.pretrain(species);

    REQUIRE(result.epochs_completed == 2);
    REQUIRE(result.loss_history.size() == 2);
    REQUIRE(result.recon_loss_history.size() == 2);
    REQUIRE(result.kl_loss_history.size() == 2);
    REQUIRE(result.total_time_seconds > 0.0f);
}
