#pragma once

// TraitNet: a trait-based multi-species network. The environment of a plot and
// the traits of every species are encoded separately and combined per species
// into one prediction per species.

#include "resolve/types.hpp"  // TraitInteractionMode

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// Bilinear interaction between environment and traits.
class BilinearTraitInteractionImpl : public torch::nn::Module {
public:
    BilinearTraitInteractionImpl(
        int64_t env_dim,
        int64_t trait_dim,
        int64_t output_dim
    );

    // env: (batch, env_dim) - environmental features
    // traits: (n_species, trait_dim) - species traits
    // Returns: (batch, n_species, output_dim)
    torch::Tensor forward(torch::Tensor env, torch::Tensor traits);

private:
    int64_t output_dim_;

    // Bilinear weight: (output_dim, env_dim, trait_dim)
    torch::Tensor weight_;
    torch::Tensor bias_;
};

TORCH_MODULE(BilinearTraitInteraction);

// A trait encoder with its own weights for every species: one
// (trait_dim, hidden_dim) matrix and bias per species per layer, applied with
// an einsum, where the shared encoder applies one matrix to every species. This
// is what TraitNetConfig::shared_trait_encoder = false asks for, and it costs
// n_species times the parameters, so it suits a small species pool rather than
// a whole regional vocabulary.
class PerSpeciesTraitEncoderImpl : public torch::nn::Module {
public:
    PerSpeciesTraitEncoderImpl(
        int64_t n_species,
        int64_t trait_dim,
        int64_t hidden_dim,
        int64_t n_layers = 2,
        float dropout = 0.1f
    );

    // traits: (n_species, trait_dim) -> (n_species, hidden_dim)
    torch::Tensor forward(torch::Tensor traits);

private:
    int64_t n_species_;
    int64_t trait_dim_;

    // Per layer: weight (n_species, in_dim, hidden_dim), bias (n_species,
    // hidden_dim), plus a normalization shared across species.
    std::vector<torch::Tensor> weights_;
    std::vector<torch::Tensor> biases_;
    torch::nn::ModuleList norms_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(PerSpeciesTraitEncoder);

class TraitNetEncoderImpl : public torch::nn::Module {
public:
    TraitNetEncoderImpl(
        int64_t env_dim,           // Number of environmental features
        int64_t trait_dim,         // Number of trait features per species
        int64_t n_species,         // Number of species
        int64_t hidden_dim = 128,
        int64_t n_layers = 2,
        float dropout = 0.1f,
        // TraitNetConfig::interaction_dim: the width the environment-trait
        // combination produces, which the head then reads.
        int64_t interaction_dim = 0,  // 0 = hidden_dim
        // TraitNetConfig::interaction: how the two are combined.
        TraitInteractionMode interaction = TraitInteractionMode::Bilinear,
        // TraitNetConfig::shared_trait_encoder: one trait encoder for every
        // species, or one per species.
        bool shared_trait_encoder = true
    );

    // env: (batch, env_dim) - environmental features
    // traits: (n_species, trait_dim); undefined reads the matrix set_traits()
    //   stored.
    // Returns: (batch, n_species) - species predictions
    torch::Tensor forward(torch::Tensor env, torch::Tensor traits = {});

    // Store the species trait matrix (n_species, trait_dim) the forward reads
    // when given none. It is a buffer of the module: it follows the module's
    // device and is saved with the weights.
    void set_traits(torch::Tensor traits);

    [[nodiscard]] bool has_traits() const { return traits_.size(0) > 0; }

    // The buffer set_traits() fills, (0, trait_dim) until then.
    [[nodiscard]] const torch::Tensor& traits_buffer() const noexcept { return traits_; }

    [[nodiscard]] int64_t output_dim() const noexcept { return n_species_; }

private:
    int64_t trait_dim_;
    int64_t n_species_;

    torch::Tensor traits_;

    torch::nn::Sequential env_encoder_{nullptr};

    // Trait encoder: one of the two, by shared_trait_encoder.
    torch::nn::Sequential trait_encoder_{nullptr};
    PerSpeciesTraitEncoder per_species_trait_encoder_{nullptr};

    // How the environment and a species' traits are combined, and which of the
    // three below is built.
    TraitInteractionMode interaction_mode_;
    int64_t interaction_dim_;

    // Bilinear interaction
    BilinearTraitInteraction interaction_{nullptr};
    // MLP interaction: the two representations concatenated and projected.
    torch::nn::Linear interaction_mlp_{nullptr};
    // Attention interaction: the environment queries each species' traits and
    // scales that species' value vector by the attention it receives.
    torch::nn::Linear interaction_query_{nullptr};
    torch::nn::Linear interaction_key_{nullptr};
    torch::nn::Linear interaction_value_{nullptr};

    torch::nn::Linear output_proj_{nullptr};
};

TORCH_MODULE(TraitNetEncoder);

}  // namespace resolve
