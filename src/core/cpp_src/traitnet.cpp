#include "resolve/traitnet.hpp"
#include "resolve/ln_mlp.hpp"

#include <cmath>
#include <string>

namespace resolve {

// =============================================================================
// Bilinear Trait Interaction
// =============================================================================

BilinearTraitInteractionImpl::BilinearTraitInteractionImpl(
    int64_t env_dim,
    int64_t trait_dim,
    int64_t output_dim
) : output_dim_(output_dim) {
    // out = env^T * W * traits
    weight_ = register_parameter("weight",
        torch::randn({output_dim, env_dim, trait_dim}) * 0.01f);
    bias_ = register_parameter("bias", torch::zeros({output_dim}));
}

torch::Tensor BilinearTraitInteractionImpl::forward(
    torch::Tensor env,
    torch::Tensor traits
) {
    // (batch, env) x (out, env, trait) -> (batch, out, trait)
    auto env_transformed = torch::einsum("be,oet->bot", {env, weight_});
    // (batch, out, trait) x (species, trait) -> (batch, out, species)
    auto output = torch::einsum("bot,st->bos", {env_transformed, traits});
    output = output + bias_.view({1, output_dim_, 1});
    return output.permute({0, 2, 1});  // (batch, species, out)
}

// =============================================================================
// Per-species trait encoder
// =============================================================================

PerSpeciesTraitEncoderImpl::PerSpeciesTraitEncoderImpl(
    int64_t n_species,
    int64_t trait_dim,
    int64_t hidden_dim,
    int64_t n_layers,
    float dropout
) : n_species_(n_species), trait_dim_(trait_dim) {
    TORCH_CHECK(n_species > 0,
        "PerSpeciesTraitEncoder needs at least one species, got ", n_species);
    TORCH_CHECK(n_layers > 0,
        "PerSpeciesTraitEncoder needs at least one layer, got ", n_layers);

    norms_ = register_module("norms", torch::nn::ModuleList());
    int64_t current_dim = trait_dim;
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        // Kaiming-style scale on the fan-in, as nn::Linear initializes.
        const double bound = 1.0 / std::sqrt(static_cast<double>(current_dim));
        weights_.push_back(register_parameter(
            "weight_" + std::to_string(layer),
            (torch::rand({n_species, current_dim, hidden_dim}) * 2.0 - 1.0) * bound));
        biases_.push_back(register_parameter(
            "bias_" + std::to_string(layer),
            (torch::rand({n_species, hidden_dim}) * 2.0 - 1.0) * bound));
        norms_->push_back(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
        current_dim = hidden_dim;
    }
    if (dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    }
}

torch::Tensor PerSpeciesTraitEncoderImpl::forward(torch::Tensor traits) {
    TORCH_CHECK(traits.size(0) == n_species_,
        "PerSpeciesTraitEncoder was built for ", n_species_,
        " species, got ", traits.size(0));
    TORCH_CHECK(traits.size(1) == trait_dim_,
        "PerSpeciesTraitEncoder was built for ", trait_dim_,
        " trait features, got ", traits.size(1));

    auto h = traits;
    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        // Each species reads its own matrix: (species, in) x (species, in, out).
        h = torch::einsum("si,sio->so", {h, weights_[layer]}) + biases_[layer];
        h = norms_->ptr(layer)->as<torch::nn::LayerNormImpl>()->forward(h);
        h = torch::gelu(h);
        if (dropout_) h = dropout_->forward(h);
    }
    return h;
}

// =============================================================================
// TraitNet Encoder
// =============================================================================

TraitNetEncoderImpl::TraitNetEncoderImpl(
    int64_t env_dim,
    int64_t trait_dim,
    int64_t n_species,
    int64_t hidden_dim,
    int64_t n_layers,
    float dropout,
    int64_t interaction_dim,
    TraitInteractionMode interaction,
    bool shared_trait_encoder
) : trait_dim_(trait_dim), n_species_(n_species), interaction_mode_(interaction),
    interaction_dim_(interaction_dim > 0 ? interaction_dim : hidden_dim) {
    traits_ = register_buffer("traits", torch::zeros({0, trait_dim}));

    const std::vector<int64_t> stages(static_cast<size_t>(n_layers), hidden_dim);
    env_encoder_ = register_module("env_encoder", make_ln_gelu_mlp(env_dim, stages, dropout));

    // Trait encoder: one set of weights for every species, or one set per
    // species (TraitNetConfig::shared_trait_encoder).
    if (shared_trait_encoder) {
        trait_encoder_ = register_module("trait_encoder",
            make_ln_gelu_mlp(trait_dim, stages, dropout));
    } else {
        per_species_trait_encoder_ = register_module("per_species_trait_encoder",
            PerSpeciesTraitEncoder(n_species, trait_dim, hidden_dim, n_layers,
                                   dropout));
    }

    // How the environment and a species' traits are combined into the
    // per-species representation the head reads. All three produce
    // (batch, n_species, interaction_dim).
    switch (interaction_mode_) {
        case TraitInteractionMode::Bilinear:
            interaction_ = register_module("interaction",
                BilinearTraitInteraction(hidden_dim, hidden_dim, interaction_dim_));
            break;
        case TraitInteractionMode::MLP:
            // The two representations side by side, projected once.
            interaction_mlp_ = register_module("interaction_mlp",
                torch::nn::Linear(2 * hidden_dim, interaction_dim_));
            break;
        case TraitInteractionMode::Attention:
            // The environment queries the species: how much attention a
            // species draws scales its own value vector.
            interaction_query_ = register_module("interaction_query",
                torch::nn::Linear(hidden_dim, hidden_dim));
            interaction_key_ = register_module("interaction_key",
                torch::nn::Linear(hidden_dim, hidden_dim));
            interaction_value_ = register_module("interaction_value",
                torch::nn::Linear(hidden_dim, interaction_dim_));
            break;
    }

    output_proj_ = register_module("output_proj",
        torch::nn::Linear(interaction_dim_, 1));
}

void TraitNetEncoderImpl::set_traits(torch::Tensor traits) {
    TORCH_CHECK(traits.dim() == 2, "traits must be (n_species, trait_dim), got ",
        traits.sizes());
    TORCH_CHECK(traits.size(0) == n_species_,
        "traits must have ", n_species_, " species, got ", traits.size(0));
    TORCH_CHECK(traits.size(1) == trait_dim_,
        "traits must have ", trait_dim_, " features, got ", traits.size(1));
    torch::NoGradGuard no_grad;
    traits_.set_data(traits.detach().to(traits_.device(), torch::kFloat32).clone());
}

torch::Tensor TraitNetEncoderImpl::forward(
    torch::Tensor env,
    torch::Tensor traits
) {
    if (!traits.defined()) {
        TORCH_CHECK(has_traits(), "traits not provided and not set via set_traits()");
        traits = traits_;
    }

    // (batch, env_dim) -> (batch, hidden_dim)
    auto env_encoded = env_encoder_->forward(env);

    // (n_species, trait_dim) -> (n_species, hidden_dim), with one set of
    // weights for all species or one per species.
    auto traits_encoded = trait_encoder_
        ? trait_encoder_->forward(traits)
        : per_species_trait_encoder_->forward(traits);

    // Combine the two into (batch, n_species, interaction_dim).
    torch::Tensor interaction;
    switch (interaction_mode_) {
        case TraitInteractionMode::Bilinear:
            interaction = interaction_->forward(env_encoded, traits_encoded);
            break;
        case TraitInteractionMode::MLP: {
            const int64_t batch = env_encoded.size(0);
            const int64_t n_species = traits_encoded.size(0);
            auto env_rows = env_encoded.unsqueeze(1).expand(
                {batch, n_species, env_encoded.size(1)});
            auto trait_rows = traits_encoded.unsqueeze(0).expand(
                {batch, n_species, traits_encoded.size(1)});
            interaction = interaction_mlp_->forward(
                torch::cat({env_rows, trait_rows}, /*dim=*/2));
            break;
        }
        case TraitInteractionMode::Attention: {
            // (batch, hidden) x (n_species, hidden) -> (batch, n_species)
            auto query = interaction_query_->forward(env_encoded);
            auto key = interaction_key_->forward(traits_encoded);
            auto value = interaction_value_->forward(traits_encoded);
            const auto scale = 1.0 / std::sqrt(static_cast<double>(query.size(1)));
            auto weights = torch::softmax(
                torch::matmul(query, key.transpose(0, 1)) * scale, /*dim=*/1);
            interaction = weights.unsqueeze(2) * value.unsqueeze(0);
            break;
        }
    }

    // (batch, n_species, interaction_dim) -> (batch, n_species)
    return output_proj_->forward(torch::gelu(interaction)).squeeze(-1);
}

}  // namespace resolve
