#pragma once

#include "resolve/types.hpp"
#include "resolve/encoder.hpp"
#include <torch/torch.h>
#include <variant>

namespace resolve {

// Forward result with optional MoE auxiliary loss
struct ModelForwardResult {
    std::unordered_map<std::string, torch::Tensor> outputs;
    torch::Tensor moe_aux_loss;  // Empty if MoE not used
};

// Full Resolve model: shared encoder + multiple task heads
// Supports three encoding modes:
// - Hash: Feature hashing for species (PlotEncoder)
// - Embed: Learnable embeddings for top-k species (PlotEncoderEmbed)
// - Sparse: Explicit species vector input (PlotEncoderSparse)
// Each mode can optionally use Mixture of Experts (MoE)
class ResolveModelImpl : public torch::nn::Module {
public:
    ResolveModelImpl(
        const ResolveSchema& schema,
        const ModelConfig& config = ModelConfig{}
    );

    // Forward pass for all targets
    // Use appropriate inputs based on encoding mode:
    // - Hash: continuous (includes hash embedding), genus_ids, family_ids
    // - Embed: continuous, species_ids, genus_ids, family_ids
    // - Sparse: continuous, species_vector, genus_ids, family_ids
    std::unordered_map<std::string, torch::Tensor> forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Forward pass returning outputs + MoE auxiliary loss (for training with MoE)
    ModelForwardResult forward_with_aux(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Forward pass for single target
    torch::Tensor forward_single(
        const std::string& target,
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Get latent representation (without heads)
    torch::Tensor get_latent(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Forward pass that returns intermediate activations (for diagnostics)
    // Only works with hash encoder for now
    std::pair<torch::Tensor, std::vector<torch::Tensor>> encode_with_activations(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Get MoE gating probabilities for analysis (only valid when MoE enabled)
    torch::Tensor get_gate_probs(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Accessors
    [[nodiscard]] const ResolveSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] int64_t latent_dim() const;
    [[nodiscard]] SpeciesEncodingMode species_encoding() const noexcept { return config_.species_encoding; }
    [[nodiscard]] bool uses_explicit_vector() const noexcept { return config_.uses_explicit_vector; }
    [[nodiscard]] bool uses_moe() const noexcept { return config_.moe_routing != MoERoutingType::None; }
    [[nodiscard]] int n_experts() const noexcept { return uses_moe() ? config_.n_experts : 0; }

    // Get task head by name
    [[nodiscard]] TaskHead& head(const std::string& name);

private:
    // Internal forward through encoder based on mode (returns latent only)
    torch::Tensor encode(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor species_ids,
        torch::Tensor species_vector
    );

    // Internal forward with aux loss (for MoE)
    std::pair<torch::Tensor, torch::Tensor> encode_with_aux(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor species_ids,
        torch::Tensor species_vector
    );

    ResolveSchema schema_;
    ModelConfig config_;

    // Standard encoders (one will be used based on encoding mode)
    PlotEncoder encoder_hash_{nullptr};
    PlotEncoderEmbed encoder_embed_{nullptr};
    PlotEncoderSparse encoder_sparse_{nullptr};

    // MoE encoder (used when moe_routing != None)
    PlotEncoderMoE encoder_moe_{nullptr};

    std::unordered_map<std::string, TaskHead> heads_;
};

TORCH_MODULE(ResolveModel);

// Alias for backwards compatibility
using SpaccModel = ResolveModel;
using SpaccModelImpl = ResolveModelImpl;

} // namespace resolve
