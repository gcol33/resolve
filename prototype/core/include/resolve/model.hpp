#pragma once

#include "types.hpp"
#include "encoder.hpp"

#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <variant>

namespace resolve {

/**
 * Full RESOLVE model: shared encoder + multiple task heads
 *
 * Supports three encoding modes:
 *   - Hash: Feature hashing for species (PlotEncoder)
 *   - Embed: Learnable embeddings for top-k species (PlotEncoderEmbed)
 *   - Sparse: Explicit species vector input (PlotEncoderSparse)
 */
class ResolveModelImpl : public torch::nn::Module {
public:
    ResolveModelImpl(
        const ResolveSchema& schema,
        const ModelConfig& config = ModelConfig{}
    );

    /**
     * Forward pass for all targets.
     *
     * Use appropriate inputs based on encoding mode:
     *   - Hash: continuous (includes hash embedding), genus_ids, family_ids
     *   - Embed: continuous, species_ids, genus_ids, family_ids
     *   - Sparse: continuous, species_vector, genus_ids, family_ids
     */
    std::unordered_map<std::string, torch::Tensor> forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    /**
     * Forward pass for single target.
     */
    torch::Tensor forward_single(
        const std::string& target,
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    /**
     * Get latent representation (without heads).
     */
    torch::Tensor get_latent(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    // Accessors
    const ResolveSchema& schema() const { return schema_; }
    const ModelConfig& config() const { return config_; }
    int64_t latent_dim() const;
    SpeciesEncodingMode species_encoding() const { return config_.species_encoding; }
    bool uses_explicit_vector() const { return config_.uses_explicit_vector; }

    // Get task head by name
    TaskHead& head(const std::string& name);

private:
    // Internal forward through encoder based on mode
    torch::Tensor encode(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor species_ids,
        torch::Tensor species_vector
    );

    ResolveSchema schema_;
    ModelConfig config_;

    // One of these will be used based on encoding mode
    PlotEncoder encoder_hash_{nullptr};
    PlotEncoderEmbed encoder_embed_{nullptr};
    PlotEncoderSparse encoder_sparse_{nullptr};

    std::vector<std::pair<std::string, TaskHead>> heads_;
};

TORCH_MODULE(ResolveModel);

} // namespace resolve
