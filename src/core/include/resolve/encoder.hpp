#pragma once

#include "resolve/types.hpp"
#include "resolve/experts.hpp"
#include "resolve/tabm.hpp"
#include <torch/torch.h>
#include <utility>

namespace resolve {

// =============================================================================
// Constants
// =============================================================================

// Default MLP architecture
inline const std::vector<int64_t> kDefaultHiddenDims = {2048, 1024, 512, 256, 128, 64};

// Default embedding dimensions
constexpr int kDefaultGenusEmbDim = 8;
constexpr int kDefaultFamilyEmbDim = 8;
constexpr int kDefaultSpeciesEmbDim = 32;
constexpr int kDefaultTopK = 3;
constexpr int kDefaultTopKSpecies = 10;

// =============================================================================
// Custom Normalization Layers
// =============================================================================

// Root Mean Square Layer Normalization
// More efficient than LayerNorm - no mean centering, just scale normalization
// Popular in LLMs (LLaMA, etc.)
class RMSNormImpl : public torch::nn::Module {
public:
    explicit RMSNormImpl(int64_t dim, float eps = 1e-8f);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor weight_;  // Learnable scale parameter
    float eps_;
};

TORCH_MODULE(RMSNorm);

// =============================================================================
// Configurable Architecture Types
// =============================================================================

// Configuration for building MLP blocks
struct MLPBlockConfig {
    ActivationType activation = ActivationType::GELU;
    NormLayerType normalization = NormLayerType::BatchNorm;
    int norm_groups = kDefaultNormGroups;
    float dropout = kDefaultDropout;
    bool use_residual = false;
    float leaky_relu_slope = kDefaultLeakyReLUSlope;
    float elu_alpha = kDefaultELUAlpha;

    // Create from ModelConfig
    static MLPBlockConfig from_model_config(const ModelConfig& cfg) {
        MLPBlockConfig block;
        block.activation = cfg.activation;
        block.normalization = cfg.normalization;
        block.norm_groups = cfg.norm_groups;
        block.dropout = cfg.dropout;
        block.use_residual = cfg.use_residual;
        block.leaky_relu_slope = cfg.leaky_relu_slope;
        block.elu_alpha = cfg.elu_alpha;
        return block;
    }
};

// Result from building a configurable MLP
struct MLPBuildResult {
    torch::nn::Sequential mlp;
    int64_t output_dim;
    std::vector<size_t> activation_indices;  // Indices of activation layers in sequential
};

// =============================================================================
// Layer Factory Functions
// =============================================================================

// Create activation module based on type
// Note: PReLU has learnable parameters and requires dim for initialization
torch::nn::AnyModule make_activation(
    ActivationType type,
    int64_t dim = 0,  // Only needed for PReLU
    float leaky_relu_slope = kDefaultLeakyReLUSlope,
    float elu_alpha = kDefaultELUAlpha
);

// Create normalization module based on type
// Returns empty AnyModule for NormLayerType::None
torch::nn::AnyModule make_normalization(
    NormLayerType type,
    int64_t dim,
    int norm_groups = kDefaultNormGroups
);

// =============================================================================
// Residual Block
// =============================================================================

// Single residual block: Linear -> Norm -> Activation -> Dropout + skip connection
// Projects input if dimensions don't match
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(
        int64_t input_dim,
        int64_t output_dim,
        const MLPBlockConfig& config
    );

    torch::Tensor forward(torch::Tensor x);

    // Get last activation output (for diagnostics)
    [[nodiscard]] torch::Tensor last_activation() const { return last_activation_; }

private:
    torch::nn::Linear linear_{nullptr};

    // Typed norm modules (only one will be used based on norm_type_)
    torch::nn::BatchNorm1d norm_bn_{nullptr};
    torch::nn::LayerNorm norm_ln_{nullptr};
    torch::nn::GroupNorm norm_gn_{nullptr};
    RMSNorm norm_rms_{nullptr};

    // Activation: AnyModule for functional dispatch, PReLU stored separately for parameters
    torch::nn::AnyModule activation_;
    torch::nn::PReLU prelu_{nullptr};

    torch::nn::Dropout dropout_{nullptr};
    torch::nn::Linear projection_{nullptr};  // For dimension mismatch

    bool needs_projection_;
    bool has_norm_;
    NormLayerType norm_type_;
    ActivationType activation_type_;
    mutable torch::Tensor last_activation_;  // Stored for diagnostics
};

TORCH_MODULE(ResidualBlock);

// =============================================================================
// Configurable MLP Builder
// =============================================================================

// Build MLP with configurable activation, normalization, and optional residuals
// Returns the sequential module, output dimension, and activation layer indices
[[nodiscard]] MLPBuildResult build_mlp_configurable(
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config
);

// =============================================================================
// Parallel Layers Block
// =============================================================================

// Single branch in a parallel block
class ParallelBranchImpl : public torch::nn::Module {
public:
    ParallelBranchImpl(
        int64_t input_dim,
        const ParallelBranchConfig& config
    );

    torch::Tensor forward(torch::Tensor x);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }

private:
    torch::nn::Sequential mlp_{nullptr};
    int64_t output_dim_;
};

TORCH_MODULE(ParallelBranch);

// Attention-based aggregation for parallel branches
class BranchAttentionImpl : public torch::nn::Module {
public:
    BranchAttentionImpl(int64_t branch_dim, int n_branches, int n_heads = 4);

    // Input: (batch, n_branches, branch_dim)
    // Output: (batch, branch_dim)
    torch::Tensor forward(torch::Tensor branch_outputs);

private:
    int n_heads_;
    int64_t head_dim_;
    torch::nn::Linear query_{nullptr};
    torch::nn::Linear key_{nullptr};
    torch::nn::Linear value_{nullptr};
    torch::nn::Linear output_proj_{nullptr};
};

TORCH_MODULE(BranchAttention);

// Gated aggregation for parallel branches
class GatedAggregationImpl : public torch::nn::Module {
public:
    GatedAggregationImpl(int64_t input_dim, int n_branches, int64_t branch_dim);

    // Input: original input + (batch, n_branches, branch_dim)
    // Output: (batch, branch_dim)
    torch::Tensor forward(torch::Tensor input, torch::Tensor branch_outputs);

private:
    int n_branches_;
    torch::nn::Linear gate_proj_{nullptr};
};

TORCH_MODULE(GatedAggregation);

// Parallel Block: runs multiple branches in parallel and aggregates
class ParallelBlockImpl : public torch::nn::Module {
public:
    ParallelBlockImpl(
        int64_t input_dim,
        const ParallelLayersConfig& config
    );

    torch::Tensor forward(torch::Tensor x);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int n_branches() const noexcept { return static_cast<int>(branches_.size()); }

private:
    std::vector<ParallelBranch> branches_;
    ParallelAggregation aggregation_;
    int64_t output_dim_;
    int64_t branch_output_dim_;  // Output dim of each branch (for non-concat modes)
    bool use_residual_;

    // Aggregation-specific modules
    BranchAttention attention_{nullptr};
    GatedAggregation gated_{nullptr};
    torch::nn::Linear residual_proj_{nullptr};  // For dimension mismatch
};

TORCH_MODULE(ParallelBlock);

// =============================================================================
// Fused Embedding Table (Performance Optimization)
// =============================================================================

// Fused embedding for multiple positions in a single kernel launch
// Instead of K separate embedding lookups, we use a single large table
// with position offsets: id_for_position_k = id + k * vocab_size
// This reduces CUDA kernel launch overhead from O(K) to O(1)
class FusedPositionalEmbeddingImpl : public torch::nn::Module {
public:
    // vocab_size: number of unique items (e.g., species)
    // n_positions: number of positions (e.g., top_k_species)
    // embed_dim: embedding dimension per position
    FusedPositionalEmbeddingImpl(
        int64_t vocab_size,
        int n_positions,
        int embed_dim
    );

    // Forward pass: single batched embedding lookup
    // ids: (batch, n_positions) - integer IDs per position
    // Returns: (batch, n_positions * embed_dim) - flattened embeddings
    torch::Tensor forward(torch::Tensor ids);

    [[nodiscard]] int64_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] int n_positions() const noexcept { return n_positions_; }
    [[nodiscard]] int embed_dim() const noexcept { return embed_dim_; }
    [[nodiscard]] int64_t total_output_dim() const noexcept { return n_positions_ * embed_dim_; }

    // Get the underlying embedding table (for checkpoint migration)
    [[nodiscard]] torch::nn::Embedding& embedding() { return embedding_; }

private:
    int64_t vocab_size_;
    int n_positions_;
    int embed_dim_;
    torch::nn::Embedding embedding_{nullptr};  // Size: (vocab_size * n_positions, embed_dim)
    torch::Tensor position_offsets_;           // Pre-computed: [0, vocab_size, 2*vocab_size, ...]
};

TORCH_MODULE(FusedPositionalEmbedding);

// =============================================================================
// Legacy Helper (backward compatibility)
// =============================================================================

// Helper function to build MLP layers - reduces duplication across encoder implementations
// DEPRECATED: Use build_mlp_configurable for new code
[[nodiscard]] inline std::pair<torch::nn::Sequential, int64_t> build_mlp(
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) noexcept(false) {
    torch::nn::Sequential mlp;
    int64_t prev_dim = input_dim;
    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
        mlp->push_back(torch::nn::BatchNorm1d(hidden_dims[i]));
        mlp->push_back(torch::nn::GELU());
        mlp->push_back(torch::nn::Dropout(dropout));
        prev_dim = hidden_dims[i];
    }
    int64_t latent_dim = hidden_dims.empty() ? input_dim : hidden_dims.back();
    return {mlp, latent_dim};
}

// PlotEncoder: shared encoder for all tasks (hash mode)
// Architecture: learned taxonomy embeddings + MLP
class PlotEncoderImpl : public torch::nn::Module {
public:
    // New constructor with configurable architecture
    PlotEncoderImpl(
        int64_t n_continuous,
        int64_t n_genera,
        int64_t n_families,
        int genus_emb_dim,
        int family_emb_dim,
        int top_k,
        const std::vector<int64_t>& hidden_dims,
        const MLPBlockConfig& mlp_config,
        const TabMConfig& tabm_config = TabMConfig{}
    );

    // Legacy constructor (backward compatibility)
    PlotEncoderImpl(
        int64_t n_continuous,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    // Forward pass
    // continuous: (batch, n_continuous) - coords + covariates + hash embedding
    // genus_ids: (batch, top_k) - optional
    // family_ids: (batch, top_k) - optional
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Forward pass that also returns intermediate activations (after each GELU)
    // Used for network diagnostics - only call after training
    std::pair<torch::Tensor, std::vector<torch::Tensor>> forward_with_activations(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }
    [[nodiscard]] const std::vector<int64_t>& hidden_dims() const noexcept { return hidden_dims_; }

private:
    bool has_taxonomy_;
    int top_k_;
    int64_t latent_dim_;
    std::vector<int64_t> hidden_dims_;
    MLPBlockConfig mlp_config_;
    std::vector<size_t> activation_indices_;  // For diagnostics

    // Taxonomy embeddings (one per rank position)
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // MLP layers (standard or TabM)
    torch::nn::Sequential mlp_{nullptr};
    TabMEncoder tabm_encoder_{nullptr};
    bool use_tabm_ = false;

    // Helper for constructor implementation
    void init(int64_t n_continuous, int64_t n_genera, int64_t n_families,
              int genus_emb_dim, int family_emb_dim, int top_k,
              const std::vector<int64_t>& hidden_dims, const MLPBlockConfig& config,
              const TabMConfig& tabm_config = TabMConfig{});
};

TORCH_MODULE(PlotEncoder);


// PlotEncoderEmbed: learnable embeddings for top-k species
// Used when species_encoding="embed"
class PlotEncoderEmbedImpl : public torch::nn::Module {
public:
    // Constructor with configurable architecture (always uses fused embeddings)
    PlotEncoderEmbedImpl(
        int64_t n_continuous,
        int64_t n_species,
        int64_t n_genera,
        int64_t n_families,
        int species_embed_dim,
        int genus_emb_dim,
        int family_emb_dim,
        int top_k_species,
        int top_k_taxonomy,
        const std::vector<int64_t>& hidden_dims,
        const MLPBlockConfig& mlp_config,
        const TabMConfig& tabm_config = TabMConfig{}
    );

    // Legacy constructor (backward compatibility)
    PlotEncoderEmbedImpl(
        int64_t n_continuous,
        int64_t n_species,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int species_embed_dim = 32,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k_species = 10,
        int top_k_taxonomy = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    // Forward pass
    // continuous: (batch, n_continuous) - coords + covariates (NO hash embedding)
    // species_ids: (batch, top_k_species) - integer IDs
    // genus_ids: (batch, top_k_taxonomy) - optional
    // family_ids: (batch, top_k_taxonomy) - optional
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_ids,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }

private:
    bool has_taxonomy_;
    int top_k_species_;
    int top_k_taxonomy_;
    int64_t latent_dim_;
    MLPBlockConfig mlp_config_;

    // Fused positional embeddings (single embedding table per type)
    FusedPositionalEmbedding fused_species_{nullptr};
    FusedPositionalEmbedding fused_genus_{nullptr};
    FusedPositionalEmbedding fused_family_{nullptr};

    // MLP layers (standard or TabM)
    torch::nn::Sequential mlp_{nullptr};
    TabMEncoder tabm_encoder_{nullptr};
    bool use_tabm_ = false;

    // Helper for constructor implementation
    void init(int64_t n_continuous, int64_t n_species, int64_t n_genera, int64_t n_families,
              int species_embed_dim, int genus_emb_dim, int family_emb_dim,
              int top_k_species, int top_k_taxonomy,
              const std::vector<int64_t>& hidden_dims, const MLPBlockConfig& config,
              const TabMConfig& tabm_config = TabMConfig{});
};

TORCH_MODULE(PlotEncoderEmbed);


// PlotEncoderSparse: explicit species abundance vectors
// Used for selection="all" or selection="presence_absence"
class PlotEncoderSparseImpl : public torch::nn::Module {
public:
    // New constructor with configurable architecture
    PlotEncoderSparseImpl(
        int64_t n_continuous,
        int64_t n_species,
        int species_embed_dim,
        int64_t n_genera,
        int64_t n_families,
        int genus_emb_dim,
        int family_emb_dim,
        int top_k,
        const std::vector<int64_t>& hidden_dims,
        const MLPBlockConfig& mlp_config,
        const TabMConfig& tabm_config = TabMConfig{}
    );

    // Legacy constructor (backward compatibility)
    PlotEncoderSparseImpl(
        int64_t n_continuous,
        int64_t n_species,
        int species_embed_dim = 64,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f
    );

    // Forward pass
    // continuous: (batch, n_continuous) - coords + covariates
    // species_vector: (batch, n_species) - abundance or presence/absence vector
    // genus_ids: (batch, top_k) - optional
    // family_ids: (batch, top_k) - optional
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_vector,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }
    [[nodiscard]] int64_t n_species() const noexcept { return n_species_; }

private:
    bool has_taxonomy_;
    int64_t n_species_;
    int top_k_;
    int64_t latent_dim_;
    MLPBlockConfig mlp_config_;

    // Linear projection from species abundances to embedding
    torch::nn::Linear species_projection_{nullptr};

    // Taxonomy embeddings (one per rank position)
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // MLP layers (standard or TabM)
    torch::nn::Sequential mlp_{nullptr};
    TabMEncoder tabm_encoder_{nullptr};
    bool use_tabm_ = false;

    // Helper for constructor implementation
    void init(int64_t n_continuous, int64_t n_species, int species_embed_dim,
              int64_t n_genera, int64_t n_families, int genus_emb_dim, int family_emb_dim,
              int top_k, const std::vector<int64_t>& hidden_dims, const MLPBlockConfig& config,
              const TabMConfig& tabm_config = TabMConfig{});
};

TORCH_MODULE(PlotEncoderSparse);


// Task head: prediction head for a single target
// Supports optional hidden layers for deeper task-specific processing
class TaskHeadImpl : public torch::nn::Module {
public:
    // Legacy constructor (single linear layer)
    TaskHeadImpl(
        int64_t latent_dim,
        TaskType task,
        int num_classes = 0,
        TransformType transform = TransformType::None
    );

    // New constructor with configurable hidden layers
    TaskHeadImpl(
        int64_t latent_dim,
        TaskType task,
        int num_classes,
        TransformType transform,
        const std::vector<int64_t>& hidden_dims,
        ActivationType activation = ActivationType::GELU,
        float dropout = 0.0f
    );

    // Forward pass - returns raw output
    torch::Tensor forward(torch::Tensor latent);

    // Predict with inverse transform
    torch::Tensor predict(torch::Tensor latent);

    // Inverse transform for predictions
    torch::Tensor inverse_transform(torch::Tensor predictions);

    [[nodiscard]] TaskType task() const noexcept { return task_; }
    [[nodiscard]] TransformType transform() const noexcept { return transform_; }
    [[nodiscard]] bool has_hidden_layers() const noexcept { return !hidden_dims_.empty(); }

private:
    void init_output_layer(int64_t input_dim, int num_classes);

    TaskType task_;
    TransformType transform_;
    std::vector<int64_t> hidden_dims_;
    torch::nn::Sequential head_mlp_{nullptr};  // Hidden layers (if any)
    torch::nn::Linear output_{nullptr};         // Final projection
};

TORCH_MODULE(TaskHead);


// =============================================================================
// MoE-Enabled Encoders
// =============================================================================

// PlotEncoderMoE: Hash mode encoder with Mixture of Experts
// Adds MoE layer after the shared MLP backbone
class PlotEncoderMoEImpl : public torch::nn::Module {
public:
    // New constructor with configurable architecture
    PlotEncoderMoEImpl(
        int64_t n_continuous,
        int64_t n_genera,
        int64_t n_families,
        int genus_emb_dim,
        int family_emb_dim,
        int top_k,
        const std::vector<int64_t>& hidden_dims,
        const MLPBlockConfig& mlp_config,
        // MoE configuration
        int n_experts,
        const std::vector<int64_t>& expert_hidden_dims,
        MoERoutingType moe_routing,
        int moe_top_k,
        float moe_noise_std
    );

    // Legacy constructor (backward compatibility)
    PlotEncoderMoEImpl(
        int64_t n_continuous,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int genus_emb_dim = 8,
        int family_emb_dim = 8,
        int top_k = 3,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        float dropout = 0.3f,
        // MoE configuration
        int n_experts = 4,
        const std::vector<int64_t>& expert_hidden_dims = {256, 128},
        MoERoutingType moe_routing = MoERoutingType::Soft,
        int moe_top_k = 2,
        float moe_noise_std = 0.1f
    );

    // Forward pass returning latent + auxiliary MoE loss
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Forward pass returning only latent (for inference)
    torch::Tensor forward_simple(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    // Get gating probabilities for analysis
    torch::Tensor get_gate_probs(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }
    [[nodiscard]] int n_experts() const noexcept { return n_experts_; }
    [[nodiscard]] MoERoutingType routing_type() const noexcept { return moe_routing_; }

private:
    torch::Tensor encode_input(
        torch::Tensor continuous,
        torch::Tensor genus_ids,
        torch::Tensor family_ids
    );

    // Helper for constructor implementation
    void init(int64_t n_continuous, int64_t n_genera, int64_t n_families,
              int genus_emb_dim, int family_emb_dim, int top_k,
              const std::vector<int64_t>& hidden_dims, const MLPBlockConfig& config,
              int n_experts, const std::vector<int64_t>& expert_hidden_dims,
              MoERoutingType moe_routing, int moe_top_k, float moe_noise_std);

    bool has_taxonomy_;
    int top_k_;
    int64_t latent_dim_;
    int n_experts_;
    MoERoutingType moe_routing_;
    MLPBlockConfig mlp_config_;

    // Taxonomy embeddings
    std::vector<torch::nn::Embedding> genus_embeddings_;
    std::vector<torch::nn::Embedding> family_embeddings_;

    // Shared backbone MLP (smaller than original, MoE adds capacity)
    torch::nn::Sequential backbone_{nullptr};
    int64_t backbone_output_dim_;

    // Mixture of Experts layer
    MixtureOfExperts moe_{nullptr};
};

TORCH_MODULE(PlotEncoderMoE);

} // namespace resolve
