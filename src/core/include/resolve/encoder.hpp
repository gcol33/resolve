#pragma once

#include "resolve/types.hpp"
#include "resolve/experts.hpp"
#include "resolve/tabm.hpp"
#include <torch/torch.h>
#include <utility>
#include <functional>

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
// fp32 Normalization Under AMP Autocast
// =============================================================================

// Run a normalization layer in float32 even when CUDA autocast (AMP/fp16) is
// active, then return its (fp32) output for the surrounding autocast region to
// re-cast as needed.
//
// BatchNorm computes its batch statistics and updates its running mean/variance
// buffers from the layer input. Under fp16 autocast those statistics are
// computed and accumulated in fp16, so the running buffers drift: the model
// trains correctly (train mode uses fresh per-batch statistics) but collapses
// in eval mode (which uses the corrupted running statistics). Computing the
// normalization in fp32 keeps the statistics accurate while the surrounding
// Linear/embedding matmuls stay in fp16, preserving the AMP speed/memory win.
//
// A no-op (calls norm_fwd directly) when CUDA autocast is inactive (AMP off or
// CPU build) or when the environment variable RESOLVE_FP32_NORM=0 is set. The
// env toggle exists to A/B the fix against the original collapse in a single
// build. See gcol33/resolve#21.
torch::Tensor run_norm_fp32(
    const std::function<torch::Tensor(torch::Tensor)>& norm_fwd,
    torch::Tensor x
);

// Wraps a normalization module so its forward runs through run_norm_fp32. Used
// by the non-residual nn::Sequential MLP path, where the norm module is invoked
// opaquely by Sequential::forward and there is no other interception point.
class Fp32NormImpl : public torch::nn::Module {
public:
    explicit Fp32NormImpl(torch::nn::AnyModule inner);
    torch::Tensor forward(torch::Tensor x);
    [[nodiscard]] bool is_empty() const { return inner_.is_empty(); }
    // The wrapped normalization module. Exposed so inference-time BN fusion
    // (Predictor::optimize_for_inference) can reach the inner BatchNorm1d that
    // this wrapper would otherwise hide from a dynamic_pointer_cast.
    [[nodiscard]] std::shared_ptr<torch::nn::Module> inner_module() const {
        return inner_.is_empty() ? nullptr : inner_.ptr();
    }

private:
    torch::nn::AnyModule inner_;
};

TORCH_MODULE(Fp32Norm);

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
    [[nodiscard]] const torch::nn::Embedding& embedding() const { return embedding_; }

private:
    int64_t vocab_size_;
    int n_positions_;
    int embed_dim_;
    torch::nn::Embedding embedding_{nullptr};  // Size: (vocab_size * n_positions, embed_dim)
    torch::Tensor position_offsets_;           // Pre-computed: [0, vocab_size, 2*vocab_size, ...]
};

TORCH_MODULE(FusedPositionalEmbedding);

// =============================================================================
// Shared Encoder Helpers (reduce duplication across encoder types)
// =============================================================================

// Register per-rank taxonomy embeddings on a module.
// Populates genus_embeddings and family_embeddings vectors.
// Returns the input dimension contribution (top_k * (genus_emb_dim + family_emb_dim)).
int64_t register_per_rank_embeddings(
    torch::nn::Module& module,
    std::vector<torch::nn::Embedding>& genus_embeddings,
    std::vector<torch::nn::Embedding>& family_embeddings,
    int64_t n_genera, int64_t n_families,
    int genus_emb_dim, int family_emb_dim, int top_k
);

// Validate that a concat-based encoder was given the taxonomy tensors it
// reserved concat width for. A concat encoder (fused embed / per-rank hash /
// sparse / MoE) allocates fixed input width for BOTH genus and family at
// construction, so once taxonomy is enabled its forward pass needs both tensors;
// supplying only one (or an empty tensor) would otherwise produce an x narrower
// than the MLP's input_dim and an opaque torch::cat/Linear shape error. This
// throws a clear message instead. Unlike the additive transformer/rank-pool
// encoders (issue #90), independent gating cannot apply here -- a missing column
// cannot be "not added" without shrinking the concat (issue #99). No-op when
// taxonomy is disabled.
void check_concat_taxonomy(
    bool has_taxonomy,
    const torch::Tensor& genus_ids,
    const torch::Tensor& family_ids,
    const char* encoder_name
);

// Collect per-rank embedding lookups into parts vector.
// Appends top_k genus embeddings then top_k family embeddings.
void embed_per_rank_taxonomy(
    std::vector<torch::Tensor>& parts,
    std::vector<torch::nn::Embedding>& genus_embeddings,
    std::vector<torch::nn::Embedding>& family_embeddings,
    torch::Tensor genus_ids, torch::Tensor family_ids,
    int top_k, bool has_taxonomy
);

// Build and register MLP or TabM backbone on a module.
// Returns (latent_dim, activation_indices).
struct BackboneSetupResult {
    int64_t latent_dim;
    std::vector<size_t> activation_indices;
    bool use_tabm;
};

BackboneSetupResult build_and_register_backbone(
    torch::nn::Module& module,
    torch::nn::Sequential& mlp,
    TabMEncoder& tabm_encoder,
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config
);

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

    // Forward pass intended to also return intermediate activations for network
    // diagnostics. NOTE: this PlotEncoder path returns the final output with an
    // EMPTY activations vector -- libtorch's type-erased nn::Sequential storage
    // does not allow per-layer capture here. Only the hash-mode encoder path
    // (encode_with_activations) actually populates activations; compute_diagnostics
    // bails out for the other encoders accordingly.
    std::pair<torch::Tensor, std::vector<torch::Tensor>> forward_with_activations(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }
    [[nodiscard]] const std::vector<int64_t>& hidden_dims() const noexcept { return hidden_dims_; }

    // Embedding weight extraction (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

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

    // Embedding weight extraction (averaged across positions from fused tables)
    [[nodiscard]] torch::Tensor get_species_weights() const;
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

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

    // Embedding weight extraction (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

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


// =============================================================================
// Cover Dropout Helper (shared by RankPool and Transformer encoders)
// =============================================================================

// Apply cover dropout: randomly zero out weights and has_cover for a fraction
// of training samples. Tensors are cloned before modification.
void apply_cover_dropout(
    bool training,
    float cover_dropout,
    int64_t batch_size,
    torch::Device device,
    torch::Tensor& weights,
    const torch::Tensor& mask,
    torch::Tensor& has_cover
);


// =============================================================================
// PlotEncoderRankPool: weighted mean pooling over variable-length species
// =============================================================================

// Used when species_encoding="rank_pool". Single shared embedding tables,
// weighted mean pooling, cover dropout for robustness.
class PlotEncoderRankPoolImpl : public torch::nn::Module {
public:
    PlotEncoderRankPoolImpl(
        int64_t n_continuous,
        int64_t n_species,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int species_embed_dim = 64,
        int genus_embed_dim = 16,
        int family_embed_dim = 16,
        const std::vector<int64_t>& hidden_dims = {2048, 1024, 512, 256, 128, 64},
        const MLPBlockConfig& mlp_config = MLPBlockConfig{},
        float cover_dropout = 0.0f,
        const TabMConfig& tabm_config = TabMConfig{}
    );

    // Forward pass with weighted mean pooling
    // continuous: (batch, n_continuous)
    // species_ids: (batch, max_species) int64, padded with 0
    // genus_ids, family_ids: (batch, max_species) int64, optional
    // weights: (batch, max_species) float, optional
    // mask: (batch, max_species) bool, optional (True=valid)
    // has_cover: (batch,) float, optional (defaults to 1.0)
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_ids,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor weights = {},
        torch::Tensor mask = {},
        torch::Tensor has_cover = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }

    [[nodiscard]] torch::Tensor get_species_weights() const;
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

private:
    bool has_taxonomy_;
    float cover_dropout_;
    int64_t latent_dim_;

    // Single shared embedding tables (NOT per-rank)
    torch::nn::Embedding species_embedding_{nullptr};
    torch::nn::Embedding genus_embedding_{nullptr};
    torch::nn::Embedding family_embedding_{nullptr};

    // MLP backbone
    torch::nn::Sequential mlp_{nullptr};
    TabMEncoder tabm_encoder_{nullptr};
    bool use_tabm_ = false;
};

TORCH_MODULE(PlotEncoderRankPool);


// =============================================================================
// PlotEncoderTransformer: self-attention over species tokens
// =============================================================================

// Used when species_encoding="transformer". Additive embeddings in d_model space,
// optional self-attention, attention or CLS pooling, cover dropout.
class PlotEncoderTransformerImpl : public torch::nn::Module {
public:
    PlotEncoderTransformerImpl(
        int64_t n_continuous,
        int64_t n_species,
        int64_t n_genera = 0,
        int64_t n_families = 0,
        int d_model = 128,
        int n_heads = 4,
        int n_attention_layers = 0,
        int transformer_ff_dim = 256,
        const std::string& transformer_pooling = "attention",
        float transformer_dropout = 0.1f,
        const std::vector<int64_t>& hidden_dims = {1024, 512},
        const MLPBlockConfig& mlp_config = MLPBlockConfig{},
        float cover_dropout = 0.0f,
        const TabMConfig& tabm_config = TabMConfig{}
    );

    // Forward pass: species tokens → self-attention → pooling → MLP → latent
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor species_ids,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor weights = {},
        torch::Tensor mask = {},
        torch::Tensor has_cover = {},
        torch::Tensor masked_positions = {}
    );

    // Get pre-pooling token embeddings for MLM pretraining
    // Returns (batch, max_species, d_model) after self-attention
    torch::Tensor forward_tokens(
        torch::Tensor species_ids,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor weights,
        torch::Tensor mask,
        torch::Tensor masked_positions = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] int d_model() const noexcept { return d_model_; }
    [[nodiscard]] bool has_taxonomy() const noexcept { return has_taxonomy_; }

    [[nodiscard]] torch::Tensor get_species_weights() const;
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

private:
    // Build additive token embeddings from species/genus/family/weights
    torch::Tensor build_tokens(
        torch::Tensor species_ids,
        torch::Tensor genus_ids,
        torch::Tensor family_ids,
        torch::Tensor weights,
        torch::Tensor masked_positions
    );

    int d_model_;
    int n_attention_layers_;
    std::string transformer_pooling_;
    bool has_taxonomy_;
    float cover_dropout_;
    int64_t latent_dim_;

    // Embeddings (all d_model-dimensional, additive)
    torch::nn::Embedding species_embedding_{nullptr};
    torch::nn::Embedding genus_embedding_{nullptr};
    torch::nn::Embedding family_embedding_{nullptr};
    torch::nn::Linear weight_proj_{nullptr};
    torch::Tensor mask_embedding_;

    // Self-attention
    torch::nn::TransformerEncoder transformer_encoder_{nullptr};

    // Attention pooling
    torch::Tensor pool_query_;
    torch::nn::MultiheadAttention pool_attn_{nullptr};
    torch::nn::LayerNorm pool_norm_{nullptr};

    // CLS pooling
    torch::Tensor cls_token_;

    // MLP backbone
    torch::nn::Sequential mlp_{nullptr};
    TabMEncoder tabm_encoder_{nullptr};
    bool use_tabm_ = false;
};

TORCH_MODULE(PlotEncoderTransformer);


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

    // Embedding weight extraction (averaged across positions)
    [[nodiscard]] torch::Tensor get_genus_weights() const;
    [[nodiscard]] torch::Tensor get_family_weights() const;

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
