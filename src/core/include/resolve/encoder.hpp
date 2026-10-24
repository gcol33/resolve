#pragma once

// The building blocks of the species encoders (species_encoder.hpp) and the
// tabular adapter: normalization under AMP, the configurable MLP and its
// residual block, the parallel block, the fused positional embedding, and the
// encoder tail every species encoder ends in.

#include "resolve/types.hpp"
#include "resolve/experts.hpp"
#include "resolve/mlp_block.hpp"
#include "resolve/tabm.hpp"
#include <torch/torch.h>
#include <functional>
#include <vector>

namespace resolve {

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
// fp32 Islands Under AMP Autocast
// =============================================================================

// Run `fn` on `inputs` in float32 while CUDA autocast (AMP/fp16) is active:
// autocast is disabled for the call and every floating-point input is cast
// to float32 (integer and boolean inputs, such as a padding mask, pass as they
// are); autocast is re-enabled on the way out, also when `fn` throws, and the
// float32 output flows on through the surrounding autocast region, which
// re-casts it where an op asks for fp16. When autocast is inactive (AMP off or
// a CPU build) fn(inputs) runs on the inputs unchanged.
//
// For an op that is wrong or missing in fp16: a normalization's statistics
// (run_norm_fp32), and libtorch's multi_head_attention_forward, whose
// key-padding fill has no Half kernel.
torch::Tensor run_fp32_under_autocast(
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& fn,
    std::vector<torch::Tensor> inputs
);

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
// Calls norm_fwd directly when CUDA autocast is inactive (AMP off or a CPU
// build).
torch::Tensor run_norm_fp32(
    const std::function<torch::Tensor(torch::Tensor)>& norm_fwd,
    torch::Tensor x
);

// Wraps a normalization module so its forward runs through run_norm_fp32. Used
// by the nn::Sequential MLP path, where the norm module is invoked opaquely by
// Sequential::forward and there is no other interception point.
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

// Description of the mixture of experts that may stand in for an encoder's MLP
// tail. Mirrors MLPBlockConfig: a torch-free projection of the MoE knobs on
// ModelConfig, so an encoder takes one argument rather than five.
struct MoETailConfig {
    MoERoutingType routing = MoERoutingType::None;
    int n_experts = 4;
    std::vector<int64_t> expert_hidden_dims = {256, 128};
    int top_k = 2;
    float noise_std = 0.1f;

    [[nodiscard]] bool enabled() const noexcept {
        return routing != MoERoutingType::None;
    }

    // Reads the MoE knobs only when the run asked for them in the tail. A
    // MoEPlacement::Post run leaves routing at None here and the model builds
    // the mixture over the finished latent instead, so the placement rule lives
    // in exactly one place and no encoder has to know about it.
    static MoETailConfig from_model_config(const ModelConfig& cfg) {
        MoETailConfig moe;
        if (cfg.moe_placement != MoEPlacement::Tail) return moe;
        moe.routing = cfg.moe_routing;
        moe.n_experts = cfg.n_experts;
        moe.expert_hidden_dims = cfg.expert_hidden_dims;
        moe.top_k = cfg.moe_top_k;
        moe.noise_std = cfg.moe_noise_std;
        return moe;
    }
};

// Result from building a configurable MLP
struct MLPBuildResult {
    torch::nn::Sequential mlp;
    int64_t output_dim;
    // Positions in `mlp` of each hidden layer's activation (a residual block's
    // position, its output being the layer's).
    std::vector<size_t> activation_indices;
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

// The normalization module of `type` over the last dimension `dim` (GroupNorm
// over `norm_groups` groups, reduced until they divide `dim`). Empty for
// NormLayerType::None. Its forward is not routed through run_norm_fp32: a
// caller runs it through run_norm_fp32, or wraps it in Fp32Norm.
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

private:
    torch::nn::Linear linear_{nullptr};
    torch::nn::AnyModule norm_;        // "norm"; empty without normalization
    torch::nn::AnyModule activation_;  // "activation"
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::Linear projection_{nullptr};  // For dimension mismatch
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

// `mlp` applied to `x`. With `activations`, the output of each module at
// `activation_indices` is appended to it.
torch::Tensor run_mlp(torch::nn::Sequential& mlp,
                      const std::vector<size_t>& activation_indices,
                      torch::Tensor x,
                      std::vector<torch::Tensor>* activations);

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

    // With `activations`, each hidden layer's activation is appended to it.
    torch::Tensor forward(torch::Tensor x, std::vector<torch::Tensor>* activations = nullptr);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    // What this branch is worth in the aggregation
    // (ParallelBranchConfig::branch_weight).
    [[nodiscard]] float branch_weight() const noexcept { return branch_weight_; }

private:
    torch::nn::Sequential mlp_{nullptr};
    std::vector<size_t> activation_indices_;
    int64_t output_dim_;
    float branch_weight_ = 1.0f;
};

TORCH_MODULE(ParallelBranch);

// Attention-based aggregation for parallel branches
class BranchAttentionImpl : public torch::nn::Module {
public:
    // The branch count comes from the input tensor at forward time, so it is
    // not a construction parameter.
    BranchAttentionImpl(int64_t branch_dim, int n_heads = 4);

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
    // n_branches sizes the gate head; the branch width comes from the input
    // tensor at forward time.
    GatedAggregationImpl(int64_t input_dim, int n_branches);

    // Input: original input + (batch, n_branches, branch_dim)
    // Output: (batch, branch_dim)
    torch::Tensor forward(torch::Tensor input, torch::Tensor branch_outputs);

private:
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

    // With `activations`, the hidden-layer activations of every branch, in
    // branch order, are appended to it.
    torch::Tensor forward(torch::Tensor x, std::vector<torch::Tensor>* activations = nullptr);

    [[nodiscard]] int64_t output_dim() const noexcept { return output_dim_; }
    [[nodiscard]] int n_branches() const noexcept { return static_cast<int>(branches_.size()); }

private:
    std::vector<ParallelBranch> branches_;
    ParallelAggregation aggregation_;
    int64_t output_dim_;
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

    // The underlying (vocab_size * n_positions, embed_dim) table.
    [[nodiscard]] torch::nn::Embedding& embedding() { return embedding_; }
    [[nodiscard]] const torch::nn::Embedding& embedding() const { return embedding_; }

    // (vocab_size, embed_dim): each item's rows averaged over the positions,
    // detached.
    [[nodiscard]] torch::Tensor position_average() const;

private:
    int64_t vocab_size_;
    int n_positions_;
    int embed_dim_;
    torch::nn::Embedding embedding_{nullptr};  // Size: (vocab_size * n_positions, embed_dim)
    torch::Tensor position_offsets_;           // Pre-computed: [0, vocab_size, 2*vocab_size, ...]
};

TORCH_MODULE(FusedPositionalEmbedding);

// =============================================================================
// Taxonomy input check
// =============================================================================

// Validate that a concat-based encoder was given the taxonomy tensors it
// reserved concat width for. Such an encoder (fixed taxonomy slots, the pooled
// rank_pool tables) allocates input width for BOTH genus and family at
// construction, so once taxonomy is enabled its forward pass needs both
// tensors; supplying only one (or an empty tensor) would otherwise produce an x
// narrower than the MLP's input_dim and an opaque torch::cat/Linear shape
// error. This throws a clear message instead. The transformer encoding adds its
// taxonomy embeddings to the tokens and gates each independently. No-op when
// taxonomy is disabled.
void check_concat_taxonomy(
    bool has_taxonomy,
    const torch::Tensor& genus_ids,
    const torch::Tensor& family_ids,
    const char* encoder_name
);

// =============================================================================
// Encoder tail
// =============================================================================

// What an encoder tail produces. aux_loss and gate_probs are undefined unless
// the tail is a mixture of experts; a caller that only wants the representation
// reads .latent and ignores the rest.
struct TailOutput {
    torch::Tensor latent;
    torch::Tensor aux_loss;
    torch::Tensor gate_probs;
};

// The last stage of every species encoder -- hash, embed, sparse, rank_pool and
// transformer all end here. It is one of four things: a plain MLP, a TabM
// ensemble, a backbone MLP whose final stage is a mixture of experts, or a
// parallel block of MLP branches (ParallelLayersConfig::enabled).
//
// The modules are registered onto the OWNING encoder rather than onto a
// sub-module of their own, so the parameter names stay flat ("mlp", "tabm",
// "backbone" + "moe", "parallel").
struct EncoderTail {
    torch::nn::Sequential mlp{nullptr};
    TabMEncoder tabm{nullptr};

    // Parallel tail: several MLP branches over the same input, aggregated.
    ParallelBlock parallel{nullptr};

    // MoE tail: backbone MLP feeding the mixture.
    torch::nn::Sequential backbone{nullptr};
    MixtureOfExperts moe{nullptr};

    int64_t latent_dim = 0;
    // Activation positions in `mlp`, or in `backbone` under a mixture.
    std::vector<size_t> activation_indices;

    [[nodiscard]] bool has_moe() const noexcept { return static_cast<bool>(moe); }
    [[nodiscard]] bool has_tabm() const noexcept { return static_cast<bool>(tabm); }
    [[nodiscard]] bool has_parallel() const noexcept {
        return static_cast<bool>(parallel);
    }
};

// Build the tail on `owner` and return its latent dimension (also stored on the
// tail). With a mixture, hidden_dims splits: all but the last two stages become
// the backbone and the mixture projects to hidden_dims.back(), so the experts
// take over the encoder's final capacity instead of being stacked on top of it.
int64_t build_encoder_tail(
    torch::nn::Module& owner,
    EncoderTail& tail,
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config = MoETailConfig{},
    const ParallelLayersConfig& parallel_config = ParallelLayersConfig{}
);

// Run whichever tail was built. With `activations`, each hidden layer's output
// is appended to it as a (rows, neurons) tensor: the activations of the plain
// MLP, the mixture's backbone or each parallel branch, and each TabM layer with
// its ensemble members as rows.
TailOutput forward_encoder_tail(EncoderTail& tail, torch::Tensor x,
                                std::vector<torch::Tensor>* activations = nullptr);

} // namespace resolve
