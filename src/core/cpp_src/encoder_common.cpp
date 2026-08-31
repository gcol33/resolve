#include "resolve/encoder.hpp"
#include "resolve/env.hpp"
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <ATen/autocast_mode.h>

namespace resolve {

// =============================================================================
// fp32 Normalization Under AMP Autocast (see encoder.hpp / gcol33/resolve#21)
// =============================================================================

namespace {
#ifdef RESOLVE_HAS_CUDA
// Read RESOLVE_FP32_NORM once. Default enabled; only the literal "0" disables.
// Guarded with its only caller in run_norm_fp32: the fp32 branch exists for
// CUDA autocast, so a CPU build has no use for the switch.
bool fp32_norm_enabled() {
    static const bool enabled = !env_flag_disabled("RESOLVE_FP32_NORM");
    return enabled;
}
#endif

// GroupNorm needs a group count that divides the channel dim. Shrink the
// requested count until it does (down to 1). Single source for both norm
// construction paths (make_normalization and ResidualBlockImpl).
int groupnorm_groups_for(int requested, int64_t dim) {
    int groups = requested;
    while (groups > 1 && dim % groups != 0) {
        groups--;
    }
    return groups;
}
}  // namespace

torch::Tensor run_norm_fp32(
    const std::function<torch::Tensor(torch::Tensor)>& norm_fwd,
    torch::Tensor x
) {
#ifdef RESOLVE_HAS_CUDA
    if (fp32_norm_enabled() && at::autocast::is_autocast_enabled(at::kCUDA)) {
        // One-time confirmation that the fp32 branch is actually reached under a
        // live autocast region (diagnostic for gcol33/resolve#21).
        static const bool amp_dbg = env_flag_enabled("RESOLVE_AMP_DEBUG");
        static bool announced = false;
        if (amp_dbg && !announced) {
            announced = true;
            std::fprintf(stderr, "[amp_dbg] run_norm_fp32: fp32 branch active "
                                 "(autocast enabled during forward)\n");
            std::fflush(stderr);
        }
        // Locally disable autocast so the normalization (and, for BatchNorm, its
        // running-statistic update) runs in fp32. The guard re-enables autocast
        // even if norm_fwd throws; the autocast cast cache is keyed by nesting
        // level, which is unchanged, so the outer region resumes cleanly.
        struct ReenableAutocast {
            ~ReenableAutocast() { at::autocast::set_autocast_enabled(at::kCUDA, true); }
        };
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        ReenableAutocast guard;
        return norm_fwd(x.to(torch::kFloat32));
    }
#endif
    return norm_fwd(std::move(x));
}

Fp32NormImpl::Fp32NormImpl(torch::nn::AnyModule inner)
    : inner_(std::move(inner)) {
    if (!inner_.is_empty()) {
        register_module("inner", inner_.ptr());
    }
}

torch::Tensor Fp32NormImpl::forward(torch::Tensor x) {
    return run_norm_fp32(
        [this](torch::Tensor t) { return inner_.forward(t); },
        std::move(x)
    );
}

// =============================================================================
// RMSNorm Implementation
// =============================================================================

RMSNormImpl::RMSNormImpl(int64_t dim, float eps)
    : eps_(eps)
{
    weight_ = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    // RMS = sqrt(mean(x^2) + eps)
    auto rms = torch::sqrt(x.pow(2).mean(-1, /*keepdim=*/true) + eps_);
    return (x / rms) * weight_;
}

// =============================================================================
// FusedPositionalEmbedding Implementation
// =============================================================================

FusedPositionalEmbeddingImpl::FusedPositionalEmbeddingImpl(
    int64_t vocab_size,
    int n_positions,
    int embed_dim
) : vocab_size_(vocab_size),
    n_positions_(n_positions),
    embed_dim_(embed_dim)
{
    // Create a single large embedding table: (vocab_size * n_positions, embed_dim)
    // Each position k uses rows [k*vocab_size, (k+1)*vocab_size)
    // We use padding_idx=0 for the first position only (position 0)
    embedding_ = register_module(
        "embedding",
        torch::nn::Embedding(
            torch::nn::EmbeddingOptions(vocab_size * n_positions, embed_dim).padding_idx(0)
        )
    );

    // Pre-compute position offsets: [0, vocab_size, 2*vocab_size, ...]
    // These are added to IDs to select the right embedding region
    std::vector<int64_t> offsets(n_positions);
    for (int k = 0; k < n_positions; ++k) {
        offsets[k] = k * vocab_size;
    }
    // Register as buffer (not a parameter, but moves with the model)
    position_offsets_ = register_buffer(
        "position_offsets",
        torch::tensor(offsets, torch::kLong)
    );
}

torch::Tensor FusedPositionalEmbeddingImpl::forward(torch::Tensor ids) {
    // ids: (batch, n_positions) - integer IDs per position
    // Returns: (batch, n_positions * embed_dim) - flattened embeddings

    auto batch_size = ids.size(0);

    // Add position offsets: ids + [0, vocab_size, 2*vocab_size, ...]
    // Broadcasting: (batch, n_positions) + (n_positions,) -> (batch, n_positions)
    auto offset_ids = ids + position_offsets_.to(ids.device());

    // Flatten for single embedding lookup: (batch * n_positions,)
    auto flat_ids = offset_ids.flatten();

    // Single embedding lookup: (batch * n_positions, embed_dim)
    auto flat_emb = embedding_(flat_ids);

    // Zero out padding/UNK slots. The fused table has a single padding_idx (0),
    // which only freezes position 0's id-0 row; for positions k>0 the id-0 slot
    // offsets to row k*vocab_size, an ordinary learnable row. Mask the lookup by
    // (original id != 0) so an absent/UNK entry at any position contributes the
    // zero vector and receives no gradient, matching the per-rank
    // nn::Embedding(..., padding_idx=0) tables used by the other encoders.
    auto pad_mask = (ids.flatten() != 0).to(flat_emb.dtype()).unsqueeze(-1);
    flat_emb = flat_emb * pad_mask;

    // Reshape to (batch, n_positions * embed_dim)
    return flat_emb.view({batch_size, n_positions_ * embed_dim_});
}

// =============================================================================
// Activation Factory
// =============================================================================

torch::nn::AnyModule make_activation(
    ActivationType type,
    int64_t dim,
    float leaky_relu_slope,
    float elu_alpha
) {
    switch (type) {
        case ActivationType::ReLU:
            return torch::nn::AnyModule(torch::nn::ReLU());
        case ActivationType::LeakyReLU:
            return torch::nn::AnyModule(
                torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(leaky_relu_slope))
            );
        case ActivationType::GELU:
            return torch::nn::AnyModule(torch::nn::GELU());
        case ActivationType::SiLU:
            return torch::nn::AnyModule(torch::nn::SiLU());
        case ActivationType::Tanh:
            return torch::nn::AnyModule(torch::nn::Tanh());
        case ActivationType::Mish:
            // Mish: x * tanh(softplus(x)) - use functional form via lambda
            return torch::nn::AnyModule(torch::nn::Mish());
        case ActivationType::ELU:
            return torch::nn::AnyModule(
                torch::nn::ELU(torch::nn::ELUOptions().alpha(elu_alpha))
            );
        case ActivationType::SELU:
            return torch::nn::AnyModule(torch::nn::SELU());
        case ActivationType::Softplus:
            return torch::nn::AnyModule(torch::nn::Softplus());
        case ActivationType::PReLU:
            // PReLU has learnable parameters, needs dim
            if (dim <= 0) {
                throw std::invalid_argument("PReLU requires dim > 0 for initialization");
            }
            return torch::nn::AnyModule(torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(dim)));
        default:
            return torch::nn::AnyModule(torch::nn::GELU());
    }
}

// =============================================================================
// Normalization Factory
// =============================================================================

torch::nn::AnyModule make_normalization(
    NormLayerType type,
    int64_t dim,
    int norm_groups
) {
    // Wrap each norm in Fp32Norm so it runs in fp32 under AMP autocast (see
    // run_norm_fp32 / gcol33/resolve#21). Sequential::forward invokes the norm
    // opaquely, so the wrapper is the only place to enforce fp32 on this path.
    switch (type) {
        case NormLayerType::BatchNorm:
            return torch::nn::AnyModule(Fp32Norm(
                torch::nn::AnyModule(torch::nn::BatchNorm1d(dim))));
        case NormLayerType::LayerNorm:
            return torch::nn::AnyModule(Fp32Norm(
                torch::nn::AnyModule(torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({dim})))));
        case NormLayerType::GroupNorm:
            return torch::nn::AnyModule(Fp32Norm(
                torch::nn::AnyModule(torch::nn::GroupNorm(
                    groupnorm_groups_for(norm_groups, dim), dim))));
        case NormLayerType::RMSNorm:
            return torch::nn::AnyModule(Fp32Norm(
                torch::nn::AnyModule(RMSNorm(dim))));
        case NormLayerType::None:
        default:
            return torch::nn::AnyModule();  // Empty module
    }
}

// =============================================================================
// ResidualBlock Implementation
// =============================================================================

ResidualBlockImpl::ResidualBlockImpl(
    int64_t input_dim,
    int64_t output_dim,
    const MLPBlockConfig& config
) : needs_projection_(input_dim != output_dim),
    has_norm_(config.normalization != NormLayerType::None),
    norm_type_(config.normalization),
    activation_type_(config.activation)
{
    // Main transformation
    linear_ = register_module("linear", torch::nn::Linear(input_dim, output_dim));

    // Normalization (if any)
    // We create typed modules and store them appropriately to ensure proper registration
    if (has_norm_) {
        switch (config.normalization) {
            case NormLayerType::BatchNorm:
                norm_bn_ = register_module("norm", torch::nn::BatchNorm1d(output_dim));
                break;
            case NormLayerType::LayerNorm:
                norm_ln_ = register_module("norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({output_dim})));
                break;
            case NormLayerType::GroupNorm:
                norm_gn_ = register_module("norm", torch::nn::GroupNorm(
                    groupnorm_groups_for(config.norm_groups, output_dim), output_dim));
                break;
            case NormLayerType::RMSNorm:
                norm_rms_ = register_module("norm", RMSNorm(output_dim));
                break;
            default:
                has_norm_ = false;  // None
                break;
        }
    }

    // Activation - create typed modules for ones with parameters
    if (config.activation == ActivationType::PReLU) {
        prelu_ = register_module("activation", torch::nn::PReLU(
            torch::nn::PReLUOptions().num_parameters(output_dim)));
    }
    // Other activations are functional (no learnable parameters), stored in AnyModule for dispatch
    activation_ = make_activation(config.activation, output_dim, config.leaky_relu_slope, config.elu_alpha);

    // Dropout
    if (config.dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(config.dropout));
    }

    // Projection for residual if dimensions don't match
    if (needs_projection_) {
        projection_ = register_module("projection", torch::nn::Linear(input_dim, output_dim));
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    auto identity = x;

    // Main path: Linear -> Norm -> Activation -> Dropout
    auto out = linear_->forward(x);

    // Apply normalization using typed module, in fp32 under AMP autocast
    // (gcol33/resolve#21). Typed members keep their "norm" checkpoint key; only
    // the forward is routed through run_norm_fp32.
    if (has_norm_) {
        switch (norm_type_) {
            case NormLayerType::BatchNorm:
                out = run_norm_fp32([this](torch::Tensor t) { return norm_bn_->forward(t); }, out);
                break;
            case NormLayerType::LayerNorm:
                out = run_norm_fp32([this](torch::Tensor t) { return norm_ln_->forward(t); }, out);
                break;
            case NormLayerType::GroupNorm:
                out = run_norm_fp32([this](torch::Tensor t) { return norm_gn_->forward(t); }, out);
                break;
            case NormLayerType::RMSNorm:
                out = run_norm_fp32([this](torch::Tensor t) { return norm_rms_->forward(t); }, out);
                break;
            default:
                break;
        }
    }

    // Apply activation - PReLU uses registered module, others use AnyModule
    if (activation_type_ == ActivationType::PReLU) {
        out = prelu_->forward(out);
    } else {
        out = activation_.forward(out);
    }
    last_activation_ = out;  // Store for diagnostics

    if (dropout_) {
        out = dropout_->forward(out);
    }

    // Residual connection
    if (needs_projection_) {
        identity = projection_->forward(identity);
    }

    return out + identity;
}

// =============================================================================
// Configurable MLP Builder
// =============================================================================

MLPBuildResult build_mlp_configurable(
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config
) {
    MLPBuildResult result;
    result.output_dim = hidden_dims.empty() ? input_dim : hidden_dims.back();

    torch::nn::Sequential mlp;
    int64_t prev_dim = input_dim;

    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        int64_t out_dim = hidden_dims[i];

        if (config.use_residual) {
            // Use ResidualBlock for this layer
            mlp->push_back(ResidualBlock(prev_dim, out_dim, config));
            // Track activation index (ResidualBlock stores activation internally)
            result.activation_indices.push_back(mlp->size() - 1);
        } else {
            // Standard layer: Linear -> Norm -> Activation -> Dropout
            mlp->push_back(torch::nn::Linear(prev_dim, out_dim));

            // Normalization
            if (config.normalization != NormLayerType::None) {
                auto norm = make_normalization(config.normalization, out_dim, config.norm_groups);
                if (!norm.is_empty()) {
                    mlp->push_back(norm);
                }
            }

            // Activation
            auto act = make_activation(config.activation, out_dim, config.leaky_relu_slope, config.elu_alpha);
            result.activation_indices.push_back(mlp->size());  // Index of activation
            mlp->push_back(act);

            // Dropout
            if (config.dropout > 0) {
                mlp->push_back(torch::nn::Dropout(config.dropout));
            }
        }

        prev_dim = out_dim;
    }

    result.mlp = mlp;
    return result;
}

// =============================================================================
// Shared Encoder Helpers
// =============================================================================

int64_t register_per_rank_embeddings(
    torch::nn::Module& module,
    std::vector<torch::nn::Embedding>& genus_embeddings,
    std::vector<torch::nn::Embedding>& family_embeddings,
    int64_t n_genera, int64_t n_families,
    int genus_emb_dim, int family_emb_dim, int top_k
) {
    // padding_idx(0) zeros and freezes row 0 (the reserved UNK / empty-slot id),
    // matching the fused embed / rank-pool / transformer tables. Without it the
    // padding slots of species-poor plots and every UNK-genus species pull a
    // learnable ~N(0,1) row that accumulates gradient and injects a constant
    // per-slot signal into the MLP concat.
    for (int k = 0; k < top_k; ++k) {
        genus_embeddings.push_back(module.register_module(
            "genus_emb_" + std::to_string(k),
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_genera, genus_emb_dim).padding_idx(0))
        ));
        family_embeddings.push_back(module.register_module(
            "family_emb_" + std::to_string(k),
            torch::nn::Embedding(torch::nn::EmbeddingOptions(n_families, family_emb_dim).padding_idx(0))
        ));
    }
    return static_cast<int64_t>(top_k) * (genus_emb_dim + family_emb_dim);
}

void check_concat_taxonomy(
    bool has_taxonomy,
    const torch::Tensor& genus_ids,
    const torch::Tensor& family_ids,
    const char* encoder_name
) {
    if (!has_taxonomy) return;
    const bool have_genus = genus_ids.defined() && genus_ids.numel() > 0;
    const bool have_family = family_ids.defined() && family_ids.numel() > 0;
    if (!have_genus || !have_family) {
        throw std::runtime_error(
            std::string(encoder_name) +
            ": taxonomy is enabled but was not given both taxonomy inputs (genus=" +
            (have_genus ? "yes" : "no") + ", family=" + (have_family ? "yes" : "no") +
            "). This encoder reserves concat width for both genus and family; pass "
            "both taxonomy tensors, or construct the model without taxonomy.");
    }
}

void embed_per_rank_taxonomy(
    std::vector<torch::Tensor>& parts,
    std::vector<torch::nn::Embedding>& genus_embeddings,
    std::vector<torch::nn::Embedding>& family_embeddings,
    torch::Tensor genus_ids, torch::Tensor family_ids,
    int top_k, bool has_taxonomy
) {
    check_concat_taxonomy(has_taxonomy, genus_ids, family_ids,
                          "PlotEncoder (per-rank taxonomy)");
    if (!has_taxonomy) return;
    for (int k = 0; k < top_k; ++k) {
        parts.push_back(genus_embeddings[k](genus_ids.select(1, k)));
    }
    for (int k = 0; k < top_k; ++k) {
        parts.push_back(family_embeddings[k](family_ids.select(1, k)));
    }
}

namespace {
// Split hidden_dims into the backbone that feeds the mixture and the width the
// mixture projects to. All but the final two stages stay in the backbone and
// the mixture produces hidden_dims.back(); a spec too short to split that way
// keeps its first stage as the backbone. An empty spec leaves an identity
// backbone, which build_mlp_configurable already tolerates, and the mixture
// then maps the encoder's input width to itself.
std::vector<int64_t> moe_backbone_dims(const std::vector<int64_t>& hidden_dims) {
    std::vector<int64_t> backbone_dims;
    if (hidden_dims.size() > 2) {
        backbone_dims.assign(hidden_dims.begin(), hidden_dims.end() - 2);
    } else if (!hidden_dims.empty()) {
        backbone_dims.push_back(hidden_dims.front());
    }
    return backbone_dims;
}
}  // namespace

int64_t build_encoder_tail(
    torch::nn::Module& owner,
    EncoderTail& tail,
    int64_t input_dim,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    // TabM and a mixture are both replacements for the same MLP tail, so only
    // one of them can have it. Requesting both used to drop TabM without a
    // word, since the MoE encoder took no TabMConfig at all.
    if (tabm_config.enabled && moe_config.enabled()) {
        throw std::invalid_argument(
            "encoder tail: tabm.enabled and moe_routing are both set, but both "
            "replace the encoder's MLP tail. Choose one: disable TabM, set "
            "moe_routing=none, or move the mixture off the tail with "
            "moe_placement=post.");
    }

    if (tabm_config.enabled) {
        tail.tabm = owner.register_module("tabm", TabMEncoder(
            input_dim, hidden_dims, tabm_config.n_ensembles,
            config.dropout, tabm_config.aggregation));
        tail.latent_dim = tail.tabm->output_dim();
        return tail.latent_dim;
    }

    if (moe_config.enabled()) {
        auto build = build_mlp_configurable(
            input_dim, moe_backbone_dims(hidden_dims), config);
        tail.backbone = owner.register_module("backbone", build.mlp);

        const int64_t moe_output_dim =
            hidden_dims.empty() ? build.output_dim : hidden_dims.back();
        tail.moe = owner.register_module("moe", MixtureOfExperts(
            build.output_dim,
            moe_config.expert_hidden_dims,
            moe_output_dim,
            moe_config.n_experts,
            moe_config.routing,
            moe_config.top_k,
            moe_config.noise_std,
            config.dropout));
        tail.latent_dim = moe_output_dim;
        return tail.latent_dim;
    }

    auto build = build_mlp_configurable(input_dim, hidden_dims, config);
    tail.mlp = owner.register_module("mlp", build.mlp);
    tail.latent_dim = build.output_dim;
    tail.activation_indices = build.activation_indices;
    return tail.latent_dim;
}

TailOutput forward_encoder_tail(EncoderTail& tail, torch::Tensor x) {
    if (tail.has_tabm()) {
        return {tail.tabm->forward(std::move(x)), {}, {}};
    }
    if (tail.has_moe()) {
        auto result = tail.moe->forward(tail.backbone->forward(std::move(x)));
        return {result.output, result.aux_loss, result.gate_probs};
    }
    return {tail.mlp->forward(std::move(x)), {}, {}};
}

// =============================================================================
// Embedding Weight Extraction (per-position -> averaged)
// =============================================================================

// Helper: stack per-position embedding weights and average
torch::Tensor average_embedding_weights(
    const std::vector<torch::nn::Embedding>& embeddings
) {
    if (embeddings.empty()) return torch::Tensor();
    std::vector<torch::Tensor> weights;
    weights.reserve(embeddings.size());
    for (const auto& emb : embeddings) {
        weights.push_back(emb->weight);
    }
    // (top_k, vocab_size, emb_dim) -> mean(0) -> (vocab_size, emb_dim)
    return torch::stack(weights, 0).mean(0);
}

// Helper: extract from FusedPositionalEmbedding and average across positions
torch::Tensor average_fused_weights(
    const FusedPositionalEmbedding& fused
) {
    if (!fused) return torch::Tensor();
    auto weight = fused->embedding()->weight;  // (vocab_size * n_positions, embed_dim)
    auto n_pos = fused->n_positions();
    auto vocab = fused->vocab_size();
    auto dim = fused->embed_dim();
    // (n_positions, vocab_size, embed_dim) -> mean(0)
    return weight.view({n_pos, vocab, dim}).mean(0);
}

// =============================================================================
// Parallel Branch Implementation
// =============================================================================

ParallelBranchImpl::ParallelBranchImpl(
    int64_t input_dim,
    const ParallelBranchConfig& config
) {
    // Build MLP for this branch
    MLPBlockConfig mlp_config;
    mlp_config.activation = config.activation;
    mlp_config.normalization = config.normalization;
    mlp_config.dropout = config.dropout;

    auto result = build_mlp_configurable(input_dim, config.hidden_dims, mlp_config);
    mlp_ = register_module("mlp", result.mlp);
    output_dim_ = result.output_dim;
}

torch::Tensor ParallelBranchImpl::forward(torch::Tensor x) {
    return mlp_->forward(x);
}

// =============================================================================
// Branch Attention Implementation
// =============================================================================

BranchAttentionImpl::BranchAttentionImpl(int64_t branch_dim, int n_heads)
    : n_heads_(n_heads)
{
    // Ensure head_dim is valid
    if (branch_dim % n_heads != 0) {
        throw std::invalid_argument(
            "branch_dim (" + std::to_string(branch_dim) +
            ") must be divisible by n_heads (" + std::to_string(n_heads) + ")"
        );
    }
    head_dim_ = branch_dim / n_heads;

    // Query from learned context vector, keys and values from branches
    query_ = register_module("query", torch::nn::Linear(branch_dim, branch_dim));
    key_ = register_module("key", torch::nn::Linear(branch_dim, branch_dim));
    value_ = register_module("value", torch::nn::Linear(branch_dim, branch_dim));
    output_proj_ = register_module("output_proj", torch::nn::Linear(branch_dim, branch_dim));
}

torch::Tensor BranchAttentionImpl::forward(torch::Tensor branch_outputs) {
    // branch_outputs: (batch, n_branches, branch_dim)
    int64_t batch_size = branch_outputs.size(0);
    int64_t n_branches = branch_outputs.size(1);
    int64_t branch_dim = branch_outputs.size(2);

    // Use mean as query (learned aggregation)
    auto query_input = branch_outputs.mean(/*dim=*/1);  // (batch, branch_dim)
    auto q = query_->forward(query_input);  // (batch, branch_dim)
    q = q.view({batch_size, 1, n_heads_, head_dim_}).transpose(1, 2);  // (batch, n_heads, 1, head_dim)

    // Keys and values from all branches
    auto k = key_->forward(branch_outputs);  // (batch, n_branches, branch_dim)
    auto v = value_->forward(branch_outputs);

    // Reshape for multi-head attention
    k = k.view({batch_size, n_branches, n_heads_, head_dim_}).transpose(1, 2);  // (batch, n_heads, n_branches, head_dim)
    v = v.view({batch_size, n_branches, n_heads_, head_dim_}).transpose(1, 2);

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn_weights = torch::matmul(q, k.transpose(-2, -1)) * scale;  // (batch, n_heads, 1, n_branches)
    attn_weights = torch::softmax(attn_weights, /*dim=*/-1);

    // Apply attention to values
    auto attn_output = torch::matmul(attn_weights, v);  // (batch, n_heads, 1, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, branch_dim});  // (batch, branch_dim)

    return output_proj_->forward(attn_output);
}

// =============================================================================
// Gated Aggregation Implementation
// =============================================================================

GatedAggregationImpl::GatedAggregationImpl(int64_t input_dim, int n_branches)
{
    // Project input to gate weights for each branch
    gate_proj_ = register_module("gate_proj", torch::nn::Linear(input_dim, n_branches));
}

torch::Tensor GatedAggregationImpl::forward(torch::Tensor input, torch::Tensor branch_outputs) {
    // input: (batch, input_dim)
    // branch_outputs: (batch, n_branches, branch_dim)

    // Compute gate weights from input
    auto gate_logits = gate_proj_->forward(input);  // (batch, n_branches)
    auto gate_weights = torch::softmax(gate_logits, /*dim=*/-1);  // (batch, n_branches)

    // Weighted sum of branches
    // gate_weights: (batch, n_branches) -> (batch, n_branches, 1)
    gate_weights = gate_weights.unsqueeze(-1);
    auto weighted = branch_outputs * gate_weights;  // (batch, n_branches, branch_dim)

    return weighted.sum(/*dim=*/1);  // (batch, branch_dim)
}

// =============================================================================
// Parallel Block Implementation
// =============================================================================

ParallelBlockImpl::ParallelBlockImpl(
    int64_t input_dim,
    const ParallelLayersConfig& config
) : aggregation_(config.aggregation),
    use_residual_(config.use_residual)
{
    if (config.branches.empty()) {
        throw std::invalid_argument("ParallelBlock requires at least one branch configuration");
    }

    // Create branches
    int64_t total_output_dim = 0;
    int64_t first_branch_dim = 0;
    bool all_same_dim = true;

    for (size_t i = 0; i < config.branches.size(); ++i) {
        auto branch = ParallelBranch(input_dim, config.branches[i]);
        register_module("branch_" + std::to_string(i), branch);
        branches_.push_back(branch);

        int64_t branch_dim = branch->output_dim();
        total_output_dim += branch_dim;

        if (i == 0) {
            first_branch_dim = branch_dim;
        } else if (branch_dim != first_branch_dim) {
            all_same_dim = false;
        }
    }

    branch_output_dim_ = first_branch_dim;

    // Validate aggregation mode vs branch dimensions
    if ((aggregation_ == ParallelAggregation::Sum ||
         aggregation_ == ParallelAggregation::Mean ||
         aggregation_ == ParallelAggregation::Attention ||
         aggregation_ == ParallelAggregation::Gated) && !all_same_dim) {
        throw std::invalid_argument(
            "Sum/Mean/Attention/Gated aggregation requires all branches to have the same output dimension"
        );
    }

    // Calculate output dimension based on aggregation
    switch (aggregation_) {
        case ParallelAggregation::Concat:
            output_dim_ = total_output_dim;
            break;
        case ParallelAggregation::Sum:
        case ParallelAggregation::Mean:
        case ParallelAggregation::Attention:
        case ParallelAggregation::Gated:
            output_dim_ = first_branch_dim;
            break;
    }

    // Create aggregation-specific modules
    if (aggregation_ == ParallelAggregation::Attention) {
        attention_ = register_module("attention",
            BranchAttention(first_branch_dim, config.attention_heads));
    } else if (aggregation_ == ParallelAggregation::Gated) {
        gated_ = register_module("gated",
            GatedAggregation(input_dim, static_cast<int>(branches_.size())));
    }

    // Residual projection if needed
    if (use_residual_ && input_dim != output_dim_) {
        residual_proj_ = register_module("residual_proj", torch::nn::Linear(input_dim, output_dim_));
    }
}

torch::Tensor ParallelBlockImpl::forward(torch::Tensor x) {
    auto identity = x;

    // Run all branches in parallel (conceptually - PyTorch will batch them)
    std::vector<torch::Tensor> branch_outputs;
    for (auto& branch : branches_) {
        branch_outputs.push_back(branch->forward(x));
    }

    torch::Tensor aggregated;

    switch (aggregation_) {
        case ParallelAggregation::Concat:
            // Concatenate all branch outputs
            aggregated = torch::cat(branch_outputs, /*dim=*/1);
            break;

        case ParallelAggregation::Sum:
            // Element-wise sum
            aggregated = branch_outputs[0];
            for (size_t i = 1; i < branch_outputs.size(); ++i) {
                aggregated = aggregated + branch_outputs[i];
            }
            break;

        case ParallelAggregation::Mean:
            // Element-wise mean
            aggregated = branch_outputs[0];
            for (size_t i = 1; i < branch_outputs.size(); ++i) {
                aggregated = aggregated + branch_outputs[i];
            }
            aggregated = aggregated / static_cast<float>(branch_outputs.size());
            break;

        case ParallelAggregation::Attention:
            // Stack and apply attention
            {
                auto stacked = torch::stack(branch_outputs, /*dim=*/1);  // (batch, n_branches, branch_dim)
                aggregated = attention_->forward(stacked);
            }
            break;

        case ParallelAggregation::Gated:
            // Gated combination using input
            {
                auto stacked = torch::stack(branch_outputs, /*dim=*/1);
                aggregated = gated_->forward(x, stacked);
            }
            break;
    }

    // Add residual connection
    if (use_residual_) {
        if (residual_proj_) {
            identity = residual_proj_->forward(identity);
        }
        aggregated = aggregated + identity;
    }

    return aggregated;
}

} // namespace resolve
