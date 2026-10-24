#include "resolve/encoder.hpp"
#include <cmath>
#include <stdexcept>
#include <ATen/autocast_mode.h>

namespace resolve {

// =============================================================================
// fp32 Normalization Under AMP Autocast
// =============================================================================

namespace {
// GroupNorm needs a group count that divides the channel dim. Shrink the
// requested count until it does (down to 1).
int groupnorm_groups_for(int requested, int64_t dim) {
    int groups = requested;
    while (groups > 1 && dim % groups != 0) {
        groups--;
    }
    return groups;
}
}  // namespace

torch::Tensor run_fp32_under_autocast(
    const std::function<torch::Tensor(const std::vector<torch::Tensor>&)>& fn,
    std::vector<torch::Tensor> inputs
) {
#ifdef RESOLVE_HAS_CUDA
    if (at::autocast::is_autocast_enabled(at::kCUDA)) {
        // The guard re-enables autocast even if fn throws; the autocast cast
        // cache is keyed by nesting level, which is unchanged, so the outer
        // region resumes cleanly.
        struct ReenableAutocast {
            ~ReenableAutocast() { at::autocast::set_autocast_enabled(at::kCUDA, true); }
        };
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        ReenableAutocast guard;
        for (auto& input : inputs) {
            if (input.defined() && input.is_floating_point()) {
                input = input.to(torch::kFloat32);
            }
        }
        return fn(inputs);
    }
#endif
    return fn(inputs);
}

torch::Tensor run_norm_fp32(
    const std::function<torch::Tensor(torch::Tensor)>& norm_fwd,
    torch::Tensor x
) {
    // The normalization (and, for BatchNorm, its running-statistic update)
    // runs in fp32.
    return run_fp32_under_autocast(
        [&norm_fwd](const std::vector<torch::Tensor>& in) { return norm_fwd(in[0]); },
        {std::move(x)});
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
    // zero vector and receives no gradient, matching the per-slot
    // nn::Embedding(..., padding_idx=0) tables used by the other encoders.
    auto pad_mask = (ids.flatten() != 0).to(flat_emb.dtype()).unsqueeze(-1);
    flat_emb = flat_emb * pad_mask;

    // Reshape to (batch, n_positions * embed_dim)
    return flat_emb.view({batch_size, n_positions_ * embed_dim_});
}

torch::Tensor FusedPositionalEmbeddingImpl::position_average() const {
    return embedding_->weight.detach().view({n_positions_, vocab_size_, embed_dim_}).mean(0);
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
    }
    return torch::nn::AnyModule(torch::nn::GELU());
}

// =============================================================================
// Normalization Factory
// =============================================================================

torch::nn::AnyModule make_normalization(
    NormLayerType type,
    int64_t dim,
    int norm_groups
) {
    switch (type) {
        case NormLayerType::BatchNorm:
            return torch::nn::AnyModule(torch::nn::BatchNorm1d(dim));
        case NormLayerType::LayerNorm:
            return torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        case NormLayerType::GroupNorm:
            return torch::nn::AnyModule(
                torch::nn::GroupNorm(groupnorm_groups_for(norm_groups, dim), dim));
        case NormLayerType::RMSNorm:
            return torch::nn::AnyModule(RMSNorm(dim));
        case NormLayerType::None:
            break;
    }
    return torch::nn::AnyModule();
}

// =============================================================================
// ResidualBlock Implementation
// =============================================================================

ResidualBlockImpl::ResidualBlockImpl(
    int64_t input_dim,
    int64_t output_dim,
    const MLPBlockConfig& config
) {
    linear_ = register_module("linear", torch::nn::Linear(input_dim, output_dim));

    norm_ = make_normalization(config.normalization, output_dim, config.norm_groups);
    if (!norm_.is_empty()) {
        register_module("norm", norm_.ptr());
    }

    activation_ = make_activation(config.activation, output_dim, config.leaky_relu_slope,
                                  config.elu_alpha);
    register_module("activation", activation_.ptr());

    if (config.dropout > 0) {
        dropout_ = register_module("dropout", torch::nn::Dropout(config.dropout));
    }

    if (input_dim != output_dim) {
        projection_ = register_module("projection", torch::nn::Linear(input_dim, output_dim));
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    // Main path: Linear -> Norm (in fp32 under AMP autocast) -> Activation -> Dropout
    auto out = linear_->forward(x);
    if (!norm_.is_empty()) {
        out = run_norm_fp32([this](torch::Tensor t) { return norm_.forward(t); }, out);
    }
    out = activation_.forward(out);
    if (dropout_) {
        out = dropout_->forward(out);
    }
    return out + (projection_ ? projection_->forward(x) : x);
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
            mlp->push_back(ResidualBlock(prev_dim, out_dim, config));
            result.activation_indices.push_back(mlp->size() - 1);
        } else {
            // Standard layer: Linear -> Norm -> Activation -> Dropout. The norm
            // runs in fp32 under AMP autocast through its Fp32Norm wrapper,
            // since Sequential::forward invokes it opaquely.
            mlp->push_back(torch::nn::Linear(prev_dim, out_dim));

            auto norm = make_normalization(config.normalization, out_dim, config.norm_groups);
            if (!norm.is_empty()) {
                mlp->push_back(Fp32Norm(std::move(norm)));
            }

            auto act = make_activation(config.activation, out_dim, config.leaky_relu_slope, config.elu_alpha);
            result.activation_indices.push_back(mlp->size());
            mlp->push_back(act);

            if (config.dropout > 0) {
                mlp->push_back(torch::nn::Dropout(config.dropout));
            }
        }

        prev_dim = out_dim;
    }

    result.mlp = mlp;
    return result;
}

torch::Tensor run_mlp(torch::nn::Sequential& mlp,
                      const std::vector<size_t>& activation_indices,
                      torch::Tensor x,
                      std::vector<torch::Tensor>* activations) {
    if (activations == nullptr) return mlp->forward(std::move(x));
    auto next = activation_indices.begin();
    size_t position = 0;
    for (auto& module : *mlp) {
        x = module.forward<torch::Tensor>(x);
        if (next != activation_indices.end() && *next == position) {
            activations->push_back(x);
            ++next;
        }
        ++position;
    }
    return x;
}

// =============================================================================
// Taxonomy input check
// =============================================================================

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

// =============================================================================
// Encoder tail
// =============================================================================

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
    const MoETailConfig& moe_config,
    const ParallelLayersConfig& parallel_config
) {
    // TabM, a mixture and a parallel block are all replacements for the same
    // MLP tail, so only one of them can have it.
    if (tabm_config.enabled && moe_config.enabled()) {
        throw std::invalid_argument(
            "encoder tail: tabm.enabled and moe_routing are both set, but both "
            "replace the encoder's MLP tail. Choose one: disable TabM, set "
            "moe_routing=none, or move the mixture off the tail with "
            "moe_placement=post.");
    }
    if (parallel_config.enabled && (tabm_config.enabled || moe_config.enabled())) {
        throw std::invalid_argument(
            "encoder tail: parallel_layers.enabled is set together with " +
            std::string(tabm_config.enabled ? "tabm.enabled" : "moe_routing") +
            ", but both replace the encoder's MLP tail. Choose one, or move the "
            "mixture off the tail with moe_placement=post.");
    }

    if (parallel_config.enabled) {
        // Branches over the encoder's finished features, aggregated as
        // ParallelLayersConfig::aggregation says. The block is the tail: the
        // plain MLP is not built at all, so the branches carry the encoder's
        // final capacity rather than sitting on top of a full-depth stack --
        // the same rule the mixture follows.
        if (parallel_config.branches.empty()) {
            throw std::invalid_argument(
                "encoder tail: parallel_layers.enabled is set but carries no "
                "branches. Add at least one ParallelBranchConfig, or leave "
                "parallel_layers disabled.");
        }
        tail.parallel = owner.register_module("parallel",
                                              ParallelBlock(input_dim, parallel_config));
        tail.latent_dim = tail.parallel->output_dim();
        return tail.latent_dim;
    }

    if (tabm_config.enabled) {
        tail.tabm = owner.register_module("tabm", TabMEncoder(
            input_dim, hidden_dims, tabm_config.n_ensembles, config,
            tabm_config.aggregation));
        tail.latent_dim = tail.tabm->output_dim();
        return tail.latent_dim;
    }

    if (moe_config.enabled()) {
        auto build = build_mlp_configurable(
            input_dim, moe_backbone_dims(hidden_dims), config);
        tail.backbone = owner.register_module("backbone", build.mlp);
        tail.activation_indices = build.activation_indices;

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

TailOutput forward_encoder_tail(EncoderTail& tail, torch::Tensor x,
                                std::vector<torch::Tensor>* activations) {
    if (tail.has_parallel()) {
        return {tail.parallel->forward(std::move(x), activations), {}, {}};
    }
    if (tail.has_tabm()) {
        return {tail.tabm->forward(std::move(x), activations), {}, {}};
    }
    if (tail.has_moe()) {
        auto result = tail.moe->forward(
            run_mlp(tail.backbone, tail.activation_indices, std::move(x), activations));
        return {result.output, result.aux_loss, result.gate_probs};
    }
    return {run_mlp(tail.mlp, tail.activation_indices, std::move(x), activations), {}, {}};
}

// =============================================================================
// Parallel Branch Implementation
// =============================================================================

ParallelBranchImpl::ParallelBranchImpl(
    int64_t input_dim,
    const ParallelBranchConfig& config
) : branch_weight_(config.branch_weight) {
    MLPBlockConfig mlp_config;
    mlp_config.activation = config.activation;
    mlp_config.normalization = config.normalization;
    mlp_config.dropout = config.dropout;

    auto result = build_mlp_configurable(input_dim, config.hidden_dims, mlp_config);
    mlp_ = register_module("mlp", result.mlp);
    activation_indices_ = result.activation_indices;
    output_dim_ = result.output_dim;
}

torch::Tensor ParallelBranchImpl::forward(torch::Tensor x,
                                          std::vector<torch::Tensor>* activations) {
    auto out = run_mlp(mlp_, activation_indices_, std::move(x), activations);
    // What this branch is worth in the aggregation
    // (ParallelBranchConfig::branch_weight). A weight of 1 -- the default --
    // leaves the output untouched, and 0 takes the branch out of the sum
    // without removing its parameters. Applied here rather than in the
    // aggregation so every mode weighs a branch the same way, the learned ones
    // included: attention and gating then see a branch scaled by what the
    // configuration says it is worth.
    return branch_weight_ == 1.0f ? out : out * branch_weight_;
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

    // Query projected from the mean of the branch outputs, keys and values
    // from each branch
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

    // The query is projected from the mean over branches
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

torch::Tensor ParallelBlockImpl::forward(torch::Tensor x,
                                         std::vector<torch::Tensor>* activations) {
    auto identity = x;

    std::vector<torch::Tensor> branch_outputs;
    for (auto& branch : branches_) {
        branch_outputs.push_back(branch->forward(x, activations));
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
