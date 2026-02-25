#include "resolve/encoder.hpp"
#include <cmath>
#include <stdexcept>

namespace resolve {

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

    // Reshape to (batch, n_positions * embed_dim)
    return flat_emb.view({batch_size, n_positions_ * embed_dim_});
}

// =============================================================================
// Mish Activation (not built into libtorch)
// =============================================================================

inline torch::Tensor mish(torch::Tensor x) {
    return x * torch::tanh(torch::softplus(x));
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
    switch (type) {
        case NormLayerType::BatchNorm:
            return torch::nn::AnyModule(torch::nn::BatchNorm1d(dim));
        case NormLayerType::LayerNorm:
            return torch::nn::AnyModule(torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({dim})
            ));
        case NormLayerType::GroupNorm:
            // Ensure groups divides dim evenly
            {
                int groups = norm_groups;
                while (groups > 1 && dim % groups != 0) {
                    groups--;
                }
                return torch::nn::AnyModule(torch::nn::GroupNorm(groups, dim));
            }
        case NormLayerType::RMSNorm:
            return torch::nn::AnyModule(RMSNorm(dim));
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
            case NormLayerType::GroupNorm: {
                int groups = config.norm_groups;
                while (groups > 1 && output_dim % groups != 0) {
                    groups--;
                }
                norm_gn_ = register_module("norm", torch::nn::GroupNorm(groups, output_dim));
                break;
            }
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

    // Apply normalization using typed module
    if (has_norm_) {
        switch (norm_type_) {
            case NormLayerType::BatchNorm:
                out = norm_bn_->forward(out);
                break;
            case NormLayerType::LayerNorm:
                out = norm_ln_->forward(out);
                break;
            case NormLayerType::GroupNorm:
                out = norm_gn_->forward(out);
                break;
            case NormLayerType::RMSNorm:
                out = norm_rms_->forward(out);
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
// PlotEncoderImpl implementation (hash mode)
// =============================================================================

// New constructor with configurable architecture
PlotEncoderImpl::PlotEncoderImpl(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& mlp_config,
    const TabMConfig& tabm_config
) {
    init(n_continuous, n_genera, n_families, genus_emb_dim, family_emb_dim,
         top_k, hidden_dims, mlp_config, tabm_config);
}

// Legacy constructor (backward compatibility)
PlotEncoderImpl::PlotEncoderImpl(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) {
    MLPBlockConfig config;
    config.dropout = dropout;
    init(n_continuous, n_genera, n_families, genus_emb_dim, family_emb_dim,
         top_k, hidden_dims, config);
}

void PlotEncoderImpl::init(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config
) {
    has_taxonomy_ = (n_genera > 0 && n_families > 0);
    top_k_ = top_k;
    hidden_dims_ = hidden_dims;
    mlp_config_ = config;

    // Calculate input dimension
    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        // Create separate embedding for each rank position
        for (int k = 0; k < top_k_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(n_genera, genus_emb_dim)
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(n_families, family_emb_dim)
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_ * (genus_emb_dim + family_emb_dim);
    }

    // Build MLP backbone (standard or TabM)
    use_tabm_ = tabm_config.enabled;
    if (use_tabm_) {
        tabm_encoder_ = register_module("tabm", TabMEncoder(
            input_dim, hidden_dims, tabm_config.n_ensembles,
            config.dropout, tabm_config.aggregation));
        latent_dim_ = tabm_encoder_->output_dim();
    } else {
        auto result = build_mlp_configurable(input_dim, hidden_dims, config);
        mlp_ = register_module("mlp", result.mlp);
        latent_dim_ = result.output_dim;
        activation_indices_ = result.activation_indices;
    }
}

torch::Tensor PlotEncoderImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        // Get embeddings for each position
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
}



std::pair<torch::Tensor, std::vector<torch::Tensor>> PlotEncoderImpl::forward_with_activations(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    // Prepare input (same as forward)
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);

    // Run through MLP manually, capturing activations after each GELU
    // MLP structure: [Linear, BatchNorm1d, GELU, Dropout] x n_layers
    std::vector<torch::Tensor> activations;
    size_t n_layers = hidden_dims_.size();

    for (size_t layer = 0; layer < n_layers; ++layer) {
        size_t base_idx = layer * 4;  // 4 modules per layer
        // Linear
        x = mlp_->ptr(base_idx)->as<torch::nn::Linear>()->forward(x);
        // BatchNorm1d
        x = mlp_->ptr(base_idx + 1)->as<torch::nn::BatchNorm1d>()->forward(x);
        // GELU
        x = mlp_->ptr(base_idx + 2)->as<torch::nn::GELU>()->forward(x);
        // Capture activation after GELU (before dropout)
        activations.push_back(x.clone());
        // Dropout
        x = mlp_->ptr(base_idx + 3)->as<torch::nn::Dropout>()->forward(x);
    }

    return {x, activations};
}


// PlotEncoderEmbedImpl implementation (embed mode with learnable species embeddings)

// New constructor with configurable architecture and optional fused embeddings
PlotEncoderEmbedImpl::PlotEncoderEmbedImpl(
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
    const TabMConfig& tabm_config
) {
    init(n_continuous, n_species, n_genera, n_families, species_embed_dim,
         genus_emb_dim, family_emb_dim, top_k_species, top_k_taxonomy,
         hidden_dims, mlp_config, tabm_config);
}

// Legacy constructor (backward compatibility)
PlotEncoderEmbedImpl::PlotEncoderEmbedImpl(
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
    float dropout
) {
    MLPBlockConfig config;
    config.dropout = dropout;
    init(n_continuous, n_species, n_genera, n_families, species_embed_dim,
         genus_emb_dim, family_emb_dim, top_k_species, top_k_taxonomy,
         hidden_dims, config);
}

void PlotEncoderEmbedImpl::init(
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
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config
) {
    has_taxonomy_ = (n_genera > 0 && n_families > 0);
    top_k_species_ = top_k_species;
    top_k_taxonomy_ = top_k_taxonomy;
    mlp_config_ = config;

    // Calculate input dimension
    int64_t input_dim = n_continuous;

    // Fused positional embeddings: single embedding table per type
    // This reduces CUDA kernel launches from O(K) to O(1)
    fused_species_ = register_module(
        "fused_species",
        FusedPositionalEmbedding(n_species, top_k_species_, species_embed_dim)
    );
    input_dim += fused_species_->total_output_dim();

    // Taxonomy embeddings (fused)
    if (has_taxonomy_) {
        fused_genus_ = register_module(
            "fused_genus",
            FusedPositionalEmbedding(n_genera, top_k_taxonomy_, genus_emb_dim)
        );
        fused_family_ = register_module(
            "fused_family",
            FusedPositionalEmbedding(n_families, top_k_taxonomy_, family_emb_dim)
        );
        input_dim += fused_genus_->total_output_dim() + fused_family_->total_output_dim();
    }

    // Build MLP backbone (standard or TabM)
    use_tabm_ = tabm_config.enabled;
    if (use_tabm_) {
        tabm_encoder_ = register_module("tabm", TabMEncoder(
            input_dim, hidden_dims, tabm_config.n_ensembles,
            config.dropout, tabm_config.aggregation));
        latent_dim_ = tabm_encoder_->output_dim();
    } else {
        auto result = build_mlp_configurable(input_dim, hidden_dims, config);
        mlp_ = register_module("mlp", result.mlp);
        latent_dim_ = result.output_dim;
    }
}

torch::Tensor PlotEncoderEmbedImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    // Fused embedding lookups: single kernel launch per embedding type
    // species_ids: (batch, top_k_species) -> (batch, top_k_species * embed_dim)
    parts.push_back(fused_species_->forward(species_ids));

    // Embed taxonomy if available
    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        parts.push_back(fused_genus_->forward(genus_ids));
        parts.push_back(fused_family_->forward(family_ids));
    }

    auto x = torch::cat(parts, /*dim=*/1);
    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
}


// PlotEncoderSparseImpl implementation (explicit species vector mode)

// New constructor with configurable architecture
PlotEncoderSparseImpl::PlotEncoderSparseImpl(
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
    const TabMConfig& tabm_config
) {
    init(n_continuous, n_species, species_embed_dim, n_genera, n_families,
         genus_emb_dim, family_emb_dim, top_k, hidden_dims, mlp_config, tabm_config);
}

// Legacy constructor (backward compatibility)
PlotEncoderSparseImpl::PlotEncoderSparseImpl(
    int64_t n_continuous,
    int64_t n_species,
    int species_embed_dim,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    float dropout
) {
    MLPBlockConfig config;
    config.dropout = dropout;
    init(n_continuous, n_species, species_embed_dim, n_genera, n_families,
         genus_emb_dim, family_emb_dim, top_k, hidden_dims, config);
}

void PlotEncoderSparseImpl::init(
    int64_t n_continuous,
    int64_t n_species,
    int species_embed_dim,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    const TabMConfig& tabm_config
) {
    has_taxonomy_ = (n_genera > 0 && n_families > 0);
    n_species_ = n_species;
    top_k_ = top_k;
    mlp_config_ = config;

    // Calculate input dimension
    int64_t input_dim = n_continuous + species_embed_dim;

    // Linear projection from species abundances to embedding space
    species_projection_ = register_module(
        "species_projection",
        torch::nn::Linear(n_species, species_embed_dim)
    );

    // Taxonomy embeddings
    if (has_taxonomy_) {
        for (int k = 0; k < top_k_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(n_genera, genus_emb_dim)
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(n_families, family_emb_dim)
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_ * (genus_emb_dim + family_emb_dim);
    }

    // Build MLP backbone (standard or TabM)
    use_tabm_ = tabm_config.enabled;
    if (use_tabm_) {
        tabm_encoder_ = register_module("tabm", TabMEncoder(
            input_dim, hidden_dims, tabm_config.n_ensembles,
            config.dropout, tabm_config.aggregation));
        latent_dim_ = tabm_encoder_->output_dim();
    } else {
        auto result = build_mlp_configurable(input_dim, hidden_dims, config);
        mlp_ = register_module("mlp", result.mlp);
        latent_dim_ = result.output_dim;
    }
}

torch::Tensor PlotEncoderSparseImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    // Project species vector to embedding space
    auto species_emb = species_projection_->forward(species_vector);

    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    parts.push_back(species_emb);

    // Embed taxonomy if available
    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
}


// TaskHeadImpl implementation

// Helper to initialize output layer
void TaskHeadImpl::init_output_layer(int64_t input_dim, int num_classes) {
    int64_t out_features = (task_ == TaskType::Classification) ? num_classes : 1;
    output_ = register_module("output", torch::nn::Linear(input_dim, out_features));
}

// Legacy constructor (single linear layer)
TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform
) : task_(task), transform_(transform)
{
    init_output_layer(latent_dim, num_classes);
}

// New constructor with configurable hidden layers
TaskHeadImpl::TaskHeadImpl(
    int64_t latent_dim,
    TaskType task,
    int num_classes,
    TransformType transform,
    const std::vector<int64_t>& hidden_dims,
    ActivationType activation,
    float dropout
) : task_(task), transform_(transform), hidden_dims_(hidden_dims)
{
    int64_t prev_dim = latent_dim;

    // Build hidden layers if specified
    if (!hidden_dims.empty()) {
        torch::nn::Sequential mlp;
        for (size_t i = 0; i < hidden_dims.size(); ++i) {
            mlp->push_back(torch::nn::Linear(prev_dim, hidden_dims[i]));
            // Add activation
            auto act = make_activation(activation, hidden_dims[i]);
            mlp->push_back(act);
            // Add dropout if specified
            if (dropout > 0) {
                mlp->push_back(torch::nn::Dropout(dropout));
            }
            prev_dim = hidden_dims[i];
        }
        head_mlp_ = register_module("head_mlp", mlp);
    }

    // Final output layer
    init_output_layer(prev_dim, num_classes);
}

torch::Tensor TaskHeadImpl::forward(torch::Tensor latent) {
    auto x = latent;
    if (head_mlp_) {
        x = head_mlp_->forward(x);
    }
    return output_->forward(x);
}

torch::Tensor TaskHeadImpl::predict(torch::Tensor latent) {
    auto output = forward(latent);

    if (task_ == TaskType::Classification) {
        return torch::argmax(output, /*dim=*/1);
    } else {
        output = output.squeeze(-1);
        return inverse_transform(output);
    }
}

torch::Tensor TaskHeadImpl::inverse_transform(torch::Tensor predictions) {
    if (transform_ == TransformType::Log1p) {
        return torch::expm1(torch::clamp(predictions, kExpClampMin, kExpClampMax));
    }
    return predictions;
}

// =============================================================================
// PlotEncoderMoE Implementation (hash mode with Mixture of Experts)
// =============================================================================

// New constructor with configurable architecture
PlotEncoderMoEImpl::PlotEncoderMoEImpl(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& mlp_config,
    int n_experts,
    const std::vector<int64_t>& expert_hidden_dims,
    MoERoutingType moe_routing,
    int moe_top_k,
    float moe_noise_std
) {
    init(n_continuous, n_genera, n_families, genus_emb_dim, family_emb_dim,
         top_k, hidden_dims, mlp_config, n_experts, expert_hidden_dims,
         moe_routing, moe_top_k, moe_noise_std);
}

// Legacy constructor (backward compatibility)
PlotEncoderMoEImpl::PlotEncoderMoEImpl(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    float dropout,
    int n_experts,
    const std::vector<int64_t>& expert_hidden_dims,
    MoERoutingType moe_routing,
    int moe_top_k,
    float moe_noise_std
) {
    MLPBlockConfig config;
    config.dropout = dropout;
    init(n_continuous, n_genera, n_families, genus_emb_dim, family_emb_dim,
         top_k, hidden_dims, config, n_experts, expert_hidden_dims,
         moe_routing, moe_top_k, moe_noise_std);
}

void PlotEncoderMoEImpl::init(
    int64_t n_continuous,
    int64_t n_genera,
    int64_t n_families,
    int genus_emb_dim,
    int family_emb_dim,
    int top_k,
    const std::vector<int64_t>& hidden_dims,
    const MLPBlockConfig& config,
    int n_experts,
    const std::vector<int64_t>& expert_hidden_dims,
    MoERoutingType moe_routing,
    int moe_top_k,
    float moe_noise_std
) {
    has_taxonomy_ = (n_genera > 0 && n_families > 0);
    top_k_ = top_k;
    n_experts_ = n_experts;
    moe_routing_ = moe_routing;
    mlp_config_ = config;

    // Calculate input dimension
    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        // Create separate embedding for each rank position
        for (int k = 0; k < top_k_; ++k) {
            auto genus_emb = register_module(
                "genus_emb_" + std::to_string(k),
                torch::nn::Embedding(n_genera, genus_emb_dim)
            );
            genus_embeddings_.push_back(genus_emb);

            auto family_emb = register_module(
                "family_emb_" + std::to_string(k),
                torch::nn::Embedding(n_families, family_emb_dim)
            );
            family_embeddings_.push_back(family_emb);
        }
        input_dim += top_k_ * (genus_emb_dim + family_emb_dim);
    }

    // Build backbone MLP (use all but last layer from hidden_dims)
    // The MoE layer will replace the final layers
    std::vector<int64_t> backbone_dims;
    if (hidden_dims.size() > 2) {
        // Use first N-2 layers as backbone, MoE handles the rest
        for (size_t i = 0; i < hidden_dims.size() - 2; ++i) {
            backbone_dims.push_back(hidden_dims[i]);
        }
    } else {
        // If hidden_dims is small, use just the first layer as backbone
        backbone_dims.push_back(hidden_dims[0]);
    }

    // Build backbone with configurable architecture
    auto result = build_mlp_configurable(input_dim, backbone_dims, config);
    backbone_ = register_module("backbone", result.mlp);
    backbone_output_dim_ = result.output_dim;

    // Determine final output dimension from hidden_dims
    int64_t moe_output_dim = hidden_dims.empty() ? backbone_output_dim_ : hidden_dims.back();
    latent_dim_ = moe_output_dim;

    // Create MoE layer
    moe_ = register_module("moe", MixtureOfExperts(
        backbone_output_dim_,
        expert_hidden_dims,
        moe_output_dim,
        n_experts,
        moe_routing,
        moe_top_k,
        moe_noise_std,
        config.dropout
    ));
}

torch::Tensor PlotEncoderMoEImpl::encode_input(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    if (has_taxonomy_ && genus_ids.defined() && family_ids.defined()) {
        for (int k = 0; k < top_k_; ++k) {
            auto g_emb = genus_embeddings_[k](genus_ids.select(1, k));
            parts.push_back(g_emb);
        }
        for (int k = 0; k < top_k_; ++k) {
            auto f_emb = family_embeddings_[k](family_ids.select(1, k));
            parts.push_back(f_emb);
        }
    }

    auto x = torch::cat(parts, /*dim=*/1);
    return backbone_->forward(x);
}

std::pair<torch::Tensor, torch::Tensor> PlotEncoderMoEImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    auto backbone_out = encode_input(continuous, genus_ids, family_ids);
    auto moe_result = moe_->forward(backbone_out);
    return {moe_result.output, moe_result.aux_loss};
}

torch::Tensor PlotEncoderMoEImpl::forward_simple(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    auto backbone_out = encode_input(continuous, genus_ids, family_ids);
    return moe_->forward_simple(backbone_out);
}

torch::Tensor PlotEncoderMoEImpl::get_gate_probs(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    auto backbone_out = encode_input(continuous, genus_ids, family_ids);
    auto moe_result = moe_->forward(backbone_out);
    return moe_result.gate_probs;
}

// =============================================================================
// Embedding Weight Extraction (per-position → averaged)
// =============================================================================

// Helper: stack per-position embedding weights and average
static torch::Tensor average_embedding_weights(
    const std::vector<torch::nn::Embedding>& embeddings
) {
    if (embeddings.empty()) return torch::Tensor();
    std::vector<torch::Tensor> weights;
    weights.reserve(embeddings.size());
    for (const auto& emb : embeddings) {
        weights.push_back(emb->weight);
    }
    // (top_k, vocab_size, emb_dim) → mean(0) → (vocab_size, emb_dim)
    return torch::stack(weights, 0).mean(0);
}

// Helper: extract from FusedPositionalEmbedding and average across positions
static torch::Tensor average_fused_weights(
    const FusedPositionalEmbedding& fused
) {
    if (!fused) return torch::Tensor();
    auto weight = fused->embedding().weight;  // (vocab_size * n_positions, embed_dim)
    auto n_pos = fused->n_positions();
    auto vocab = fused->vocab_size();
    auto dim = fused->embed_dim();
    // (n_positions, vocab_size, embed_dim) → mean(0)
    return weight.view({n_pos, vocab, dim}).mean(0);
}

// PlotEncoder (hash mode, per-position)
torch::Tensor PlotEncoderImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(genus_embeddings_);
}

torch::Tensor PlotEncoderImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(family_embeddings_);
}

// PlotEncoderEmbed (embed mode, fused)
torch::Tensor PlotEncoderEmbedImpl::get_species_weights() const {
    return average_fused_weights(fused_species_);
}

torch::Tensor PlotEncoderEmbedImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_fused_weights(fused_genus_);
}

torch::Tensor PlotEncoderEmbedImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_fused_weights(fused_family_);
}

// PlotEncoderSparse (sparse mode, per-position)
torch::Tensor PlotEncoderSparseImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(genus_embeddings_);
}

torch::Tensor PlotEncoderSparseImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(family_embeddings_);
}

// PlotEncoderMoE (MoE mode, per-position)
torch::Tensor PlotEncoderMoEImpl::get_genus_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(genus_embeddings_);
}

torch::Tensor PlotEncoderMoEImpl::get_family_weights() const {
    if (!has_taxonomy_) return torch::Tensor();
    return average_embedding_weights(family_embeddings_);
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

BranchAttentionImpl::BranchAttentionImpl(int64_t branch_dim, int n_branches, int n_heads)
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

GatedAggregationImpl::GatedAggregationImpl(int64_t input_dim, int n_branches, int64_t branch_dim)
    : n_branches_(n_branches)
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
            BranchAttention(first_branch_dim, static_cast<int>(branches_.size()), config.attention_heads));
    } else if (aggregation_ == ParallelAggregation::Gated) {
        gated_ = register_module("gated",
            GatedAggregation(input_dim, static_cast<int>(branches_.size()), first_branch_dim));
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
