#include "resolve/encoder.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace resolve {

// Forward declarations for weight extraction helpers (defined in encoder_common.cpp)
torch::Tensor average_embedding_weights(const std::vector<torch::nn::Embedding>& embeddings);
torch::Tensor average_fused_weights(const FusedPositionalEmbedding& fused);

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

    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        input_dim += register_per_rank_embeddings(
            *this, genus_embeddings_, family_embeddings_,
            n_genera, n_families, genus_emb_dim, family_emb_dim, top_k_);
    }

    auto bb = build_and_register_backbone(
        *this, mlp_, tabm_encoder_, input_dim, hidden_dims, config, tabm_config);
    use_tabm_ = bb.use_tabm;
    latent_dim_ = bb.latent_dim;
    activation_indices_ = bb.activation_indices;
}

torch::Tensor PlotEncoderImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    embed_per_rank_taxonomy(parts, genus_embeddings_, family_embeddings_,
                            genus_ids, family_ids, top_k_, has_taxonomy_);

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
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    embed_per_rank_taxonomy(parts, genus_embeddings_, family_embeddings_,
                            genus_ids, family_ids, top_k_, has_taxonomy_);

    auto x = torch::cat(parts, /*dim=*/1);

    // Run full MLP forward pass and capture final output
    // Activation capture requires per-layer iteration which is not supported
    // by libtorch Sequential's type-erased storage. Return the final output
    // with empty activations when using non-standard MLP configurations.
    std::vector<torch::Tensor> activations;
    x = mlp_->forward(x);

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
    // Genus OR family real entries enable taxonomy (matches the transform gate
    // in EmbeddingEncoder), so family-only datasets keep family embeddings.
    has_taxonomy_ = (n_genera > 1 || n_families > 1);
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

    auto bb = build_and_register_backbone(
        *this, mlp_, tabm_encoder_, input_dim, hidden_dims, config, tabm_config);
    use_tabm_ = bb.use_tabm;
    latent_dim_ = bb.latent_dim;
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

    int64_t input_dim = n_continuous + species_embed_dim;

    species_projection_ = register_module(
        "species_projection",
        torch::nn::Linear(n_species, species_embed_dim)
    );

    if (has_taxonomy_) {
        input_dim += register_per_rank_embeddings(
            *this, genus_embeddings_, family_embeddings_,
            n_genera, n_families, genus_emb_dim, family_emb_dim, top_k_);
    }

    auto bb = build_and_register_backbone(
        *this, mlp_, tabm_encoder_, input_dim, hidden_dims, config, tabm_config);
    use_tabm_ = bb.use_tabm;
    latent_dim_ = bb.latent_dim;
}

torch::Tensor PlotEncoderSparseImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    auto species_emb = species_projection_->forward(species_vector);

    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    parts.push_back(species_emb);
    embed_per_rank_taxonomy(parts, genus_embeddings_, family_embeddings_,
                            genus_ids, family_ids, top_k_, has_taxonomy_);

    auto x = torch::cat(parts, /*dim=*/1);
    if (use_tabm_) {
        return tabm_encoder_->forward(x);
    }
    return mlp_->forward(x);
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

    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        input_dim += register_per_rank_embeddings(
            *this, genus_embeddings_, family_embeddings_,
            n_genera, n_families, genus_emb_dim, family_emb_dim, top_k_);
    }

    // Build backbone MLP (use all but last layer from hidden_dims)
    // The MoE layer will replace the final layers
    std::vector<int64_t> backbone_dims;
    if (hidden_dims.size() > 2) {
        // Use first N-2 layers as backbone, MoE handles the rest
        for (size_t i = 0; i < hidden_dims.size() - 2; ++i) {
            backbone_dims.push_back(hidden_dims[i]);
        }
    } else if (!hidden_dims.empty()) {
        // If hidden_dims is small (but non-empty), use just the first layer as backbone
        backbone_dims.push_back(hidden_dims[0]);
    }
    // else: empty hidden_dims -> identity backbone (backbone_dims stays empty),
    // matching build_mlp_configurable's tolerance for an empty spec.

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
    embed_per_rank_taxonomy(parts, genus_embeddings_, family_embeddings_,
                            genus_ids, family_ids, top_k_, has_taxonomy_);

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
// Weight Extraction for Hash/Embed/Sparse/MoE Encoders
// =============================================================================

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

} // namespace resolve
