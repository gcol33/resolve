#include "resolve/encoder.hpp"
#include <cmath>
#include <set>
#include <stdexcept>

namespace resolve {

// Forward declarations for weight extraction helpers (defined in encoder_common.cpp)
torch::Tensor average_embedding_weights(const std::vector<torch::nn::Embedding>& embeddings);
torch::Tensor average_fused_weights(const FusedPositionalEmbedding& fused);

namespace {
// Per-position taxonomy weight accessor shared by the hash and sparse
// encoders: an undefined tensor when taxonomy is disabled, else the position-
// averaged embedding table. Single source for the identical bodies (issue #99).
torch::Tensor taxonomy_weights(bool has_taxonomy,
                               const std::vector<torch::nn::Embedding>& embeddings) {
    if (!has_taxonomy) return torch::Tensor();
    return average_embedding_weights(embeddings);
}
}  // namespace

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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    init(n_continuous, n_genera, n_families, genus_emb_dim, family_emb_dim,
         top_k, hidden_dims, mlp_config, tabm_config, moe_config);
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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    has_taxonomy_ = (n_genera > 1 || n_families > 1);
    top_k_ = top_k;
    hidden_dims_ = hidden_dims;
    mlp_config_ = config;

    int64_t input_dim = n_continuous;

    if (has_taxonomy_) {
        input_dim += register_per_rank_embeddings(
            *this, genus_embeddings_, family_embeddings_,
            n_genera, n_families, genus_emb_dim, family_emb_dim, top_k_);
    }

    latent_dim_ = build_encoder_tail(
        *this, tail_, input_dim, hidden_dims, config, tabm_config, moe_config);
    activation_indices_ = tail_.activation_indices;
}

TailOutput PlotEncoderImpl::encode(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);
    embed_per_rank_taxonomy(parts, genus_embeddings_, family_embeddings_,
                            genus_ids, family_ids, top_k_, has_taxonomy_);

    return forward_encoder_tail(tail_, torch::cat(parts, /*dim=*/1));
}

torch::Tensor PlotEncoderImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    return encode(std::move(continuous), std::move(genus_ids),
                  std::move(family_ids)).latent;
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

    // Run the full forward pass and capture the final output.
    // Activation capture requires per-layer iteration which is not supported
    // by libtorch Sequential's type-erased storage. Return the final output
    // with empty activations when using non-standard MLP configurations.
    std::vector<torch::Tensor> activations;
    x = forward_encoder_tail(tail_, x).latent;

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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    init(n_continuous, n_species, n_genera, n_families, species_embed_dim,
         genus_emb_dim, family_emb_dim, top_k_species, top_k_taxonomy,
         hidden_dims, mlp_config, tabm_config, moe_config);
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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
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

    latent_dim_ = build_encoder_tail(
        *this, tail_, input_dim, hidden_dims, config, tabm_config, moe_config);
}

torch::Tensor PlotEncoderEmbedImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    return encode(std::move(continuous), std::move(species_ids),
                  std::move(genus_ids), std::move(family_ids)).latent;
}

TailOutput PlotEncoderEmbedImpl::encode(
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

    // Embed taxonomy. A taxonomy-enabled embed model reserves fixed concat width
    // for both genus and family, so both must be present; check_concat_taxonomy
    // throws a clear error instead of the opaque cat/Linear shape error a partial
    // or missing taxonomy input would otherwise trigger (issue #99).
    check_concat_taxonomy(has_taxonomy_, genus_ids, family_ids, "PlotEncoderEmbed");
    if (has_taxonomy_) {
        parts.push_back(fused_genus_->forward(genus_ids));
        parts.push_back(fused_family_->forward(family_ids));
    }

    return forward_encoder_tail(tail_, torch::cat(parts, /*dim=*/1));
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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    init(n_continuous, n_species, species_embed_dim, n_genera, n_families,
         genus_emb_dim, family_emb_dim, top_k, hidden_dims, mlp_config,
         tabm_config, moe_config);
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
    const TabMConfig& tabm_config,
    const MoETailConfig& moe_config
) {
    has_taxonomy_ = (n_genera > 1 || n_families > 1);
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

    latent_dim_ = build_encoder_tail(
        *this, tail_, input_dim, hidden_dims, config, tabm_config, moe_config);
}

torch::Tensor PlotEncoderSparseImpl::forward(
    torch::Tensor continuous,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    return encode(std::move(continuous), std::move(species_vector),
                  std::move(genus_ids), std::move(family_ids)).latent;
}

TailOutput PlotEncoderSparseImpl::encode(
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

    return forward_encoder_tail(tail_, torch::cat(parts, /*dim=*/1));
}


// =============================================================================
// Weight Extraction for Hash/Embed/Sparse Encoders
// =============================================================================

// PlotEncoder (hash mode, per-position)
torch::Tensor PlotEncoderImpl::get_genus_weights() const {
    return taxonomy_weights(has_taxonomy_, genus_embeddings_);
}

torch::Tensor PlotEncoderImpl::get_family_weights() const {
    return taxonomy_weights(has_taxonomy_, family_embeddings_);
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
    return taxonomy_weights(has_taxonomy_, genus_embeddings_);
}

torch::Tensor PlotEncoderSparseImpl::get_family_weights() const {
    return taxonomy_weights(has_taxonomy_, family_embeddings_);
}

} // namespace resolve
