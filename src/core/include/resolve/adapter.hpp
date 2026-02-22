#pragma once

#include "resolve/types.hpp"
#include "resolve/encoder.hpp"
#include "resolve/attention.hpp"
#include <torch/torch.h>

namespace resolve {

// =============================================================================
// TabularAdapter: bridges RESOLVE's species-encoding pipeline to generic
// tabular architectures (FT-Transformer, TabNet, SAINT, GNN, TraitNet)
// =============================================================================

// The adapter:
// 1. Takes same inputs as PlotEncoder (continuous, genus_ids, family_ids,
//    species_ids, species_vector)
// 2. Produces species embedding via shared embedding logic
// 3. Feeds combined features as numerical + taxonomy IDs as categoricals
//    to the selected tabular architecture
// 4. Returns latent vector of the same shape as PlotEncoder output

class TabularAdapterImpl : public torch::nn::Module {
public:
    TabularAdapterImpl(
        const ResolveSchema& schema,
        const ModelConfig& config
    );

    // Forward pass (same signature as PlotEncoder variants)
    torch::Tensor forward(
        torch::Tensor continuous,
        torch::Tensor genus_ids = {},
        torch::Tensor family_ids = {},
        torch::Tensor species_ids = {},
        torch::Tensor species_vector = {}
    );

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] EncoderArchitecture architecture() const noexcept { return architecture_; }

private:
    // Prepare numerical features from species encoding + continuous
    torch::Tensor prepare_numerical(
        torch::Tensor continuous,
        torch::Tensor species_ids,
        torch::Tensor species_vector
    );

    // Prepare categorical features from taxonomy IDs
    std::vector<torch::Tensor> prepare_categoricals(
        torch::Tensor genus_ids,
        torch::Tensor family_ids
    );

    EncoderArchitecture architecture_;
    SpeciesEncodingMode species_encoding_;
    int64_t latent_dim_;
    int64_t n_numerical_;
    int64_t n_categoricals_;

    // Species embedding for embed mode
    FusedPositionalEmbedding fused_species_{nullptr};
    int top_k_species_ = 0;
    int species_embed_dim_ = 0;

    // Species projection for sparse mode
    torch::nn::Linear species_projection_{nullptr};
    int species_project_dim_ = 0;

    // Output projection from architecture output_dim to desired latent_dim
    torch::nn::Linear output_proj_{nullptr};
    bool needs_output_proj_ = false;

    // Architecture-specific encoders (only one is used)
    FTTransformerEncoder ft_transformer_{nullptr};
    TabNetEncoder tabnet_{nullptr};
    SAINTEncoder saint_{nullptr};
    GNNEncoder gnn_{nullptr};
    ExcelFormerEncoder excelformer_{nullptr};
    HeterogeneousGNNEncoder hetero_gnn_{nullptr};

    // For GNN: adjacency matrix builder
    int k_neighbors_ = 10;

    // For Heterogeneous GNN: stored graph structure
    torch::Tensor hetero_edge_index_;  // (2, n_edges)
    torch::Tensor hetero_edge_type_;   // (n_edges,)
    bool hetero_graph_set_ = false;

public:
    // Set the species graph for HeterogeneousGNN mode
    // Must be called before forward() when using HeterogeneousGNN architecture
    void set_species_graph(torch::Tensor edge_index, torch::Tensor edge_type);
    [[nodiscard]] bool has_species_graph() const noexcept { return hetero_graph_set_; }
};

TORCH_MODULE(TabularAdapter);

} // namespace resolve
