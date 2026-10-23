#pragma once

#include "resolve/aux_loss.hpp"
#include "resolve/encoder.hpp"
#include "resolve/excelformer.hpp"
#include "resolve/ft_transformer.hpp"
#include "resolve/gnn.hpp"
#include "resolve/hetero_gnn.hpp"
#include "resolve/plot_inputs.hpp"
#include "resolve/saint.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/tabnet.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <memory>
#include <vector>

namespace resolve {

// =============================================================================
// TabularAdapter: bridges RESOLVE's species-encoding pipeline to the generic
// tabular architectures (FT-Transformer, TabNet, SAINT, GNN, ExcelFormer,
// HeterogeneousGNN)
// =============================================================================

// The adapter:
// 1. Takes the plot inputs the model forward receives (with the categorical
//    embeddings already fused into `continuous`)
// 2. Turns the species encoding into numerical features: the features of its
//    SpeciesFeatures (species_encoder.hpp) -- none for hash, whose hash
//    embedding is already in the continuous block; the fused species slots for
//    embed; the projected species vector for sparse; the pooled species
//    embeddings (with their pooled genus / family embeddings) and the
//    has_cover flag for rank_pool and transformer
// 3. Feeds those with the continuous block as numerical features, and the
//    fixed taxonomy slots of the hash / embed / sparse encodings as
//    categoricals (embedded per slot for the architectures that read one flat
//    feature vector), to the selected architecture
// 4. Returns a latent vector the model's heads read
//
// The construction refuses an architecture that reads an input the encoding
// does not provide: HeterogeneousGNN and the co-occurrence GNN graph read the
// species vector (sparse), the taxonomic GNN graph the fixed taxonomy slots.

// What one adapter forward produces: the latent, and the auxiliary objective
// terms the architecture adds to the training loss (TabNet's sparsity term).
struct AdapterOutput {
    torch::Tensor latent;
    AuxLosses aux_losses;
};

class TabularAdapterImpl : public torch::nn::Module {
public:
    TabularAdapterImpl(
        const ResolveSchema& schema,
        const ModelConfig& config
    );

    // `reference` is read by SAINT's row attention in eval mode: the training
    // rows every plot attends to (row_reference_rows()). Every other
    // architecture ignores it.
    AdapterOutput forward(const PlotInputs& inputs, const PlotInputs& reference = {});

    [[nodiscard]] int64_t latent_dim() const noexcept { return latent_dim_; }
    [[nodiscard]] EncoderArchitecture architecture() const noexcept { return architecture_; }

    // How many training rows SAINT's inference attends over; 0 for every other
    // architecture and for SAINT without row attention.
    [[nodiscard]] int64_t row_reference_rows() const noexcept {
        return saint_ ? saint_->reference_rows() : 0;
    }

    // The species graph a HeterogeneousGNN passes messages on.
    void set_species_graph(const torch::Tensor& edge_index, const torch::Tensor& edge_type);
    [[nodiscard]] bool has_species_graph() const;
    [[nodiscard]] torch::Tensor species_graph_edge_index() const;
    [[nodiscard]] torch::Tensor species_graph_edge_type() const;

    // Buffers that hold data the model was trained with rather than a
    // quantity of fixed shape: their shape is the data's (ResolveModel::
    // data_buffers).
    [[nodiscard]] std::vector<torch::Tensor> data_buffers() const;

private:
    // Continuous block plus the species-encoding features.
    torch::Tensor prepare_numerical(const PlotInputs& inputs);

    // (batch, 2 * n_taxonomy_slots) taxonomy codes, genus slots then family
    // slots; undefined without taxonomy.
    torch::Tensor taxonomy_codes(const PlotInputs& inputs) const;

    // The numerical features with the taxonomy slots embedded and appended:
    // the one flat vector TabNet and the GNN read.
    torch::Tensor flat_features(const torch::Tensor& numerical, const PlotInputs& inputs);

    // The per-plot features the GNN measures neighbourhood on, chosen by
    // GNNConfig::graph_mode: the coordinates, the taxonomic composition, or the
    // species vector. Refuses a mode the data cannot supply rather than
    // measuring distance on whatever happens to sit in those columns.
    torch::Tensor graph_features(const PlotInputs& inputs) const;

    EncoderArchitecture architecture_;
    int64_t latent_dim_ = 0;
    int64_t n_numerical_ = 0;
    bool has_taxonomy_ = false;
    int n_taxonomy_slots_ = 0;

    // The species encoding's features, registered on this module; null for
    // HeterogeneousGNN.
    std::unique_ptr<SpeciesFeatures> species_features_;

    // Projection from a transformer-family encoder's d_model to latent_dim
    torch::nn::Linear output_proj_{nullptr};

    // Per-slot taxonomy embeddings for the flat-input architectures (TabNet,
    // GNN), which cannot read integer codes as magnitudes. The attention
    // encoders embed the codes in their own tokenizer instead.
    std::unique_ptr<SlotTaxonomy> slot_taxonomy_;

    // Architecture-specific encoders (only one is built)
    FTTransformerEncoder ft_transformer_{nullptr};
    TabNetEncoder tabnet_{nullptr};
    SAINTEncoder saint_{nullptr};
    GNNEncoder gnn_{nullptr};
    ExcelFormerEncoder excelformer_{nullptr};
    HeterogeneousGNNEncoder hetero_gnn_{nullptr};

    float tabnet_sparsity_coefficient_ = 0.0f;

    // GNN graph construction (GNNConfig::k_neighbors / graph_mode /
    // use_edge_features).
    int k_neighbors_ = 10;
    GraphConstructionMode graph_mode_ = GraphConstructionMode::Spatial;
    bool use_edge_features_ = false;
    // Vocabulary widths for the taxonomic composition vector.
    int64_t genus_vocab_size_ = 0;
    int64_t family_vocab_size_ = 0;
    // The spatial graph reads the first two continuous columns, which are the
    // coordinates only when the dataset has them.
    bool has_coordinates_ = false;
};

TORCH_MODULE(TabularAdapter);

} // namespace resolve
