#include "resolve/adapter.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/sparse_rows.hpp"

#include <stdexcept>
#include <string>

namespace resolve {

namespace {

// Multi-hot taxonomic composition: column j counts the plot's slots naming
// genus j, and the family block after it does the same for families. Slot id 0
// is the reserved <UNK>/padding row and contributes nothing, so a plot with
// fewer genera than slots is not pulled towards every other short plot.
torch::Tensor taxonomy_composition(const torch::Tensor& genus_ids,
                                   const torch::Tensor& family_ids,
                                   int64_t n_genera, int64_t n_families) {
    const torch::Tensor& present = genus_ids.defined() && genus_ids.numel() > 0
                                       ? genus_ids : family_ids;
    const int64_t n_rows = present.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32)
                       .device(present.device());
    auto composition = torch::zeros({n_rows, n_genera + n_families}, options);

    auto add_block = [&](const torch::Tensor& ids, int64_t offset, int64_t vocab) {
        if (!ids.defined() || ids.numel() == 0 || vocab <= 0) return;
        auto slots = ids.to(torch::kLong).clamp(0, vocab - 1);
        auto counts = (slots > 0).to(torch::kFloat32);
        auto block = composition.slice(/*dim=*/1, offset, offset + vocab);
        block.scatter_add_(/*dim=*/1, slots, counts);
    };
    add_block(genus_ids, 0, n_genera);
    add_block(family_ids, n_genera, n_families);
    return composition;
}

// Refuses an architecture that reads an input the species encoding does not
// provide, naming what would work.
void require_encoding_supported(EncoderArchitecture architecture, SpeciesEncodingMode encoding,
                                const ModelConfig& config) {
    const auto& traits = species_encoding_traits(encoding);
    auto refuse = [&](const std::string& needs, const std::string& remedy) {
        throw std::invalid_argument(
            std::string("encoder_architecture=") + encoder_architecture_to_string(architecture) +
            " " + needs + ", which species_encoding=" + species_encoding_to_string(encoding) +
            " does not provide. " + remedy);
    };
    if (architecture == EncoderArchitecture::HeterogeneousGNN &&
        encoding != SpeciesEncodingMode::Sparse) {
        refuse("aggregates its species-graph embeddings through the plot's species vector",
               "Use species_encoding=sparse.");
    }
    if (architecture == EncoderArchitecture::GNN) {
        if (config.gnn.graph_mode == GraphConstructionMode::CoOccurrence &&
            encoding != SpeciesEncodingMode::Sparse) {
            refuse("with graph_mode=cooccurrence builds its graph from the plot's species vector",
                   "Use species_encoding=sparse, or another graph_mode.");
        }
        if (config.gnn.graph_mode == GraphConstructionMode::Taxonomic && !traits.taxonomy_slots) {
            refuse("with graph_mode=taxonomic builds its graph from the fixed genus / family slots",
                   "Use species_encoding=hash, embed or sparse, or another graph_mode.");
        }
    }
}

}  // namespace

TabularAdapterImpl::TabularAdapterImpl(
    const ResolveSchema& schema,
    const ModelConfig& config
) : architecture_(config.encoder_architecture),
    n_taxonomy_slots_(config.n_taxonomy_slots),
    has_coordinates_(schema.has_coordinates)
{
    // =========================================================================
    // Numerical features: the continuous block plus the species encoding's
    // features (species_encoder.hpp)
    // =========================================================================

    const SpeciesEncodingMode encoding = effective_species_encoding(config);
    require_encoding_supported(architecture_, encoding, config);
    const auto& traits = species_encoding_traits(encoding);
    const auto spec = SpeciesEncoderSpec::from_schema(schema, config);
    has_taxonomy_ = traits.taxonomy_slots && spec.has_taxonomy();

    const int64_t n_continuous = schema.continuous_width();
    n_numerical_ = n_continuous + (traits.hash_in_continuous ? config.hash_dim : 0);
    // HeterogeneousGNN builds plot features from the species graph and the
    // continuous block directly (see forward()).
    if (architecture_ != EncoderArchitecture::HeterogeneousGNN) {
        species_features_ = traits.make_features(*this, spec);
        n_numerical_ += species_features_->width();
    }

    // =========================================================================
    // Categorical features: one per taxonomy slot
    // =========================================================================

    // Genus slots then family slots. Codes run 0..n_genera-1 with the reserved
    // <UNK> = 0, so the cardinality is n_genera, the height of the model's
    // taxonomy tables.
    std::vector<int64_t> cat_cardinalities;
    if (has_taxonomy_) {
        cat_cardinalities.assign(static_cast<size_t>(n_taxonomy_slots_),
                                 schema.genus_table_size());
        cat_cardinalities.insert(cat_cardinalities.end(),
                                 static_cast<size_t>(n_taxonomy_slots_),
                                 schema.family_table_size());
    }

    // A transformer-family encoder's d_model output, projected to the model's
    // latent dim (the last hidden dim) when the two differ.
    auto project_to_latent = [&](int64_t arch_output) {
        latent_dim_ = config.hidden_dims.empty() ? arch_output : config.hidden_dims.back();
        if (arch_output != latent_dim_) {
            output_proj_ = register_module("output_proj",
                torch::nn::Linear(arch_output, latent_dim_));
        }
    };

    // Per-slot taxonomy embeddings for the flat-input architectures; returns
    // the width they add to the feature vector.
    auto register_taxonomy_embeddings = [&]() -> int64_t {
        if (!has_taxonomy_) return 0;
        slot_taxonomy_ = make_per_rank_slot_taxonomy(*this, spec);
        return slot_taxonomy_->width();
    };

    // =========================================================================
    // Architecture-specific encoder
    // =========================================================================

    switch (architecture_) {
        case EncoderArchitecture::FTTransformer: {
            const auto& cfg = config.ft_transformer;
            // The two configured rates act where they are named: attention
            // dropout on the attention weights, FFN dropout inside the
            // feed-forward hidden layer and on each sublayer's residual branch
            // (the non-attention dropout of Gorishniy et al. 2021).
            TransformerBlockConfig block{
                /*d_model=*/cfg.d_model,
                /*n_heads=*/cfg.n_heads,
                /*d_ff=*/cfg.ffn_multiplier * cfg.d_model,
                /*attention_dropout=*/cfg.attention_dropout,
                /*ffn_dropout=*/cfg.ffn_dropout,
                /*residual_dropout=*/cfg.ffn_dropout,
                /*pre_norm=*/cfg.pre_norm};
            ft_transformer_ = register_module("ft_transformer",
                FTTransformerEncoder(n_numerical_, cat_cardinalities, block, cfg.n_layers));
            project_to_latent(cfg.d_model);
            break;
        }

        case EncoderArchitecture::TabNet: {
            const auto& cfg = config.tabnet;
            const int64_t input_dim = n_numerical_ + register_taxonomy_embeddings();
            tabnet_ = register_module("tabnet",
                TabNetEncoder(input_dim, cfg.n_steps, cfg.n_d, cfg.n_a,
                              cfg.relaxation_factor, cfg.use_sparsemax,
                              cfg.virtual_batch_size));
            tabnet_sparsity_coefficient_ = cfg.sparsity_coefficient;
            latent_dim_ = cfg.n_d;
            break;
        }

        case EncoderArchitecture::SAINT: {
            const auto& cfg = config.saint;
            // SAINT configures one rate, which every dropout in the block runs
            // at.
            auto block = TransformerBlockConfig::uniform(
                cfg.d_model, cfg.n_heads, /*d_ff=*/4 * cfg.d_model,
                cfg.attention_dropout);
            saint_ = register_module("saint",
                SAINTEncoder(n_numerical_, cat_cardinalities, block, cfg.n_layers,
                             cfg.use_row_attention ? cfg.row_attention_rows : 0));
            project_to_latent(cfg.d_model);
            break;
        }

        case EncoderArchitecture::GNN: {
            const auto& cfg = config.gnn;
            k_neighbors_ = cfg.k_neighbors;
            graph_mode_ = cfg.graph_mode;
            use_edge_features_ = cfg.use_edge_features;
            genus_vocab_size_ = schema.genus_table_size();
            family_vocab_size_ = schema.family_table_size();
            const int64_t input_dim = n_numerical_ + register_taxonomy_embeddings();
            gnn_ = register_module("gnn",
                GNNEncoder(input_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.n_layers,
                           cfg.gnn_type, cfg.n_heads, cfg.edge_dropout));
            latent_dim_ = cfg.hidden_dim;
            break;
        }

        case EncoderArchitecture::ExcelFormer: {
            const auto& cfg = config.excelformer;
            auto block = TransformerBlockConfig::uniform(
                cfg.d_model, cfg.n_heads,
                /*d_ff=*/cfg.ffn_multiplier * cfg.d_model,
                cfg.attention_dropout, cfg.pre_norm);
            excelformer_ = register_module("excelformer",
                ExcelFormerEncoder(n_numerical_, cat_cardinalities, block, cfg.n_layers,
                                   cfg.importance_threshold));
            project_to_latent(cfg.d_model);
            break;
        }

        case EncoderArchitecture::HeterogeneousGNN: {
            const auto& cfg = config.heterogeneous_gnn;
            // Species embeddings learned on the species graph, aggregated per
            // plot through the species vector (sparse encoding).
            hetero_gnn_ = register_module("hetero_gnn",
                HeterogeneousGNNEncoder(schema.n_species_vocab, cfg.hidden_dim,
                                        cfg.output_dim, cfg.n_layers, cfg.n_edge_types,
                                        cfg.n_heads, cfg.dropout));
            // The continuous block concatenated with the per-plot species-graph
            // embedding, so covariates reach the plot representation.
            latent_dim_ = n_continuous + cfg.output_dim;
            break;
        }

        case EncoderArchitecture::TraitNet:
        case EncoderArchitecture::MLP:
            throw std::invalid_argument(
                std::string("TabularAdapter builds the tabular architectures; "
                            "encoder_architecture=") +
                (architecture_ == EncoderArchitecture::MLP ? "mlp" : "traitnet") +
                " is built by ResolveModel itself.");
    }
}

torch::Tensor TabularAdapterImpl::prepare_numerical(const PlotInputs& inputs) {
    if (!species_features_ || species_features_->width() == 0) return inputs.continuous;
    return torch::cat({inputs.continuous, species_features_->forward(inputs, is_training())},
                      /*dim=*/1);
}

torch::Tensor TabularAdapterImpl::taxonomy_codes(const PlotInputs& inputs) const {
    if (!has_taxonomy_) return {};
    check_concat_taxonomy(true, inputs.genus_ids, inputs.family_ids, "TabularAdapter");
    return torch::cat({inputs.genus_ids, inputs.family_ids}, /*dim=*/1);
}

torch::Tensor TabularAdapterImpl::flat_features(const torch::Tensor& numerical,
                                                const PlotInputs& inputs) {
    if (!slot_taxonomy_) return numerical;
    std::vector<torch::Tensor> parts{numerical};
    slot_taxonomy_->append(parts, inputs);
    return torch::cat(parts, /*dim=*/1);
}

torch::Tensor TabularAdapterImpl::graph_features(const PlotInputs& inputs) const {
    switch (graph_mode_) {
        case GraphConstructionMode::Spatial: {
            // The first two continuous columns are the plot coordinates ONLY
            // when the dataset carries them (continuous is laid out as
            // [coordinates | coordinate flag | covariates | covariate flags |
            // unknown_* | categorical_embed | ...], see continuous_block.hpp;
            // the flags are present under MissingValuePolicy::Indicate).
            // Without coordinates those columns are covariates, and a
            // "spatial" kNN graph built from them would be meaningless.
            TORCH_CHECK(has_coordinates_,
                "GNN encoder requires coordinates to build its spatial graph, "
                "but the dataset has none. Provide longitude/latitude roles, "
                "select another GNNConfig::graph_mode, or use a non-GNN encoder "
                "architecture.");
            return inputs.continuous.slice(/*dim=*/1, 0, 2);
        }
        case GraphConstructionMode::Taxonomic: {
            const bool has_genus = inputs.genus_ids.defined() && inputs.genus_ids.numel() > 0;
            const bool has_family = inputs.family_ids.defined() && inputs.family_ids.numel() > 0;
            TORCH_CHECK(has_genus || has_family,
                "GNNConfig::graph_mode = taxonomic builds its graph from the "
                "genus/family composition of each plot, but this forward "
                "carries no taxonomy IDs. Provide genus/family roles with "
                "DatasetConfig::use_taxonomy, or select another graph_mode.");
            TORCH_CHECK(genus_vocab_size_ > 0 || family_vocab_size_ > 0,
                "GNNConfig::graph_mode = taxonomic needs a fitted taxonomy "
                "vocabulary, but the schema reports none.");
            return taxonomy_composition(inputs.genus_ids, inputs.family_ids,
                                        genus_vocab_size_, family_vocab_size_);
        }
        case GraphConstructionMode::CoOccurrence: {
            TORCH_CHECK(inputs.species_vector.defined() && inputs.species_vector.numel() > 0,
                "GNNConfig::graph_mode = cooccurrence builds its graph from "
                "the per-plot species vector, which only the sparse species "
                "encoding provides. Set DatasetConfig::species_encoding = "
                "sparse, or select another graph_mode.");
            return sparse_rows_to_dense(inputs.species_vector).to(torch::kFloat32);
        }
    }
    throw std::invalid_argument(
        "unknown GraphConstructionMode value: " +
        std::to_string(static_cast<int>(graph_mode_)));
}

AdapterOutput TabularAdapterImpl::forward(const PlotInputs& inputs,
                                          const PlotInputs& reference) {
    AdapterOutput out;

    switch (architecture_) {
        case EncoderArchitecture::FTTransformer:
            out.latent = ft_transformer_->forward(prepare_numerical(inputs),
                                                  taxonomy_codes(inputs));
            break;

        case EncoderArchitecture::TabNet: {
            auto tabnet = tabnet_->forward(flat_features(prepare_numerical(inputs), inputs));
            out.latent = tabnet.output;
            out.aux_losses[kTabNetSparsityLoss] =
                tabnet_sparsity_coefficient_ * tabnet.mask_entropy;
            break;
        }

        case EncoderArchitecture::SAINT: {
            torch::Tensor reference_numerical;
            torch::Tensor reference_categoricals;
            if (!is_training() && reference.n_plots() > 0) {
                reference_numerical = prepare_numerical(reference);
                reference_categoricals = taxonomy_codes(reference);
            }
            out.latent = saint_->forward(prepare_numerical(inputs), taxonomy_codes(inputs),
                                         reference_numerical, reference_categoricals);
            break;
        }

        case EncoderArchitecture::ExcelFormer:
            out.latent = excelformer_->forward(prepare_numerical(inputs),
                                               taxonomy_codes(inputs));
            break;

        case EncoderArchitecture::GNN: {
            // The kNN graph is built over the plots in the current forward. For
            // this architecture BOTH training and inference forward the full node
            // set in a single batch (ResolveModel::requires_full_batch_training
            // forces full-batch training; Predictor::predict forces a single
            // full-batch inference forward), so the graph is the global
            // structure over all plots -- each plot attends to its true nearest
            // neighbours -- not an arbitrary mini-batch neighbourhood.
            //
            // Coordinates are a straight-line distance; a composition vector is
            // compared by the angle between plots, so that the number of
            // species recorded does not stand in for their identity.
            const GraphMetric metric = graph_mode_ == GraphConstructionMode::Spatial
                                           ? GraphMetric::Euclidean : GraphMetric::Cosine;
            auto graph = build_knn_graph(graph_features(inputs), k_neighbors_, metric,
                                         use_edge_features_);
            out.latent = gnn_->forward(flat_features(prepare_numerical(inputs), inputs), graph);
            break;
        }

        case EncoderArchitecture::HeterogeneousGNN: {
            TORCH_CHECK(inputs.species_vector.defined(),
                "HeterogeneousGNN requires species_vector (sparse encoding mode).");
            auto species_emb = hetero_gnn_->forward();  // (n_species, output_dim)
            auto plot_emb = HeterogeneousGNNEncoderImpl::aggregate_for_plots(
                species_emb, inputs.species_vector);
            out.latent = torch::cat({inputs.continuous, plot_emb}, /*dim=*/1);
            break;
        }

        case EncoderArchitecture::TraitNet:
        case EncoderArchitecture::MLP:
            throw std::logic_error("TabularAdapter holds no encoder for this architecture");
    }

    if (output_proj_) {
        out.latent = output_proj_->forward(out.latent);
    }
    return out;
}

void TabularAdapterImpl::set_species_graph(const torch::Tensor& edge_index,
                                           const torch::Tensor& edge_type) {
    TORCH_CHECK(hetero_gnn_, "set_species_graph(): only a HeterogeneousGNN reads a species graph");
    hetero_gnn_->set_graph(edge_index, edge_type);
}

bool TabularAdapterImpl::has_species_graph() const {
    return hetero_gnn_ && hetero_gnn_->has_graph();
}

torch::Tensor TabularAdapterImpl::species_graph_edge_index() const {
    return has_species_graph() ? hetero_gnn_->graph_edge_index() : torch::Tensor();
}

torch::Tensor TabularAdapterImpl::species_graph_edge_type() const {
    return has_species_graph() ? hetero_gnn_->graph_edge_type() : torch::Tensor();
}

std::vector<torch::Tensor> TabularAdapterImpl::data_buffers() const {
    if (!hetero_gnn_) return {};
    return {hetero_gnn_->graph_edge_index(), hetero_gnn_->graph_edge_type()};
}

} // namespace resolve
