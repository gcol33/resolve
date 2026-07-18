#include "resolve/adapter.hpp"
#include <stdexcept>

namespace resolve {

TabularAdapterImpl::TabularAdapterImpl(
    const ResolveSchema& schema,
    const ModelConfig& config
) : architecture_(config.encoder_architecture),
    species_encoding_(config.species_encoding),
    has_coordinates_(schema.has_coordinates)
{
    // =========================================================================
    // Step 1: Determine numerical feature count from species encoding
    // =========================================================================

    int64_t n_coords = schema.has_coordinates ? 2 : 0;
    int64_t n_unknown_features = schema.track_unknown_fraction ? 1 : 0;
    if (schema.track_unknown_count) n_unknown_features += 1;
    // Include the categorical-covariate embedding width: ResolveModel fuses the
    // CategoricalEmbedder output into `continuous` (fuse_categoricals_) BEFORE
    // calling the adapter, so the numerical block the adapter receives is wider
    // by exactly this amount. Omitting it made the FeatureTokenizer read only
    // the leading columns (silently dropping features) and crashed TabNet/GNN on
    // the shape mismatch.
    int64_t n_continuous_base = n_coords + static_cast<int64_t>(schema.covariate_names.size())
                                + n_unknown_features + schema.categorical_embed_width();

    // Species features contribute to numerical features
    int64_t n_species_features = 0;

    if (config.species_encoding == SpeciesEncodingMode::Hash && !config.uses_explicit_vector) {
        // Hash mode: hash embedding is already in continuous
        n_species_features = config.hash_dim;
        n_numerical_ = n_continuous_base + n_species_features;
    }
    else if (config.species_encoding == SpeciesEncodingMode::Embed) {
        // Embed mode: learnable species embeddings become numerical features
        top_k_species_ = config.top_k_species;
        species_embed_dim_ = config.species_embed_dim;
        fused_species_ = register_module("fused_species",
            FusedPositionalEmbedding(schema.n_species_vocab, top_k_species_, species_embed_dim_));
        n_species_features = top_k_species_ * species_embed_dim_;
        n_numerical_ = n_continuous_base + n_species_features;
    }
    else if (architecture_ == EncoderArchitecture::HeterogeneousGNN) {
        // HeterogeneousGNN builds plot features from the species graph and the
        // continuous block directly (see forward()); it never runs a
        // species-vector projection. Registering one here would add an untrained
        // n_species_vocab x embed_dim parameter block and a wasted per-batch GEMM.
        n_numerical_ = n_continuous_base;
    }
    else {
        // Sparse mode: project species vector to embedding
        species_project_dim_ = config.species_embed_dim;
        species_projection_ = register_module("species_projection",
            torch::nn::Linear(schema.n_species_vocab, species_project_dim_));
        n_species_features = species_project_dim_;
        n_numerical_ = n_continuous_base + n_species_features;
    }

    // =========================================================================
    // Step 2: Determine categorical features from taxonomy
    // =========================================================================

    std::vector<int64_t> cat_cardinalities;
    if (schema.has_taxonomy) {
        // Each taxonomy slot is a categorical feature
        int n_slots = config.n_taxonomy_slots;
        for (int k = 0; k < n_slots; ++k) {
            cat_cardinalities.push_back(schema.n_genera + 1);  // +1 for UNK
        }
        for (int k = 0; k < n_slots; ++k) {
            cat_cardinalities.push_back(schema.n_families + 1);
        }
    }
    n_categoricals_ = static_cast<int64_t>(cat_cardinalities.size());

    // Project a transformer-family encoder's d_model output down to the model's
    // latent dim (the last hidden dim, or the arch output when no hidden dims).
    // Shared by FT-Transformer / SAINT / ExcelFormer so the block lives once.
    auto setup_output_proj = [&](int64_t arch_output) {
        latent_dim_ = config.hidden_dims.empty() ? arch_output : config.hidden_dims.back();
        if (arch_output != latent_dim_) {
            output_proj_ = register_module("output_proj",
                torch::nn::Linear(arch_output, latent_dim_));
            needs_output_proj_ = true;
        } else {
            latent_dim_ = arch_output;
        }
    };

    // =========================================================================
    // Step 3: Create architecture-specific encoder
    // =========================================================================

    switch (architecture_) {
        case EncoderArchitecture::FTTransformer: {
            const auto& cfg = config.ft_transformer;
            int64_t d_ff = cfg.ffn_multiplier * cfg.d_model;
            ft_transformer_ = register_module("ft_transformer",
                FTTransformerEncoder(
                    n_numerical_,
                    cat_cardinalities,
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_layers,
                    d_ff,
                    cfg.attention_dropout,
                    /*use_cls_token=*/true,
                    cfg.pre_norm
                ));
            setup_output_proj(cfg.d_model);
            break;
        }

        case EncoderArchitecture::TabNet: {
            const auto& cfg = config.tabnet;
            // TabNet takes numerical features only (categoricals not embedded)
            tabnet_ = register_module("tabnet",
                TabNetEncoder(
                    n_numerical_,
                    cfg.n_steps,
                    cfg.n_d,
                    cfg.n_a,
                    cfg.relaxation_factor,
                    cfg.sparsity_coefficient
                ));
            latent_dim_ = cfg.n_d;
            break;
        }

        case EncoderArchitecture::SAINT: {
            const auto& cfg = config.saint;
            int64_t d_ff = 4 * cfg.d_model;
            saint_ = register_module("saint",
                SAINTEncoder(
                    n_numerical_,
                    cat_cardinalities,
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_layers,
                    d_ff,
                    cfg.attention_dropout,
                    cfg.use_row_attention,
                    /*use_cls_token=*/true
                ));
            setup_output_proj(cfg.d_model);
            break;
        }

        case EncoderArchitecture::GNN: {
            const auto& cfg = config.gnn;
            k_neighbors_ = cfg.k_neighbors;
            GNNEncoderImpl::GNNType gnn_type;
            switch (cfg.gnn_type) {
                case GNNType::GCN: gnn_type = GNNEncoderImpl::GNNType::GCN; break;
                case GNNType::GAT: gnn_type = GNNEncoderImpl::GNNType::GAT; break;
                case GNNType::GraphSAGE: gnn_type = GNNEncoderImpl::GNNType::GraphSAGE; break;
            }
            int64_t gnn_input = n_numerical_ + n_categoricals_;  // Flatten all features
            gnn_ = register_module("gnn",
                GNNEncoder(
                    gnn_input,
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    cfg.n_layers,
                    gnn_type,
                    cfg.n_heads,
                    cfg.edge_dropout
                ));
            latent_dim_ = cfg.hidden_dim;
            break;
        }

        case EncoderArchitecture::ExcelFormer: {
            const auto& cfg = config.excelformer;
            int64_t d_ff = cfg.ffn_multiplier * cfg.d_model;
            excelformer_ = register_module("excelformer",
                ExcelFormerEncoder(
                    n_numerical_,
                    cat_cardinalities,
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.n_layers,
                    d_ff,
                    cfg.attention_dropout,
                    cfg.importance_threshold,
                    /*use_cls_token=*/true
                ));
            setup_output_proj(cfg.d_model);
            break;
        }

        case EncoderArchitecture::HeterogeneousGNN: {
            const auto& cfg = config.heterogeneous_gnn;
            // HeterogeneousGNN learns species embeddings from a graph,
            // then aggregates them using species_vector to produce plot features.
            // Requires sparse species encoding mode.
            hetero_gnn_ = register_module("hetero_gnn",
                HeterogeneousGNNEncoder(
                    schema.n_species_vocab,
                    cfg.hidden_dim,
                    cfg.output_dim,
                    cfg.n_layers,
                    cfg.n_edge_types,
                    cfg.n_heads,
                    cfg.dropout
                ));
            // Output is the environmental (continuous) block concatenated with
            // the per-plot species-graph embedding, so covariates influence the
            // plot representation. Previously only the graph output was returned,
            // silently dropping every continuous/environmental feature.
            latent_dim_ = n_continuous_base + cfg.output_dim;
            break;
        }

        case EncoderArchitecture::TraitNet:
            throw std::runtime_error(
                "TraitNet requires a trait-environment architecture that is not compatible "
                "with TabularAdapter. TraitNet support is planned for a future release.");
        case EncoderArchitecture::MLP:
        default:
            throw std::runtime_error(
                "TabularAdapter does not handle EncoderArchitecture::MLP. "
                "Use the standard PlotEncoder for MLP mode.");
    }
}

torch::Tensor TabularAdapterImpl::prepare_numerical(
    torch::Tensor continuous,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    std::vector<torch::Tensor> parts;
    parts.push_back(continuous);

    if (species_encoding_ == SpeciesEncodingMode::Embed && fused_species_) {
        // Embed mode: look up species embeddings
        auto species_emb = fused_species_->forward(species_ids);
        parts.push_back(species_emb);
    }
    else if (species_encoding_ == SpeciesEncodingMode::Sparse && species_projection_) {
        // Sparse mode: project species vector
        auto species_emb = species_projection_->forward(species_vector);
        parts.push_back(species_emb);
    }
    // Hash mode: hash embedding is already in continuous

    return torch::cat(parts, /*dim=*/1);
}

std::vector<torch::Tensor> TabularAdapterImpl::prepare_categoricals(
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    std::vector<torch::Tensor> cats;
    if (genus_ids.defined() && family_ids.defined()) {
        // Each taxonomy slot becomes a separate categorical
        int n_slots = genus_ids.size(1);
        for (int k = 0; k < n_slots; ++k) {
            cats.push_back(genus_ids.select(1, k));
        }
        for (int k = 0; k < n_slots; ++k) {
            cats.push_back(family_ids.select(1, k));
        }
    }
    return cats;
}

torch::Tensor TabularAdapterImpl::forward(
    torch::Tensor continuous,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor species_ids,
    torch::Tensor species_vector
) {
    // HeterogeneousGNN ignores the numerical block (it uses continuous + the
    // species-graph embedding), so skip the projection GEMM for it.
    torch::Tensor numerical;
    if (architecture_ != EncoderArchitecture::HeterogeneousGNN) {
        numerical = prepare_numerical(continuous, species_ids, species_vector);
    }
    auto categoricals = prepare_categoricals(genus_ids, family_ids);

    torch::Tensor output;

    switch (architecture_) {
        case EncoderArchitecture::FTTransformer:
            output = ft_transformer_->forward(numerical, categoricals);
            break;

        case EncoderArchitecture::TabNet: {
            // TabNet needs flat input; embed categoricals and concat
            // For now, pass numerical directly (categoricals are optional for TabNet)
            auto [tabnet_out, feature_importance] = tabnet_->forward(numerical);
            output = tabnet_out;
            break;
        }

        case EncoderArchitecture::SAINT:
            output = saint_->forward(numerical, categoricals);
            break;

        case EncoderArchitecture::ExcelFormer:
            output = excelformer_->forward(numerical, categoricals);
            break;

        case EncoderArchitecture::GNN: {
            // The first two continuous columns are the plot coordinates ONLY
            // when the dataset carries them (continuous is laid out as
            // [coordinates | covariates | unknown_* | categorical_embed | ...]).
            // Without coordinates those columns are covariates, and building a
            // "spatial" kNN graph from them is meaningless -- refuse rather than
            // silently corrupt the graph (issue #73).
            TORCH_CHECK(has_coordinates_,
                "GNN encoder requires coordinates to build its spatial graph, "
                "but the dataset has none. Provide longitude/latitude roles or "
                "use a non-GNN encoder architecture.");
            auto coords = continuous.slice(/*dim=*/1, 0, 2);
            // NOTE: the kNN graph is built over the plots in the CURRENT forward
            // batch, so during mini-batch training each plot attends only to its
            // batch-mates, not its true spatial neighbors. Inference forces a
            // single full-batch forward for this architecture (see
            // Predictor::predict) so predictions are batch-composition
            // independent, but the training-time locality is a known limitation
            // of this experimental encoder; a batch-composition-independent
            // spatial GNN needs neighbor-sampling over a fixed global graph
            // (issue #73, tracked separately).
            auto adj = build_knn_adjacency(coords, k_neighbors_);
            // Flatten features for GNN
            std::vector<torch::Tensor> all_feats;
            all_feats.push_back(numerical);
            for (auto& cat : categoricals) {
                all_feats.push_back(cat.unsqueeze(1).to(torch::kFloat32));
            }
            auto flat = torch::cat(all_feats, /*dim=*/1);
            auto gnn_out = gnn_->forward(flat, adj);
            // GNN returns per-node (per-sample) features already
            output = gnn_out;
            break;
        }

        case EncoderArchitecture::HeterogeneousGNN: {
            TORCH_CHECK(hetero_graph_set_,
                "Species graph not set. Call set_species_graph() before forward().");
            TORCH_CHECK(species_vector.defined(),
                "HeterogeneousGNN requires species_vector (sparse encoding mode).");

            // Move graph to same device as input
            auto ei = hetero_edge_index_.to(continuous.device());
            auto et = hetero_edge_type_.to(continuous.device());

            // Run GNN on species graph to get species embeddings
            auto species_emb = hetero_gnn_->forward(ei, et);  // (n_species, output_dim)

            // Aggregate per plot using species abundance vector, then prepend
            // the environmental (continuous) features so they are not dropped.
            auto gnn_plot = HeterogeneousGNNEncoderImpl::aggregate_for_plots(
                species_emb, species_vector);  // (batch, output_dim)
            output = torch::cat({continuous, gnn_plot}, /*dim=*/1);  // (batch, n_cont + output_dim)
            break;
        }

        default:
            throw std::runtime_error("Unsupported architecture in TabularAdapter::forward");
    }

    // Project to desired latent dimension if needed
    if (needs_output_proj_) {
        output = output_proj_->forward(output);
    }

    return output;
}

void TabularAdapterImpl::set_species_graph(
    torch::Tensor edge_index,
    torch::Tensor edge_type
) {
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
        "edge_index must be (2, n_edges), got ", edge_index.sizes());
    TORCH_CHECK(edge_type.dim() == 1 && edge_type.size(0) == edge_index.size(1),
        "edge_type must be (n_edges,) matching edge_index, got ", edge_type.sizes());

    hetero_edge_index_ = edge_index;
    hetero_edge_type_ = edge_type;
    hetero_graph_set_ = true;
}

} // namespace resolve
