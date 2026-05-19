#include "bindings_common.hpp"

void register_types(nb::module_& m) {
    // Role Mapping and Dataset Configuration
    nb::class_<resolve::RoleMapping>(m, "RoleMapping")
        .def(nb::init<>())
        .def_rw("plot_id", &resolve::RoleMapping::plot_id)
        .def_rw("species_id", &resolve::RoleMapping::species_id)
        .def_rw("abundance", &resolve::RoleMapping::abundance)
        .def_rw("longitude", &resolve::RoleMapping::longitude)
        .def_rw("latitude", &resolve::RoleMapping::latitude)
        .def_rw("genus", &resolve::RoleMapping::genus)
        .def_rw("family", &resolve::RoleMapping::family)
        .def_rw("covariates", &resolve::RoleMapping::covariates)
        .def_rw("categoricals", &resolve::RoleMapping::categoricals)
        .def_rw("targets", &resolve::RoleMapping::targets)
        .def("has_coordinates", &resolve::RoleMapping::has_coordinates)
        .def("has_taxonomy", &resolve::RoleMapping::has_taxonomy)
        .def("has_abundance", &resolve::RoleMapping::has_abundance)
        .def("has_categoricals", &resolve::RoleMapping::has_categoricals);

    nb::class_<resolve::TargetSpec>(m, "TargetSpec")
        .def(nb::init<>())
        .def_rw("column_name", &resolve::TargetSpec::column_name)
        .def_rw("target_name", &resolve::TargetSpec::target_name)
        .def_rw("task", &resolve::TargetSpec::task)
        .def_rw("transform", &resolve::TargetSpec::transform)
        .def_rw("num_classes", &resolve::TargetSpec::num_classes)
        .def_rw("weight", &resolve::TargetSpec::weight)
        // Optional explicit string -> int mapping for classification target
        // columns. When empty (default), the loader auto-fits the mapping
        // from the data. Mirrors the POC's `cfg["mapping"]`.
        .def_rw("class_mapping", &resolve::TargetSpec::class_mapping)
        .def_static("regression", &resolve::TargetSpec::regression,
                    nb::arg("column"), nb::arg("transform") = resolve::TransformType::None)
        .def_static("classification", &resolve::TargetSpec::classification,
                    nb::arg("column"), nb::arg("num_classes"))
        .def_static("classification_with_mapping",
                    &resolve::TargetSpec::classification_with_mapping,
                    nb::arg("column"), nb::arg("mapping"));

    nb::class_<resolve::DatasetConfig>(m, "DatasetConfig")
        .def(nb::init<>())
        .def_rw("species_encoding", &resolve::DatasetConfig::species_encoding)
        .def_rw("hash_dim", &resolve::DatasetConfig::hash_dim)
        .def_rw("top_k", &resolve::DatasetConfig::top_k)
        .def_rw("top_k_species", &resolve::DatasetConfig::top_k_species)
        .def_rw("selection", &resolve::DatasetConfig::selection)
        .def_rw("representation", &resolve::DatasetConfig::representation)
        .def_rw("normalization", &resolve::DatasetConfig::normalization)
        .def_rw("aggregation", &resolve::DatasetConfig::aggregation)
        .def_rw("track_unknown_fraction", &resolve::DatasetConfig::track_unknown_fraction)
        .def_rw("track_unknown_count", &resolve::DatasetConfig::track_unknown_count)
        .def_rw("use_taxonomy", &resolve::DatasetConfig::use_taxonomy)
        .def_rw("use_cuda_hash", &resolve::DatasetConfig::use_cuda_hash)
        // Per-species weight scheme for rank_pool / transformer encoders
        // (binary, abundance, log1p, norm, rank). Ignored otherwise.
        .def_rw("pool_weighting", &resolve::DatasetConfig::pool_weighting)
        // Cap on species-per-plot for rank_pool / transformer encoders. See
        // DatasetConfig::pool_species_cap doc for the sentinel meanings:
        // 0 = no cap (default), -1 = auto p99, >0 = manual cap.
        .def_rw("pool_species_cap", &resolve::DatasetConfig::pool_species_cap);

    // Configuration structs
    nb::class_<resolve::TargetConfig>(m, "TargetConfig")
        .def(nb::init<>())
        .def_rw("name", &resolve::TargetConfig::name)
        .def_rw("task", &resolve::TargetConfig::task)
        .def_rw("transform", &resolve::TargetConfig::transform)
        .def_rw("num_classes", &resolve::TargetConfig::num_classes)
        .def_rw("weight", &resolve::TargetConfig::weight)
        .def_rw("class_weights", &resolve::TargetConfig::class_weights)
        // Ordered class vocabulary (index == code). Populated by
        // ResolveDataset.from_csv when the classification target column
        // arrived as strings and the loader factorized it. Empty for
        // regression or for already-integer-encoded classification columns.
        .def_rw("class_names", &resolve::TargetConfig::class_names);

    nb::class_<resolve::ResolveSchema>(m, "ResolveSchema")
        .def(nb::init<>())
        .def_rw("n_plots", &resolve::ResolveSchema::n_plots)
        .def_rw("n_species", &resolve::ResolveSchema::n_species)
        .def_rw("n_species_vocab", &resolve::ResolveSchema::n_species_vocab)
        .def_rw("has_coordinates", &resolve::ResolveSchema::has_coordinates)
        .def_rw("has_abundance", &resolve::ResolveSchema::has_abundance)
        .def_rw("has_taxonomy", &resolve::ResolveSchema::has_taxonomy)
        .def_rw("n_genera", &resolve::ResolveSchema::n_genera)
        .def_rw("n_families", &resolve::ResolveSchema::n_families)
        .def_rw("n_genera_vocab", &resolve::ResolveSchema::n_genera_vocab)
        .def_rw("n_families_vocab", &resolve::ResolveSchema::n_families_vocab)
        .def_rw("covariate_names", &resolve::ResolveSchema::covariate_names)
        .def_rw("targets", &resolve::ResolveSchema::targets)
        .def_rw("track_unknown_fraction", &resolve::ResolveSchema::track_unknown_fraction)
        .def_rw("track_unknown_count", &resolve::ResolveSchema::track_unknown_count)
        .def_rw("categorical_names", &resolve::ResolveSchema::categorical_names)
        .def_rw("categorical_vocab_sizes", &resolve::ResolveSchema::categorical_vocab_sizes)
        .def_rw("categorical_embed_dim", &resolve::ResolveSchema::categorical_embed_dim)
        .def("has_categoricals", &resolve::ResolveSchema::has_categoricals)
        .def("n_categoricals", &resolve::ResolveSchema::n_categoricals);

    // Alias for backwards compatibility
    m.attr("SpaccSchema") = m.attr("ResolveSchema");

    // Architecture-specific config structs
    nb::class_<resolve::FTTransformerConfig>(m, "FTTransformerConfig")
        .def(nb::init<>())
        .def_rw("d_model", &resolve::FTTransformerConfig::d_model)
        .def_rw("n_heads", &resolve::FTTransformerConfig::n_heads)
        .def_rw("n_layers", &resolve::FTTransformerConfig::n_layers)
        .def_rw("attention_dropout", &resolve::FTTransformerConfig::attention_dropout)
        .def_rw("ffn_dropout", &resolve::FTTransformerConfig::ffn_dropout)
        .def_rw("ffn_multiplier", &resolve::FTTransformerConfig::ffn_multiplier)
        .def_rw("pre_norm", &resolve::FTTransformerConfig::pre_norm);

    nb::class_<resolve::TabNetConfig>(m, "TabNetConfig")
        .def(nb::init<>())
        .def_rw("n_steps", &resolve::TabNetConfig::n_steps)
        .def_rw("n_d", &resolve::TabNetConfig::n_d)
        .def_rw("n_a", &resolve::TabNetConfig::n_a)
        .def_rw("relaxation_factor", &resolve::TabNetConfig::relaxation_factor)
        .def_rw("sparsity_coefficient", &resolve::TabNetConfig::sparsity_coefficient)
        .def_rw("virtual_batch_size", &resolve::TabNetConfig::virtual_batch_size)
        .def_rw("use_sparsemax", &resolve::TabNetConfig::use_sparsemax);

    nb::class_<resolve::SAINTConfig>(m, "SAINTConfig")
        .def(nb::init<>())
        .def_rw("d_model", &resolve::SAINTConfig::d_model)
        .def_rw("n_heads", &resolve::SAINTConfig::n_heads)
        .def_rw("n_layers", &resolve::SAINTConfig::n_layers)
        .def_rw("attention_dropout", &resolve::SAINTConfig::attention_dropout)
        .def_rw("use_row_attention", &resolve::SAINTConfig::use_row_attention)
        .def_rw("use_contrastive_pretrain", &resolve::SAINTConfig::use_contrastive_pretrain)
        .def_rw("mixup_alpha", &resolve::SAINTConfig::mixup_alpha);

    nb::class_<resolve::GNNConfig>(m, "GNNConfig")
        .def(nb::init<>())
        .def_rw("gnn_type", &resolve::GNNConfig::gnn_type)
        .def_rw("n_layers", &resolve::GNNConfig::n_layers)
        .def_rw("hidden_dim", &resolve::GNNConfig::hidden_dim)
        .def_rw("n_heads", &resolve::GNNConfig::n_heads)
        .def_rw("k_neighbors", &resolve::GNNConfig::k_neighbors)
        .def_rw("graph_mode", &resolve::GNNConfig::graph_mode)
        .def_rw("edge_dropout", &resolve::GNNConfig::edge_dropout)
        .def_rw("use_edge_features", &resolve::GNNConfig::use_edge_features);

    nb::class_<resolve::TraitNetConfig>(m, "TraitNetConfig")
        .def(nb::init<>())
        .def_rw("env_dim", &resolve::TraitNetConfig::env_dim)
        .def_rw("trait_dim", &resolve::TraitNetConfig::trait_dim)
        .def_rw("interaction_dim", &resolve::TraitNetConfig::interaction_dim)
        .def_rw("interaction", &resolve::TraitNetConfig::interaction)
        .def_rw("shared_trait_encoder", &resolve::TraitNetConfig::shared_trait_encoder);

    // ExcelFormer configuration
    nb::class_<resolve::ExcelFormerConfig>(m, "ExcelFormerConfig")
        .def(nb::init<>())
        .def_rw("d_model", &resolve::ExcelFormerConfig::d_model)
        .def_rw("n_heads", &resolve::ExcelFormerConfig::n_heads)
        .def_rw("n_layers", &resolve::ExcelFormerConfig::n_layers)
        .def_rw("attention_dropout", &resolve::ExcelFormerConfig::attention_dropout)
        .def_rw("ffn_multiplier", &resolve::ExcelFormerConfig::ffn_multiplier)
        .def_rw("importance_threshold", &resolve::ExcelFormerConfig::importance_threshold)
        .def_rw("pre_norm", &resolve::ExcelFormerConfig::pre_norm);

    // Heterogeneous GNN configuration
    nb::class_<resolve::HeterogeneousGNNConfig>(m, "HeterogeneousGNNConfig")
        .def(nb::init<>())
        .def_rw("hidden_dim", &resolve::HeterogeneousGNNConfig::hidden_dim)
        .def_rw("output_dim", &resolve::HeterogeneousGNNConfig::output_dim)
        .def_rw("n_layers", &resolve::HeterogeneousGNNConfig::n_layers)
        .def_rw("n_edge_types", &resolve::HeterogeneousGNNConfig::n_edge_types)
        .def_rw("n_heads", &resolve::HeterogeneousGNNConfig::n_heads)
        .def_rw("dropout", &resolve::HeterogeneousGNNConfig::dropout)
        .def_rw("k_cooccurrence", &resolve::HeterogeneousGNNConfig::k_cooccurrence)
        .def_rw("cooccurrence_threshold", &resolve::HeterogeneousGNNConfig::cooccurrence_threshold)
        .def_rw("use_taxonomic_edges", &resolve::HeterogeneousGNNConfig::use_taxonomic_edges)
        .def_rw("use_cooccurrence_edges", &resolve::HeterogeneousGNNConfig::use_cooccurrence_edges);

    // TabM configuration
    nb::class_<resolve::TabMConfig>(m, "TabMConfig")
        .def(nb::init<>())
        .def_rw("enabled", &resolve::TabMConfig::enabled)
        .def_rw("n_ensembles", &resolve::TabMConfig::n_ensembles)
        .def_rw("aggregation", &resolve::TabMConfig::aggregation);

    // Parallel layers configuration
    nb::class_<resolve::ParallelBranchConfig>(m, "ParallelBranchConfig")
        .def(nb::init<>())
        .def_rw("hidden_dims", &resolve::ParallelBranchConfig::hidden_dims)
        .def_rw("activation", &resolve::ParallelBranchConfig::activation)
        .def_rw("normalization", &resolve::ParallelBranchConfig::normalization)
        .def_rw("dropout", &resolve::ParallelBranchConfig::dropout)
        .def_rw("branch_weight", &resolve::ParallelBranchConfig::branch_weight);

    nb::class_<resolve::ParallelLayersConfig>(m, "ParallelLayersConfig")
        .def(nb::init<>())
        .def_rw("enabled", &resolve::ParallelLayersConfig::enabled)
        .def_rw("branches", &resolve::ParallelLayersConfig::branches)
        .def_rw("aggregation", &resolve::ParallelLayersConfig::aggregation)
        .def_rw("attention_heads", &resolve::ParallelLayersConfig::attention_heads)
        .def_rw("use_residual", &resolve::ParallelLayersConfig::use_residual);

    nb::class_<resolve::ModelConfig>(m, "ModelConfig")
        .def(nb::init<>())
        .def_rw("species_encoding", &resolve::ModelConfig::species_encoding)
        .def_rw("uses_explicit_vector", &resolve::ModelConfig::uses_explicit_vector)
        .def_rw("hash_dim", &resolve::ModelConfig::hash_dim)
        .def_rw("species_embed_dim", &resolve::ModelConfig::species_embed_dim)
        .def_rw("genus_emb_dim", &resolve::ModelConfig::genus_emb_dim)
        .def_rw("family_emb_dim", &resolve::ModelConfig::family_emb_dim)
        .def_rw("categorical_embed_dim", &resolve::ModelConfig::categorical_embed_dim)
        .def_rw("top_k", &resolve::ModelConfig::top_k)
        .def_rw("top_k_species", &resolve::ModelConfig::top_k_species)
        .def_rw("n_taxonomy_slots", &resolve::ModelConfig::n_taxonomy_slots)
        .def_rw("hidden_dims", &resolve::ModelConfig::hidden_dims)
        .def_rw("dropout", &resolve::ModelConfig::dropout)
        // MoE configuration
        .def_rw("moe_routing", &resolve::ModelConfig::moe_routing)
        .def_rw("n_experts", &resolve::ModelConfig::n_experts)
        .def_rw("expert_hidden_dims", &resolve::ModelConfig::expert_hidden_dims)
        .def_rw("moe_top_k", &resolve::ModelConfig::moe_top_k)
        .def_rw("moe_noise_std", &resolve::ModelConfig::moe_noise_std)
        .def_rw("moe_aux_loss_weight", &resolve::ModelConfig::moe_aux_loss_weight)
        // Configurable architecture
        .def_rw("activation", &resolve::ModelConfig::activation)
        .def_rw("normalization", &resolve::ModelConfig::normalization)
        .def_rw("norm_groups", &resolve::ModelConfig::norm_groups)
        .def_rw("use_residual", &resolve::ModelConfig::use_residual)
        .def_rw("leaky_relu_slope", &resolve::ModelConfig::leaky_relu_slope)
        .def_rw("elu_alpha", &resolve::ModelConfig::elu_alpha)
        // Multi-layer heads
        .def_rw("head_hidden_dims", &resolve::ModelConfig::head_hidden_dims)
        .def_rw("head_activation", &resolve::ModelConfig::head_activation)
        .def_rw("head_dropout", &resolve::ModelConfig::head_dropout)
        // Advanced architecture (v2.0)
        .def_rw("encoder_architecture", &resolve::ModelConfig::encoder_architecture)
        .def_rw("ft_transformer", &resolve::ModelConfig::ft_transformer)
        .def_rw("tabnet", &resolve::ModelConfig::tabnet)
        .def_rw("saint", &resolve::ModelConfig::saint)
        .def_rw("gnn", &resolve::ModelConfig::gnn)
        .def_rw("trait_net", &resolve::ModelConfig::trait_net)
        .def_rw("excelformer", &resolve::ModelConfig::excelformer)
        .def_rw("heterogeneous_gnn", &resolve::ModelConfig::heterogeneous_gnn)
        // Parallel layers
        .def_rw("parallel_layers", &resolve::ModelConfig::parallel_layers)
        // TabM configuration
        .def_rw("tabm", &resolve::ModelConfig::tabm)
        // RankPool / Transformer encoder fields
        .def_rw("cover_dropout", &resolve::ModelConfig::cover_dropout)
        .def_rw("d_model", &resolve::ModelConfig::d_model)
        .def_rw("n_heads", &resolve::ModelConfig::n_heads)
        .def_rw("n_attention_layers", &resolve::ModelConfig::n_attention_layers)
        .def_rw("transformer_ff_dim", &resolve::ModelConfig::transformer_ff_dim)
        .def_rw("transformer_pooling", &resolve::ModelConfig::transformer_pooling)
        .def_rw("transformer_dropout", &resolve::ModelConfig::transformer_dropout);

    nb::class_<resolve::TrainConfig>(m, "TrainConfig")
        .def(nb::init<>())
        .def_rw("batch_size", &resolve::TrainConfig::batch_size)
        .def_rw("max_epochs", &resolve::TrainConfig::max_epochs)
        .def_rw("patience", &resolve::TrainConfig::patience)
        .def_rw("lr", &resolve::TrainConfig::lr)
        .def_rw("weight_decay", &resolve::TrainConfig::weight_decay)
        .def_rw("phase_boundaries", &resolve::TrainConfig::phase_boundaries)
        .def_rw("loss_config", &resolve::TrainConfig::loss_config)
        .def_rw("lr_scheduler", &resolve::TrainConfig::lr_scheduler)
        .def_rw("lr_step_size", &resolve::TrainConfig::lr_step_size)
        .def_rw("lr_gamma", &resolve::TrainConfig::lr_gamma)
        .def_rw("lr_min", &resolve::TrainConfig::lr_min)
        // Automatic Mixed Precision (AMP)
        .def_rw("use_amp", &resolve::TrainConfig::use_amp)
        .def_rw("amp_init_scale", &resolve::TrainConfig::amp_init_scale)
        .def_rw("amp_growth_factor", &resolve::TrainConfig::amp_growth_factor)
        .def_rw("amp_backoff_factor", &resolve::TrainConfig::amp_backoff_factor)
        .def_rw("amp_growth_interval", &resolve::TrainConfig::amp_growth_interval)
        // CUDA performance optimizations
        .def_rw("cudnn_benchmark", &resolve::TrainConfig::cudnn_benchmark)
        .def_rw("allow_tf32", &resolve::TrainConfig::allow_tf32)
        .def_rw("vram_fraction", &resolve::TrainConfig::vram_fraction)
        // Device property (string-based for Python convenience)
        .def_prop_rw("device",
            [](const resolve::TrainConfig& c) {
                return c.device.is_cuda() ? "cuda" : "cpu";
            },
            [](resolve::TrainConfig& c, const std::string& dev) {
                c.device = (dev == "cuda") ? torch::kCUDA : torch::kCPU;
            });

    // Baseline metrics for comparison against naive baselines
    nb::class_<resolve::BaselineMetrics>(m, "BaselineMetrics")
        .def(nb::init<>())
        .def_ro("baseline_mse", &resolve::BaselineMetrics::baseline_mse)
        .def_ro("baseline_mae", &resolve::BaselineMetrics::baseline_mae)
        .def_ro("model_mse", &resolve::BaselineMetrics::model_mse)
        .def_ro("model_mae", &resolve::BaselineMetrics::model_mae)
        .def_ro("skill_score", &resolve::BaselineMetrics::skill_score)
        .def_ro("r_squared", &resolve::BaselineMetrics::r_squared)
        .def_ro("baseline_accuracy", &resolve::BaselineMetrics::baseline_accuracy)
        .def_ro("model_accuracy", &resolve::BaselineMetrics::model_accuracy)
        .def_ro("accuracy_lift", &resolve::BaselineMetrics::accuracy_lift)
        .def_ro("training_mean", &resolve::BaselineMetrics::training_mean)
        .def_ro("training_mode", &resolve::BaselineMetrics::training_mode);

    // Layer diagnostics for detecting dead/saturated neurons
    nb::class_<resolve::LayerDiagnostics>(m, "LayerDiagnostics")
        .def(nb::init<>())
        .def_ro("name", &resolve::LayerDiagnostics::name)
        .def_ro("n_neurons", &resolve::LayerDiagnostics::n_neurons)
        .def_ro("n_dead", &resolve::LayerDiagnostics::n_dead)
        .def_ro("n_saturated", &resolve::LayerDiagnostics::n_saturated)
        .def_ro("dead_fraction", &resolve::LayerDiagnostics::dead_fraction)
        .def_ro("saturated_fraction", &resolve::LayerDiagnostics::saturated_fraction)
        .def_ro("mean_activation", &resolve::LayerDiagnostics::mean_activation)
        .def_ro("std_activation", &resolve::LayerDiagnostics::std_activation)
        .def_ro("sparsity", &resolve::LayerDiagnostics::sparsity);

    // Network-wide diagnostics
    nb::class_<resolve::NetworkDiagnostics>(m, "NetworkDiagnostics")
        .def(nb::init<>())
        .def_ro("layers", &resolve::NetworkDiagnostics::layers)
        .def_ro("total_neurons", &resolve::NetworkDiagnostics::total_neurons)
        .def_ro("total_dead", &resolve::NetworkDiagnostics::total_dead)
        .def_ro("total_saturated", &resolve::NetworkDiagnostics::total_saturated)
        .def_ro("overall_dead_fraction", &resolve::NetworkDiagnostics::overall_dead_fraction)
        .def_ro("overall_saturated_fraction", &resolve::NetworkDiagnostics::overall_saturated_fraction)
        .def_ro("has_issues", &resolve::NetworkDiagnostics::has_issues)
        .def_ro("summary", &resolve::NetworkDiagnostics::summary);

    // Result structs
    nb::class_<resolve::TrainResult>(m, "TrainResult")
        .def(nb::init<>())
        .def_ro("best_epoch", &resolve::TrainResult::best_epoch)
        .def_ro("final_metrics", &resolve::TrainResult::final_metrics)
        .def_ro("train_loss_history", &resolve::TrainResult::train_loss_history)
        .def_ro("test_loss_history", &resolve::TrainResult::test_loss_history)
        .def_ro("train_time_seconds", &resolve::TrainResult::train_time_seconds)
        .def_ro("resumed_from_epoch", &resolve::TrainResult::resumed_from_epoch)
        .def_ro("baselines", &resolve::TrainResult::baselines)
        .def_ro("diagnostics", &resolve::TrainResult::diagnostics);

    nb::class_<resolve::ResolvePredictions>(m, "ResolvePredictions")
        .def(nb::init<>())
        .def_prop_ro("predictions", [](const resolve::ResolvePredictions& p) {
            return tensor_map_to_dict(p.predictions);
        })
        .def_prop_ro("targets", [](const resolve::ResolvePredictions& p) {
            return tensor_map_to_dict(p.targets);
        })
        .def_ro("plot_ids", &resolve::ResolvePredictions::plot_ids)
        .def_prop_ro("latent", [](const resolve::ResolvePredictions& p) {
            if (p.latent.defined()) {
                auto cpu_tensor = p.latent.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        });

    m.attr("SpaccPredictions") = m.attr("ResolvePredictions");

    nb::class_<resolve::Scalers>(m, "Scalers")
        .def(nb::init<>())
        .def_prop_ro("continuous_mean", [](const resolve::Scalers& s) {
            if (s.continuous_mean.defined()) {
                auto cpu_tensor = s.continuous_mean.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        })
        .def_prop_ro("continuous_scale", [](const resolve::Scalers& s) {
            if (s.continuous_scale.defined()) {
                auto cpu_tensor = s.continuous_scale.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        });

    // Calibration types for classification evaluation
    nb::class_<resolve::CalibrationBin>(m, "CalibrationBin")
        .def(nb::init<>())
        .def_ro("bin_start", &resolve::CalibrationBin::bin_start)
        .def_ro("bin_end", &resolve::CalibrationBin::bin_end)
        .def_ro("mean_predicted_prob", &resolve::CalibrationBin::mean_predicted_prob)
        .def_ro("actual_frequency", &resolve::CalibrationBin::actual_frequency)
        .def_ro("count", &resolve::CalibrationBin::count);

    nb::class_<resolve::CalibrationResult>(m, "CalibrationResult")
        .def(nb::init<>())
        .def_ro("target_name", &resolve::CalibrationResult::target_name)
        .def_ro("class_idx", &resolve::CalibrationResult::class_idx)
        .def_ro("bins", &resolve::CalibrationResult::bins)
        .def_ro("expected_calibration_error", &resolve::CalibrationResult::expected_calibration_error)
        .def_ro("max_calibration_error", &resolve::CalibrationResult::max_calibration_error);

    // Residual analysis for regression evaluation
    nb::class_<resolve::ResidualAnalysis>(m, "ResidualAnalysis")
        .def(nb::init<>())
        .def_ro("target_name", &resolve::ResidualAnalysis::target_name)
        .def_ro("predictions", &resolve::ResidualAnalysis::predictions)
        .def_ro("actuals", &resolve::ResidualAnalysis::actuals)
        .def_ro("residuals", &resolve::ResidualAnalysis::residuals)
        .def_ro("mean_residual", &resolve::ResidualAnalysis::mean_residual)
        .def_ro("std_residual", &resolve::ResidualAnalysis::std_residual)
        .def_ro("skewness", &resolve::ResidualAnalysis::skewness)
        .def_ro("kurtosis", &resolve::ResidualAnalysis::kurtosis)
        .def_ro("q05", &resolve::ResidualAnalysis::q05)
        .def_ro("q25", &resolve::ResidualAnalysis::q25)
        .def_ro("q50", &resolve::ResidualAnalysis::q50)
        .def_ro("q75", &resolve::ResidualAnalysis::q75)
        .def_ro("q95", &resolve::ResidualAnalysis::q95);

    // Spatial block configuration
    nb::class_<resolve::SpatialBlockConfig>(m, "SpatialBlockConfig")
        .def(nb::init<>())
        .def_rw("lat_size", &resolve::SpatialBlockConfig::lat_size)
        .def_rw("lon_size", &resolve::SpatialBlockConfig::lon_size)
        .def_rw("balance", &resolve::SpatialBlockConfig::balance);

    // Cross-validation result
    nb::class_<resolve::CrossValidationResult>(m, "CrossValidationResult")
        .def(nb::init<>())
        .def_ro("n_folds", &resolve::CrossValidationResult::n_folds)
        .def_ro("mean_metrics", &resolve::CrossValidationResult::mean_metrics)
        .def_ro("std_metrics", &resolve::CrossValidationResult::std_metrics)
        .def_ro("fold_results", &resolve::CrossValidationResult::fold_results)
        .def_ro("total_time_seconds", &resolve::CrossValidationResult::total_time_seconds);

    nb::class_<resolve::RunMetadata>(m, "RunMetadata")
        .def(nb::init<>())
        .def_rw("resolve_version", &resolve::RunMetadata::resolve_version)
        .def_rw("created_at", &resolve::RunMetadata::created_at)
        .def_rw("completed_at", &resolve::RunMetadata::completed_at)
        .def_rw("train_time_seconds", &resolve::RunMetadata::train_time_seconds)
        .def_rw("n_plots_train", &resolve::RunMetadata::n_plots_train)
        .def_rw("n_plots_test", &resolve::RunMetadata::n_plots_test)
        .def_rw("best_epoch", &resolve::RunMetadata::best_epoch)
        .def_rw("total_epochs", &resolve::RunMetadata::total_epochs)
        .def_rw("final_metrics", &resolve::RunMetadata::final_metrics);

    nb::class_<resolve::ModelForwardResult>(m, "ModelForwardResult")
        .def(nb::init<>())
        .def_prop_ro("outputs", [](const resolve::ModelForwardResult& r) {
            return tensor_map_to_dict(r.outputs);
        })
        .def_prop_ro("moe_aux_loss", [](const resolve::ModelForwardResult& r) {
            if (r.moe_aux_loss.defined()) {
                auto cpu_tensor = r.moe_aux_loss.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        });
}
