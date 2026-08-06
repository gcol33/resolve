#include "bindings_common.hpp"

#include "resolve/config_registry.hpp"

#include <type_traits>

namespace {

// Registers one field-registry row as a read/write Python attribute. The
// training device is reachable through a string property (torch::Device has no
// natural Python spelling) and the log callback has no Python form at all, so
// both are skipped by TYPE rather than by being quietly left off a list -- the
// omission a reader would have to notice.
template <typename Class, typename Cfg, typename T>
void bind_config_field(Class& cls, const char* name, T Cfg::*member) {
    if constexpr (resolve::is_python_bindable_field_v<T>) {
        cls.def_rw(name, member);
    } else {
        (void)cls;
        (void)name;
        (void)member;
    }
}

}  // namespace

// Expands one registry row into a def_rw. `cls` and `Cfg` come from the
// enclosing block, so a struct is bound by opening a scope, naming it once, and
// handing its field list to this macro.
#define RESOLVE_BIND_FIELD(member, key) bind_config_field(cls, #member, &Cfg::member);

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
        // from the data.
        .def_rw("class_mapping", &resolve::TargetSpec::class_mapping)
        .def_static("regression", &resolve::TargetSpec::regression,
                    nb::arg("column"), nb::arg("transform") = resolve::TransformType::None)
        .def_static("classification", &resolve::TargetSpec::classification,
                    nb::arg("column"), nb::arg("num_classes"))
        .def_static("classification_with_mapping",
                    &resolve::TargetSpec::classification_with_mapping,
                    nb::arg("column"), nb::arg("mapping"));

    // Loader configuration. Attribute names come from the field registry, so
    // `pool_weighting` (the rank_pool / transformer per-species weight scheme)
    // and `pool_species_cap` (0 = no cap, -1 = auto p99, >0 = manual) reach
    // Python the moment they exist on the struct.
    {
        using Cfg = resolve::DatasetConfig;
        auto cls = nb::class_<Cfg>(m, "DatasetConfig");
        cls.def(nb::init<>());
        RESOLVE_DATASET_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }

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
        .def_rw("pool_weighting", &resolve::ResolveSchema::pool_weighting)
        .def_rw("pool_species_cap", &resolve::ResolveSchema::pool_species_cap)
        // Remaining DatasetConfig knobs the loader consumed (issue #102). #38
        // restored pool_weighting only; without these an inference-side
        // DatasetConfig silently reverted to the struct defaults.
        .def_rw("top_k_species", &resolve::ResolveSchema::top_k_species)
        .def_rw("selection", &resolve::ResolveSchema::selection)
        .def_rw("representation", &resolve::ResolveSchema::representation)
        .def_rw("normalization", &resolve::ResolveSchema::normalization)
        .def_rw("aggregation", &resolve::ResolveSchema::aggregation)
        .def_rw("use_taxonomy", &resolve::ResolveSchema::use_taxonomy)
        // Fitted species / genus / family vocabularies, index = integer code,
        // [0] = "<UNK>" (issue #102). Empty on a pre-fix checkpoint.
        .def_rw("species_vocab", &resolve::ResolveSchema::species_vocab)
        .def_rw("genus_vocab", &resolve::ResolveSchema::genus_vocab)
        .def_rw("family_vocab", &resolve::ResolveSchema::family_vocab)
        .def("has_categoricals", &resolve::ResolveSchema::has_categoricals)
        .def("n_categoricals", &resolve::ResolveSchema::n_categoricals)
        .def("has_species_vocab", &resolve::ResolveSchema::has_species_vocab)
        .def("has_taxonomy_vocab", &resolve::ResolveSchema::has_taxonomy_vocab);

    // ExternalVocabs (issue #102) is registered in register_dataset, after the
    // TaxonomyVocab / CategoricalVocab classes it holds.

    // Alias for backwards compatibility
    m.attr("SpaccSchema") = m.attr("ResolveSchema");

    // Architecture-specific config structs. Each is bound from its field
    // registry, so a hyperparameter added to the struct becomes a Python
    // attribute in the same edit that adds it -- no second list to keep in step.
    // The sub-configs are registered before ModelConfig, which exposes them as
    // attributes of its own.
    {
        using Cfg = resolve::FTTransformerConfig;
        auto cls = nb::class_<Cfg>(m, "FTTransformerConfig");
        cls.def(nb::init<>());
        RESOLVE_FT_TRANSFORMER_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::TabNetConfig;
        auto cls = nb::class_<Cfg>(m, "TabNetConfig");
        cls.def(nb::init<>());
        RESOLVE_TABNET_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::SAINTConfig;
        auto cls = nb::class_<Cfg>(m, "SAINTConfig");
        cls.def(nb::init<>());
        RESOLVE_SAINT_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::GNNConfig;
        auto cls = nb::class_<Cfg>(m, "GNNConfig");
        cls.def(nb::init<>());
        RESOLVE_GNN_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::TraitNetConfig;
        auto cls = nb::class_<Cfg>(m, "TraitNetConfig");
        cls.def(nb::init<>());
        RESOLVE_TRAIT_NET_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::ExcelFormerConfig;
        auto cls = nb::class_<Cfg>(m, "ExcelFormerConfig");
        cls.def(nb::init<>());
        RESOLVE_EXCELFORMER_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::HeterogeneousGNNConfig;
        auto cls = nb::class_<Cfg>(m, "HeterogeneousGNNConfig");
        cls.def(nb::init<>());
        RESOLVE_HETEROGENEOUS_GNN_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::TabMConfig;
        auto cls = nb::class_<Cfg>(m, "TabMConfig");
        cls.def(nb::init<>());
        RESOLVE_TABM_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::ParallelBranchConfig;
        auto cls = nb::class_<Cfg>(m, "ParallelBranchConfig");
        cls.def(nb::init<>());
        RESOLVE_PARALLEL_BRANCH_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }
    {
        using Cfg = resolve::ParallelLayersConfig;
        auto cls = nb::class_<Cfg>(m, "ParallelLayersConfig");
        cls.def(nb::init<>());
        RESOLVE_PARALLEL_LAYERS_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }

    {
        using Cfg = resolve::ModelConfig;
        auto cls = nb::class_<Cfg>(m, "ModelConfig");
        cls.def(nb::init<>());
        RESOLVE_MODEL_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
    }

    {
        using Cfg = resolve::TrainConfig;
        auto cls = nb::class_<Cfg>(m, "TrainConfig");
        cls.def(nb::init<>());
        RESOLVE_TRAIN_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
        // The device is the one field the registry cannot bind directly:
        // torch::Device has no natural Python spelling, so it is exposed as the
        // string "cuda" / "cpu". The log callback has no Python form at all.
        cls.def_prop_rw("device",
            [](const Cfg& c) {
                return c.device.is_cuda() ? "cuda" : "cpu";
            },
            [](Cfg& c, const std::string& dev) {
                c.device = (dev == "cuda") ? torch::kCUDA : torch::kCPU;
            });
    }

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
        .def_ro("effective_batch_size", &resolve::TrainResult::effective_batch_size)
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
            return nb::none();
        });

    m.attr("SpaccPredictions") = m.attr("ResolvePredictions");

    nb::class_<resolve::Scalers>(m, "Scalers")
        .def(nb::init<>())
        .def_prop_ro("continuous_mean", [](const resolve::Scalers& s) {
            if (s.continuous_mean.defined()) {
                auto cpu_tensor = s.continuous_mean.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
        })
        .def_prop_ro("continuous_scale", [](const resolve::Scalers& s) {
            if (s.continuous_scale.defined()) {
                auto cpu_tensor = s.continuous_scale.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
        })
        // Per-target regression scaling as { name: (mean, scale) }. Mirrors the
        // C-ABI scalers_to_value target_scalers marshal so both bindings expose
        // the target scaling that save_scalers/load_scalers persist.
        .def_prop_ro("target_scalers", [](const resolve::Scalers& s) {
            nb::dict out;
            for (const auto& [name, ms] : s.target_scalers) {
                double mean = ms.first.defined() ? ms.first.item<double>() : 0.0;
                double scale = ms.second.defined() ? ms.second.item<double>() : 0.0;
                out[name.c_str()] = nb::make_tuple(mean, scale);
            }
            return out;
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

    // Per-plot classification predictions for test-fold scoring. The tensor
    // fields are exposed lazily as torch tensors (None when undefined, e.g.
    // for a non-classification target).
    nb::class_<resolve::ClassificationPredictions>(m, "ClassificationPredictions")
        .def(nb::init<>())
        .def_ro("target_name", &resolve::ClassificationPredictions::target_name)
        .def_ro("class_names", &resolve::ClassificationPredictions::class_names)
        .def_prop_ro("predicted_classes", [](const resolve::ClassificationPredictions& c) {
            if (c.predicted_classes.defined()) {
                auto cpu_tensor = c.predicted_classes.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
        })
        .def_prop_ro("probabilities", [](const resolve::ClassificationPredictions& c) {
            if (c.probabilities.defined()) {
                auto cpu_tensor = c.probabilities.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
        })
        .def_prop_ro("actuals", [](const resolve::ClassificationPredictions& c) {
            if (c.actuals.defined()) {
                auto cpu_tensor = c.actuals.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
        });

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
            return nb::none();
        });
}

#undef RESOLVE_BIND_FIELD
