#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <torch/torch.h>

#include "resolve/resolve.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/dataset.hpp"

namespace nb = nanobind;

// Helper to convert Python dict to unordered_map of tensors
std::unordered_map<std::string, torch::Tensor> dict_to_tensor_map(const nb::dict& d) {
    std::unordered_map<std::string, torch::Tensor> result;
    for (auto item : d) {
        result[nb::cast<std::string>(item.first)] = nb::cast<torch::Tensor>(item.second);
    }
    return result;
}

// Helper to convert unordered_map of tensors to Python dict
nb::dict tensor_map_to_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    nb::dict result;
    for (const auto& [key, value] : m) {
        result[nb::str(key.c_str())] = value;
    }
    return result;
}

NB_MODULE(_resolve_core, m) {
    m.doc() = "RESOLVE C++ core library for species-composition based prediction";

    // ==========================================================================
    // Enums
    // ==========================================================================

    nb::enum_<resolve::TaskType>(m, "TaskType")
        .value("Regression", resolve::TaskType::Regression)
        .value("Classification", resolve::TaskType::Classification)
        .export_values();

    nb::enum_<resolve::TransformType>(m, "TransformType")
        .value("None_", resolve::TransformType::None)
        .value("Log1p", resolve::TransformType::Log1p)
        .export_values();

    nb::enum_<resolve::SpeciesEncodingMode>(m, "SpeciesEncodingMode")
        .value("Hash", resolve::SpeciesEncodingMode::Hash)
        .value("Embed", resolve::SpeciesEncodingMode::Embed)
        .value("Sparse", resolve::SpeciesEncodingMode::Sparse)
        .export_values();

    nb::enum_<resolve::LossConfigMode>(m, "LossConfigMode")
        .value("MAE", resolve::LossConfigMode::MAE)
        .value("SMAPE", resolve::LossConfigMode::SMAPE)
        .value("Combined", resolve::LossConfigMode::Combined)
        .export_values();

    nb::enum_<resolve::SelectionMode>(m, "SelectionMode")
        .value("Top", resolve::SelectionMode::Top)
        .value("Bottom", resolve::SelectionMode::Bottom)
        .value("TopBottom", resolve::SelectionMode::TopBottom)
        .value("All", resolve::SelectionMode::All)
        .export_values();

    nb::enum_<resolve::RepresentationMode>(m, "RepresentationMode")
        .value("Abundance", resolve::RepresentationMode::Abundance)
        .value("PresenceAbsence", resolve::RepresentationMode::PresenceAbsence)
        .export_values();

    nb::enum_<resolve::NormalizationMode>(m, "NormalizationMode")
        .value("Raw", resolve::NormalizationMode::Raw)
        .value("Norm", resolve::NormalizationMode::Norm)
        .value("Log1p", resolve::NormalizationMode::Log1p)
        .export_values();

    nb::enum_<resolve::AggregationMode>(m, "AggregationMode")
        .value("Abundance", resolve::AggregationMode::Abundance)
        .value("Count", resolve::AggregationMode::Count)
        .export_values();

    nb::enum_<resolve::LRSchedulerType>(m, "LRSchedulerType")
        .value("None_", resolve::LRSchedulerType::None)
        .value("StepLR", resolve::LRSchedulerType::StepLR)
        .value("CosineAnnealing", resolve::LRSchedulerType::CosineAnnealing)
        .export_values();

    // ==========================================================================
    // Role Mapping and Dataset Configuration
    // ==========================================================================

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
        .def_rw("targets", &resolve::RoleMapping::targets)
        .def("has_coordinates", &resolve::RoleMapping::has_coordinates)
        .def("has_taxonomy", &resolve::RoleMapping::has_taxonomy)
        .def("has_abundance", &resolve::RoleMapping::has_abundance);

    nb::class_<resolve::TargetSpec>(m, "TargetSpec")
        .def(nb::init<>())
        .def_rw("column_name", &resolve::TargetSpec::column_name)
        .def_rw("target_name", &resolve::TargetSpec::target_name)
        .def_rw("task", &resolve::TargetSpec::task)
        .def_rw("transform", &resolve::TargetSpec::transform)
        .def_rw("num_classes", &resolve::TargetSpec::num_classes)
        .def_rw("weight", &resolve::TargetSpec::weight)
        .def_static("regression", &resolve::TargetSpec::regression,
                    nb::arg("column"), nb::arg("transform") = resolve::TransformType::None)
        .def_static("classification", &resolve::TargetSpec::classification,
                    nb::arg("column"), nb::arg("num_classes"));

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
        .def_rw("use_taxonomy", &resolve::DatasetConfig::use_taxonomy);

    // ==========================================================================
    // Configuration structs
    // ==========================================================================

    nb::class_<resolve::TargetConfig>(m, "TargetConfig")
        .def(nb::init<>())
        .def_rw("name", &resolve::TargetConfig::name)
        .def_rw("task", &resolve::TargetConfig::task)
        .def_rw("transform", &resolve::TargetConfig::transform)
        .def_rw("num_classes", &resolve::TargetConfig::num_classes)
        .def_rw("weight", &resolve::TargetConfig::weight)
        .def_rw("class_weights", &resolve::TargetConfig::class_weights);

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
        .def_rw("track_unknown_count", &resolve::ResolveSchema::track_unknown_count);

    // Alias for backwards compatibility
    m.attr("SpaccSchema") = m.attr("ResolveSchema");

    nb::class_<resolve::ModelConfig>(m, "ModelConfig")
        .def(nb::init<>())
        .def_rw("species_encoding", &resolve::ModelConfig::species_encoding)
        .def_rw("uses_explicit_vector", &resolve::ModelConfig::uses_explicit_vector)
        .def_rw("hash_dim", &resolve::ModelConfig::hash_dim)
        .def_rw("species_embed_dim", &resolve::ModelConfig::species_embed_dim)
        .def_rw("genus_emb_dim", &resolve::ModelConfig::genus_emb_dim)
        .def_rw("family_emb_dim", &resolve::ModelConfig::family_emb_dim)
        .def_rw("top_k", &resolve::ModelConfig::top_k)
        .def_rw("top_k_species", &resolve::ModelConfig::top_k_species)
        .def_rw("n_taxonomy_slots", &resolve::ModelConfig::n_taxonomy_slots)
        .def_rw("hidden_dims", &resolve::ModelConfig::hidden_dims)
        .def_rw("dropout", &resolve::ModelConfig::dropout);

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
        .def_rw("lr_min", &resolve::TrainConfig::lr_min);

    // ==========================================================================
    // Result structs
    // ==========================================================================

    nb::class_<resolve::TrainResult>(m, "TrainResult")
        .def(nb::init<>())
        .def_ro("best_epoch", &resolve::TrainResult::best_epoch)
        .def_ro("final_metrics", &resolve::TrainResult::final_metrics)
        .def_ro("train_loss_history", &resolve::TrainResult::train_loss_history)
        .def_ro("test_loss_history", &resolve::TrainResult::test_loss_history)
        .def_ro("train_time_seconds", &resolve::TrainResult::train_time_seconds)
        .def_ro("resumed_from_epoch", &resolve::TrainResult::resumed_from_epoch);

    nb::class_<resolve::ResolvePredictions>(m, "ResolvePredictions")
        .def(nb::init<>())
        .def_prop_ro("predictions", [](const resolve::ResolvePredictions& p) {
            return tensor_map_to_dict(p.predictions);
        })
        .def_ro("plot_ids", &resolve::ResolvePredictions::plot_ids)
        .def_ro("latent", &resolve::ResolvePredictions::latent);

    m.attr("SpaccPredictions") = m.attr("ResolvePredictions");

    // ==========================================================================
    // Dataset
    // ==========================================================================

    nb::class_<resolve::ResolveDataset>(m, "ResolveDataset")
        .def_static("from_csv", &resolve::ResolveDataset::from_csv,
                    nb::arg("header_path"),
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset from header CSV and species CSV files")
        .def_static("from_species_csv", &resolve::ResolveDataset::from_species_csv,
                    nb::arg("species_path"),
                    nb::arg("roles"),
                    nb::arg("targets"),
                    nb::arg("config") = resolve::DatasetConfig{},
                    "Load dataset from a single species CSV file")
        .def_prop_ro("coordinates", &resolve::ResolveDataset::coordinates)
        .def_prop_ro("covariates", &resolve::ResolveDataset::covariates)
        .def_prop_ro("hash_embedding", &resolve::ResolveDataset::hash_embedding)
        .def_prop_ro("species_ids", &resolve::ResolveDataset::species_ids)
        .def_prop_ro("species_vector", &resolve::ResolveDataset::species_vector)
        .def_prop_ro("genus_ids", &resolve::ResolveDataset::genus_ids)
        .def_prop_ro("family_ids", &resolve::ResolveDataset::family_ids)
        .def_prop_ro("unknown_fraction", &resolve::ResolveDataset::unknown_fraction)
        .def_prop_ro("unknown_count", &resolve::ResolveDataset::unknown_count)
        .def_prop_ro("targets", [](const resolve::ResolveDataset& d) {
            return tensor_map_to_dict(d.targets());
        })
        .def_prop_ro("schema", &resolve::ResolveDataset::schema)
        .def_prop_ro("plot_ids", &resolve::ResolveDataset::plot_ids)
        .def_prop_ro("species_vocab", &resolve::ResolveDataset::species_vocab)
        .def_prop_ro("n_plots", &resolve::ResolveDataset::n_plots)
        .def_prop_ro("config", &resolve::ResolveDataset::config);

    nb::class_<resolve::Scalers>(m, "Scalers")
        .def(nb::init<>())
        .def_rw("continuous_mean", &resolve::Scalers::continuous_mean)
        .def_rw("continuous_scale", &resolve::Scalers::continuous_scale);

    // ==========================================================================
    // Species Encoding
    // ==========================================================================

    nb::class_<resolve::TaxonomyVocab>(m, "TaxonomyVocab")
        .def(nb::init<>())
        .def("fit", &resolve::TaxonomyVocab::fit)
        .def("encode_genus", &resolve::TaxonomyVocab::encode_genus)
        .def("encode_family", &resolve::TaxonomyVocab::encode_family)
        .def("n_genera", &resolve::TaxonomyVocab::n_genera)
        .def("n_families", &resolve::TaxonomyVocab::n_families)
        .def("save", &resolve::TaxonomyVocab::save)
        .def_static("load", &resolve::TaxonomyVocab::load);

    nb::class_<resolve::SpeciesRecord>(m, "SpeciesRecord")
        .def(nb::init<>())
        .def_rw("species_id", &resolve::SpeciesRecord::species_id)
        .def_rw("genus", &resolve::SpeciesRecord::genus)
        .def_rw("family", &resolve::SpeciesRecord::family)
        .def_rw("abundance", &resolve::SpeciesRecord::abundance)
        .def_rw("plot_id", &resolve::SpeciesRecord::plot_id);

    nb::class_<resolve::EncodedSpecies>(m, "EncodedSpecies")
        .def(nb::init<>())
        .def_ro("hash_embedding", &resolve::EncodedSpecies::hash_embedding)
        .def_ro("genus_ids", &resolve::EncodedSpecies::genus_ids)
        .def_ro("family_ids", &resolve::EncodedSpecies::family_ids)
        .def_ro("unknown_fraction", &resolve::EncodedSpecies::unknown_fraction)
        .def_ro("unknown_count", &resolve::EncodedSpecies::unknown_count)
        .def_ro("species_vector", &resolve::EncodedSpecies::species_vector)
        .def_ro("plot_ids", &resolve::EncodedSpecies::plot_ids);

    // ==========================================================================
    // Model
    // ==========================================================================

    nb::class_<resolve::ResolveModel>(m, "ResolveModel")
        .def(nb::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
             nb::arg("schema"), nb::arg("config") = resolve::ModelConfig{})
        .def("forward", [](resolve::ResolveModel& self,
                          torch::Tensor continuous,
                          torch::Tensor genus_ids,
                          torch::Tensor family_ids,
                          torch::Tensor species_ids,
                          torch::Tensor species_vector) {
            return tensor_map_to_dict(self->forward(continuous, genus_ids, family_ids, species_ids, species_vector));
        }, nb::arg("continuous"),
           nb::arg("genus_ids") = torch::Tensor(),
           nb::arg("family_ids") = torch::Tensor(),
           nb::arg("species_ids") = torch::Tensor(),
           nb::arg("species_vector") = torch::Tensor())
        .def("get_latent", [](resolve::ResolveModel& self,
                              torch::Tensor continuous,
                              torch::Tensor genus_ids,
                              torch::Tensor family_ids,
                              torch::Tensor species_ids,
                              torch::Tensor species_vector) {
            return self->get_latent(continuous, genus_ids, family_ids, species_ids, species_vector);
        }, nb::arg("continuous"),
           nb::arg("genus_ids") = torch::Tensor(),
           nb::arg("family_ids") = torch::Tensor(),
           nb::arg("species_ids") = torch::Tensor(),
           nb::arg("species_vector") = torch::Tensor())
        .def("train", [](resolve::ResolveModel& self, bool mode) { self->train(mode); }, nb::arg("mode") = true)
        .def("eval", [](resolve::ResolveModel& self) { self->eval(); })
        .def("to", [](resolve::ResolveModel& self, const std::string& device) {
            if (device == "cuda") {
                self->to(torch::kCUDA);
            } else {
                self->to(torch::kCPU);
            }
        })
        .def_prop_ro("schema", [](resolve::ResolveModel& self) { return self->schema(); })
        .def_prop_ro("config", [](resolve::ResolveModel& self) { return self->config(); })
        .def_prop_ro("latent_dim", [](resolve::ResolveModel& self) { return self->latent_dim(); })
        .def_prop_ro("species_encoding", [](resolve::ResolveModel& self) { return self->species_encoding(); })
        .def_prop_ro("uses_explicit_vector", [](resolve::ResolveModel& self) { return self->uses_explicit_vector(); });

    m.attr("SpaccModel") = m.attr("ResolveModel");

    // ==========================================================================
    // Trainer
    // ==========================================================================

    nb::class_<resolve::Trainer>(m, "Trainer")
        .def(nb::init<resolve::ResolveModel, const resolve::TrainConfig&>(),
             nb::arg("model"), nb::arg("config") = resolve::TrainConfig{})
        .def("prepare_data", [](resolve::Trainer& self,
                               const resolve::ResolveDataset& dataset,
                               float test_size,
                               int seed) {
            self.prepare_data(dataset, test_size, seed);
        }, nb::arg("dataset"),
           nb::arg("test_size") = 0.2f,
           nb::arg("seed") = 42,
           "Prepare data from a ResolveDataset (preferred API)")
        .def("prepare_data_raw", [](resolve::Trainer& self,
                               torch::Tensor coordinates,
                               torch::Tensor covariates,
                               torch::Tensor hash_embedding,
                               torch::Tensor species_ids,
                               torch::Tensor species_vector,
                               torch::Tensor genus_ids,
                               torch::Tensor family_ids,
                               torch::Tensor unknown_fraction,
                               torch::Tensor unknown_count,
                               const nb::dict& targets,
                               float test_size,
                               int seed) {
            self.prepare_data(coordinates, covariates, hash_embedding,
                            species_ids, species_vector, genus_ids, family_ids,
                            unknown_fraction, unknown_count,
                            dict_to_tensor_map(targets), test_size, seed);
        }, nb::arg("coordinates"),
           nb::arg("covariates"),
           nb::arg("hash_embedding"),
           nb::arg("species_ids"),
           nb::arg("species_vector"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("unknown_fraction"),
           nb::arg("unknown_count"),
           nb::arg("targets"),
           nb::arg("test_size") = 0.2f,
           nb::arg("seed") = 42,
           "Prepare data from raw tensors (backwards compatible API)")
        .def("fit", &resolve::Trainer::fit)
        .def("save", &resolve::Trainer::save)
        .def_static("load", &resolve::Trainer::load,
                    nb::arg("path"), nb::arg("device") = torch::kCPU)
        .def_prop_ro("model", &resolve::Trainer::model)
        .def_prop_ro("scalers", &resolve::Trainer::scalers)
        .def_prop_ro("config", &resolve::Trainer::config);

    // ==========================================================================
    // Predictor
    // ==========================================================================

    nb::class_<resolve::Predictor>(m, "Predictor")
        .def(nb::init<resolve::ResolveModel, resolve::Scalers, torch::Device>(),
             nb::arg("model"), nb::arg("scalers"),
             nb::arg("device") = torch::kCPU)
        .def_static("load", &resolve::Predictor::load,
                    nb::arg("path"), nb::arg("device") = torch::kCPU)
        .def("predict", &resolve::Predictor::predict,
             nb::arg("coordinates"),
             nb::arg("covariates"),
             nb::arg("hash_embedding"),
             nb::arg("genus_ids"),
             nb::arg("family_ids"),
             nb::arg("return_latent") = false)
        .def("get_embeddings", &resolve::Predictor::get_embeddings,
             nb::arg("coordinates"),
             nb::arg("covariates"),
             nb::arg("hash_embedding"),
             nb::arg("genus_ids"),
             nb::arg("family_ids"))
        .def("get_genus_embeddings", &resolve::Predictor::get_genus_embeddings)
        .def("get_family_embeddings", &resolve::Predictor::get_family_embeddings)
        .def_prop_ro("model", &resolve::Predictor::model)
        .def_prop_ro("scalers", &resolve::Predictor::scalers);

    // ==========================================================================
    // Metrics
    // ==========================================================================

    nb::class_<resolve::ClassificationMetrics>(m, "ClassificationMetrics")
        .def(nb::init<>())
        .def_ro("accuracy", &resolve::ClassificationMetrics::accuracy)
        .def_ro("macro_f1", &resolve::ClassificationMetrics::macro_f1)
        .def_ro("weighted_f1", &resolve::ClassificationMetrics::weighted_f1)
        .def_ro("per_class_precision", &resolve::ClassificationMetrics::per_class_precision)
        .def_ro("per_class_recall", &resolve::ClassificationMetrics::per_class_recall)
        .def_ro("per_class_f1", &resolve::ClassificationMetrics::per_class_f1)
        .def_ro("per_class_support", &resolve::ClassificationMetrics::per_class_support)
        .def_ro("confusion_matrix", &resolve::ClassificationMetrics::confusion_matrix);

    nb::class_<resolve::ConfidenceMetrics>(m, "ConfidenceMetrics")
        .def(nb::init<>())
        .def_ro("accuracy", &resolve::ConfidenceMetrics::accuracy)
        .def_ro("coverage", &resolve::ConfidenceMetrics::coverage)
        .def_ro("n_samples", &resolve::ConfidenceMetrics::n_samples)
        .def_ro("n_total", &resolve::ConfidenceMetrics::n_total);

    nb::class_<resolve::Metrics>(m, "Metrics")
        .def_static("band_accuracy", &resolve::Metrics::band_accuracy,
                    nb::arg("pred"), nb::arg("target"), nb::arg("threshold") = 0.25f)
        .def_static("mae", &resolve::Metrics::mae)
        .def_static("rmse", &resolve::Metrics::rmse)
        .def_static("smape", &resolve::Metrics::smape,
                    nb::arg("pred"), nb::arg("target"), nb::arg("eps") = 1e-8f)
        .def_static("r_squared", &resolve::Metrics::r_squared,
                    nb::arg("pred"), nb::arg("target"))
        .def_static("accuracy", &resolve::Metrics::accuracy)
        .def_static("confusion_matrix", &resolve::Metrics::confusion_matrix,
                    nb::arg("pred"), nb::arg("target"), nb::arg("num_classes"))
        .def_static("classification_metrics", &resolve::Metrics::classification_metrics,
                    nb::arg("pred"), nb::arg("target"), nb::arg("num_classes"))
        .def_static("accuracy_at_threshold", &resolve::Metrics::accuracy_at_threshold,
                    nb::arg("pred"), nb::arg("target"), nb::arg("confidence"), nb::arg("threshold"))
        .def_static("accuracy_coverage_curve", &resolve::Metrics::accuracy_coverage_curve,
                    nb::arg("pred"), nb::arg("target"), nb::arg("confidence"),
                    nb::arg("thresholds") = std::vector<float>{0.0f, 0.5f, 0.8f, 0.9f, 0.95f})
        .def_static("compute", &resolve::Metrics::compute,
                    nb::arg("pred"), nb::arg("target"), nb::arg("task"),
                    nb::arg("transform") = resolve::TransformType::None,
                    nb::arg("band_thresholds") = std::vector<float>{0.25f, 0.50f, 0.75f},
                    nb::arg("num_classes") = 0);

    // ==========================================================================
    // Version
    // ==========================================================================

    m.attr("__version__") = resolve::VERSION;
}
