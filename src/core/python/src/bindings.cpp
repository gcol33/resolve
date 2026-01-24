#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

#include "resolve/resolve.hpp"

namespace py = pybind11;

// Helper to convert Python dict to unordered_map of tensors
std::unordered_map<std::string, torch::Tensor> dict_to_tensor_map(const py::dict& d) {
    std::unordered_map<std::string, torch::Tensor> result;
    for (auto item : d) {
        result[item.first.cast<std::string>()] = item.second.cast<torch::Tensor>();
    }
    return result;
}

// Helper to convert unordered_map of tensors to Python dict
py::dict tensor_map_to_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    py::dict result;
    for (const auto& [key, value] : m) {
        result[py::str(key)] = value;
    }
    return result;
}

PYBIND11_MODULE(_resolve_core, m) {
    m.doc() = "RESOLVE C++ core library for species-composition based prediction";

    // ==========================================================================
    // Enums
    // ==========================================================================

    py::enum_<resolve::TaskType>(m, "TaskType")
        .value("Regression", resolve::TaskType::Regression)
        .value("Classification", resolve::TaskType::Classification)
        .export_values();

    py::enum_<resolve::TransformType>(m, "TransformType")
        .value("None_", resolve::TransformType::None)
        .value("Log1p", resolve::TransformType::Log1p)
        .export_values();

    py::enum_<resolve::SpeciesEncodingMode>(m, "SpeciesEncodingMode")
        .value("Hash", resolve::SpeciesEncodingMode::Hash)
        .value("Embed", resolve::SpeciesEncodingMode::Embed)
        .value("Sparse", resolve::SpeciesEncodingMode::Sparse)
        .export_values();

    py::enum_<resolve::SelectionMode>(m, "SelectionMode")
        .value("Top", resolve::SelectionMode::Top)
        .value("Bottom", resolve::SelectionMode::Bottom)
        .value("TopBottom", resolve::SelectionMode::TopBottom)
        .value("All", resolve::SelectionMode::All)
        .export_values();

    py::enum_<resolve::RepresentationMode>(m, "RepresentationMode")
        .value("Abundance", resolve::RepresentationMode::Abundance)
        .value("PresenceAbsence", resolve::RepresentationMode::PresenceAbsence)
        .export_values();

    py::enum_<resolve::NormalizationMode>(m, "NormalizationMode")
        .value("Raw", resolve::NormalizationMode::Raw)
        .value("Norm", resolve::NormalizationMode::Norm)
        .value("Log1p", resolve::NormalizationMode::Log1p)
        .export_values();

    py::enum_<resolve::AggregationMode>(m, "AggregationMode")
        .value("Abundance", resolve::AggregationMode::Abundance)
        .value("Count", resolve::AggregationMode::Count)
        .export_values();

    // ==========================================================================
    // Configuration structs
    // ==========================================================================

    py::class_<resolve::TargetConfig>(m, "TargetConfig")
        .def(py::init<>())
        .def_readwrite("name", &resolve::TargetConfig::name)
        .def_readwrite("task", &resolve::TargetConfig::task)
        .def_readwrite("transform", &resolve::TargetConfig::transform)
        .def_readwrite("num_classes", &resolve::TargetConfig::num_classes)
        .def_readwrite("weight", &resolve::TargetConfig::weight);

    py::class_<resolve::ResolveSchema>(m, "ResolveSchema")
        .def(py::init<>())
        .def_readwrite("n_plots", &resolve::ResolveSchema::n_plots)
        .def_readwrite("n_species", &resolve::ResolveSchema::n_species)
        .def_readwrite("n_species_vocab", &resolve::ResolveSchema::n_species_vocab)
        .def_readwrite("has_coordinates", &resolve::ResolveSchema::has_coordinates)
        .def_readwrite("has_abundance", &resolve::ResolveSchema::has_abundance)
        .def_readwrite("has_taxonomy", &resolve::ResolveSchema::has_taxonomy)
        .def_readwrite("n_genera", &resolve::ResolveSchema::n_genera)
        .def_readwrite("n_families", &resolve::ResolveSchema::n_families)
        .def_readwrite("n_genera_vocab", &resolve::ResolveSchema::n_genera_vocab)
        .def_readwrite("n_families_vocab", &resolve::ResolveSchema::n_families_vocab)
        .def_readwrite("covariate_names", &resolve::ResolveSchema::covariate_names)
        .def_readwrite("targets", &resolve::ResolveSchema::targets)
        .def_readwrite("track_unknown_fraction", &resolve::ResolveSchema::track_unknown_fraction)
        .def_readwrite("track_unknown_count", &resolve::ResolveSchema::track_unknown_count);

    // Alias for backwards compatibility
    m.attr("SpaccSchema") = m.attr("ResolveSchema");

    py::class_<resolve::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("species_encoding", &resolve::ModelConfig::species_encoding)
        .def_readwrite("uses_explicit_vector", &resolve::ModelConfig::uses_explicit_vector)
        .def_readwrite("hash_dim", &resolve::ModelConfig::hash_dim)
        .def_readwrite("species_embed_dim", &resolve::ModelConfig::species_embed_dim)
        .def_readwrite("genus_emb_dim", &resolve::ModelConfig::genus_emb_dim)
        .def_readwrite("family_emb_dim", &resolve::ModelConfig::family_emb_dim)
        .def_readwrite("top_k", &resolve::ModelConfig::top_k)
        .def_readwrite("top_k_species", &resolve::ModelConfig::top_k_species)
        .def_readwrite("n_taxonomy_slots", &resolve::ModelConfig::n_taxonomy_slots)
        .def_readwrite("hidden_dims", &resolve::ModelConfig::hidden_dims)
        .def_readwrite("dropout", &resolve::ModelConfig::dropout);

    py::class_<resolve::TrainConfig>(m, "TrainConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &resolve::TrainConfig::batch_size)
        .def_readwrite("max_epochs", &resolve::TrainConfig::max_epochs)
        .def_readwrite("patience", &resolve::TrainConfig::patience)
        .def_readwrite("lr", &resolve::TrainConfig::lr)
        .def_readwrite("weight_decay", &resolve::TrainConfig::weight_decay)
        .def_readwrite("phase_boundaries", &resolve::TrainConfig::phase_boundaries);

    // ==========================================================================
    // Result structs
    // ==========================================================================

    py::class_<resolve::TrainResult>(m, "TrainResult")
        .def(py::init<>())
        .def_readonly("best_epoch", &resolve::TrainResult::best_epoch)
        .def_readonly("final_metrics", &resolve::TrainResult::final_metrics)
        .def_readonly("train_loss_history", &resolve::TrainResult::train_loss_history)
        .def_readonly("test_loss_history", &resolve::TrainResult::test_loss_history)
        .def_readonly("train_time_seconds", &resolve::TrainResult::train_time_seconds)
        .def_readonly("resumed_from_epoch", &resolve::TrainResult::resumed_from_epoch);

    py::class_<resolve::ResolvePredictions>(m, "ResolvePredictions")
        .def(py::init<>())
        .def_property_readonly("predictions", [](const resolve::ResolvePredictions& p) {
            return tensor_map_to_dict(p.predictions);
        })
        .def_readonly("plot_ids", &resolve::ResolvePredictions::plot_ids)
        .def_readonly("latent", &resolve::ResolvePredictions::latent);

    m.attr("SpaccPredictions") = m.attr("ResolvePredictions");

    py::class_<resolve::Scalers>(m, "Scalers")
        .def(py::init<>())
        .def_readwrite("continuous_mean", &resolve::Scalers::continuous_mean)
        .def_readwrite("continuous_scale", &resolve::Scalers::continuous_scale);

    // ==========================================================================
    // Species Encoding
    // ==========================================================================

    py::class_<resolve::TaxonomyVocab>(m, "TaxonomyVocab")
        .def(py::init<>())
        .def("fit", &resolve::TaxonomyVocab::fit)
        .def("encode_genus", &resolve::TaxonomyVocab::encode_genus)
        .def("encode_family", &resolve::TaxonomyVocab::encode_family)
        .def("n_genera", &resolve::TaxonomyVocab::n_genera)
        .def("n_families", &resolve::TaxonomyVocab::n_families)
        .def("save", &resolve::TaxonomyVocab::save)
        .def_static("load", &resolve::TaxonomyVocab::load);

    py::class_<resolve::SpeciesRecord>(m, "SpeciesRecord")
        .def(py::init<>())
        .def_readwrite("species_id", &resolve::SpeciesRecord::species_id)
        .def_readwrite("genus", &resolve::SpeciesRecord::genus)
        .def_readwrite("family", &resolve::SpeciesRecord::family)
        .def_readwrite("abundance", &resolve::SpeciesRecord::abundance)
        .def_readwrite("plot_id", &resolve::SpeciesRecord::plot_id);

    py::class_<resolve::EncodedSpecies>(m, "EncodedSpecies")
        .def(py::init<>())
        .def_readonly("hash_embedding", &resolve::EncodedSpecies::hash_embedding)
        .def_readonly("genus_ids", &resolve::EncodedSpecies::genus_ids)
        .def_readonly("family_ids", &resolve::EncodedSpecies::family_ids)
        .def_readonly("unknown_fraction", &resolve::EncodedSpecies::unknown_fraction)
        .def_readonly("unknown_count", &resolve::EncodedSpecies::unknown_count)
        .def_readonly("species_vector", &resolve::EncodedSpecies::species_vector)
        .def_readonly("plot_ids", &resolve::EncodedSpecies::plot_ids);


    // ==========================================================================
    // Model
    // ==========================================================================

    py::class_<resolve::ResolveModelImpl, std::shared_ptr<resolve::ResolveModelImpl>>(m, "ResolveModelImpl")
        .def(py::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
             py::arg("schema"), py::arg("config") = resolve::ModelConfig{})
        .def("forward", [](resolve::ResolveModelImpl& self,
                          torch::Tensor continuous,
                          torch::Tensor genus_ids,
                          torch::Tensor family_ids,
                          torch::Tensor species_ids,
                          torch::Tensor species_vector) {
            return tensor_map_to_dict(self.forward(continuous, genus_ids, family_ids, species_ids, species_vector));
        }, py::arg("continuous"),
           py::arg("genus_ids") = torch::Tensor(),
           py::arg("family_ids") = torch::Tensor(),
           py::arg("species_ids") = torch::Tensor(),
           py::arg("species_vector") = torch::Tensor())
        .def("get_latent", &resolve::ResolveModelImpl::get_latent,
             py::arg("continuous"),
             py::arg("genus_ids") = torch::Tensor(),
             py::arg("family_ids") = torch::Tensor(),
             py::arg("species_ids") = torch::Tensor(),
             py::arg("species_vector") = torch::Tensor())
        .def("schema", &resolve::ResolveModelImpl::schema)
        .def("config", &resolve::ResolveModelImpl::config)
        .def("latent_dim", &resolve::ResolveModelImpl::latent_dim)
        .def("species_encoding", &resolve::ResolveModelImpl::species_encoding)
        .def("uses_explicit_vector", &resolve::ResolveModelImpl::uses_explicit_vector);

    py::class_<resolve::ResolveModel>(m, "ResolveModel")
        .def(py::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
             py::arg("schema"), py::arg("config") = resolve::ModelConfig{})
        .def("forward", [](resolve::ResolveModel& self,
                          torch::Tensor continuous,
                          torch::Tensor genus_ids,
                          torch::Tensor family_ids,
                          torch::Tensor species_ids,
                          torch::Tensor species_vector) {
            return tensor_map_to_dict(self->forward(continuous, genus_ids, family_ids, species_ids, species_vector));
        }, py::arg("continuous"),
           py::arg("genus_ids") = torch::Tensor(),
           py::arg("family_ids") = torch::Tensor(),
           py::arg("species_ids") = torch::Tensor(),
           py::arg("species_vector") = torch::Tensor())
        .def("get_latent", [](resolve::ResolveModel& self,
                              torch::Tensor continuous,
                              torch::Tensor genus_ids,
                              torch::Tensor family_ids,
                              torch::Tensor species_ids,
                              torch::Tensor species_vector) {
            return self->get_latent(continuous, genus_ids, family_ids, species_ids, species_vector);
        }, py::arg("continuous"),
           py::arg("genus_ids") = torch::Tensor(),
           py::arg("family_ids") = torch::Tensor(),
           py::arg("species_ids") = torch::Tensor(),
           py::arg("species_vector") = torch::Tensor())
        .def("train", [](resolve::ResolveModel& self, bool mode) { self->train(mode); }, py::arg("mode") = true)
        .def("eval", [](resolve::ResolveModel& self) { self->eval(); })
        .def("to", [](resolve::ResolveModel& self, const std::string& device) {
            if (device == "cuda") {
                self->to(torch::kCUDA);
            } else {
                self->to(torch::kCPU);
            }
        })
        .def_property_readonly("schema", [](resolve::ResolveModel& self) { return self->schema(); })
        .def_property_readonly("config", [](resolve::ResolveModel& self) { return self->config(); })
        .def_property_readonly("latent_dim", [](resolve::ResolveModel& self) { return self->latent_dim(); })
        .def_property_readonly("species_encoding", [](resolve::ResolveModel& self) { return self->species_encoding(); })
        .def_property_readonly("uses_explicit_vector", [](resolve::ResolveModel& self) { return self->uses_explicit_vector(); });

    m.attr("SpaccModel") = m.attr("ResolveModel");

    // ==========================================================================
    // Trainer
    // ==========================================================================

    py::class_<resolve::Trainer>(m, "Trainer")
        .def(py::init<resolve::ResolveModel, const resolve::TrainConfig&>(),
             py::arg("model"), py::arg("config") = resolve::TrainConfig{})
        .def("prepare_data", &resolve::Trainer::prepare_data,
             py::arg("coordinates"),
             py::arg("covariates"),
             py::arg("hash_embedding"),
             py::arg("species_ids"),
             py::arg("species_vector"),
             py::arg("genus_ids"),
             py::arg("family_ids"),
             py::arg("unknown_fraction"),
             py::arg("unknown_count"),
             py::arg("targets"),
             py::arg("test_size") = 0.2f,
             py::arg("seed") = 42)
        .def("fit", &resolve::Trainer::fit)
        .def("save", &resolve::Trainer::save)
        .def_static("load", &resolve::Trainer::load,
                    py::arg("path"), py::arg("device") = torch::kCPU)
        .def_property_readonly("model", &resolve::Trainer::model)
        .def_property_readonly("scalers", &resolve::Trainer::scalers)
        .def_property_readonly("config", &resolve::Trainer::config);

    // ==========================================================================
    // Predictor
    // ==========================================================================

    py::class_<resolve::Predictor>(m, "Predictor")
        .def(py::init<resolve::ResolveModel, resolve::Scalers, torch::Device>(),
             py::arg("model"), py::arg("scalers"),
             py::arg("device") = torch::kCPU)
        .def_static("load", &resolve::Predictor::load,
                    py::arg("path"), py::arg("device") = torch::kCPU)
        .def("predict", &resolve::Predictor::predict,
             py::arg("coordinates"),
             py::arg("covariates"),
             py::arg("hash_embedding"),
             py::arg("genus_ids"),
             py::arg("family_ids"),
             py::arg("return_latent") = false)
        .def("get_embeddings", &resolve::Predictor::get_embeddings,
             py::arg("coordinates"),
             py::arg("covariates"),
             py::arg("hash_embedding"),
             py::arg("genus_ids"),
             py::arg("family_ids"))
        .def("get_genus_embeddings", &resolve::Predictor::get_genus_embeddings)
        .def("get_family_embeddings", &resolve::Predictor::get_family_embeddings)
        .def_property_readonly("model", &resolve::Predictor::model)
        .def_property_readonly("scalers", &resolve::Predictor::scalers);

    // ==========================================================================
    // Metrics
    // ==========================================================================

    py::class_<resolve::Metrics>(m, "Metrics")
        .def_static("band_accuracy", &resolve::Metrics::band_accuracy,
                    py::arg("pred"), py::arg("target"), py::arg("threshold") = 0.25f)
        .def_static("mae", &resolve::Metrics::mae)
        .def_static("rmse", &resolve::Metrics::rmse)
        .def_static("smape", &resolve::Metrics::smape,
                    py::arg("pred"), py::arg("target"), py::arg("eps") = 1e-8f)
        .def_static("accuracy", &resolve::Metrics::accuracy)
        .def_static("compute", &resolve::Metrics::compute,
                    py::arg("pred"), py::arg("target"), py::arg("task"),
                    py::arg("transform") = resolve::TransformType::None);

    // ==========================================================================
    // Version
    // ==========================================================================

    m.attr("__version__") = resolve::VERSION;
}
