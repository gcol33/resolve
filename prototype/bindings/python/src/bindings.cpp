/**
 * pybind11 bindings for RESOLVE C++ core.
 *
 * This exposes the C++ implementation to Python, allowing:
 *   - Full training and inference from Python
 *   - Interoperability with NumPy arrays
 *   - Same API as the pure Python implementation
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <resolve/types.hpp>
#include <resolve/vocab.hpp>
#include <resolve/dataset.hpp>
#include <resolve/loss.hpp>
#include <resolve/trainer.hpp>
#include <resolve/encoder.hpp>
#include <resolve/model.hpp>
#include <resolve/plot_encoder.hpp>

namespace py = pybind11;

// ============================================================================
// Helper functions for NumPy <-> Torch conversion
// ============================================================================

torch::Tensor numpy_to_tensor(py::array_t<float> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<float*>(buf.ptr);

    std::vector<int64_t> shape;
    for (auto dim : buf.shape) {
        shape.push_back(static_cast<int64_t>(dim));
    }

    return torch::from_blob(ptr, shape, torch::kFloat32).clone();
}

py::array_t<float> tensor_to_numpy(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().contiguous();
    auto* data = cpu_tensor.data_ptr<float>();

    std::vector<ssize_t> shape;
    for (int64_t dim : cpu_tensor.sizes()) {
        shape.push_back(static_cast<ssize_t>(dim));
    }

    return py::array_t<float>(shape, data);
}


// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(_core, m) {
    m.doc() = "RESOLVE C++ core bindings";

    // ========================================================================
    // Enums
    // ========================================================================

    py::enum_<resolve::TaskType>(m, "TaskType")
        .value("Regression", resolve::TaskType::Regression)
        .value("Classification", resolve::TaskType::Classification)
        .export_values();

    py::enum_<resolve::TransformType>(m, "TransformType")
        .value("NONE", resolve::TransformType::None)
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

    py::enum_<resolve::NormalizationMode>(m, "NormalizationMode")
        .value("Raw", resolve::NormalizationMode::Raw)
        .value("Norm", resolve::NormalizationMode::Norm)
        .value("Log1p", resolve::NormalizationMode::Log1p)
        .export_values();

    py::enum_<resolve::RepresentationMode>(m, "RepresentationMode")
        .value("Abundance", resolve::RepresentationMode::Abundance)
        .value("PresenceAbsence", resolve::RepresentationMode::PresenceAbsence)
        .export_values();

    py::enum_<resolve::AggregationMode>(m, "AggregationMode")
        .value("Abundance", resolve::AggregationMode::Abundance)
        .value("Count", resolve::AggregationMode::Count)
        .export_values();

    // ========================================================================
    // Configurations
    // ========================================================================

    py::class_<resolve::TargetConfig>(m, "TargetConfig")
        .def(py::init<>())
        .def_readwrite("name", &resolve::TargetConfig::name)
        .def_readwrite("column", &resolve::TargetConfig::column)
        .def_readwrite("task", &resolve::TargetConfig::task)
        .def_readwrite("transform", &resolve::TargetConfig::transform)
        .def_readwrite("num_classes", &resolve::TargetConfig::num_classes)
        .def_readwrite("weight", &resolve::TargetConfig::weight)
        .def_static("regression", &resolve::TargetConfig::regression,
            py::arg("name"), py::arg("column"),
            py::arg("transform") = resolve::TransformType::None,
            py::arg("weight") = 1.0f)
        .def_static("classification", &resolve::TargetConfig::classification,
            py::arg("name"), py::arg("column"),
            py::arg("num_classes"), py::arg("weight") = 1.0f);

    py::class_<resolve::RoleMapping>(m, "RoleMapping")
        .def(py::init<>())
        .def_readwrite("plot_id", &resolve::RoleMapping::plot_id)
        .def_readwrite("species_id", &resolve::RoleMapping::species_id)
        .def_readwrite("species_plot_id", &resolve::RoleMapping::species_plot_id)
        .def_readwrite("coords_lat", &resolve::RoleMapping::coords_lat)
        .def_readwrite("coords_lon", &resolve::RoleMapping::coords_lon)
        .def_readwrite("abundance", &resolve::RoleMapping::abundance)
        .def_readwrite("taxonomy_genus", &resolve::RoleMapping::taxonomy_genus)
        .def_readwrite("taxonomy_family", &resolve::RoleMapping::taxonomy_family)
        .def_readwrite("covariates", &resolve::RoleMapping::covariates)
        .def("has_coordinates", &resolve::RoleMapping::has_coordinates)
        .def("has_abundance", &resolve::RoleMapping::has_abundance)
        .def("has_taxonomy", &resolve::RoleMapping::has_taxonomy)
        .def("validate", &resolve::RoleMapping::validate);

    py::class_<resolve::ResolveSchema>(m, "ResolveSchema")
        .def(py::init<>())
        .def_readwrite("n_plots", &resolve::ResolveSchema::n_plots)
        .def_readwrite("n_species", &resolve::ResolveSchema::n_species)
        .def_readwrite("n_continuous", &resolve::ResolveSchema::n_continuous)
        .def_readwrite("has_coordinates", &resolve::ResolveSchema::has_coordinates)
        .def_readwrite("has_abundance", &resolve::ResolveSchema::has_abundance)
        .def_readwrite("has_taxonomy", &resolve::ResolveSchema::has_taxonomy)
        .def_readwrite("n_genera", &resolve::ResolveSchema::n_genera)
        .def_readwrite("n_families", &resolve::ResolveSchema::n_families)
        .def_readwrite("covariate_names", &resolve::ResolveSchema::covariate_names)
        .def_readwrite("targets", &resolve::ResolveSchema::targets)
        .def_readwrite("n_species_vocab", &resolve::ResolveSchema::n_species_vocab)
        .def_readwrite("n_genera_vocab", &resolve::ResolveSchema::n_genera_vocab)
        .def_readwrite("n_families_vocab", &resolve::ResolveSchema::n_families_vocab)
        .def_readwrite("track_unknown_fraction", &resolve::ResolveSchema::track_unknown_fraction)
        .def_readwrite("track_unknown_count", &resolve::ResolveSchema::track_unknown_count);

    py::class_<resolve::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("species_encoding", &resolve::ModelConfig::species_encoding)
        .def_readwrite("uses_explicit_vector", &resolve::ModelConfig::uses_explicit_vector)
        .def_readwrite("hash_dim", &resolve::ModelConfig::hash_dim)
        .def_readwrite("species_embed_dim", &resolve::ModelConfig::species_embed_dim)
        .def_readwrite("top_k_species", &resolve::ModelConfig::top_k_species)
        .def_readwrite("genus_emb_dim", &resolve::ModelConfig::genus_emb_dim)
        .def_readwrite("family_emb_dim", &resolve::ModelConfig::family_emb_dim)
        .def_readwrite("top_k", &resolve::ModelConfig::top_k)
        .def_readwrite("n_taxonomy_slots", &resolve::ModelConfig::n_taxonomy_slots)
        .def_readwrite("hidden_dims", &resolve::ModelConfig::hidden_dims)
        .def_readwrite("dropout", &resolve::ModelConfig::dropout);

    py::class_<resolve::TrainResult>(m, "TrainResult")
        .def(py::init<>())
        .def_readwrite("best_epoch", &resolve::TrainResult::best_epoch)
        .def_readwrite("total_epochs", &resolve::TrainResult::total_epochs)
        .def_readwrite("best_loss", &resolve::TrainResult::best_loss)
        .def_readwrite("final_metrics", &resolve::TrainResult::final_metrics)
        .def_readwrite("train_loss_history", &resolve::TrainResult::train_loss_history)
        .def_readwrite("test_loss_history", &resolve::TrainResult::test_loss_history)
        .def_readwrite("train_time_seconds", &resolve::TrainResult::train_time_seconds)
        .def_readwrite("resumed_from_epoch", &resolve::TrainResult::resumed_from_epoch)
        .def_readwrite("early_stopped", &resolve::TrainResult::early_stopped);

    // ========================================================================
    // Vocabularies
    // ========================================================================

    py::class_<resolve::SpeciesVocab>(m, "SpeciesVocab")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, int64_t>>())
        .def("n_species", &resolve::SpeciesVocab::n_species)
        .def("encode", &resolve::SpeciesVocab::encode)
        .def("encode_batch", &resolve::SpeciesVocab::encode_batch)
        .def_static("from_species_data", &resolve::SpeciesVocab::from_species_data,
            py::arg("species_ids"), py::arg("min_count") = 1)
        .def("save", &resolve::SpeciesVocab::save)
        .def_static("load", &resolve::SpeciesVocab::load)
        .def("mapping", &resolve::SpeciesVocab::mapping);

    py::class_<resolve::TaxonomyVocab>(m, "TaxonomyVocab")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, int64_t>,
                      std::unordered_map<std::string, int64_t>>())
        .def("n_genera", &resolve::TaxonomyVocab::n_genera)
        .def("n_families", &resolve::TaxonomyVocab::n_families)
        .def("encode_genus", &resolve::TaxonomyVocab::encode_genus)
        .def("encode_family", &resolve::TaxonomyVocab::encode_family)
        .def_static("from_species_data", &resolve::TaxonomyVocab::from_species_data)
        .def("save", &resolve::TaxonomyVocab::save)
        .def_static("load", &resolve::TaxonomyVocab::load);

    // ========================================================================
    // Plot Encoder (Generalized)
    // ========================================================================

    py::enum_<resolve::EncodingType>(m, "EncodingType")
        .value("Numeric", resolve::EncodingType::Numeric)
        .value("Raw", resolve::EncodingType::Raw)
        .value("Hash", resolve::EncodingType::Hash)
        .value("Embed", resolve::EncodingType::Embed)
        .value("OneHot", resolve::EncodingType::OneHot)
        .export_values();

    py::enum_<resolve::DataSource>(m, "DataSource")
        .value("Plot", resolve::DataSource::Plot)
        .value("Observation", resolve::DataSource::Observation)
        .export_values();

    py::class_<resolve::ColumnSpec>(m, "ColumnSpec")
        .def(py::init<>())
        .def_readwrite("name", &resolve::ColumnSpec::name)
        .def_readwrite("columns", &resolve::ColumnSpec::columns)
        .def_readwrite("type", &resolve::ColumnSpec::type)
        .def_readwrite("source", &resolve::ColumnSpec::source)
        .def_readwrite("dim", &resolve::ColumnSpec::dim)
        .def_readwrite("top_k", &resolve::ColumnSpec::top_k)
        .def_readwrite("bottom_k", &resolve::ColumnSpec::bottom_k)
        .def_readwrite("rank_by", &resolve::ColumnSpec::rank_by)
        .def("n_slots", &resolve::ColumnSpec::n_slots)
        .def("has_selection", &resolve::ColumnSpec::has_selection);

    py::class_<resolve::PlotRecord>(m, "PlotRecord")
        .def(py::init<>())
        .def_readwrite("plot_id", &resolve::PlotRecord::plot_id)
        .def_readwrite("categorical", &resolve::PlotRecord::categorical)
        .def_readwrite("numeric", &resolve::PlotRecord::numeric);

    py::class_<resolve::ObservationRecord>(m, "ObservationRecord")
        .def(py::init<>())
        .def_readwrite("plot_id", &resolve::ObservationRecord::plot_id)
        .def_readwrite("categorical", &resolve::ObservationRecord::categorical)
        .def_readwrite("numeric", &resolve::ObservationRecord::numeric);

    py::class_<resolve::EncodedColumn>(m, "EncodedColumn")
        .def(py::init<>())
        .def_readonly("name", &resolve::EncodedColumn::name)
        .def_readonly("values", &resolve::EncodedColumn::values)
        .def_readonly("is_embedding_ids", &resolve::EncodedColumn::is_embedding_ids)
        .def_readonly("vocab_size", &resolve::EncodedColumn::vocab_size)
        .def_readonly("embed_dim", &resolve::EncodedColumn::embed_dim)
        .def_readonly("n_slots", &resolve::EncodedColumn::n_slots);

    py::class_<resolve::EncodedPlotData>(m, "EncodedPlotData")
        .def(py::init<>())
        .def_readonly("columns", &resolve::EncodedPlotData::columns)
        .def_readonly("unknown_fraction", &resolve::EncodedPlotData::unknown_fraction)
        .def_readonly("plot_ids", &resolve::EncodedPlotData::plot_ids)
        .def("continuous_features", &resolve::EncodedPlotData::continuous_features)
        .def("embedding_ids", &resolve::EncodedPlotData::embedding_ids)
        .def("embedding_specs", &resolve::EncodedPlotData::embedding_specs);

    py::class_<resolve::PlotEncoder>(m, "PlotEncoder")
        .def(py::init<>())
        .def("add_numeric", &resolve::PlotEncoder::add_numeric,
            py::arg("name"),
            py::arg("columns"),
            py::arg("source") = resolve::DataSource::Plot)
        .def("add_raw", &resolve::PlotEncoder::add_raw,
            py::arg("name"),
            py::arg("columns"),
            py::arg("source") = resolve::DataSource::Plot)
        .def("add_hash", &resolve::PlotEncoder::add_hash,
            py::arg("name"),
            py::arg("columns"),
            py::arg("dim") = 32,
            py::arg("top_k") = 0,
            py::arg("bottom_k") = 0,
            py::arg("rank_by") = "",
            py::arg("source") = resolve::DataSource::Observation)
        .def("add_embed", &resolve::PlotEncoder::add_embed,
            py::arg("name"),
            py::arg("columns"),
            py::arg("dim") = 16,
            py::arg("top_k") = 0,
            py::arg("bottom_k") = 0,
            py::arg("rank_by") = "",
            py::arg("source") = resolve::DataSource::Observation)
        .def("add_onehot", &resolve::PlotEncoder::add_onehot,
            py::arg("name"),
            py::arg("columns"),
            py::arg("source") = resolve::DataSource::Plot)
        .def("fit", &resolve::PlotEncoder::fit,
            py::arg("plot_data"),
            py::arg("obs_data") = std::vector<resolve::ObservationRecord>{})
        .def("transform", &resolve::PlotEncoder::transform,
            py::arg("plot_data"),
            py::arg("obs_data"),
            py::arg("plot_ids"))
        .def("fit_transform", &resolve::PlotEncoder::fit_transform,
            py::arg("plot_data"),
            py::arg("obs_data"),
            py::arg("plot_ids"))
        .def("specs", &resolve::PlotEncoder::specs)
        .def("is_fitted", &resolve::PlotEncoder::is_fitted)
        .def("vocab_size", &resolve::PlotEncoder::vocab_size)
        .def("continuous_dim", &resolve::PlotEncoder::continuous_dim)
        .def("embedding_configs", &resolve::PlotEncoder::embedding_configs)
        .def("save", &resolve::PlotEncoder::save)
        .def_static("load", &resolve::PlotEncoder::load);

    // ========================================================================
    // Loss
    // ========================================================================

    py::class_<resolve::PhaseConfig>(m, "PhaseConfig")
        .def(py::init<>())
        .def(py::init([](float mae, float mse, float huber, float smape, float band) {
            return resolve::PhaseConfig{mae, mse, huber, smape, band};
        }), py::arg("mae") = 0.0f, py::arg("mse") = 0.0f,
            py::arg("huber") = 0.0f, py::arg("smape") = 0.0f, py::arg("band") = 0.0f)
        .def_readwrite("mae", &resolve::PhaseConfig::mae)
        .def_readwrite("mse", &resolve::PhaseConfig::mse)
        .def_readwrite("huber", &resolve::PhaseConfig::huber)
        .def_readwrite("smape", &resolve::PhaseConfig::smape)
        .def_readwrite("band", &resolve::PhaseConfig::band)
        .def_readwrite("huber_delta", &resolve::PhaseConfig::huber_delta)
        .def_readwrite("band_threshold", &resolve::PhaseConfig::band_threshold)
        .def("is_valid", &resolve::PhaseConfig::is_valid)
        .def("needs_original_scale", &resolve::PhaseConfig::needs_original_scale)
        .def_static("mae_only", &resolve::PhaseConfig::mae_only)
        .def_static("combined", &resolve::PhaseConfig::combined,
            py::arg("mae_w") = 0.7f, py::arg("smape_w") = 0.2f, py::arg("band_w") = 0.1f);

    py::class_<resolve::Metrics>(m, "Metrics")
        .def_static("band_accuracy", &resolve::Metrics::band_accuracy,
            py::arg("pred"), py::arg("target"), py::arg("threshold") = 0.25f)
        .def_static("mae", &resolve::Metrics::mae)
        .def_static("rmse", &resolve::Metrics::rmse)
        .def_static("smape", &resolve::Metrics::smape,
            py::arg("pred"), py::arg("target"), py::arg("eps") = 1e-8f)
        .def_static("accuracy", &resolve::Metrics::accuracy)
        .def_static("compute_regression", &resolve::Metrics::compute_regression)
        .def_static("compute_classification", &resolve::Metrics::compute_classification);

    // ========================================================================
    // Model
    // ========================================================================

    py::class_<resolve::ResolveModel>(m, "ResolveModel")
        .def(py::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
            py::arg("schema"), py::arg("config") = resolve::ModelConfig{})
        .def("forward", &resolve::ResolveModelImpl::forward,
            py::arg("continuous"),
            py::arg("genus_ids") = torch::Tensor(),
            py::arg("family_ids") = torch::Tensor(),
            py::arg("species_ids") = torch::Tensor(),
            py::arg("species_vector") = torch::Tensor())
        .def("forward_single", &resolve::ResolveModelImpl::forward_single)
        .def("get_latent", &resolve::ResolveModelImpl::get_latent)
        .def("latent_dim", &resolve::ResolveModelImpl::latent_dim)
        .def("species_encoding", &resolve::ResolveModelImpl::species_encoding)
        .def("uses_explicit_vector", &resolve::ResolveModelImpl::uses_explicit_vector);

    // ========================================================================
    // Trainer
    // ========================================================================

    py::class_<resolve::Trainer>(m, "Trainer")
        .def(py::init<
            resolve::SpeciesEncodingMode,
            int, int,
            resolve::SelectionMode,
            resolve::NormalizationMode,
            std::vector<int64_t>,
            int, int, float,
            int, int, int,
            float, float, float,
            const std::string&,
            std::optional<std::unordered_map<int, resolve::PhaseConfig>>,
            std::optional<std::vector<int>>,
            const std::string&,
            int, bool,
            torch::Device
        >(),
            py::arg("species_encoding") = resolve::SpeciesEncodingMode::Hash,
            py::arg("hash_dim") = 32,
            py::arg("top_k") = 5,
            py::arg("selection") = resolve::SelectionMode::Top,
            py::arg("normalization") = resolve::NormalizationMode::Norm,
            py::arg("hidden_dims") = std::vector<int64_t>{2048, 1024, 512, 256, 128, 64},
            py::arg("genus_emb_dim") = 8,
            py::arg("family_emb_dim") = 8,
            py::arg("dropout") = 0.3f,
            py::arg("batch_size") = 4096,
            py::arg("max_epochs") = 500,
            py::arg("patience") = 50,
            py::arg("lr") = 1e-3f,
            py::arg("weight_decay") = 1e-4f,
            py::arg("max_grad_norm") = 1.0f,
            py::arg("loss_preset") = "mae",
            py::arg("custom_phases") = std::nullopt,
            py::arg("phase_boundaries") = std::nullopt,
            py::arg("checkpoint_dir") = "",
            py::arg("checkpoint_every") = 50,
            py::arg("resume") = true,
            py::arg("device") = torch::kCPU
        )
        // .def("fit", &resolve::Trainer::fit)  // TODO: Complete implementation
        // .def("predict", &resolve::Trainer::predict)
        .def("save", &resolve::Trainer::save)
        .def("load", &resolve::Trainer::load)
        .def("n_params", &resolve::Trainer::n_params)
        .def("is_fitted", &resolve::Trainer::is_fitted);

    // ========================================================================
    // Helper functions
    // ========================================================================

    m.def("numpy_to_tensor", &numpy_to_tensor, "Convert numpy array to torch tensor");
    m.def("tensor_to_numpy", &tensor_to_numpy, "Convert torch tensor to numpy array");

    // ========================================================================
    // Version info
    // ========================================================================

    m.attr("__version__") = "0.1.0";
}
