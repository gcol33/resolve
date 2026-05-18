#include "bindings_common.hpp"

namespace {

// Convert an nb::object that may be a Python tensor or None into an at::Tensor.
// None / undefined → empty at::Tensor (matches the C++ default {} semantics).
inline at::Tensor unpack_or_empty(const nb::object& obj) {
    if (!obj.is_valid() || obj.is_none()) return at::Tensor();
    return THPVariable_Unpack(obj.ptr());
}

}  // namespace

void register_trainer(nb::module_& m) {
    nb::class_<resolve::Trainer>(m, "Trainer")
        .def(nb::init<resolve::ResolveModel, const resolve::TrainConfig&>(),
             nb::arg("model"), nb::arg("config"))
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
                               nb::object coordinates_obj,
                               nb::object covariates_obj,
                               nb::object hash_embedding_obj,
                               nb::object species_ids_obj,
                               nb::object species_vector_obj,
                               nb::object genus_ids_obj,
                               nb::object family_ids_obj,
                               nb::object unknown_fraction_obj,
                               nb::object unknown_count_obj,
                               const nb::dict& targets,
                               float test_size,
                               int seed) {
            self.prepare_data(unpack_or_empty(coordinates_obj),
                            unpack_or_empty(covariates_obj),
                            unpack_or_empty(hash_embedding_obj),
                            unpack_or_empty(species_ids_obj),
                            unpack_or_empty(species_vector_obj),
                            unpack_or_empty(genus_ids_obj),
                            unpack_or_empty(family_ids_obj),
                            unpack_or_empty(unknown_fraction_obj),
                            unpack_or_empty(unknown_count_obj),
                            dict_to_tensor_map(targets),
                            /*pool_genus_ids=*/{}, /*pool_family_ids=*/{},
                            /*pool_weights=*/{}, /*pool_mask=*/{}, /*pool_has_cover=*/{},
                            test_size, seed);
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
        .def("fit", &resolve::Trainer::fit, nb::call_guard<nb::gil_scoped_release>())
        .def("save", [](const resolve::Trainer& self, const std::string& path, nb::object metadata_obj) {
            if (metadata_obj.is_none()) {
                self.save(path);
            } else {
                auto metadata = nb::cast<resolve::RunMetadata>(metadata_obj);
                self.save(path, &metadata);
            }
        }, nb::arg("path"), nb::arg("metadata") = nb::none())
        .def_static("load", [](const std::string& path, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return resolve::Trainer::load(path, dev);
        }, nb::arg("path"), nb::arg("device") = "cpu")
        .def_prop_ro("model", [](resolve::Trainer& self) -> resolve::ResolveModel { return self.model(); })
        .def_prop_ro("scalers", [](const resolve::Trainer& self) { return resolve::Scalers(self.scalers()); })
        .def_prop_ro("config", [](const resolve::Trainer& self) { return resolve::TrainConfig(self.config()); })
        .def("compute_diagnostics", &resolve::Trainer::compute_diagnostics,
             "Compute network health diagnostics (dead neurons, saturation, etc.)")
        .def("compute_calibration", &resolve::Trainer::compute_calibration,
             nb::arg("target_name"),
             nb::arg("n_bins") = 10,
             "Compute calibration curve for a classification target")
        .def("compute_residuals", &resolve::Trainer::compute_residuals,
             nb::arg("target_name"),
             "Compute residual analysis for a regression target")
        .def("cross_validate", &resolve::Trainer::cross_validate,
             nb::arg("n_folds") = 5,
             nb::arg("seed") = 42,
             nb::call_guard<nb::gil_scoped_release>(),
             "Perform k-fold cross-validation")
        .def("cross_validate_spatial", &resolve::Trainer::cross_validate_spatial,
             nb::arg("spatial_config"),
             nb::arg("n_folds") = 5,
             nb::arg("seed") = 42,
             nb::call_guard<nb::gil_scoped_release>(),
             "Perform spatial block cross-validation")
        .def("predict", [](resolve::Trainer& self,
                          nb::object continuous_obj,
                          nb::object genus_ids_obj,
                          nb::object family_ids_obj,
                          nb::object species_ids_obj,
                          nb::object species_vector_obj,
                          nb::object pool_genus_ids_obj,
                          nb::object pool_family_ids_obj,
                          nb::object pool_weights_obj,
                          nb::object pool_mask_obj,
                          nb::object pool_has_cover_obj) {
            auto result = self.predict(unpack_or_empty(continuous_obj),
                                       unpack_or_empty(genus_ids_obj),
                                       unpack_or_empty(family_ids_obj),
                                       unpack_or_empty(species_ids_obj),
                                       unpack_or_empty(species_vector_obj),
                                       unpack_or_empty(pool_genus_ids_obj),
                                       unpack_or_empty(pool_family_ids_obj),
                                       unpack_or_empty(pool_weights_obj),
                                       unpack_or_empty(pool_mask_obj),
                                       unpack_or_empty(pool_has_cover_obj));
            return tensor_map_to_dict(result);
        }, nb::arg("continuous"),
           nb::arg("genus_ids") = nb::none(),
           nb::arg("family_ids") = nb::none(),
           nb::arg("species_ids") = nb::none(),
           nb::arg("species_vector") = nb::none(),
           nb::arg("pool_genus_ids") = nb::none(),
           nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(),
           nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           "Predict on data (runs model in eval mode)");

    nb::class_<resolve::Predictor>(m, "Predictor")
        .def("__init__", [](resolve::Predictor* self, resolve::ResolveModel model, resolve::Scalers scalers, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            new (self) resolve::Predictor(model, scalers, dev);
        }, nb::arg("model"), nb::arg("scalers"), nb::arg("device") = "cpu")
        .def_static("load", [](const std::string& path, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return resolve::Predictor::load(path, dev);
        }, nb::arg("path"), nb::arg("device") = "cpu")
        .def("predict", [](resolve::Predictor& self,
                          nb::object coordinates_obj,
                          nb::object covariates_obj,
                          nb::object hash_embedding_obj,
                          nb::object species_ids_obj,
                          nb::object species_vector_obj,
                          nb::object genus_ids_obj,
                          nb::object family_ids_obj,
                          nb::object unknown_fraction_obj,
                          nb::object unknown_count_obj,
                          nb::object pool_genus_ids_obj,
                          nb::object pool_family_ids_obj,
                          nb::object pool_weights_obj,
                          nb::object pool_mask_obj,
                          nb::object pool_has_cover_obj,
                          bool return_latent) {
            return self.predict(unpack_or_empty(coordinates_obj),
                                unpack_or_empty(covariates_obj),
                                unpack_or_empty(hash_embedding_obj),
                                unpack_or_empty(species_ids_obj),
                                unpack_or_empty(species_vector_obj),
                                unpack_or_empty(genus_ids_obj),
                                unpack_or_empty(family_ids_obj),
                                unpack_or_empty(unknown_fraction_obj),
                                unpack_or_empty(unknown_count_obj),
                                unpack_or_empty(pool_genus_ids_obj),
                                unpack_or_empty(pool_family_ids_obj),
                                unpack_or_empty(pool_weights_obj),
                                unpack_or_empty(pool_mask_obj),
                                unpack_or_empty(pool_has_cover_obj),
                                return_latent);
        }, nb::arg("coordinates"),
           nb::arg("covariates"),
           nb::arg("hash_embedding"),
           nb::arg("species_ids"),
           nb::arg("species_vector"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("unknown_fraction"),
           nb::arg("unknown_count"),
           nb::arg("pool_genus_ids") = nb::none(),
           nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(),
           nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           nb::arg("return_latent") = false)
        .def("predict_dataset", [](resolve::Predictor& self,
                                   const resolve::ResolveDataset& dataset,
                                   bool return_latent) {
            return self.predict(dataset, return_latent);
        }, nb::arg("dataset"), nb::arg("return_latent") = false)
        .def("get_embeddings", [](resolve::Predictor& self,
                                  nb::object coordinates_obj,
                                  nb::object covariates_obj,
                                  nb::object hash_embedding_obj,
                                  nb::object genus_ids_obj,
                                  nb::object family_ids_obj) {
            auto out = self.get_embeddings(unpack_or_empty(coordinates_obj),
                                           unpack_or_empty(covariates_obj),
                                           unpack_or_empty(hash_embedding_obj),
                                           unpack_or_empty(genus_ids_obj),
                                           unpack_or_empty(family_ids_obj));
            return nb::steal(THPVariable_Wrap(out));
        }, nb::arg("coordinates"),
           nb::arg("covariates"),
           nb::arg("hash_embedding"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"))
        .def("get_genus_embeddings", [](resolve::Predictor& self) {
            return nb::steal(THPVariable_Wrap(self.get_genus_embeddings()));
        })
        .def("get_family_embeddings", [](resolve::Predictor& self) {
            return nb::steal(THPVariable_Wrap(self.get_family_embeddings()));
        })
        .def("get_species_embeddings", [](resolve::Predictor& self) {
            return nb::steal(THPVariable_Wrap(self.get_species_embeddings()));
        })
        .def("optimize_for_inference", &resolve::Predictor::optimize_for_inference)
        .def_prop_ro("device", [](const resolve::Predictor& self) {
            return self.device().is_cuda() ? std::string("cuda") : std::string("cpu");
        })
        .def_prop_ro("model", [](resolve::Predictor& self) -> resolve::ResolveModel { return self.model(); })
        .def_prop_ro("scalers", [](resolve::Predictor& self) -> const resolve::Scalers& { return self.scalers(); });
}
