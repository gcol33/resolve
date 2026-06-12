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
                               nb::object categorical_ids_obj,
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
                            unpack_or_empty(categorical_ids_obj),
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
           nb::arg("categorical_ids") = nb::none(),
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
        .def_static("load", [](const std::string& path, const std::string& device, float vram_fraction) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return resolve::Trainer::load(path, dev, vram_fraction);
        }, nb::arg("path"), nb::arg("device") = "cpu", nb::arg("vram_fraction") = 1.0f)
        .def_static("load_train_config", &resolve::Trainer::load_train_config,
                    nb::arg("path"),
                    "Recover the persisted TrainConfig from a checkpoint (training "
                    "hyperparameters; fields not persisted keep TrainConfig defaults).")
        .def_static("load_run_metadata", &resolve::Trainer::load_run_metadata,
                    nb::arg("path"),
                    "Recover the persisted RunMetadata (timing, plot counts, best "
                    "epoch, and the per-target final-metric tree) from a checkpoint.")
        .def_prop_ro("model", [](resolve::Trainer& self) -> resolve::ResolveModel { return self.model(); })
        .def_prop_ro("scalers", [](const resolve::Trainer& self) { return resolve::Scalers(self.scalers()); })
        .def_prop_ro("config", [](const resolve::Trainer& self) { return resolve::TrainConfig(self.config()); })
        .def_prop_ro("categorical_vocab",
                     [](const resolve::Trainer& self) -> const resolve::CategoricalVocab& {
                         return self.categorical_vocab();
                     })
        .def("compute_diagnostics", &resolve::Trainer::compute_diagnostics,
             "Compute network health diagnostics (dead neurons, saturation, etc.)")
        .def("compute_calibration", &resolve::Trainer::compute_calibration,
             nb::arg("target_name"),
             nb::arg("n_bins") = 10,
             "Compute calibration curve for a classification target")
        .def("compute_residuals", &resolve::Trainer::compute_residuals,
             nb::arg("target_name"),
             "Compute residual analysis for a regression target")
        .def("compute_classification_predictions",
             &resolve::Trainer::compute_classification_predictions,
             nb::arg("target_name"),
             "Per-plot test-fold predictions for a classification target "
             "(predicted_classes, probabilities, actuals).")
        .def("load_state", [](resolve::Trainer& self, const std::string& path,
                              const std::string& device, float vram_fraction) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            self.load_state(path, dev, vram_fraction);
        }, nb::arg("path"), nb::arg("device") = "cpu", nb::arg("vram_fraction") = 1.0f,
           "Load checkpoint weights, scalers, and categorical vocab into this "
           "trainer in place (first-class replacement for the static load()).")
        .def("test_indices", [](const resolve::Trainer& self) {
            auto t = self.test_indices();
            if (t.defined()) {
                auto cpu_tensor = t.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        }, "Global plot indices of the held-out test fold (int64).")
        .def("train_indices", [](const resolve::Trainer& self) {
            auto t = self.train_indices();
            if (t.defined()) {
                auto cpu_tensor = t.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::steal(Py_None);
        }, "Global plot indices of the training fold (int64).")
        .def("test_plot_ids", &resolve::Trainer::test_plot_ids,
             "Plot IDs of the held-out test fold (requires prepare_data(dataset)).")
        .def("train_plot_ids", &resolve::Trainer::train_plot_ids,
             "Plot IDs of the training fold (requires prepare_data(dataset)).")
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
                          nb::object pool_has_cover_obj,
                          nb::object categorical_ids_obj) {
            auto result = self.predict(unpack_or_empty(continuous_obj),
                                       unpack_or_empty(genus_ids_obj),
                                       unpack_or_empty(family_ids_obj),
                                       unpack_or_empty(species_ids_obj),
                                       unpack_or_empty(species_vector_obj),
                                       unpack_or_empty(pool_genus_ids_obj),
                                       unpack_or_empty(pool_family_ids_obj),
                                       unpack_or_empty(pool_weights_obj),
                                       unpack_or_empty(pool_mask_obj),
                                       unpack_or_empty(pool_has_cover_obj),
                                       unpack_or_empty(categorical_ids_obj));
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
           nb::arg("categorical_ids") = nb::none(),
           "Predict on data (runs model in eval mode)");

    nb::class_<resolve::Predictor>(m, "Predictor")
        .def("__init__", [](resolve::Predictor* self, resolve::ResolveModel model, resolve::Scalers scalers, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            new (self) resolve::Predictor(model, scalers, dev);
        }, nb::arg("model"), nb::arg("scalers"), nb::arg("device") = "cpu")
        .def_static("load", [](const std::string& path, const std::string& device, float vram_fraction) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return resolve::Predictor::load(path, dev, vram_fraction);
        }, nb::arg("path"), nb::arg("device") = "cpu", nb::arg("vram_fraction") = 1.0f)
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
                          nb::object categorical_ids_obj,
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
                                unpack_or_empty(categorical_ids_obj),
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
           nb::arg("categorical_ids") = nb::none(),
           nb::arg("return_latent") = false)
        .def("predict_dataset", [](resolve::Predictor& self,
                                   const resolve::ResolveDataset& dataset,
                                   bool return_latent,
                                   int64_t batch_size) {
            return self.predict(dataset, return_latent, batch_size);
        }, nb::arg("dataset"),
           nb::arg("return_latent") = false,
           nb::arg("batch_size") = 4096,
           "Predict on a ResolveDataset. batch_size controls how the forward "
           "pass is chunked along dim 0: -1 = single forward over the whole "
           "dataset (legacy, can OOM on >150k plots at typical hidden sizes); "
           ">0 = chunked forward with results concatenated on CPU. Default "
           "4096 keeps peak VRAM bounded on 16 GiB-class GPUs.")
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
        .def_prop_ro("scalers", [](resolve::Predictor& self) -> const resolve::Scalers& { return self.scalers(); })
        .def_prop_ro("categorical_vocab",
                     [](const resolve::Predictor& self) -> const resolve::CategoricalVocab& {
                         return self.categorical_vocab();
                     });
}
