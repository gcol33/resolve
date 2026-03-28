#include "bindings_common.hpp"

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
        .def("predict", &resolve::Trainer::predict,
             nb::arg("continuous"),
             nb::arg("genus_ids") = torch::Tensor(),
             nb::arg("family_ids") = torch::Tensor(),
             nb::arg("species_ids") = torch::Tensor(),
             nb::arg("species_vector") = torch::Tensor(),
             nb::arg("pool_genus_ids") = torch::Tensor(),
             nb::arg("pool_family_ids") = torch::Tensor(),
             nb::arg("pool_weights") = torch::Tensor(),
             nb::arg("pool_mask") = torch::Tensor(),
             nb::arg("pool_has_cover") = torch::Tensor(),
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
        .def("predict",
             static_cast<resolve::ResolvePredictions (resolve::Predictor::*)(
                 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                 torch::Tensor,
                 torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                 bool)>(&resolve::Predictor::predict),
             nb::arg("coordinates"),
             nb::arg("covariates"),
             nb::arg("hash_embedding"),
             nb::arg("species_ids"),
             nb::arg("species_vector"),
             nb::arg("genus_ids"),
             nb::arg("family_ids"),
             nb::arg("unknown_fraction"),
             nb::arg("unknown_count"),
             nb::arg("pool_genus_ids") = torch::Tensor(),
             nb::arg("pool_family_ids") = torch::Tensor(),
             nb::arg("pool_weights") = torch::Tensor(),
             nb::arg("pool_mask") = torch::Tensor(),
             nb::arg("pool_has_cover") = torch::Tensor(),
             nb::arg("return_latent") = false)
        .def("predict_dataset",
             static_cast<resolve::ResolvePredictions (resolve::Predictor::*)(
                 const resolve::ResolveDataset&, bool)>(&resolve::Predictor::predict),
             nb::arg("dataset"),
             nb::arg("return_latent") = false)
        .def("get_embeddings", &resolve::Predictor::get_embeddings,
             nb::arg("coordinates"),
             nb::arg("covariates"),
             nb::arg("hash_embedding"),
             nb::arg("genus_ids"),
             nb::arg("family_ids"))
        .def("get_genus_embeddings", &resolve::Predictor::get_genus_embeddings)
        .def("get_family_embeddings", &resolve::Predictor::get_family_embeddings)
        .def("get_species_embeddings", &resolve::Predictor::get_species_embeddings)
        .def("optimize_for_inference", &resolve::Predictor::optimize_for_inference)
        .def_prop_ro("device", [](const resolve::Predictor& self) {
            return self.device().is_cuda() ? std::string("cuda") : std::string("cpu");
        })
        .def_prop_ro("model", [](resolve::Predictor& self) -> resolve::ResolveModel { return self.model(); })
        .def_prop_ro("scalers", [](resolve::Predictor& self) -> const resolve::Scalers& { return self.scalers(); });
}
