#include "bindings_common.hpp"
#include "bindings_plot_inputs.hpp"

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
           nb::call_guard<nb::gil_scoped_release>(),
           "Prepare data from a ResolveDataset, holding out a seeded random "
           "test_size fraction of the plots for early stopping.")
        .def("prepare_data", [](resolve::Trainer& self,
                               const resolve::ResolveDataset& dataset,
                               const std::vector<int64_t>& stopping_indices,
                               int seed) {
            self.prepare_data(dataset, stopping_indices, seed);
        }, nb::arg("dataset"),
           nb::arg("stopping_indices"),
           nb::arg("seed") = 42,
           nb::call_guard<nb::gil_scoped_release>(),
           "Prepare data from a ResolveDataset, holding out the plots at "
           "stopping_indices (0-based positions in dataset.plot_ids; "
           "dataset.plot_indices maps plot IDs to them) for early stopping. "
           "Early stopping, best_epoch, final_metrics, the baselines and "
           "test_indices() read exactly these plots, in this order; every other "
           "plot is a fitting row. seed drives the per-epoch shuffle. Raises "
           "ValueError for an empty fold, an index out of range, a repeated "
           "index, a fold of every plot, or a fold with no labelled plot for a "
           "masked target.")
        .def("prepare_data_raw", [](resolve::Trainer& self, nb::args args, nb::kwargs kwargs) {
            CallArgs call("prepare_data_raw", args, kwargs, {});
            const auto parts = call.continuous_parts();
            const auto inputs = call.plot_inputs(/*with_continuous=*/false);
            const auto targets = dict_to_tensor_map(call.require<nb::dict>("targets"));
            const float test_size = call.take<float>("test_size", 0.2f);
            const int seed = call.take<int>("seed", 42);
            call.finish();
            nb::gil_scoped_release nogil;
            self.prepare_data(parts, inputs, targets, test_size, seed);
        }, persistent_doc(
               "prepare_data_raw(*, targets, " +
               argument_list(continuous_part_names()) + ", " +
               argument_list(species_input_names()) + ", test_size=0.2, seed=42)\n\n"
               "Prepare data from tensors, all passed by keyword: the parts of the "
               "continuous block, the other model inputs, and targets "
               "(dict name -> (n_plots,) tensor). A seeded random test_size "
               "fraction is held out."))
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
        .def_prop_ro("run_metadata",
                     [](const resolve::Trainer& self) { return resolve::RunMetadata(self.run_metadata()); },
                     "The run as a checkpoint records it: the split prepare_data "
                     "made (n_plots_train / n_plots_test, hold_out, test_size, "
                     "split_seed) and, after fit(), its timing, epochs and final "
                     "metrics. Pass it to save() to record it in the checkpoint.")
        .def_prop_ro("effective_batch_size", &resolve::Trainer::effective_batch_size,
                     "Batch size the last fit() actually trained at. Equal to "
                     "config.batch_size on a clean run, smaller when the CUDA "
                     "auto-halve-on-OOM retry fired; 0 before the first fit().")
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
            return nb::none();
        }, "Global plot indices of the held-out test fold (int64).")
        .def("train_indices", [](const resolve::Trainer& self) {
            auto t = self.train_indices();
            if (t.defined()) {
                auto cpu_tensor = t.detach().cpu().contiguous();
                return nb::steal(THPVariable_Wrap(cpu_tensor));
            }
            return nb::none();
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
        .def("predict", [](resolve::Trainer& self, nb::args args, nb::kwargs kwargs) {
            CallArgs call("predict", args, kwargs, plot_input_names());
            const auto inputs = call.plot_inputs(/*with_continuous=*/true);
            call.finish();
            resolve::TensorMap result;
            {
                // Drop the GIL around the eval-mode forward (no Python C-API here).
                nb::gil_scoped_release nogil;
                result = self.predict(inputs);
            }
            return tensor_map_to_dict(result);
        }, persistent_doc(
               "predict(" + argument_list(plot_input_names(), 1) + ") -> dict[str, Tensor]\n\n"
               "Predict in eval mode from the raw continuous block (standardised "
               "here with the fitted scalers) and the other model inputs."));

    nb::class_<resolve::Predictor>(m, "Predictor")
        .def("__init__", [](resolve::Predictor* self, resolve::ResolveModel model, resolve::Scalers scalers, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            new (self) resolve::Predictor(model, scalers, dev);
        }, nb::arg("model"), nb::arg("scalers"), nb::arg("device") = "cpu")
        .def_static("load", [](const std::string& path, const std::string& device, float vram_fraction) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return resolve::Predictor::load(path, dev, vram_fraction);
        }, nb::arg("path"), nb::arg("device") = "cpu", nb::arg("vram_fraction") = 1.0f)
        .def("predict", [](resolve::Predictor& self, nb::args args, nb::kwargs kwargs) {
            CallArgs call("predict", args, kwargs, {});
            const auto parts = call.continuous_parts();
            const auto inputs = call.plot_inputs(/*with_continuous=*/false);
            const bool return_latent = call.take<bool>("return_latent", false);
            const int64_t batch_size = call.take<int64_t>("batch_size", 4096);
            call.finish();
            resolve::ResolvePredictions result;
            {
                // Inference over many plots can take seconds (especially on CPU);
                // release the GIL around the pure-C++ predict so concurrent
                // Python threads are not blocked. No Python C-API runs here.
                nb::gil_scoped_release nogil;
                result = self.predict(parts, inputs, return_latent, batch_size);
            }
            return result;
        }, persistent_doc(
               "predict(*, " + argument_list(continuous_part_names()) + ", " +
               argument_list(species_input_names()) +
               ", return_latent=False, batch_size=4096) -> ResolvePredictions\n\n"
               "Predict from tensors, all passed by keyword: the parts of the "
               "continuous block and the other model inputs. batch_size chunks the "
               "forward as in predict_dataset; every output is on the CPU."))
        .def("predict_dataset", [](resolve::Predictor& self,
                                   const resolve::ResolveDataset& dataset,
                                   bool return_latent,
                                   int64_t batch_size) {
            return self.predict(dataset, return_latent, batch_size);
        }, nb::arg("dataset"),
           nb::arg("return_latent") = false,
           nb::arg("batch_size") = 4096,
           nb::call_guard<nb::gil_scoped_release>(),
           "Predict on a ResolveDataset. batch_size controls how the forward "
           "pass is chunked along dim 0: -1 = single forward over the whole "
           "dataset (legacy, can OOM on >150k plots at typical hidden sizes); "
           ">0 = chunked forward with results concatenated on CPU. Default "
           "4096 keeps peak VRAM bounded on 16 GiB-class GPUs.")
        .def("get_embeddings", [](resolve::Predictor& self, nb::args args, nb::kwargs kwargs) {
            CallArgs call("get_embeddings", args, kwargs, {});
            const auto parts = call.continuous_parts();
            const auto inputs = call.plot_inputs(/*with_continuous=*/false);
            const int64_t batch_size = call.take<int64_t>("batch_size", 4096);
            call.finish();
            torch::Tensor out;
            {
                nb::gil_scoped_release nogil;
                out = self.get_embeddings(parts, inputs, batch_size);
            }
            return nb::steal(THPVariable_Wrap(out));
        }, persistent_doc(
               "get_embeddings(*, " + argument_list(continuous_part_names()) + ", " +
               argument_list(species_input_names()) + ", batch_size=4096) -> Tensor\n\n"
               "Latent embedding (n_plots, latent_dim) of each plot, on the CPU, from "
               "the same keyword inputs as predict()."))
        // Move to CPU (detached) so a CUDA predictor returns a CPU tensor, like
        // the sibling predict/scalers accessors and the C-ABI predictor_get
        // (which always emits a CPU matrix). An undefined tensor (no taxonomy /
        // non-embedding encoder) wraps to None, so only .cpu() when defined.
        .def("get_genus_embeddings", [](resolve::Predictor& self) {
            auto t = self.get_genus_embeddings();
            if (t.defined()) t = t.detach().cpu();
            return nb::steal(THPVariable_Wrap(t));
        })
        .def("get_family_embeddings", [](resolve::Predictor& self) {
            auto t = self.get_family_embeddings();
            if (t.defined()) t = t.detach().cpu();
            return nb::steal(THPVariable_Wrap(t));
        })
        .def("get_species_embeddings", [](resolve::Predictor& self) {
            auto t = self.get_species_embeddings();
            if (t.defined()) t = t.detach().cpu();
            return nb::steal(THPVariable_Wrap(t));
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
                     })
        // Everything needed to build an inference dataset in this model's
        // integer-code namespace.
        //
        //   vocabs = predictor.external_vocabs
        //   cfg    = predictor.dataset_config
        //   ds     = ResolveDataset.from_csv_with_vocabs(
        //                header, species, roles, targets, vocabs, cfg)
        //   out    = predictor.predict_dataset(ds)
        //
        // predict_dataset rejects a dataset built any other way.
        .def_prop_ro("schema",
                     [](const resolve::Predictor& self) -> resolve::ResolveSchema {
                         return self.schema();
                     })
        .def_prop_ro("external_vocabs", &resolve::Predictor::external_vocabs)
        .def_prop_ro("species_vocab",
                     [](const resolve::Predictor& self) -> const std::vector<std::string>& {
                         return self.species_vocab();
                     })
        .def_prop_ro("genus_vocab",
                     [](const resolve::Predictor& self) -> const std::vector<std::string>& {
                         return self.genus_vocab();
                     })
        .def_prop_ro("family_vocab",
                     [](const resolve::Predictor& self) -> const std::vector<std::string>& {
                         return self.family_vocab();
                     })
        // The DatasetConfig this checkpoint implies, ready to pass to a loader.
        .def_prop_ro("dataset_config",
                     [](const resolve::Predictor& self) -> resolve::DatasetConfig {
                         return resolve::dataset_config_from_checkpoint(
                             self.schema(), self.model()->config());
                     });
}
