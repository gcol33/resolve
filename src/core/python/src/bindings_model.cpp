#include "bindings_common.hpp"
#include <map>
#include <set>
#include <torch/csrc/autograd/python_variable.h>  // For THPVariable_WrapList

void register_model(nb::module_& m) {
    nb::class_<resolve::ResolveModel>(m, "ResolveModel")
        .def(nb::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
             nb::arg("schema"), nb::arg("config") = resolve::ModelConfig{})
        .def("forward", [](resolve::ResolveModel& self,
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
            // Convert Python tensors to C++ tensors using THPVariable_Unpack
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            const at::Tensor& species_ids = THPVariable_Unpack(species_ids_obj.ptr());
            const at::Tensor& species_vector = THPVariable_Unpack(species_vector_obj.ptr());
            at::Tensor pool_genus_ids = pool_genus_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_genus_ids_obj.ptr());
            at::Tensor pool_family_ids = pool_family_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_family_ids_obj.ptr());
            at::Tensor pool_weights = pool_weights_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_weights_obj.ptr());
            at::Tensor pool_mask = pool_mask_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_mask_obj.ptr());
            at::Tensor pool_has_cover = pool_has_cover_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_has_cover_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());

            auto result = self->forward(continuous, genus_ids, family_ids, species_ids, species_vector,
                                         pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                         categorical_ids);

            // Convert output tensors to Python using THPVariable_Wrap
            PyObject* py_dict = PyDict_New();
            for (const auto& [key, value] : result) {
                PyObject* py_tensor = THPVariable_Wrap(value);
                PyDict_SetItemString(py_dict, key.c_str(), py_tensor);
                Py_DECREF(py_tensor);
            }
            return nb::steal(py_dict);
        }, nb::arg("continuous"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("species_ids"),
           nb::arg("species_vector"),
           nb::arg("pool_genus_ids") = nb::none(),
           nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(),
           nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           nb::arg("categorical_ids") = nb::none())
        // __call__ makes model(inputs) work like PyTorch
        .def("__call__", [](resolve::ResolveModel& self,
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
            // Convert Python tensors to C++ tensors using THPVariable_Unpack
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            const at::Tensor& species_ids = THPVariable_Unpack(species_ids_obj.ptr());
            const at::Tensor& species_vector = THPVariable_Unpack(species_vector_obj.ptr());
            at::Tensor pool_genus_ids = pool_genus_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_genus_ids_obj.ptr());
            at::Tensor pool_family_ids = pool_family_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_family_ids_obj.ptr());
            at::Tensor pool_weights = pool_weights_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_weights_obj.ptr());
            at::Tensor pool_mask = pool_mask_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_mask_obj.ptr());
            at::Tensor pool_has_cover = pool_has_cover_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_has_cover_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());

            auto result = self->forward(continuous, genus_ids, family_ids, species_ids, species_vector,
                                         pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                         categorical_ids);

            // Convert output tensors to Python using THPVariable_Wrap
            PyObject* py_dict = PyDict_New();
            for (const auto& [key, value] : result) {
                PyObject* py_tensor = THPVariable_Wrap(value);
                PyDict_SetItemString(py_dict, key.c_str(), py_tensor);
                Py_DECREF(py_tensor);
            }
            return nb::steal(py_dict);
        }, nb::arg("continuous"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("species_ids"),
           nb::arg("species_vector"),
           nb::arg("pool_genus_ids") = nb::none(),
           nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(),
           nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           nb::arg("categorical_ids") = nb::none())
        .def("get_latent", [](resolve::ResolveModel& self,
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
            // Convert Python tensors to C++ tensors
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            const at::Tensor& species_ids = THPVariable_Unpack(species_ids_obj.ptr());
            const at::Tensor& species_vector = THPVariable_Unpack(species_vector_obj.ptr());
            at::Tensor pool_genus_ids = pool_genus_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_genus_ids_obj.ptr());
            at::Tensor pool_family_ids = pool_family_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_family_ids_obj.ptr());
            at::Tensor pool_weights = pool_weights_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_weights_obj.ptr());
            at::Tensor pool_mask = pool_mask_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_mask_obj.ptr());
            at::Tensor pool_has_cover = pool_has_cover_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_has_cover_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());

            at::Tensor result = self->get_latent(continuous, genus_ids, family_ids, species_ids, species_vector,
                                                  pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                                  categorical_ids);
            return nb::steal(THPVariable_Wrap(result));
        }, nb::arg("continuous"),
           nb::arg("genus_ids"),
           nb::arg("family_ids"),
           nb::arg("species_ids"),
           nb::arg("species_vector"),
           nb::arg("pool_genus_ids") = nb::none(),
           nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(),
           nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           nb::arg("categorical_ids") = nb::none())
        .def("train", [](resolve::ResolveModel& self, bool mode) { self->train(mode); }, nb::arg("mode") = true)
        .def("eval", [](resolve::ResolveModel& self) { self->eval(); })
        .def("to", [](resolve::ResolveModel& self, const std::string& device) {
            if (device == "cuda") {
                self->to(torch::kCUDA);
            } else {
                self->to(torch::kCPU);
            }
        })
        .def_prop_ro("schema", [](const resolve::ResolveModel& self) -> resolve::ResolveSchema { return self->schema(); })
        .def_prop_ro("config", [](const resolve::ResolveModel& self) -> resolve::ModelConfig { return self->config(); })
        .def_prop_ro("latent_dim", [](resolve::ResolveModel& self) { return self->latent_dim(); })
        .def_prop_ro("species_encoding", [](resolve::ResolveModel& self) { return self->species_encoding(); })
        .def_prop_ro("uses_explicit_vector", [](resolve::ResolveModel& self) { return self->uses_explicit_vector(); })
        // PyTorch-compatible parameter access
        .def("parameters", [](resolve::ResolveModel& self) {
            // Return list of tensors with requires_grad=True for optimizer compatibility
            // Use PyTorch's THPVariable_WrapList to convert tensors to Python
            std::vector<at::Tensor> params_vec;
            for (const auto& p : self->parameters()) {
                params_vec.push_back(p);
            }
            PyObject* py_list = THPVariable_WrapList(params_vec);
            return nb::steal(py_list);
        })
        .def("named_parameters", [](resolve::ResolveModel& self) {
            // Return dict of name -> tensor for PyTorch compatibility
            // Use THPVariable_Wrap to convert tensors to Python objects
            PyObject* py_dict = PyDict_New();
            for (const auto& pair : self->named_parameters()) {
                PyObject* py_tensor = THPVariable_Wrap(pair.value());
                PyDict_SetItemString(py_dict, pair.key().c_str(), py_tensor);
                Py_DECREF(py_tensor);  // PyDict_SetItemString increments refcount
            }
            return nb::steal(py_dict);
        })
        .def("state_dict", [](resolve::ResolveModel& self) {
            // Return ordered dict of all parameters and buffers
            // Use THPVariable_Wrap to convert tensors to Python objects
            PyObject* py_dict = PyDict_New();
            // Parameters
            for (const auto& pair : self->named_parameters()) {
                PyObject* py_tensor = THPVariable_Wrap(pair.value().clone());
                PyDict_SetItemString(py_dict, pair.key().c_str(), py_tensor);
                Py_DECREF(py_tensor);
            }
            // Buffers (e.g., BatchNorm running mean/var)
            for (const auto& pair : self->named_buffers()) {
                PyObject* py_tensor = THPVariable_Wrap(pair.value().clone());
                PyDict_SetItemString(py_dict, pair.key().c_str(), py_tensor);
                Py_DECREF(py_tensor);
            }
            return nb::steal(py_dict);
        })
        .def("load_state_dict", [](resolve::ResolveModel& self, nb::object state_dict_obj, bool strict) {
            // Load parameters from dict
            // Use Python C API to work with tensors since nanobind doesn't have torch::Tensor caster
            torch::NoGradGuard no_grad;

            std::vector<std::string> missing_keys;
            std::vector<std::string> unexpected_keys;

            PyObject* state_dict = state_dict_obj.ptr();
            if (!PyDict_Check(state_dict)) {
                throw std::runtime_error("state_dict must be a dict");
            }

            // Build set of model parameter/buffer names
            std::set<std::string> model_keys;
            for (const auto& pair : self->named_parameters()) {
                model_keys.insert(pair.key());
            }
            for (const auto& pair : self->named_buffers()) {
                model_keys.insert(pair.key());
            }

            // Track which keys from state_dict were used
            std::set<std::string> used_keys;

            // Load parameters
            for (const auto& pair : self->named_parameters()) {
                std::string key = pair.key();
                PyObject* py_tensor = PyDict_GetItemString(state_dict, key.c_str());
                if (py_tensor != nullptr) {
                    if (THPVariable_Check(py_tensor)) {
                        const at::Tensor& src = THPVariable_Unpack(py_tensor);
                        pair.value().copy_(src);
                        used_keys.insert(key);
                    }
                } else if (strict) {
                    missing_keys.push_back(key);
                }
            }

            // Load buffers
            for (const auto& pair : self->named_buffers()) {
                std::string key = pair.key();
                PyObject* py_tensor = PyDict_GetItemString(state_dict, key.c_str());
                if (py_tensor != nullptr) {
                    if (THPVariable_Check(py_tensor)) {
                        const at::Tensor& src = THPVariable_Unpack(py_tensor);
                        pair.value().copy_(src);
                        used_keys.insert(key);
                    }
                } else if (strict) {
                    missing_keys.push_back(key);
                }
            }

            // Check for unexpected keys
            if (strict) {
                PyObject *py_key, *py_value;
                Py_ssize_t pos = 0;
                while (PyDict_Next(state_dict, &pos, &py_key, &py_value)) {
                    const char* key_cstr = PyUnicode_AsUTF8(py_key);
                    if (key_cstr) {
                        std::string key(key_cstr);
                        if (used_keys.find(key) == used_keys.end()) {
                            unexpected_keys.push_back(key);
                        }
                    }
                }
            }

            // Return tuple of (missing_keys, unexpected_keys) like PyTorch
            return std::make_pair(missing_keys, unexpected_keys);
        }, nb::arg("state_dict"), nb::arg("strict") = true)
        .def("n_parameters", [](resolve::ResolveModel& self) {
            int64_t count = 0;
            for (const auto& p : self->parameters()) {
                count += p.numel();
            }
            return count;
        })
        .def("zero_grad", [](resolve::ResolveModel& self) {
            // Zero all gradients - useful for custom training loops
            self->zero_grad();
        })
        .def("requires_grad_", [](resolve::ResolveModel& self, bool requires_grad) {
            // Set requires_grad for all parameters
            for (auto& p : self->parameters()) {
                p.set_requires_grad(requires_grad);
            }
        }, nb::arg("requires_grad") = true)
        // Embedding weight extraction
        .def("get_genus_weights", [](resolve::ResolveModel& self) {
            auto t = self->get_genus_weights();
            if (!t.defined()) return nb::steal(nb::none().release().ptr());
            return nb::steal(THPVariable_Wrap(t));
        })
        .def("get_family_weights", [](resolve::ResolveModel& self) {
            auto t = self->get_family_weights();
            if (!t.defined()) return nb::steal(nb::none().release().ptr());
            return nb::steal(THPVariable_Wrap(t));
        })
        .def("get_species_weights", [](resolve::ResolveModel& self) {
            auto t = self->get_species_weights();
            if (!t.defined()) return nb::steal(nb::none().release().ptr());
            return nb::steal(THPVariable_Wrap(t));
        })
        .def("forward_with_aux", [](resolve::ResolveModel& self,
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
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            const at::Tensor& species_ids = THPVariable_Unpack(species_ids_obj.ptr());
            const at::Tensor& species_vector = THPVariable_Unpack(species_vector_obj.ptr());
            at::Tensor pool_genus_ids = pool_genus_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_genus_ids_obj.ptr());
            at::Tensor pool_family_ids = pool_family_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_family_ids_obj.ptr());
            at::Tensor pool_weights = pool_weights_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_weights_obj.ptr());
            at::Tensor pool_mask = pool_mask_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_mask_obj.ptr());
            at::Tensor pool_has_cover = pool_has_cover_obj.is_none() ? at::Tensor() : THPVariable_Unpack(pool_has_cover_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());

            auto result = self->forward_with_aux(continuous, genus_ids, family_ids, species_ids, species_vector,
                                                  pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                                  categorical_ids);
            return result;  // ModelForwardResult is already bound
        }, nb::arg("continuous"), nb::arg("genus_ids"), nb::arg("family_ids"),
           nb::arg("species_ids"), nb::arg("species_vector"),
           nb::arg("pool_genus_ids") = nb::none(), nb::arg("pool_family_ids") = nb::none(),
           nb::arg("pool_weights") = nb::none(), nb::arg("pool_mask") = nb::none(),
           nb::arg("pool_has_cover") = nb::none(),
           nb::arg("categorical_ids") = nb::none())
        .def("forward_single", [](resolve::ResolveModel& self,
                                   const std::string& target,
                                   nb::object continuous_obj,
                                   nb::object genus_ids_obj,
                                   nb::object family_ids_obj,
                                   nb::object species_ids_obj,
                                   nb::object species_vector_obj,
                                   nb::object categorical_ids_obj) {
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            const at::Tensor& species_ids = THPVariable_Unpack(species_ids_obj.ptr());
            const at::Tensor& species_vector = THPVariable_Unpack(species_vector_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());
            auto result = self->forward_single(target, continuous, genus_ids, family_ids, species_ids, species_vector,
                                                categorical_ids);
            return nb::steal(THPVariable_Wrap(result));
        }, nb::arg("target"), nb::arg("continuous"), nb::arg("genus_ids"),
           nb::arg("family_ids"), nb::arg("species_ids"), nb::arg("species_vector"),
           nb::arg("categorical_ids") = nb::none())
        .def("encode_with_activations", [](resolve::ResolveModel& self,
                                            nb::object continuous_obj,
                                            nb::object genus_ids_obj,
                                            nb::object family_ids_obj,
                                            nb::object categorical_ids_obj) {
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            at::Tensor categorical_ids = categorical_ids_obj.is_none() ? at::Tensor() : THPVariable_Unpack(categorical_ids_obj.ptr());
            auto [latent, activations] = self->encode_with_activations(continuous, genus_ids, family_ids, categorical_ids);
            nb::list act_list;
            for (const auto& a : activations) {
                act_list.append(nb::steal(THPVariable_Wrap(a)));
            }
            return std::make_pair(nb::steal(THPVariable_Wrap(latent)), act_list);
        }, nb::arg("continuous"), nb::arg("genus_ids"), nb::arg("family_ids"),
           nb::arg("categorical_ids") = nb::none())
        .def("get_gate_probs", [](resolve::ResolveModel& self,
                                   nb::object continuous_obj,
                                   nb::object genus_ids_obj,
                                   nb::object family_ids_obj) {
            const at::Tensor& continuous = THPVariable_Unpack(continuous_obj.ptr());
            const at::Tensor& genus_ids = THPVariable_Unpack(genus_ids_obj.ptr());
            const at::Tensor& family_ids = THPVariable_Unpack(family_ids_obj.ptr());
            auto result = self->get_gate_probs(continuous, genus_ids, family_ids);
            return nb::steal(THPVariable_Wrap(result));
        }, nb::arg("continuous"), nb::arg("genus_ids"), nb::arg("family_ids"))
        .def_prop_ro("uses_moe", [](resolve::ResolveModel& self) { return self->uses_moe(); })
        .def_prop_ro("n_experts", [](resolve::ResolveModel& self) { return self->n_experts(); })
        .def("set_traits", [](resolve::ResolveModel& self, nb::object traits_obj) {
            const at::Tensor& traits = THPVariable_Unpack(traits_obj.ptr());
            self->set_traits(traits);
        }, nb::arg("traits"));

    m.attr("SpaccModel") = m.attr("ResolveModel");
}
