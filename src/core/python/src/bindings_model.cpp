#include "bindings_common.hpp"
#include "bindings_plot_inputs.hpp"
#include <map>
#include <set>
#include <torch/csrc/autograd/python_variable.h>  // For THPVariable_WrapList

namespace {

// Binds `name` as a ResolveModel method taking the model inputs, positionally
// in PlotInputs field order (continuous required) or by keyword, after the
// leading arguments `lead`. `run(model, lead_args, inputs)` runs with the GIL
// released (pure C++, the inputs hold their storage) and its result goes to
// `wrap` with the GIL held.
template <typename Run, typename Wrap>
void def_model_inputs_method(nb::class_<resolve::ResolveModel>& cls, const char* name,
                             const std::vector<std::string>& lead, const std::string& returns,
                             const std::string& summary, Run run, Wrap wrap) {
    std::vector<std::string> positional = lead;
    for (const auto& field : plot_input_names()) positional.push_back(field);
    const std::string method = name;
    const std::size_t n_lead = lead.size();
    const char* doc = persistent_doc(
        method + "(" + argument_list(positional, n_lead + 1) + ") -> " + returns +
        "\n\n" + summary);
    cls.def(name, [method, positional, n_lead, run, wrap](resolve::ResolveModel& self,
                                                          nb::args args, nb::kwargs kwargs) {
        CallArgs call(method, args, kwargs, positional);
        std::vector<std::string> lead_values;
        for (std::size_t i = 0; i < n_lead; ++i) {
            lead_values.push_back(call.require<std::string>(positional[i]));
        }
        const resolve::PlotInputs inputs = call.plot_inputs(/*with_continuous=*/true);
        call.finish();
        decltype(run(self, lead_values, inputs)) result;
        {
            nb::gil_scoped_release nogil;
            result = run(self, lead_values, inputs);
        }
        return wrap(result);
    }, doc);
}

nb::object wrap_tensor(const torch::Tensor& t) { return nb::steal(THPVariable_Wrap(t)); }

nb::object wrap_tensor_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    PyObject* py_dict = PyDict_New();
    for (const auto& [key, value] : m) {
        PyObject* py_tensor = THPVariable_Wrap(value);
        PyDict_SetItemString(py_dict, key.c_str(), py_tensor);
        Py_DECREF(py_tensor);
    }
    return nb::steal(py_dict);
}

}  // namespace

void register_model(nb::module_& m) {
    using Lead = const std::vector<std::string>&;
    auto model_cls = nb::class_<resolve::ResolveModel>(m, "ResolveModel");
    model_cls.def(nb::init<const resolve::ResolveSchema&, const resolve::ModelConfig&>(),
                  nb::arg("schema"), nb::arg("config") = resolve::ModelConfig{});

    auto forward = [](resolve::ResolveModel& self, Lead, const resolve::PlotInputs& in) {
        return self->forward(in);
    };
    def_model_inputs_method(model_cls, "forward", {}, "dict[str, Tensor]",
                            "Per-target outputs: regression in standardised space, "
                            "classification as logits.",
                            forward, wrap_tensor_dict);
    // __call__ makes model(inputs) work like PyTorch
    def_model_inputs_method(model_cls, "__call__", {}, "dict[str, Tensor]",
                            "Same as forward().", forward, wrap_tensor_dict);
    def_model_inputs_method(
        model_cls, "get_latent", {}, "Tensor", "Latent representation (batch, latent_dim).",
        [](resolve::ResolveModel& self, Lead, const resolve::PlotInputs& in) {
            return self->get_latent(in);
        },
        wrap_tensor);
    def_model_inputs_method(
        model_cls, "forward_with_aux", {}, "ModelForwardResult",
        "Outputs plus the mixture-of-experts load-balancing loss.",
        [](resolve::ResolveModel& self, Lead, const resolve::PlotInputs& in) {
            return self->forward_with_aux(in);
        },
        [](const resolve::ModelForwardResult& r) { return nb::cast(r, nb::rv_policy::copy); });
    def_model_inputs_method(
        model_cls, "forward_single", {"target"}, "Tensor", "The output of one target's head.",
        [](resolve::ResolveModel& self, Lead lead, const resolve::PlotInputs& in) {
            return self->forward_single(lead.front(), in);
        },
        wrap_tensor);
    def_model_inputs_method(
        model_cls, "encode_with_activations", {}, "tuple[Tensor, list[Tensor]]",
        "The latent and the activations of every hidden layer of the species "
        "encoder's tail, each (rows, neurons); an empty list for the tabular "
        "architectures and TraitNet.",
        [](resolve::ResolveModel& self, Lead, const resolve::PlotInputs& in) {
            return self->encode_with_activations(in);
        },
        [](const std::pair<torch::Tensor, std::vector<torch::Tensor>>& r) {
            nb::list act_list;
            for (const auto& a : r.second) act_list.append(wrap_tensor(a));
            return nb::object(nb::make_tuple(wrap_tensor(r.first), act_list));
        });
    def_model_inputs_method(
        model_cls, "get_gate_probs", {}, "Tensor",
        "Expert gate probabilities (batch, n_experts); undefined without MoE.",
        [](resolve::ResolveModel& self, Lead, const resolve::PlotInputs& in) {
            return self->get_gate_probs(in);
        },
        wrap_tensor);

    model_cls
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
        .def("composition_parameters", [](resolve::ResolveModel& self) {
            nb::list out;
            for (const auto& t : self->composition_parameters()) {
                out.append(nb::steal(THPVariable_Wrap(t)));
            }
            return out;
        }, "The species, genus and family tables the species encoder reads a "
           "plot's composition through, as the parameters themselves; "
           "ModelConfig.freeze_composition keeps them at their initialisation.")
        .def_prop_ro("uses_moe", [](resolve::ResolveModel& self) { return self->uses_moe(); })
        .def_prop_ro("aux_loss_names", [](resolve::ResolveModel& self) {
            return self->aux_loss_names();
        }, "The auxiliary objective terms forward_with_aux() reports in "
           "aux_losses: 'moe' with a mixture of experts, 'tabnet_sparsity' for "
           "TabNet.")
        .def_prop_ro("row_reference_rows", [](resolve::ResolveModel& self) {
            return self->row_reference_rows();
        }, "How many training rows SAINT's row attention keeps as the "
           "reference set inference attends over; 0 for other architectures.")
        .def_prop_ro("n_experts", [](resolve::ResolveModel& self) { return self->n_experts(); })
        .def("set_traits", [](resolve::ResolveModel& self, nb::object traits_obj) {
            at::Tensor traits = unpack_required_tensor(traits_obj, "traits");
            self->set_traits(traits);
        }, nb::arg("traits"))
        // The species graph a HeterogeneousGNN passes messages on. Trainer
        // .prepare_data builds it from the dataset and a checkpoint carries
        // it, so setting one by hand is for a graph of the caller's own.
        .def("set_species_graph", [](resolve::ResolveModel& self,
                                     nb::object edge_index_obj,
                                     nb::object edge_type_obj) {
            at::Tensor edge_index = unpack_required_tensor(edge_index_obj, "edge_index");
            at::Tensor edge_type = unpack_required_tensor(edge_type_obj, "edge_type");
            self->set_species_graph(edge_index, edge_type);
        }, nb::arg("edge_index"), nb::arg("edge_type"))
        .def_prop_ro("requires_species_graph", [](resolve::ResolveModel& self) {
            return self->requires_species_graph();
        })
        .def_prop_ro("has_species_graph", [](resolve::ResolveModel& self) {
            return self->has_species_graph();
        })
        .def_prop_ro("species_graph_edge_index", [](resolve::ResolveModel& self) {
            auto t = self->species_graph_edge_index();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        })
        .def_prop_ro("species_graph_edge_type", [](resolve::ResolveModel& self) {
            auto t = self->species_graph_edge_type();
            return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
        });
}
