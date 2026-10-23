#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/python_variable.h>  // For THPVariable_Wrap/Unpack

#include "resolve/resolve.hpp"
#include "resolve/config_registry.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/dataset.hpp"

namespace nb = nanobind;

// The one rule for a tensor handed to Python: data and results (a dataset's
// tensors, predictions, scalers, indices, graphs, evaluation results) are
// detached CPU tensors, or None when undefined. A CPU tensor is handed over
// without a copy, so it shares storage with the engine object it came from.
// The exceptions act on the module itself and stay live on the model's device:
// ResolveModel's forward outputs (autograd for a custom training loop) and its
// parameter accessors (parameters, named_parameters, composition_parameters,
// get_*_weights), which an optimizer or a freeze must reach.
//
// nanobind has no caster for at::Tensor, so every tensor goes through this or
// through THPVariable_Wrap directly for the live exceptions.
inline nb::object tensor_or_none(const torch::Tensor& t) {
    if (!t.defined()) return nb::none();
    torch::Tensor host = t.detach().cpu();
    // A sparse layout (the sparse encoding's species vector) has no strides.
    if (host.layout() == torch::kStrided) host = host.contiguous();
    return nb::steal(THPVariable_Wrap(host));
}

// Registers one field-registry row as a read/write Python attribute. A device
// is reachable through a string property (bind_device_property) and the log
// callback has no Python form at all, so both are skipped by TYPE rather than
// by being quietly left off a list -- the omission a reader would have to
// notice.
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

// Expands one registry row into a def_rw. `cls` and `Cfg` come from the
// enclosing block, so a struct is bound by opening a scope, naming it once, and
// handing its field list to this macro.
#define RESOLVE_BIND_FIELD(member, key) bind_config_field(cls, #member, &Cfg::member);

// Unpack an optional tensor argument. Returns an undefined tensor for None;
// THPVariable_Unpack on Py_None reinterprets the None singleton as a
// THPVariable and reads out of bounds (UB), so callers passing None for an
// unused input (e.g. genus_ids in hash mode) must be guarded here.
inline at::Tensor unpack_optional_tensor(const nb::object& obj) {
    if (!obj.is_valid() || obj.is_none()) return at::Tensor();
    return THPVariable_Unpack(obj.ptr());
}

// Unpack a required tensor argument. Raises a clear Python-visible error for
// None or a non-tensor instead of the UB THPVariable_Unpack would hit.
inline at::Tensor unpack_required_tensor(const nb::object& obj, const char* name) {
    PyObject* p = obj.ptr();
    if (obj.is_none() || !THPVariable_Check(p)) {
        throw std::invalid_argument(std::string(name) + " must be a tensor");
    }
    return THPVariable_Unpack(p);
}

// Helper to convert Python dict to unordered_map of tensors
inline std::unordered_map<std::string, torch::Tensor> dict_to_tensor_map(const nb::dict& d) {
    std::unordered_map<std::string, torch::Tensor> result;
    for (auto item : d) {
        // Use THPVariable_Unpack to convert Python tensor to C++ tensor.
        // Reject a non-tensor value loudly instead of silently dropping it,
        // which would make a mistyped targets/inputs entry vanish.
        PyObject* py_tensor = item.second.ptr();
        auto key = nb::cast<std::string>(item.first);
        if (!THPVariable_Check(py_tensor)) {
            throw std::runtime_error(
                "dict_to_tensor_map: value for key '" + key +
                "' is not a torch.Tensor");
        }
        result[key] = THPVariable_Unpack(py_tensor);
    }
    return result;
}

// A tensor map as a dict of name -> tensor, each through tensor_or_none; an
// undefined entry is left out.
inline nb::dict tensor_map_to_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    nb::dict out;
    for (const auto& [key, value] : m) {
        if (value.defined()) out[key.c_str()] = tensor_or_none(value);
    }
    return out;
}

// A torch::Device field as a read/write `str` attribute: "cpu", "cuda" or
// "cuda:N" (resolve::parse_device; anything else raises ValueError).
template <typename Class, typename Cfg>
void bind_device_property(Class& cls, torch::Device Cfg::*member) {
    cls.def_prop_rw(
        "device", [member](const Cfg& c) { return resolve::device_name(c.*member); },
        [member](Cfg& c, const std::string& device) { c.*member = resolve::parse_device(device); });
}

// Forward declarations for binding registration functions
void register_enums(nb::module_& m);
void register_types(nb::module_& m);
void register_dataset(nb::module_& m);
void register_model(nb::module_& m);
void register_trainer(nb::module_& m);
void register_metrics(nb::module_& m);
void register_pretraining(nb::module_& m);
void register_fuzzy(nb::module_& m);
void register_suite(nb::module_& m);
