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
#include "resolve/role_mapping.hpp"
#include "resolve/dataset.hpp"

namespace nb = nanobind;

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

// Helper to convert unordered_map of tensors to Python dict
inline nb::object tensor_map_to_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    PyObject* py_dict = PyDict_New();
    for (const auto& [key, value] : m) {
        if (value.defined()) {
            // Move tensor to CPU and make contiguous for Python interop
            auto cpu_tensor = value.detach().cpu().contiguous();
            PyObject* py_tensor = THPVariable_Wrap(cpu_tensor);
            PyDict_SetItemString(py_dict, key.c_str(), py_tensor);
            Py_DECREF(py_tensor);  // PyDict_SetItemString increments refcount
        }
    }
    return nb::steal(py_dict);
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
