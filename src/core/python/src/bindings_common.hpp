#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/optional.h>
#include <torch/torch.h>

#include "resolve/resolve.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/dataset.hpp"

namespace nb = nanobind;

// Helper to convert Python dict to unordered_map of tensors
inline std::unordered_map<std::string, torch::Tensor> dict_to_tensor_map(const nb::dict& d) {
    std::unordered_map<std::string, torch::Tensor> result;
    for (auto item : d) {
        result[nb::cast<std::string>(item.first)] = nb::cast<torch::Tensor>(item.second);
    }
    return result;
}

// Helper to convert unordered_map of tensors to Python dict
inline nb::dict tensor_map_to_dict(const std::unordered_map<std::string, torch::Tensor>& m) {
    nb::dict result;
    for (const auto& [key, value] : m) {
        result[nb::str(key.c_str())] = value;
    }
    return result;
}

// Forward declarations for binding registration functions
void register_enums(nb::module_& m);
void register_types(nb::module_& m);
void register_dataset(nb::module_& m);
void register_model(nb::module_& m);
void register_trainer(nb::module_& m);
void register_metrics(nb::module_& m);
