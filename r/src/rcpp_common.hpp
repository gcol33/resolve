// rcpp_common.hpp - Shared type conversions and utilities for R bindings
#ifndef RCPP_COMMON_HPP
#define RCPP_COMMON_HPP

// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <torch/torch.h>
#include "resolve/resolve.hpp"

using namespace Rcpp;

// =============================================================================
// Type conversion: R -> Torch
// =============================================================================

inline torch::Tensor r_vec_to_tensor(NumericVector x) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t = torch::from_blob(
        x.begin(),
        {static_cast<int64_t>(x.size())},
        options
    ).clone();
    return t;
}

// Generic matrix-to-tensor: works for NumericMatrix (float) and IntegerMatrix (int64)
template <typename MatT, typename ScalarT, torch::ScalarType DType>
inline torch::Tensor r_mat_to_tensor_impl(MatT x) {
    auto options = torch::TensorOptions().dtype(DType);
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<ScalarT> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<ScalarT>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
}

inline torch::Tensor r_mat_to_tensor(NumericMatrix x) {
    return r_mat_to_tensor_impl<NumericMatrix, float, torch::kFloat32>(x);
}

inline torch::Tensor r_int_vec_to_tensor(IntegerVector x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(x.size())}, options).clone();
}

inline torch::Tensor r_int_mat_to_tensor(IntegerMatrix x) {
    return r_mat_to_tensor_impl<IntegerMatrix, int64_t, torch::kInt64>(x);
}

// =============================================================================
// Type conversion: Torch -> R
// =============================================================================

inline NumericVector tensor_to_r_vec(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    float* data = cpu.data_ptr<float>();
    return NumericVector(data, data + cpu.numel());
}

inline NumericMatrix tensor_to_r_mat(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    int nrow = cpu.size(0);
    int ncol = cpu.size(1);
    NumericMatrix out(nrow, ncol);
    float* data = cpu.data_ptr<float>();
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            out(i, j) = data[i * ncol + j];
        }
    }
    return out;
}

// =============================================================================
// Enum conversions
// =============================================================================

// Generic string-to-enum parser. Avoids duplicating the same if/stop pattern
// for every enum type. Entries is an initializer_list of {string, EnumValue}.
template <typename EnumT>
inline EnumT parse_enum(
    const std::string& s,
    std::initializer_list<std::pair<const char*, EnumT>> entries,
    const char* type_name
) {
    for (const auto& [key, val] : entries) {
        if (s == key) return val;
    }
    stop("Invalid " + std::string(type_name) + ": " + s);
}

inline resolve::SelectionMode parse_selection_mode(const std::string& s) {
    return parse_enum<resolve::SelectionMode>(s, {
        {"top", resolve::SelectionMode::Top},
        {"bottom", resolve::SelectionMode::Bottom},
        {"top_bottom", resolve::SelectionMode::TopBottom},
        {"all", resolve::SelectionMode::All},
    }, "selection mode");
}

inline resolve::RepresentationMode parse_representation_mode(const std::string& s) {
    return parse_enum<resolve::RepresentationMode>(s, {
        {"abundance", resolve::RepresentationMode::Abundance},
        {"presence_absence", resolve::RepresentationMode::PresenceAbsence},
    }, "representation mode");
}

inline resolve::NormalizationMode parse_normalization_mode(const std::string& s) {
    return parse_enum<resolve::NormalizationMode>(s, {
        {"raw", resolve::NormalizationMode::Raw},
        {"norm", resolve::NormalizationMode::Norm},
        {"log1p", resolve::NormalizationMode::Log1p},
    }, "normalization mode");
}

inline resolve::AggregationMode parse_aggregation_mode(const std::string& s) {
    return parse_enum<resolve::AggregationMode>(s, {
        {"abundance", resolve::AggregationMode::Abundance},
        {"count", resolve::AggregationMode::Count},
    }, "aggregation mode");
}

inline resolve::TaskType parse_task_type(const std::string& s) {
    return parse_enum<resolve::TaskType>(s, {
        {"regression", resolve::TaskType::Regression},
        {"classification", resolve::TaskType::Classification},
    }, "task type");
}

inline resolve::TransformType parse_transform_type(const std::string& s) {
    return parse_enum<resolve::TransformType>(s, {
        {"none", resolve::TransformType::None},
        {"log1p", resolve::TransformType::Log1p},
    }, "transform type");
}

inline resolve::SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    return parse_enum<resolve::SpeciesEncodingMode>(s, {
        {"hash", resolve::SpeciesEncodingMode::Hash},
        {"embed", resolve::SpeciesEncodingMode::Embed},
        {"sparse", resolve::SpeciesEncodingMode::Sparse},
    }, "species encoding mode");
}

inline resolve::LossConfigMode parse_loss_config_mode(const std::string& s) {
    return parse_enum<resolve::LossConfigMode>(s, {
        {"mae", resolve::LossConfigMode::MAE},
        {"smape", resolve::LossConfigMode::SMAPE},
        {"combined", resolve::LossConfigMode::Combined},
    }, "loss config mode");
}

inline resolve::LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    return parse_enum<resolve::LRSchedulerType>(s, {
        {"none", resolve::LRSchedulerType::None},
        {"step", resolve::LRSchedulerType::StepLR},
        {"cosine", resolve::LRSchedulerType::CosineAnnealing},
    }, "LR scheduler type");
}

#endif // RCPP_COMMON_HPP
