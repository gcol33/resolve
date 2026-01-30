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

inline torch::Tensor r_mat_to_tensor(NumericMatrix x) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<float> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<float>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
}

inline torch::Tensor r_int_vec_to_tensor(IntegerVector x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(x.size())}, options).clone();
}

inline torch::Tensor r_int_mat_to_tensor(IntegerMatrix x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<int64_t> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<int64_t>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
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

inline resolve::SelectionMode parse_selection_mode(const std::string& s) {
    if (s == "top") return resolve::SelectionMode::Top;
    if (s == "bottom") return resolve::SelectionMode::Bottom;
    if (s == "top_bottom") return resolve::SelectionMode::TopBottom;
    if (s == "all") return resolve::SelectionMode::All;
    stop("Invalid selection mode: " + s);
}

inline resolve::RepresentationMode parse_representation_mode(const std::string& s) {
    if (s == "abundance") return resolve::RepresentationMode::Abundance;
    if (s == "presence_absence") return resolve::RepresentationMode::PresenceAbsence;
    stop("Invalid representation mode: " + s);
}

inline resolve::NormalizationMode parse_normalization_mode(const std::string& s) {
    if (s == "raw") return resolve::NormalizationMode::Raw;
    if (s == "norm") return resolve::NormalizationMode::Norm;
    if (s == "log1p") return resolve::NormalizationMode::Log1p;
    stop("Invalid normalization mode: " + s);
}

inline resolve::AggregationMode parse_aggregation_mode(const std::string& s) {
    if (s == "abundance") return resolve::AggregationMode::Abundance;
    if (s == "count") return resolve::AggregationMode::Count;
    stop("Invalid aggregation mode: " + s);
}

inline resolve::TaskType parse_task_type(const std::string& s) {
    if (s == "regression") return resolve::TaskType::Regression;
    if (s == "classification") return resolve::TaskType::Classification;
    stop("Invalid task type: " + s);
}

inline resolve::TransformType parse_transform_type(const std::string& s) {
    if (s == "none") return resolve::TransformType::None;
    if (s == "log1p") return resolve::TransformType::Log1p;
    stop("Invalid transform type: " + s);
}

inline resolve::SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    if (s == "hash") return resolve::SpeciesEncodingMode::Hash;
    if (s == "embed") return resolve::SpeciesEncodingMode::Embed;
    if (s == "sparse") return resolve::SpeciesEncodingMode::Sparse;
    stop("Invalid species encoding mode: " + s);
}

inline resolve::LossConfigMode parse_loss_config_mode(const std::string& s) {
    if (s == "mae") return resolve::LossConfigMode::MAE;
    if (s == "smape") return resolve::LossConfigMode::SMAPE;
    if (s == "combined") return resolve::LossConfigMode::Combined;
    stop("Invalid loss config mode: " + s);
}

inline resolve::LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    if (s == "none") return resolve::LRSchedulerType::None;
    if (s == "step") return resolve::LRSchedulerType::StepLR;
    if (s == "cosine") return resolve::LRSchedulerType::CosineAnnealing;
    stop("Invalid LR scheduler type: " + s);
}

#endif // RCPP_COMMON_HPP
