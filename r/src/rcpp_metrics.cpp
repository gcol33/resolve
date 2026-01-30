// rcpp_metrics.cpp - Metrics functions for R
#include "rcpp_common.hpp"

// =============================================================================
// Metrics (static functions exported to R)
// =============================================================================

// [[Rcpp::export]]
double resolve_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::band_accuracy(pred_t, target_t, static_cast<float>(threshold));
}

// [[Rcpp::export]]
double resolve_mae(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::mae(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_rmse(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::rmse(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_smape(NumericVector pred, NumericVector target, double eps = 1e-8) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::smape(pred_t, target_t, static_cast<float>(eps));
}

// [[Rcpp::export]]
double resolve_accuracy(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::accuracy(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_r_squared(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::r_squared(pred_t, target_t);
}
