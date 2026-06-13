// rcpp_metrics.cpp - Metrics functions for R (thin C-facade clients).
#include "rcpp_common.h"

namespace {
void require_same_length(NumericVector pred, NumericVector target) {
    if (pred.size() != target.size()) {
        stop("pred and target must have the same length");
    }
}
}  // namespace

// [[Rcpp::export]]
double resolve_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_band_accuracy(
        pred.begin(), target.begin(), pred.size(), threshold, &out));
    return out;
}

// [[Rcpp::export]]
double resolve_mae(NumericVector pred, NumericVector target) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_mae(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

// [[Rcpp::export]]
double resolve_rmse(NumericVector pred, NumericVector target) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_rmse(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

// [[Rcpp::export]]
double resolve_smape(NumericVector pred, NumericVector target, double eps = 1e-8) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_smape(pred.begin(), target.begin(), pred.size(), eps, &out));
    return out;
}

// [[Rcpp::export]]
double resolve_accuracy(NumericVector pred, NumericVector target) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_accuracy(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

// [[Rcpp::export]]
double resolve_r_squared(NumericVector pred, NumericVector target) {
    require_same_length(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_r_squared(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}
