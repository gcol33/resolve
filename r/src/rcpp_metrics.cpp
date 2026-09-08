// rcpp_metrics.cpp - Metrics functions for R (thin C-facade clients).
#include "rcpp_common.h"

namespace {
// Guard every metric: the resolve_c backend must be loaded (else the metric
// forwarder is a NULL pointer), and the two vectors must match in length.
void check_metric_inputs(NumericVector pred, NumericVector target) {
    capi_require_loaded();
    if (pred.size() != target.size()) {
        stop("pred and target must have the same length");
    }
}
}  // namespace

//' Prediction metrics
//'
//' The engine's metric functions, computed over a vector of predictions and
//' the matching vector of observed values. They are the metrics
//' [resolve.train.dataset()] reports for each target, made available so a
//' prediction scored elsewhere (a reloaded checkpoint, a held-out survey) can
//' be evaluated the same way. The `resolve_c` backend must be loaded
//' ([resolve.available()]).
//'
//' @param pred Numeric vector of predicted values (for `resolve_accuracy()`,
//'   predicted class codes).
//' @param target Numeric vector of observed values, the same length as `pred`
//'   (for `resolve_accuracy()`, the true class codes).
//' @param threshold Relative half-width of the band for
//'   `resolve_band_accuracy()`: a prediction counts as correct when it lies
//'   within `target * (1 - threshold)` and `target * (1 + threshold)`.
//' @param eps Small constant added to the denominator of the symmetric mean
//'   absolute percentage error to avoid division by zero.
//'
//' @return A single number: the mean absolute error (`resolve_mae()`), the
//'   root mean squared error (`resolve_rmse()`), the symmetric mean absolute
//'   percentage error in `[0, 2]` (`resolve_smape()`), the coefficient of
//'   determination (`resolve_r_squared()`), the share of predictions inside
//'   the relative band (`resolve_band_accuracy()`), or the share of exactly
//'   matching class codes (`resolve_accuracy()`).
//'
//' @examples
//' \dontrun{
//' pred <- c(1.0, 2.2, 2.9, 4.1)
//' obs <- c(1, 2, 3, 4)
//' resolve_mae(pred, obs)
//' resolve_rmse(pred, obs)
//' resolve_r_squared(pred, obs)
//' resolve_band_accuracy(pred, obs, threshold = 0.05)
//' }
//' @name resolve_metrics
//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_band_accuracy(
        pred.begin(), target.begin(), pred.size(), threshold, &out));
    return out;
}

//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_mae(NumericVector pred, NumericVector target) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_mae(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_rmse(NumericVector pred, NumericVector target) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_rmse(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_smape(NumericVector pred, NumericVector target, double eps = 1e-8) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_smape(pred.begin(), target.begin(), pred.size(), eps, &out));
    return out;
}

//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_accuracy(NumericVector pred, NumericVector target) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_accuracy(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}

//' @rdname resolve_metrics
//' @export
// [[Rcpp::export]]
double resolve_r_squared(NumericVector pred, NumericVector target) {
    check_metric_inputs(pred, target);
    double out = 0.0;
    capi_check_status(resolve_metric_r_squared(pred.begin(), target.begin(), pred.size(), &out));
    return out;
}
