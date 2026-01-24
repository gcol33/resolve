#' @title Band Accuracy
#' @description
#' Compute the fraction of predictions within a relative threshold of targets.
#'
#' @param pred Predicted values (numeric vector)
#' @param target Target values (numeric vector)
#' @param threshold Relative error threshold (default: 0.25 for 25%)
#' @return Band accuracy (0 to 1)
#' @export
band_accuracy <- function(pred, target, threshold = 0.25) {
  metrics_band_accuracy(pred, target, threshold)
}

#' @title Mean Absolute Error
#' @description
#' Compute the mean absolute error between predictions and targets.
#'
#' @param pred Predicted values (numeric vector)
#' @param target Target values (numeric vector)
#' @return MAE value
#' @export
mae <- function(pred, target) {
  metrics_mae(pred, target)
}

#' @title Root Mean Squared Error
#' @description
#' Compute the root mean squared error between predictions and targets.
#'
#' @param pred Predicted values (numeric vector)
#' @param target Target values (numeric vector)
#' @return RMSE value
#' @export
rmse <- function(pred, target) {
  metrics_rmse(pred, target)
}

#' @title Symmetric Mean Absolute Percentage Error
#' @description
#' Compute SMAPE between predictions and targets.
#'
#' @param pred Predicted values (numeric vector)
#' @param target Target values (numeric vector)
#' @return SMAPE value (0 to 2)
#' @export
smape <- function(pred, target) {
  metrics_smape(pred, target)
}

#' @title Compute All Regression Metrics
#' @description
#' Compute all standard regression metrics.
#'
#' @param pred Predicted values (numeric vector)
#' @param target Target values (numeric vector)
#' @return Named list with mae, rmse, smape, band_10, band_25, band_50
#' @export
compute_metrics <- function(pred, target) {
  list(
    mae = mae(pred, target),
    rmse = rmse(pred, target),
    smape = smape(pred, target),
    band_10 = band_accuracy(pred, target, 0.10),
    band_25 = band_accuracy(pred, target, 0.25),
    band_50 = band_accuracy(pred, target, 0.50)
  )
}
