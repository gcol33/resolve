# Gap fitting: what a dataset's own complete plots add to the predictions for
# the plots that lack a value. Thin wrappers over the engine's fit_gaps
# (resolve_fit_gaps / resolve_fit_prediction_gaps in the C ABI).

#' Fit a Dataset's Gaps From Its Own Complete Plots
#'
#' A dataset usually records a variable on most of its plots and lacks it on
#' some. Gap fitting chooses, on the plots that carry it, how to combine the
#' model's raw prediction with the dataset's own neighbours, and applies that
#' choice to the plots that lack it (the gaps). The candidates are the complete
#' plots' centre (`local_constant`), the raw prediction (`resolve`), the raw
#' prediction calibrated by least squares (`offset_slope`), the median of the k
#' most similar complete plots in species composition (`knn_species`) or the k
#' nearest by great-circle distance (`knn_spatial`), and the raw prediction
#' combined with those medians by least squares (`stack_species`,
#' `stack_species_spatial`).
#'
#' Each candidate is judged by cross-validation over the complete plots, with
#' folds laid out to hold plots out the way the gaps are held out (k-fold
#' nearest-neighbour distance matching, Linnenbrink et al. 2024): squares of
#' an equal-area plane, survey-year periods, both, or plot by plot, the last
#' also with buffers drawn from the gaps' distances, whichever leaves the
#' held-out plots as far from the rest, in species similarity, in km and in
#' survey years, as the gaps are from the complete plots. Every fitted part
#' (k, least-squares coefficients) is fitted on the other folds, and the
#' candidate with the lowest out-of-fold error, by the target's own error (mean
#' absolute error; on a log scale that of log1p; the circular error of a
#' bearing; the median km of a location), predicts the gaps. The whole choice,
#' repeated with each fold held out, gives the error reported for it
#' (`estimated_error`, nested cross-validation). Gaps with and without
#' coordinates choose apart. The raw prediction is never replaced: the fitted
#' values come back beside it.
#'
#' Undated plots that are probably older than the dated ones, and plots
#' without coordinates that probably come from an area with no georeferenced
#' plots, are better served by the raw prediction: the complete plots cannot
#' show how far off such gaps are.
#'
#' @param raw The model's prediction per plot in the target's units: a numeric
#'   vector, or for `scale = "geographic"` a plots x 2 matrix of latitude and
#'   longitude. `NA` marks a plot the model did not impute: it is neither a gap
#'   nor a complete plot.
#' @param recorded The dataset's recorded value per plot, shaped as `raw`;
#'   `NA` marks a gap.
#' @param species A plots x species numeric matrix; an entry other than 0
#'   marks a species present.
#' @param scale How the values lie: `"linear"`, `"log"` (fitted on log1p),
#'   `"circular"` (a bearing with full turn `period`) or `"geographic"`.
#' @param period A bearing's full turn, in the target's units (`"circular"`).
#' @param coordinates `NULL` or a plots x 2 matrix of latitude and longitude
#'   (`NA` where unknown): the spatial axis.
#' @param years `NULL` or each plot's survey year (`NA` where unknown).
#' @param gapFitting `NULL` for the defaults, or a list of settings:
#'   `candidates` (method names), `k_grid` (the neighbour counts k is chosen
#'   from, default 1, 3, 5, 10, 20, 50), `spatial` (`FALSE` reads composition
#'   only), `matching_seed`, `n_folds` (default 10), `max_fold_share` (the
#'   largest share of the complete plots one fold may hold, default 0.5) and
#'   `matching_sample` (complete plots the layouts are compared on, default
#'   1000).
#' @param name The target's name, as the report states it.
#' @param folds `NULL`, or an integer fold per complete plot (a plot with both
#'   `recorded` and `raw`; in plot order,
#'   numbered from 0) replacing the matched layout (to reproduce another
#'   implementation's layout).
#'
#' @return A `resolve_gap_fit` list: `fitted` (as `raw`; a plot recording the
#'   value keeps it, a plot lacking it that was not imputed is `NA`), `method`
#'   (per plot, `"recorded"` for a plot recording the value, `"not_imputed"`
#'   for one lacking it whose `raw` is `NA`) and `report` (`n_complete`,
#'   `n_gaps`, `n_not_imputed`, the gaps' distance per axis, the fold layout, the k chosen
#'   per neighbour kind, and per group of gaps every candidate's
#'   cross-validated error, the estimated error of the whole choice, the same
#'   error per held-out fold (`fold_error`, over `fold_n_estimated` complete
#'   plots) and the chosen method's coefficients).
#' @seealso [resolve.predict.suite()] with `gapFitting`, which fits a suite's
#'   targets after combining them.
#' @examplesIf resolveR::resolve.available()
#' set.seed(1)
#' n <- 300
#' raw <- runif(n, 1950, 2020)
#' recorded <- 1990 + 0.5 * (raw - mean(raw)) + rnorm(n, 0, 2)
#' recorded[sample(n, 60)] <- NA
#' species <- matrix(rbinom(n * 20, 1, 0.2), n, 20)
#' fit <- resolve.fit_gaps(raw, recorded, species, name = "year")
#' fit$report$groups$unlocated$method
#' table(fit$method)
#' @export
resolve.fit_gaps <- function(raw, recorded, species, scale = "linear", period = 360,
                             coordinates = NULL, years = NULL, gapFitting = NULL,
                             name = "", folds = NULL) {
  if (!is.matrix(species) || !(is.numeric(species) || is.logical(species))) {
    stop("species must be a numeric matrix", call. = FALSE)
  }
  if (!is.character(scale) || length(scale) != 1L) stop("scale must be a single string", call. = FALSE)
  if (!is.null(coordinates) && (!is.matrix(coordinates) || ncol(coordinates) != 2L)) {
    stop("coordinates must be a plots x 2 matrix of latitude and longitude", call. = FALSE)
  }
  if (!is.null(folds)) {
    if (!is.numeric(folds) || anyNA(folds) || any(folds != round(folds))) {
      stop("folds must be whole numbers, one per complete plot", call. = FALSE)
    }
    folds <- as.integer(folds)
  }
  config <- .resolve_gap_config(gapFitting)
  if (is.null(config)) config <- list()
  .resolve_require_backend()
  target <- list(name = as.character(name), scale = scale, period = as.numeric(period))
  .resolve_gap_fit(.resolve_module()$fit_gaps(
    target, .resolve_na_real(raw), .resolve_na_real(recorded), species * 1,
    if (is.null(coordinates)) NULL else .resolve_na_real(coordinates),
    if (is.null(years)) NULL else .resolve_na_real(years), config, folds))
}


#' Fit a Single Model's Gaps From Its Predictions and the Dataset's Tables
#'
#' [resolve.fit_gaps()] for a single checkpoint: the species composition (every
#' named species, whatever the model's vocabulary holds), positions, survey
#' years and recorded values are read from the dataset's own tables, aligned to
#' the plots the model predicted.
#'
#' @param predictions The output of [resolve.predict.dataset()].
#' @param species The species records, one row per record (a data.frame).
#' @param header The plots, one row per plot (a data.frame), or `NULL` to read
#'   plot-level columns from each plot's first species row.
#' @param columns Named list: `plot_id`, `species_id`, optionally `latitude`,
#'   `longitude`, `survey_year`, and `recorded`, a named list of target ->
#'   recorded column (`c(latitude, longitude)` for a location).
#' @param targets Named list of target -> list(`scale`, `period`, `outputs`):
#'   the scale as in [resolve.fit_gaps()] and the prediction head(s) read
#'   (default the target's name; a bearing may read its sine and cosine heads,
#'   a location reads its latitude and longitude heads).
#' @param gapFitting As in [resolve.fit_gaps()].
#' @return A named list of `resolve_gap_fit`, one per target.
#' @export
resolve.fit_prediction_gaps <- function(predictions, species, header = NULL, columns,
                                        targets, gapFitting = NULL) {
  if (!is.list(predictions) || is.null(predictions$plot_ids) || is.null(predictions$predictions)) {
    stop("predictions must come from resolve.predict.dataset()", call. = FALSE)
  }
  if (!is.list(columns) || is.null(names(columns))) stop("columns must be a named list", call. = FALSE)
  if (!is.list(targets) || is.null(names(targets))) stop("targets must be a named list", call. = FALSE)
  .resolve_require_backend()
  requests <- unname(Map(function(spec, name) {
    spec <- as.list(spec)
    c(list(name = name), spec)
  }, targets, names(targets)))
  outputs <- lapply(predictions$predictions, .resolve_na_real)
  config <- .resolve_gap_config(gapFitting)
  if (is.null(config)) config <- list()
  species_cols <- .resolve_df_to_columns(species, "species")
  header_cols <- if (is.null(header)) list() else .resolve_df_to_columns(header, "header")
  fits <- .resolve_module()$fit_prediction_gaps(
    header_cols, species_cols, columns,
    list(plot_ids = as.character(unlist(predictions$plot_ids)), outputs = outputs),
    requests, config)
  lapply(fits, .resolve_gap_fit)
}


# A gap fitting config for the C ABI: NULL / FALSE fits nothing, TRUE the
# defaults (a named empty list travels as an empty map), a list its fields.
.resolve_gap_config <- function(gapFitting) {
  if (is.null(gapFitting) || isFALSE(gapFitting)) return(NULL)
  if (isTRUE(gapFitting)) return(structure(list(), names = character(0)))
  if (!is.list(gapFitting) || (length(gapFitting) > 0 && is.null(names(gapFitting)))) {
    stop("gapFitting must be TRUE, NULL or a named list of settings", call. = FALSE)
  }
  if (length(gapFitting) == 0) return(structure(list(), names = character(0)))
  if (!is.null(gapFitting$candidates)) gapFitting$candidates <- as.list(as.character(gapFitting$candidates))
  if (!is.null(gapFitting$k_grid)) gapFitting$k_grid <- as.integer(gapFitting$k_grid)
  gapFitting
}

# A numeric vector or matrix with NA as NaN, which the engine reads as missing.
.resolve_na_real <- function(x) {
  x <- x * 1
  x[is.na(x)] <- NaN
  x
}

# The engine's gap fit as R: fitted and method as vectors (fitted a plots x 2
# matrix for a location), the report as a nested list.
.resolve_gap_fit <- function(fit) {
  fitted <- fit$fitted
  if (is.matrix(fitted) && ncol(fitted) == 2L) colnames(fitted) <- c("latitude", "longitude")
  structure(list(fitted = fitted, method = as.character(unlist(fit$method)), report = fit$report),
            class = "resolve_gap_fit")
}
