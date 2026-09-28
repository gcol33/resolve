#' Load a Model Suite
#'
#' Load a released set of models scored as one: a directory holding a
#' `manifest.json` and the weight files it names. Each target of the suite is
#' predicted by several members (the same recipe trained under different
#' seeds), and [resolve.predict.suite()] reports their combined prediction
#' beside how far they agree.
#'
#' Loading reads and validates the manifest, compares every member file's size
#' and SHA-256 with it, loads each member and checks it against the manifest's
#' input contract (its heads, and the columns it was trained to read). Any
#' mismatch stops the load.
#'
#' @param dir Path to the suite directory.
#' @param device `"cpu"` (default) or `"cuda"`.
#' @param vramFraction Fraction of GPU memory the caching allocator may use when
#'   loading onto CUDA (default 1.0).
#' @param verify Compare every member's size and SHA-256 with the manifest
#'   before loading (default `TRUE`).
#' @param targets Character vector of suite targets to load; `NULL` (default)
#'   loads every target.
#'
#' @return A `Suite` object (an Rcpp module class). `suite$manifest()` returns
#'   the manifest as a nested list, `suite$targets()` the loaded targets.
#'
#' @seealso [resolve.predict.suite()], [resolve.verify_suite()]
#' @examples
#' \dontrun{
#' suite <- resolve.load_suite("eva_context")
#' suite$manifest()$inputs
#' }
#' @export
resolve.load_suite <- function(dir, device = "cpu", vramFraction = 1.0, verify = TRUE,
                               targets = NULL) {
  if (!is.character(dir) || length(dir) != 1) {
    stop("dir must be a single directory path")
  }
  if (!dir.exists(dir)) {
    stop(sprintf("suite directory does not exist: %s", dir))
  }
  if (!device %in% c("cpu", "cuda")) {
    stop("device must be 'cpu' or 'cuda'")
  }
  if (!is.numeric(vramFraction) || length(vramFraction) != 1 ||
      vramFraction <= 0 || vramFraction > 1) {
    stop("vramFraction must be a single number in (0, 1]")
  }
  if (!is.logical(verify) || length(verify) != 1 || is.na(verify)) {
    stop("verify must be TRUE or FALSE")
  }
  options <- list(device = device, vram_fraction = as.numeric(vramFraction),
                  verify = verify)
  if (!is.null(targets)) {
    if (!is.character(targets) || length(targets) == 0) {
      stop("targets must be a character vector of suite target names")
    }
    options$targets <- targets
  }

  .resolve_require_backend()
  .resolve_module()$Suite_load(dir, options)
}


#' Check a Model Suite Against Its Manifest
#'
#' Read and validate a suite's manifest without loading any model, then compare
#' every member file (presence, size, SHA-256) with it.
#'
#' @param dir Path to the suite directory.
#' @return A list with `manifest` (the manifest as a nested list) and `problems`
#'   (a character vector, empty when every member is intact).
#' @examples
#' \dontrun{
#' check <- resolve.verify_suite("eva_context")
#' length(check$problems) == 0
#' }
#' @export
resolve.verify_suite <- function(dir) {
  if (!is.character(dir) || length(dir) != 1) {
    stop("dir must be a single directory path")
  }
  .resolve_require_backend()
  out <- .resolve_module()$Suite_verify(dir)
  out$problems <- as.character(unlist(out$problems))
  out
}


#' Seal a Model Suite Being Built
#'
#' The last step of building a suite: read `dir/manifest.json`, whose members
#' may still lack their `sha256` and `bytes`, fill both from the weight files
#' under `dir`, validate the whole manifest, and write it back. Afterwards
#' [resolve.load_suite()] refuses any member file that changes.
#'
#' @param dir Path to the suite directory.
#' @return The sealed manifest as a nested list, invisibly.
#' @examples
#' \dontrun{
#' jsonlite::write_json(manifest, file.path(dir, "manifest.json"), auto_unbox = TRUE)
#' resolve.seal_suite(dir)
#' }
#' @export
resolve.seal_suite <- function(dir) {
  if (!is.character(dir) || length(dir) != 1) {
    stop("dir must be a single directory path")
  }
  .resolve_require_backend()
  invisible(.resolve_module()$Suite_seal(dir))
}


.resolve_suite_column_keys <- c("plot_id", "species_id", "abundance", "latitude",
                                "longitude", "genus", "family", "covariates",
                                "categoricals")


#' Predict With a Model Suite
#'
#' Score plots with every loaded target of a suite. The plots are read with
#' the columns the manifest's input contract names; `columns` renames them for
#' a table that spells them differently. No target column is read: the plots to
#' score carry no answer.
#'
#' Per plot and target the result holds the combined prediction and, each on
#' its own:
#' - for a class vote: the share of members naming the winning class
#'   (`agreement`) and the members' mean class probabilities (a tie in votes
#'   goes to the tied class with the higher mean probability);
#' - for a mean: the members' standard deviation (`dispersion`), in the
#'   target's units;
#' - for a circular mean (a bearing): the members' circular standard deviation
#'   (`dispersion`), in the target's units;
#' - how much of the plot's species list the target's vocabulary recognises:
#'   `n_species`, `n_recognised`, `recognised_share` (by distinct species) and
#'   `recognised_abundance_share` (by abundance).
#'
#' Agreement and dispersion measure how far the members disagree; they are not
#' calibrated intervals. `NA` marks a value that is undefined for the plot (a
#' plot with no recorded species has no recognised share).
#'
#' @param suite A `Suite` from [resolve.load_suite()].
#' @param species The species records, one row per record: a data.frame or the
#'   path of a CSV file.
#' @param header The plots, one row per plot: a data.frame, a CSV path, or
#'   `NULL`. Required when the suite reads plot-level columns (coordinates or
#'   covariates); when given it decides which plots are scored and their order.
#'   Must be the same kind (data.frame or path) as `species`.
#' @param columns Named list renaming the manifest's input columns, with keys
#'   among `plot_id`, `species_id`, `abundance`, `latitude`, `longitude`,
#'   `genus`, `family`, `covariates`, `categoricals`.
#' @param batchSize Forward-pass chunk size per member (default 4096).
#' @param keepMembers Also return every member's own prediction (default
#'   `FALSE`).
#'
#' @return A `resolve_suite_predictions` list: `plot_ids`, and `targets`, a
#'   named list with one entry per target holding `value` (the combined
#'   prediction; class codes for a vote, with the class labels in `label`),
#'   `agreement`, `probabilities`, `dispersion`, the recognition columns above,
#'   `members` (plots x members, with `keepMembers`), and the target's `units`,
#'   `status` and `limit`. [as.data.frame()] flattens it to one row per plot.
#'
#' @seealso [resolve.load_suite()]
#' @examples
#' \dontrun{
#' suite <- resolve.load_suite("eva_context")
#' preds <- resolve.predict.suite(suite, species = species_df, header = plots_df)
#' preds$targets$eunis$label
#' head(as.data.frame(preds))
#' }
#' @export
resolve.predict.suite <- function(suite, species, header = NULL, columns = NULL,
                                  batchSize = 4096L, keepMembers = FALSE) {
  if (!inherits(suite, "Rcpp_Suite")) {
    stop("suite must be loaded with resolve.load_suite()")
  }
  if (!is.numeric(batchSize) || length(batchSize) != 1 ||
      !(batchSize == -1 || batchSize > 0)) {
    stop("batchSize must be -1 or a positive integer")
  }
  if (!is.logical(keepMembers) || length(keepMembers) != 1 || is.na(keepMembers)) {
    stop("keepMembers must be TRUE or FALSE")
  }
  options <- list(batch_size = as.integer(batchSize), keep_members = keepMembers)
  if (!is.null(columns)) {
    if (!is.list(columns)) stop("columns must be a named list")
    .resolve_check_keys(columns, "columns", .resolve_suite_column_keys)
    for (key in names(columns)) {
      if (!is.character(columns[[key]])) {
        stop(sprintf("columns$%s must be a character vector", key))
      }
    }
    options$columns <- columns
  }

  .resolve_require_backend()
  if (is.character(species)) {
    if (length(species) != 1) stop("species must be a data.frame or one CSV path")
    if (!is.null(header) && !(is.character(header) && length(header) == 1)) {
      stop("with a species CSV path, header must be a CSV path or NULL")
    }
    raw <- suite$predict_csv(if (is.null(header)) "" else header, species, options)
  } else {
    species_cols <- .resolve_df_to_columns(species, "species")
    header_cols <- if (is.null(header)) list() else .resolve_df_to_columns(header, "header")
    raw <- suite$predict_frame(header_cols, species_cols, options)
  }
  .resolve_suite_predictions(raw)
}


# Shape the C-ABI result for R: class labels beside codes, labelled probability
# columns, and members as plots x members.
.resolve_suite_predictions <- function(raw) {
  plot_ids <- as.character(unlist(raw$plot_ids))
  targets <- lapply(raw$targets, function(t) {
    classes <- as.character(unlist(t$class_names))
    seeds <- as.integer(unlist(t$member_seeds))
    out <- list(
      value = t$value,
      label = NULL,
      agreement = t$agreement,
      probabilities = t$probabilities,
      dispersion = t$dispersion,
      n_species = t$n_species,
      n_recognised = t$n_recognised,
      recognised_share = t$recognised_share,
      recognised_abundance_share = t$recognised_abundance_share,
      members = NULL,
      combine = t$combine,
      units = t$units,
      status = t$status,
      limit = t$limit,
      class_names = classes,
      member_seeds = seeds
    )
    if (identical(t$combine, "vote")) {
      out$value <- as.integer(t$value)
      out$label <- if (length(classes) > 0) classes[out$value + 1L] else as.character(out$value)
      if (!is.null(out$probabilities) && ncol(out$probabilities) == length(classes)) {
        colnames(out$probabilities) <- classes
      }
    }
    if (!is.null(t$members)) {
      members <- t(t$members)
      colnames(members) <- paste0("seed", seeds)
      if (identical(t$combine, "vote") && length(classes) > 0) {
        members <- matrix(classes[members + 1L], nrow = nrow(members),
                          dimnames = dimnames(members))
      }
      out$members <- members
    }
    out[!vapply(out, is.null, logical(1))]
  })
  experimental <- Filter(function(t) identical(t$status, "experimental"), targets)
  for (name in names(experimental)) {
    message(sprintf("'%s' is released as experimental: %s", name, experimental[[name]]$limit))
  }
  structure(list(plot_ids = plot_ids, targets = targets),
            class = "resolve_suite_predictions")
}


#' Flatten Suite Predictions to One Row per Plot
#'
#' The same columns `resolve predict --suite` writes: `plot_id`, then per target
#' the prediction (`<target>`; for a vote also `<target>_code` and
#' `<target>_agreement`), its dispersion (`<target>_sd` for a mean,
#' `<target>_circular_sd` for a bearing), the four recognition columns, and,
#' when present, `<target>_seed<N>` for each member.
#'
#' @param x A `resolve_suite_predictions` object from [resolve.predict.suite()].
#' @param row.names,optional Ignored; present for the generic's signature.
#' @param probabilities Also include each vote target's class probabilities,
#'   `<target>_prob_<class>` (default `FALSE`).
#' @param ... Unused.
#' @return A data.frame with one row per plot.
#' @export
as.data.frame.resolve_suite_predictions <- function(x, row.names = NULL, optional = FALSE,
                                                    probabilities = FALSE, ...) {
  out <- data.frame(plot_id = x$plot_ids, stringsAsFactors = FALSE)
  for (name in names(x$targets)) {
    t <- x$targets[[name]]
    if (identical(t$combine, "vote")) {
      out[[name]] <- t$label
      out[[paste0(name, "_code")]] <- t$value
      out[[paste0(name, "_agreement")]] <- t$agreement
      if (probabilities) {
        for (k in seq_len(ncol(t$probabilities))) {
          label <- colnames(t$probabilities)[k]
          if (is.null(label)) label <- as.character(k - 1L)
          out[[paste0(name, "_prob_", label)]] <- t$probabilities[, k]
        }
      }
    } else {
      out[[name]] <- t$value
      sd_name <- if (identical(t$combine, "circular_mean")) "_circular_sd" else "_sd"
      out[[paste0(name, sd_name)]] <- t$dispersion
    }
    out[[paste0(name, "_n_species")]] <- t$n_species
    out[[paste0(name, "_n_recognised")]] <- t$n_recognised
    out[[paste0(name, "_recognised_share")]] <- t$recognised_share
    out[[paste0(name, "_recognised_abundance_share")]] <- t$recognised_abundance_share
    if (!is.null(t$members)) {
      for (k in seq_len(ncol(t$members))) {
        out[[paste0(name, "_", colnames(t$members)[k])]] <- t$members[, k]
      }
    }
  }
  out
}
