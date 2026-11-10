#' Open a Trained Model to Fine-Tune From
#'
#' A fine-tuned model starts from a trained one, its parent: a checkpoint, or a
#' member of a model suite such as the released EVA suites. It reads the
#' vegetation plots exactly as the parent does (the parent's species, genus,
#' family and categorical vocabularies, its loader settings and the scaling of
#' its coordinates and covariates), starts from the parent's encoder, and
#' trains heads for your own targets: a national habitat typology, a soil or
#' management variable, or a released target refitted on regional plots.
#'
#' Load the fine-tuning plots with the parent's vocabularies and loader
#' settings, then pass the parent to [resolve.train.dataset()] as `initFrom`:
#'
#' ```
#' parent <- resolve.parent(suite_dir, target = "habitat")
#' dataset <- resolve.dataset.csv(header, species, roles, targets,
#'                                vocabs = parent$vocabs(),
#'                                config = parent$dataset_config())
#' fit <- resolve.train.dataset(dataset, initFrom = parent, freezeEncoder = TRUE)
#' ```
#'
#' @param source Path of a checkpoint, or of a suite directory (one holding
#'   `manifest.json`).
#' @param target For a suite: the target whose member to start from. May be
#'   left `NULL` for a suite of one target.
#' @param seed For a suite: the member's seed (`NULL`, the default, takes the
#'   target's first member).
#'
#' @return A `Parent` object (an Rcpp module class): `vocabs()` and
#'   `dataset_config()` are what the fine-tuning dataset is loaded with,
#'   `model_config()` the configuration the fine-tuned model takes,
#'   `schema()` and `scalers()` the parent's, and `record()` the record a
#'   fine-tuned checkpoint keeps of it (path, SHA-256, size, engine version,
#'   and the suite, target and seed it came from). A suite member's size and
#'   SHA-256 are checked against the manifest before it is read.
#'
#' @examplesIf resolveR::resolve.available()
#' source(system.file("examples", "habitat-suite.R", package = "resolveR"))
#' resolve.seal_suite(suite_dir)
#' parent <- resolve.parent(suite_dir, target = "habitat", seed = 1L)
#' parent$record()$suite_target
#' @export
resolve.parent <- function(source, target = NULL, seed = NULL) {
  if (!is.character(source) || length(source) != 1) {
    stop("source must be the path of one checkpoint or suite directory")
  }
  config <- list()
  if (!is.null(target)) {
    if (!is.character(target) || length(target) != 1) stop("target must be one target name")
    config$suite_target <- target
  }
  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1 || is.na(seed)) stop("seed must be one integer")
    config$suite_seed <- as.integer(seed)
  }
  .resolve_require_backend()
  .resolve_module()$Parent_open(source, config)
}

# The initFrom argument as a Parent: NULL, a Parent, or a checkpoint path.
.resolve_parent_arg <- function(initFrom) {
  if (is.null(initFrom)) return(NULL)
  if (inherits(initFrom, "Rcpp_Parent")) return(initFrom)
  if (is.character(initFrom) && length(initFrom) == 1) return(resolve.parent(initFrom))
  stop("initFrom must be a parent from resolve.parent() or the path of one checkpoint")
}

# `base` with every element of `over` set on it, named lists merged element by
# element, so a sub-config given in part keeps the rest of the base's.
.resolve_overlay <- function(base, over) {
  for (key in names(over)) {
    if (is.list(base[[key]]) && !is.null(names(base[[key]])) && is.list(over[[key]])) {
      base[[key]] <- .resolve_overlay(base[[key]], over[[key]])
    } else {
      base[[key]] <- over[[key]]
    }
  }
  base
}


#' Release Fine-Tuned Models as a Suite
#'
#' Release checkpoints fine-tuned from a suite's members (see
#' [resolve.parent()]) as a model suite of their own, which
#' [resolve.load_suite()] and [resolve.predict.suite()] read like any other.
#' Each checkpoint is copied into `dir`, and the manifest written beside them
#' takes its input contract from the parent suite (the part of it the targets'
#' members read), names for each target the parent suite and target its
#' members started from, with that suite's licence and training scope, and
#' records per member the SHA-256 of its parent checkpoint, which the
#' checkpoint records too. The manifest is sealed and the suite loaded once,
#' so every member is checked against it.
#'
#' @param dir Directory the suite is written to; must not hold a manifest yet.
#' @param parent Path of the suite directory the members were fine-tuned from.
#' @param name The suite's name.
#' @param licence The licence the suite is released under.
#' @param trainingScope The plots the fine-tuned members were trained on.
#' @param targets A named list with one element per target, named by the
#'   target: a list with `members`, a named vector or list of checkpoint paths
#'   named by their seed (`c("1" = "seed1.pt", "2" = "seed2.pt")`), and
#'   optionally `outputs` (the heads read; default the head named as the
#'   target), `combine` (`"vote"`, `"mean"`, `"circular_mean"`,
#'   `"spherical_mean"`; default a vote for a classification head, a mean for
#'   a regression head), `period`, `units`, `status` (`"released"` or
#'   `"experimental"`), `limit` (required for an experimental target),
#'   `training` and `validation` (named lists recorded as they are).
#' @param description What the suite imputes.
#'
#' @return The manifest written, as a nested list, invisibly.
#'
#' @examplesIf resolveR::resolve.available()
#' source(system.file("examples", "habitat-suite.R", package = "resolveR"))
#' resolve.seal_suite(suite_dir)
#' parent <- resolve.parent(suite_dir, target = "habitat", seed = 1L)
#' tuned <- resolve.dataset.csv(plots, species, roles = roles,
#'                              targets = list(elevation = list(column = "elevation",
#'                                                              task = "regression")),
#'                              vocabs = parent$vocabs(), config = parent$dataset_config())
#' path <- file.path(tempdir(), "elevation_seed1.pt")
#' fit <- resolve.train.dataset(tuned, initFrom = parent, freezeEncoder = TRUE,
#'                              maxEpochs = 20L, savePath = path, verbose = FALSE)
#' derived <- file.path(tempdir(), "elevation_suite")
#' unlink(derived, recursive = TRUE)
#' manifest <- resolve.derive_suite(derived, parent = suite_dir, name = "elevation_example",
#'                                  licence = "CC-BY-4.0",
#'                                  trainingScope = "150 synthetic Alpine plots",
#'                                  targets = list(elevation = list(members = c("1" = path),
#'                                                                  units = "m")))
#' manifest$targets[[1]]$derived_from$target
#' @export
resolve.derive_suite <- function(dir, parent, name, licence, trainingScope, targets,
                                 description = "") {
  for (arg in c("dir", "parent", "name", "licence", "trainingScope", "description")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1) stop(sprintf("%s must be one string", arg))
  }
  if (!is.list(targets) || length(targets) == 0 || is.null(names(targets)) ||
      any(!nzchar(names(targets)))) {
    stop("targets must be a non-empty list named by target")
  }
  spec_targets <- lapply(names(targets), function(target) {
    t <- targets[[target]]
    if (!is.list(t) || is.null(t$members) || is.null(names(t$members))) {
      stop(sprintf("targets$%s must be a list whose members are checkpoint paths named by seed",
                   target))
    }
    seeds <- suppressWarnings(as.integer(names(t$members)))
    if (anyNA(seeds)) stop(sprintf("targets$%s$members must be named by integer seeds", target))
    out <- list(name = target,
                members = lapply(seq_along(seeds), function(i) {
                  list(checkpoint = as.character(t$members[[i]]), seed = seeds[[i]])
                }))
    if (!is.null(t$outputs)) out$outputs <- as.list(as.character(t$outputs))
    for (key in c("combine", "period", "units", "status", "limit", "training", "validation")) {
      if (!is.null(t[[key]])) out[[key]] <- t[[key]]
    }
    unknown <- setdiff(names(t), c("members", "outputs", "combine", "period", "units", "status",
                                   "limit", "training", "validation"))
    if (length(unknown) > 0) {
      stop(sprintf("targets$%s: unknown element(s) %s", target,
                   paste0("'", unknown, "'", collapse = ", ")))
    }
    out
  })
  spec <- list(parent = parent, name = name, description = description, licence = licence,
               training_scope = trainingScope, targets = spec_targets)
  .resolve_require_backend()
  invisible(.resolve_module()$Suite_derive(dir, spec))
}
