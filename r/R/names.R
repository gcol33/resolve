# Name resolution: species names read as the names a model learnt.

#' Resolve Species Names Against a Model's Vocabulary
#'
#' A model knows each species by the name its training records were resolved
#' to. A table names a species as it was entered in the field, by a current
#' accepted name, or misspelled, and none of these need be the name the model
#' learnt. Each name is taken, first hit wins:
#'
#' - as written, where the vocabulary holds it (`exact`);
#' - through the release's name table, which maps EVA's entered names and the
#'   synonymy of the plant backbones to the names the models learnt
#'   (`alias`, with the `source` that linked them);
#' - as the nearest known name or table name within `maxEditDistance` edits,
#'   letter case ignored, searched within the name's genus first, and only
#'   when every nearest name stands for one species (`spelling`, with the
#'   name it was read `via` and the `distance`);
#'
#' and otherwise left as written (`unresolved`), where it counts as a species
#' the model does not know. [resolve.predict.suite()] and
#' [resolve.dataset.csv()] apply the same steps through their `names`
#' argument.
#'
#' @param names Character vector of species names as written.
#' @param model Where the vocabulary comes from: a `Suite` from
#'   [resolve.load_suite()] (with `target`), a `Predictor` from
#'   [resolve.load()], or a character vector ordered as a checkpoint orders
#'   it (`"<UNK>"` first).
#' @param target The suite target whose vocabulary is used.
#' @param table The path of the release's `names.json`, or `NULL` to resolve
#'   against the vocabulary alone.
#' @param maxEditDistance The edits a misspelled name may be away from a known
#'   name (default 1; 0 turns the spelling step off; 2 can also rename species
#'   the models never saw).
#'
#' @return A data.frame, one row per name in order: `name`, `token` (the name
#'   the model reads), `route`, `source`, `via`, `distance`.
#'
#' @examplesIf resolveR::resolve.available()
#' vocabulary <- c("<UNK>", "Quercus robur", "Anemone altaica")
#' resolve.names(c("Quercus robur", "Quercus robru", "Poa annua"), vocabulary)
#'
#' @export
resolve.names <- function(names, model, target = NULL, table = NULL, maxEditDistance = 1L) {
  if (!is.character(names)) stop("names must be a character vector")
  vocabulary <- if (inherits(model, "Rcpp_Suite")) {
    if (is.null(target)) stop("a suite holds one vocabulary per target: pass target")
    vocabularies <- model$species_vocabularies()
    if (!target %in% names(vocabularies)) {
      stop(sprintf("the suite has no loaded target '%s'", target))
    }
    as.character(unlist(vocabularies[[target]]))
  } else if (inherits(model, "Rcpp_Predictor")) {
    as.character(unlist(model$vocabs()$species_vocab))
  } else if (is.character(model)) {
    model
  } else {
    stop("model must be a Suite, a Predictor or a character vocabulary")
  }
  .resolve_require_backend()
  option <- .resolve_names_option(if (is.null(table)) TRUE else table, maxEditDistance)
  .resolve_name_frame(.resolve_module()$names_resolve(vocabulary, unname(names), option))
}

# The names option of the C ABI for a `names` argument: NULL / FALSE reads
# every name as written, TRUE resolves against the vocabulary alone, a path is
# the release's names.json.
.resolve_names_option <- function(names, maxEditDistance) {
  if (is.null(names) || isFALSE(names)) return(NULL)
  if (!is.numeric(maxEditDistance) || length(maxEditDistance) != 1 || is.na(maxEditDistance) ||
      maxEditDistance < 0) {
    stop("maxEditDistance must be a non-negative integer")
  }
  option <- list(max_edit_distance = as.integer(maxEditDistance))
  if (isTRUE(names)) return(option)
  if (!is.character(names) || length(names) != 1) {
    stop("names must be the path of names.json, TRUE or NULL")
  }
  if (!file.exists(names)) stop(sprintf("names file does not exist: %s", names))
  option$table <- normalizePath(names, winslash = "/")
  option
}

# A resolved-names list from the engine as a data.frame.
.resolve_name_frame <- function(x) {
  columns <- c("name", "token", "route", "source", "via")
  cells <- lapply(columns, function(k) as.character(unlist(x[[k]])))
  names(cells) <- columns
  out <- as.data.frame(cells, stringsAsFactors = FALSE)
  out$distance <- as.integer(unlist(x$distance))
  if (!is.null(x$n_records)) out$n_records <- as.numeric(unlist(x$n_records))
  out
}
