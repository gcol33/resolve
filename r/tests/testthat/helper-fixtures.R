# Shared builders for the testthat suite: temporary files and directories that
# are removed when the test that made them ends, the synthetic plot and species
# tables several files load or train on, and the model schema the model-level
# tests construct against.

# Registers `cleanup` (a function of no arguments) to run when the frame `env`
# exits, as on.exit(add = TRUE) would inside that frame.
defer_cleanup <- function(cleanup, env) {
  do.call(base::on.exit, list(as.call(list(cleanup)), add = TRUE), envir = env)
}

# Writes `x` to a temporary CSV removed when `env` exits and returns its path.
# A data frame is written without row names and with missing cells empty; a
# character vector is written verbatim, one element per line.
local_csv <- function(x, env = parent.frame()) {
  path <- tempfile(fileext = ".csv")
  if (is.character(x)) {
    writeLines(x, path)
  } else {
    utils::write.csv(x, path, row.names = FALSE, na = "")
  }
  defer_cleanup(function() unlink(path), env)
  path
}

# The header (plot-level) and species (record-level) CSVs of one dataset.
local_csv_pair <- function(header, species, env = parent.frame()) {
  list(header = local_csv(header, env), species = local_csv(species, env))
}

# A checkpoint path; the checkpoint and the JSON sidecar saved beside it are
# removed when `env` exits.
local_model_file <- function(env = parent.frame()) {
  path <- tempfile(fileext = ".pt")
  defer_cleanup(function() unlink(c(path, sub("\\.pt$", ".json", path))), env)
  path
}

# An empty temporary directory, removed with its contents when `env` exits.
local_dir <- function(prefix = "resolve_", env = parent.frame()) {
  path <- tempfile(prefix)
  dir.create(path)
  defer_cleanup(function() unlink(path, recursive = TRUE), env)
  path
}

# Schema for the tests that construct a ResolveModel directly: `n_plots` plots
# of `n_species` hashed species with coordinates, no abundance or taxonomy, and
# one regression target. Named arguments in `...` replace or add fields.
fixture_schema <- function(..., n_plots = 50L, n_species = 20L) {
  schema <- list(
    n_plots = n_plots,
    n_species = n_species,
    has_coordinates = TRUE,
    has_abundance = FALSE,
    has_taxonomy = FALSE,
    n_genera = 0L,
    n_families = 0L,
    dataset_config = list(track_unknown_fraction = FALSE, track_unknown_count = FALSE),
    targets = list(area = list(task = "regression", transform = "none"))
  )
  overrides <- list(...)
  schema[names(overrides)] <- overrides
  schema
}

# Three plots with coordinates and an area target, and two species records per
# plot drawn from three species in two genera of Fagaceae and one of Pinaceae.
taxonomy_tables <- function() {
  list(
    header = data.frame(
      plot_id = c("p1", "p2", "p3"),
      lon = c(10.0, 11.0, 12.0),
      lat = c(50.0, 51.0, 52.0),
      area = c(100.0, 200.0, 150.0),
      stringsAsFactors = FALSE
    ),
    species = data.frame(
      plot_id = c("p1", "p1", "p2", "p2", "p3", "p3"),
      species_id = c("sp1", "sp2", "sp1", "sp3", "sp2", "sp3"),
      cover = c(0.5, 0.5, 0.8, 0.2, 0.6, 0.4),
      genus = c("Quercus", "Fagus", "Quercus", "Pinus", "Fagus", "Pinus"),
      family = c("Fagaceae", "Fagaceae", "Fagaceae", "Pinaceae", "Fagaceae", "Pinaceae"),
      stringsAsFactors = FALSE
    )
  )
}

taxonomy_roles <- function() {
  list(plot_id = "plot_id", species_id = "species_id", abundance = "cover",
       longitude = "lon", latitude = "lat", genus = "genus", family = "family")
}

# `n` plots with coordinates, two smooth covariates `cov1` and `cov2`, and a
# regression target `y = 2 cov1 - 1.5 cov2 + 3`; the species table holds one
# record per plot (cover 1) cycling through `n_species` species `sp0`, `sp1`, ...
signal_tables <- function(n, n_species = 6L) {
  i <- seq_len(n)
  cov1 <- sin(i * 0.13)
  cov2 <- cos(i * 0.07)
  list(
    header = data.frame(
      plot_id = paste0("P", i),
      lon = -5.0 + i * 0.001,
      lat = 40.0 + i * 0.001,
      cov1 = cov1,
      cov2 = cov2,
      y = 2.0 * cov1 - 1.5 * cov2 + 3.0,
      stringsAsFactors = FALSE
    ),
    species = data.frame(
      plot_id = paste0("P", i),
      species_id = paste0("sp", i %% n_species),
      cover = 1.0,
      stringsAsFactors = FALSE
    )
  )
}

# Roles for `signal_tables()`, reading the covariates named in `covariates`.
signal_roles <- function(covariates = c("cov1", "cov2"), ...) {
  c(list(plot_id = "plot_id", species_id = "species_id", abundance = "cover",
         covariates = covariates), list(...))
}

# The small hashed encoding the training tests fit quickly; settings in `...`
# replace or add fields.
small_hash_config <- function(...) {
  utils::modifyList(list(species_encoding = "hash", hash_dim = 4, top_k = 2), list(...))
}

# Writes `tables` (from `signal_tables()` or a variant) to CSVs that live as
# long as `env` and loads them as a dataset.
load_tables <- function(tables, roles, targets, config = small_hash_config(),
                        env = parent.frame(), ...) {
  files <- local_csv_pair(tables$header, tables$species, env = env)
  resolve.dataset.csv(header = files$header, species = files$species,
                      roles = roles, targets = targets, config = config, ...)
}

regression_target <- function(column = "y") {
  stats::setNames(list(list(column = column, task = "regression")), column)
}
