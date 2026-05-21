# Tests for R binding layer registration and input validation
#
# These tests verify:
#   1. Rcpp module registration (class and function names)
#   2. Version string format
#   3. Modern R wrapper input validation (works WITHOUT libtorch)
#   4. Factory function exports
#   5. Removed legacy entry points raise clear errors

# =============================================================================
# 1. Module registration
# =============================================================================

test_that("resolve_module is loaded", {
  skip_on_cran()
  mod <- tryCatch(resolve:::.resolve_module, error = function(e) NULL)
  skip_if(is.null(mod), "resolve module not available (needs libtorch)")

  # NB: `names(mod)` on an Rcpp `Module` S4 object is NOT the registration
  # list — it falls through to the underlying `.xData` environment (Rcpp
  # internals + lazily cached `$` accesses). Use the internal introspection
  # helper that reads `Module__classes_info` / `Module__functions_names`
  # directly off the module pointer.
  registered <- resolve:::.resolve_module_registered()

  # Verify core classes are registered. SpeciesEncoder was removed when the
  # unified resolve::SpeciesEncoder C++ class was split into RankPoolEncoder
  # and EmbeddingEncoder — see ?resolve.encoder for the full rationale.
  expect_false("SpeciesEncoder" %in% registered)
  expect_true("ResolveModel" %in% registered)
  expect_true("Trainer" %in% registered)
  expect_true("Predictor" %in% registered)
  expect_true("ResolveDataset" %in% registered)
})

test_that("module factory functions are registered", {
  skip_on_cran()
  mod <- tryCatch(resolve:::.resolve_module, error = function(e) NULL)
  skip_if(is.null(mod), "resolve module not available (needs libtorch)")

  registered <- resolve:::.resolve_module_registered()

  expect_true("ResolveDataset_from_csv" %in% registered)
  expect_true("ResolveDataset_from_species_csv" %in% registered)
  expect_true("Predictor_load" %in% registered)
  # SpeciesEncoder_load was removed alongside SpeciesEncoder.
  expect_false("SpeciesEncoder_load" %in% registered)
})


# =============================================================================
# 2. Version
# =============================================================================

test_that("resolve.version returns valid semver string", {
  skip_on_cran()
  v <- tryCatch(resolve.version(), error = function(e) NULL)
  skip_if(is.null(v), "resolve C++ not available")

  expect_type(v, "character")
  expect_length(v, 1)
  expect_match(v, "^[0-9]+\\.[0-9]+\\.[0-9]+$")
})


# =============================================================================
# 3. Modern R wrapper input validation (no libtorch required)
# =============================================================================

# --- resolve.load ---

test_that("resolve.load rejects non-string path", {
  expect_error(resolve.load(123), "path")
  expect_error(resolve.load(c("a.pt", "b.pt")), "path")
})

test_that("resolve.load rejects non-existent file", {
  expect_error(resolve.load("/nonexistent/model.pt"), "does not exist")
})

test_that("resolve.load rejects invalid device", {
  tmp <- tempfile(fileext = ".pt")
  writeLines("fake", tmp)
  on.exit(unlink(tmp), add = TRUE)

  expect_error(resolve.load(tmp, device = "tpu"), "device")
  expect_error(resolve.load(tmp, device = "mps"), "device")
})


# --- resolve.save ---

test_that("resolve.save rejects invalid trainer types", {
  expect_error(resolve.save("not_a_trainer", "out.pt"), "trainer must be")
  expect_error(resolve.save(42, "out.pt"), "trainer must be")
  expect_error(resolve.save(data.frame(), "out.pt"), "trainer must be")
})


# --- resolve.dataset.csv ---

test_that("resolve.dataset.csv rejects non-string header path", {
  expect_error(resolve.dataset.csv(header = 123, species = "sp.csv"), "header")
})

test_that("resolve.dataset.csv rejects non-existent header file", {
  tmp_species <- tempfile(fileext = ".csv")
  writeLines("plot_id,species_id\np1,sp1", tmp_species)
  on.exit(unlink(tmp_species), add = TRUE)

  expect_error(
    resolve.dataset.csv(
      header = "/nonexistent/header.csv",
      species = tmp_species,
      targets = list(area = list(column = "area", task = "regression"))
    ),
    "does not exist"
  )
})

test_that("resolve.dataset.csv rejects non-string species path", {
  tmp_header <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_header)
  on.exit(unlink(tmp_header), add = TRUE)

  expect_error(
    resolve.dataset.csv(header = tmp_header, species = 42),
    "species"
  )
})

test_that("resolve.dataset.csv rejects non-existent species file", {
  tmp_header <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_header)
  on.exit(unlink(tmp_header), add = TRUE)

  expect_error(
    resolve.dataset.csv(
      header = tmp_header,
      species = "/nonexistent/species.csv",
      targets = list(area = list(column = "area", task = "regression"))
    ),
    "does not exist"
  )
})

test_that("resolve.dataset.csv rejects non-list roles", {
  tmp_h <- tempfile(fileext = ".csv")
  tmp_s <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_h)
  writeLines("plot_id,species_id\np1,sp1", tmp_s)
  on.exit({unlink(tmp_h); unlink(tmp_s)}, add = TRUE)

  expect_error(
    resolve.dataset.csv(
      header = tmp_h,
      species = tmp_s,
      roles = "not_a_list",
      targets = list(area = list(column = "area", task = "regression"))
    ),
    "roles"
  )
})

test_that("resolve.dataset.csv rejects empty targets", {
  tmp_h <- tempfile(fileext = ".csv")
  tmp_s <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_h)
  writeLines("plot_id,species_id\np1,sp1", tmp_s)
  on.exit({unlink(tmp_h); unlink(tmp_s)}, add = TRUE)

  expect_error(
    resolve.dataset.csv(header = tmp_h, species = tmp_s, targets = list()),
    "targets must not be empty"
  )
})

test_that("resolve.dataset.csv rejects target without column", {
  tmp_h <- tempfile(fileext = ".csv")
  tmp_s <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_h)
  writeLines("plot_id,species_id\np1,sp1", tmp_s)
  on.exit({unlink(tmp_h); unlink(tmp_s)}, add = TRUE)

  expect_error(
    resolve.dataset.csv(
      header = tmp_h,
      species = tmp_s,
      targets = list(area = list(task = "regression"))
    ),
    "column"
  )
})

test_that("resolve.dataset.csv rejects invalid target task", {
  tmp_h <- tempfile(fileext = ".csv")
  tmp_s <- tempfile(fileext = ".csv")
  writeLines("plot_id,area\np1,100", tmp_h)
  writeLines("plot_id,species_id\np1,sp1", tmp_s)
  on.exit({unlink(tmp_h); unlink(tmp_s)}, add = TRUE)

  expect_error(
    resolve.dataset.csv(
      header = tmp_h,
      species = tmp_s,
      targets = list(area = list(column = "area", task = "segmentation"))
    ),
    "task"
  )
})


# --- resolve.train.dataset ---

test_that("resolve.train.dataset rejects non-ResolveDataset input", {
  expect_error(
    resolve.train.dataset(data.frame()),
    "resolve\\.dataset\\.csv"
  )
  expect_error(
    resolve.train.dataset(list()),
    "resolve\\.dataset\\.csv"
  )
})

test_that("resolve.train.dataset rejects invalid maxEpochs", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, maxEpochs = 0), "maxEpochs")
  expect_error(resolve.train.dataset(fake_ds, maxEpochs = -1), "maxEpochs")
})

test_that("resolve.train.dataset rejects invalid patience", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, patience = 0), "patience")
})

test_that("resolve.train.dataset rejects invalid lr", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, lr = 0), "lr")
  expect_error(resolve.train.dataset(fake_ds, lr = -0.01), "lr")
})

test_that("resolve.train.dataset rejects invalid batchSize", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, batchSize = 0), "batchSize")
})

test_that("resolve.train.dataset rejects invalid device", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, device = "tpu"), "device")
})

test_that("resolve.train.dataset rejects invalid testSize", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, testSize = 0), "testSize")
  expect_error(resolve.train.dataset(fake_ds, testSize = 1), "testSize")
  expect_error(resolve.train.dataset(fake_ds, testSize = 1.5), "testSize")
})

test_that("resolve.train.dataset rejects invalid lossConfig", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(resolve.train.dataset(fake_ds, lossConfig = "mse"), "lossConfig")
})


# --- resolve.predict.dataset ---

test_that("resolve.predict.dataset rejects invalid predictor type", {
  fake_ds <- structure(list(), class = "Rcpp_ResolveDataset")

  expect_error(
    resolve.predict.dataset("not_a_predictor", fake_ds),
    "resolve\\.load"
  )
})

test_that("resolve.predict.dataset rejects invalid dataset type", {
  fake_pred <- structure(list(), class = "Rcpp_Predictor")

  expect_error(
    resolve.predict.dataset(fake_pred, data.frame()),
    "resolve\\.dataset\\.csv"
  )
})


# =============================================================================
# 4. Factory function exports in namespace
# =============================================================================

test_that("all exported R wrapper functions exist", {
  ns <- asNamespace("resolve")

  # Legacy facades retained as stub errors (so explicit "removed" messages
  # show up at call time, rather than "could not find function").
  expect_true(exists("resolve.encoder", envir = ns))
  expect_true(exists("resolve.dataset", envir = ns))
  expect_true(exists("resolve.train", envir = ns))
  expect_true(exists("resolve.predict", envir = ns))

  # Live API.
  expect_true(exists("resolve.load", envir = ns))
  expect_true(exists("resolve.save", envir = ns))
  expect_true(exists("resolve.progress", envir = ns))
  expect_true(exists("resolve.version", envir = ns))
  expect_true(exists("resolve.dataset.csv", envir = ns))
  expect_true(exists("resolve.train.dataset", envir = ns))
  expect_true(exists("resolve.predict.dataset", envir = ns))
})

test_that("all exported metric functions exist", {
  ns <- asNamespace("resolve")

  expect_true(exists("resolve_mae", envir = ns))
  expect_true(exists("resolve_rmse", envir = ns))
  expect_true(exists("resolve_smape", envir = ns))
  expect_true(exists("resolve_r_squared", envir = ns))
  expect_true(exists("resolve_band_accuracy", envir = ns))
  expect_true(exists("resolve_accuracy", envir = ns))
})

test_that("exported functions are actual functions, not stubs", {
  ns <- asNamespace("resolve")

  expect_true(is.function(get("resolve.load", envir = ns)))
  expect_true(is.function(get("resolve.dataset.csv", envir = ns)))
  expect_true(is.function(get("resolve.train.dataset", envir = ns)))
  expect_true(is.function(get("resolve.predict.dataset", envir = ns)))
  expect_true(is.function(get("resolve_mae", envir = ns)))
})


# =============================================================================
# 5. Wrapper function signatures
# =============================================================================

test_that("resolve.train.dataset has expected formal arguments", {
  args <- names(formals(resolve.train.dataset))

  expect_true("dataset" %in% args)
  expect_true("hiddenDims" %in% args)
  expect_true("maxEpochs" %in% args)
  expect_true("patience" %in% args)
  expect_true("lr" %in% args)
  expect_true("batchSize" %in% args)
  expect_true("device" %in% args)
  expect_true("lossConfig" %in% args)
  # RankPool/Transformer options
  expect_true("coverDropout" %in% args)
  expect_true("dModel" %in% args)
  expect_true("nHeads" %in% args)
  expect_true("nAttentionLayers" %in% args)
  expect_true("transformerFfDim" %in% args)
  expect_true("transformerPooling" %in% args)
  expect_true("transformerDropout" %in% args)
})

test_that("resolve.dataset.csv has expected formal arguments", {
  args <- names(formals(resolve.dataset.csv))

  expect_true("header" %in% args)
  expect_true("species" %in% args)
  expect_true("roles" %in% args)
  expect_true("targets" %in% args)
  expect_true("config" %in% args)
})

test_that("resolve.load has expected formal arguments", {
  args <- names(formals(resolve.load))

  expect_true("path" %in% args)
  expect_true("device" %in% args)
})


# =============================================================================
# 6. Default argument values
# =============================================================================

test_that("resolve.load defaults are sensible", {
  defaults <- formals(resolve.load)

  expect_equal(eval(defaults$device), "cpu")
})
