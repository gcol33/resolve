# Tests for R binding layer registration and input validation
#
# These tests verify:
#   1. Rcpp module registration (class and function names)
#   2. Version string format
#   3. R wrapper input validation (works WITHOUT libtorch)
#   4. Factory function exports
#   5. Wrapper function argument handling


# =============================================================================
# 1. Module registration
# =============================================================================

test_that("resolve_module is loaded", {
  skip_on_cran()
  mod <- tryCatch(resolve:::.resolve_module, error = function(e) NULL)
  skip_if(is.null(mod), "resolve module not available (needs libtorch)")

  # Verify core classes are registered
  expect_true("SpeciesEncoder" %in% names(mod))
  expect_true("ResolveModel" %in% names(mod))
  expect_true("Trainer" %in% names(mod))
  expect_true("Predictor" %in% names(mod))
  expect_true("ResolveDataset" %in% names(mod))
})

test_that("module factory functions are registered", {
  skip_on_cran()
  mod <- tryCatch(resolve:::.resolve_module, error = function(e) NULL)
  skip_if(is.null(mod), "resolve module not available (needs libtorch)")

  expect_true("ResolveDataset_from_csv" %in% names(mod))
  expect_true("ResolveDataset_from_species_csv" %in% names(mod))
  expect_true("Predictor_load" %in% names(mod))
  expect_true("SpeciesEncoder_load" %in% names(mod))
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
# 3. R wrapper input validation (no libtorch required)
# =============================================================================

# --- resolve.encoder ---

test_that("resolve.encoder rejects invalid hashDim", {
  expect_error(resolve.encoder(hashDim = 0), "hashDim")
  expect_error(resolve.encoder(hashDim = -5), "hashDim")
  expect_error(resolve.encoder(hashDim = "abc"), "hashDim")
})

test_that("resolve.encoder rejects invalid topK", {
  expect_error(resolve.encoder(topK = 0), "topK")
  expect_error(resolve.encoder(topK = -1), "topK")
})

test_that("resolve.encoder rejects invalid aggregation", {
  expect_error(resolve.encoder(aggregation = "bad"), "aggregation")
})

test_that("resolve.encoder rejects invalid normalization", {
  expect_error(resolve.encoder(normalization = "bad"), "normalization")
})

test_that("resolve.encoder rejects invalid selection", {
  expect_error(resolve.encoder(selection = "invalid"), "selection")
})

test_that("resolve.encoder rejects invalid representation", {
  expect_error(resolve.encoder(representation = "bad"), "representation")
})

test_that("resolve.encoder rejects invalid minSpeciesFrequency", {
  expect_error(resolve.encoder(minSpeciesFrequency = 0), "minSpeciesFrequency")
  expect_error(resolve.encoder(minSpeciesFrequency = -1), "minSpeciesFrequency")
})

test_that("resolve.encoder accepts all valid parameter combinations", {
  # These should only fail at the C++ level (module not loaded), not R validation
  for (agg in c("abundance", "count")) {
    for (norm in c("raw", "norm", "log1p")) {
      for (sel in c("top", "bottom", "top_bottom", "all")) {
        for (rep in c("abundance", "presence_absence")) {
          err <- tryCatch(
            resolve.encoder(
              aggregation = agg,
              normalization = norm,
              selection = sel,
              representation = rep
            ),
            error = function(e) e$message
          )
          # If it errors, the error must NOT be about R-level validation
          if (is.character(err)) {
            expect_false(grepl("aggregation|normalization|selection|representation", err),
                         info = sprintf("agg=%s norm=%s sel=%s rep=%s", agg, norm, sel, rep))
          }
        }
      }
    }
  }
})


# --- resolve.train ---

test_that("resolve.train rejects non-resolve.dataset input", {
  expect_error(
    resolve.train(data.frame(x = 1)),
    "resolve\\.dataset"
  )
  expect_error(
    resolve.train(list(a = 1)),
    "resolve\\.dataset"
  )
})

test_that("resolve.train rejects invalid speciesEncoding", {
  # Create a minimal resolve.dataset-like object to pass the class check
  fake_ds <- structure(list(), class = "resolve.dataset")

  expect_error(
    resolve.train(fake_ds, speciesEncoding = "invalid"),
    "speciesEncoding"
  )
  expect_error(
    resolve.train(fake_ds, speciesEncoding = ""),
    "speciesEncoding"
  )
  expect_error(
    resolve.train(fake_ds, speciesEncoding = "hashing"),
    "speciesEncoding"
  )
})

test_that("resolve.train accepts all valid speciesEncoding values", {
  fake_ds <- structure(list(speciesIds = matrix(1L)), class = "resolve.dataset")

  for (enc in c("hash", "embed", "sparse", "rank_pool", "transformer")) {
    err <- tryCatch(
      resolve.train(fake_ds, speciesEncoding = enc),
      error = function(e) e$message
    )
    # Should fail downstream (no encoder, no C++), but NOT on speciesEncoding
    expect_false(grepl("speciesEncoding", err),
                 info = paste("encoding:", enc))
  }
})

test_that("resolve.train rejects invalid lossConfig", {
  fake_ds <- structure(list(), class = "resolve.dataset")

  expect_error(
    resolve.train(fake_ds, lossConfig = "invalid"),
    "lossConfig"
  )
})

test_that("resolve.train accepts all valid lossConfig values", {
  fake_ds <- structure(list(), class = "resolve.dataset")

  for (loss in c("mae", "smape", "combined")) {
    err <- tryCatch(
      resolve.train(fake_ds, lossConfig = loss),
      error = function(e) e$message
    )
    # Should fail downstream, but NOT on lossConfig
    expect_false(grepl("lossConfig", err),
                 info = paste("loss:", loss))
  }
})

test_that("resolve.train requires speciesIds for embed mode", {
  fake_ds <- structure(list(speciesIds = NULL), class = "resolve.dataset")

  expect_error(
    resolve.train(fake_ds, speciesEncoding = "embed"),
    "speciesIds"
  )
})


# --- resolve.load ---

test_that("resolve.load rejects non-string path", {
  expect_error(resolve.load(123), "path")
  expect_error(resolve.load(c("a.pt", "b.pt")), "path")
})

test_that("resolve.load rejects non-existent file", {
  expect_error(resolve.load("/nonexistent/model.pt"), "does not exist")
})

test_that("resolve.load rejects invalid device", {
  # Use a path that doesn't exist - device check should come after path check
  # but let's test with a temp file to isolate the device error
  tmp <- tempfile(fileext = ".pt")
  writeLines("fake", tmp)
  on.exit(unlink(tmp), add = TRUE)

  expect_error(resolve.load(tmp, device = "tpu"), "device")
  expect_error(resolve.load(tmp, device = "mps"), "device")
})


# --- resolve.predict ---

test_that("resolve.predict rejects invalid outputSpace", {
  fake_predictor <- structure(list(), class = "Rcpp_Predictor")
  fake_ds <- structure(list(), class = "resolve.dataset")

  expect_error(
    resolve.predict(fake_predictor, fake_ds, outputSpace = "bad"),
    "outputSpace"
  )
})

test_that("resolve.predict rejects invalid model types", {
  fake_ds <- structure(list(), class = "resolve.dataset")

  expect_error(
    resolve.predict("not_a_model", fake_ds),
    "model must be"
  )
  expect_error(
    resolve.predict(42, fake_ds),
    "model must be"
  )
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

  # Main API

  expect_true(exists("resolve.encoder", envir = ns))
  expect_true(exists("resolve.dataset", envir = ns))
  expect_true(exists("resolve.train", envir = ns))
  expect_true(exists("resolve.predict", envir = ns))
  expect_true(exists("resolve.load", envir = ns))
  expect_true(exists("resolve.save", envir = ns))
  expect_true(exists("resolve.progress", envir = ns))
  expect_true(exists("resolve.version", envir = ns))

  # C++ API wrappers
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

  expect_true(is.function(get("resolve.encoder", envir = ns)))
  expect_true(is.function(get("resolve.train", envir = ns)))
  expect_true(is.function(get("resolve.predict", envir = ns)))
  expect_true(is.function(get("resolve.load", envir = ns)))
  expect_true(is.function(get("resolve.dataset.csv", envir = ns)))
  expect_true(is.function(get("resolve_mae", envir = ns)))
})


# =============================================================================
# 5. Wrapper function signatures
# =============================================================================

test_that("resolve.encoder has expected formal arguments", {
  args <- names(formals(resolve.encoder))

  expect_true("hashDim" %in% args)
  expect_true("topK" %in% args)
  expect_true("aggregation" %in% args)
  expect_true("normalization" %in% args)
  expect_true("trackUnknownCount" %in% args)
  expect_true("selection" %in% args)
  expect_true("representation" %in% args)
  expect_true("minSpeciesFrequency" %in% args)
})

test_that("resolve.train has expected formal arguments", {
  args <- names(formals(resolve.train))

  expect_true("dataset" %in% args)
  expect_true("speciesEncoding" %in% args)
  expect_true("hiddenDims" %in% args)
  expect_true("maxEpochs" %in% args)
  expect_true("patience" %in% args)
  expect_true("lr" %in% args)
  expect_true("batchSize" %in% args)
  expect_true("device" %in% args)
  expect_true("savePath" %in% args)
  expect_true("lossConfig" %in% args)
  expect_true("verbose" %in% args)
})

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

test_that("resolve.predict has expected formal arguments", {
  args <- names(formals(resolve.predict))

  expect_true("model" %in% args)
  expect_true("dataset" %in% args)
  expect_true("returnLatent" %in% args)
  expect_true("outputSpace" %in% args)
  expect_true("confidenceThreshold" %in% args)
})

test_that("resolve.load has expected formal arguments", {
  args <- names(formals(resolve.load))

  expect_true("path" %in% args)
  expect_true("device" %in% args)
})


# =============================================================================
# 6. Default argument values
# =============================================================================

test_that("resolve.encoder defaults are sensible", {
  defaults <- formals(resolve.encoder)

  expect_equal(eval(defaults$hashDim), 32L)
  expect_equal(eval(defaults$topK), 3L)
  expect_equal(eval(defaults$aggregation), "abundance")
  expect_equal(eval(defaults$normalization), "norm")
  expect_equal(eval(defaults$trackUnknownCount), FALSE)
  expect_equal(eval(defaults$selection), "top")
  expect_equal(eval(defaults$representation), "abundance")
  expect_equal(eval(defaults$minSpeciesFrequency), 1L)
})

test_that("resolve.train defaults are sensible", {
  defaults <- formals(resolve.train)

  expect_equal(eval(defaults$speciesEncoding), "hash")
  expect_equal(eval(defaults$maxEpochs), 500L)
  expect_equal(eval(defaults$patience), 50L)
  expect_equal(eval(defaults$lr), 1e-3)
  expect_equal(eval(defaults$batchSize), 4096L)
  expect_equal(eval(defaults$device), "cpu")
  expect_equal(eval(defaults$testSize), 0.2)
  expect_equal(eval(defaults$seed), 42L)
  expect_equal(eval(defaults$lossConfig), "mae")
  expect_equal(eval(defaults$verbose), TRUE)
})

test_that("resolve.load defaults are sensible", {
  defaults <- formals(resolve.load)

  expect_equal(eval(defaults$device), "cpu")
})

test_that("resolve.predict defaults are sensible", {
  defaults <- formals(resolve.predict)

  expect_equal(eval(defaults$returnLatent), FALSE)
  expect_equal(eval(defaults$outputSpace), "raw")
  expect_equal(eval(defaults$confidenceThreshold), 0.0)
})
