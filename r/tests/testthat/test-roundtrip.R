# End-to-end round trip through the C ABI (issue #89): the R bindings exist for
# the Windows resolve_c path (issue #17), so the train -> save -> load -> predict
# chain that actually matters had no coverage (test-trainer.R only constructs a
# Trainer). This drives the whole chain on a synthetic signal and asserts the
# reloaded model produces finite predictions correlated with the truth.

test_that("train -> save -> load -> predict round trip recovers a signal", {
  skip_on_cran()

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")
  model_file <- tempfile(fileext = ".pt")
  on.exit({
    unlink(header_file)
    unlink(species_file)
    unlink(model_file)
    unlink(sub("\\.pt$", ".json", model_file))
  }, add = TRUE)

  # Deterministic linear signal in two covariates.
  n <- 400L
  c1 <- sin(seq_len(n) * 0.13)
  c2 <- cos(seq_len(n) * 0.07)
  y <- 2.0 * c1 - 1.5 * c2 + 3.0

  header_data <- data.frame(
    plot_id = paste0("P", seq_len(n)),
    lon = -5.0 + seq_len(n) * 0.001,
    lat = 40.0 + seq_len(n) * 0.001,
    cov1 = c1,
    cov2 = c2,
    y = y,
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)

  species_data <- data.frame(
    plot_id = paste0("P", seq_len(n)),
    species_id = paste0("sp", seq_len(n) %% 8L),
    cover = 1.0,
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  dataset <- resolve.dataset.csv(
    header = header_file,
    species = species_file,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id",
      abundance = "cover",
      longitude = "lon",
      latitude = "lat",
      covariates = c("cov1", "cov2")
    ),
    targets = list(
      y = list(column = "y", task = "regression")
    ),
    config = list(species_encoding = "hash", hash_dim = 4, top_k = 2)
  )
  expect_equal(dataset$n_plots(), n)

  fit <- resolve.train.dataset(
    dataset,
    hiddenDims = c(32L, 16L),
    maxEpochs = 300L,
    patience = 40L,
    lr = 1e-2,
    batchSize = 64L,
    testSize = 0.25,
    seed = 11L,
    savePath = model_file,
    verbose = FALSE
  )
  expect_true(file.exists(model_file))
  expect_true(!is.null(fit$trainer))

  # Reload the saved checkpoint and predict on the full dataset.
  predictor <- resolve.load(model_file, device = "cpu")
  preds <- resolve.predict.dataset(predictor, dataset)

  expect_true("y" %in% names(preds))
  pred_y <- as.numeric(preds$y)
  expect_equal(length(pred_y), n)
  expect_true(all(is.finite(pred_y)))
  # The reloaded model recovers the signal (an untrained/broken load would be ~0).
  expect_gt(cor(pred_y, y), 0.9)
})

test_that("load_train_config round-trips the training hyperparameters", {
  skip_on_cran()

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")
  model_file <- tempfile(fileext = ".pt")
  on.exit({
    unlink(header_file)
    unlink(species_file)
    unlink(model_file)
    unlink(sub("\\.pt$", ".json", model_file))
  }, add = TRUE)

  n <- 120L
  header_data <- data.frame(
    plot_id = paste0("P", seq_len(n)),
    cov1 = sin(seq_len(n) * 0.11),
    y = 1.7 * sin(seq_len(n) * 0.11) + 2.0,
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)
  species_data <- data.frame(
    plot_id = paste0("P", seq_len(n)),
    species_id = paste0("sp", seq_len(n) %% 6L),
    cover = 1.0,
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  dataset <- resolve.dataset.csv(
    header = header_file, species = species_file,
    roles = list(plot_id = "plot_id", species_id = "species_id",
                 abundance = "cover", covariates = c("cov1")),
    targets = list(y = list(column = "y", task = "regression")),
    config = list(species_encoding = "hash", hash_dim = 4, top_k = 2)
  )

  resolve.train.dataset(dataset, hiddenDims = c(16L), maxEpochs = 30L,
                        patience = 10L, batchSize = 32L, testSize = 0.25,
                        seed = 3L, savePath = model_file, verbose = FALSE)

  cfg <- resolve.load_train_config(model_file)
  # Enums come back as strings and the requested batch size is recoverable
  # (issues #86 / #87): the config list must round-trip, not carry integer enum
  # codes or drop the batch size.
  expect_equal(cfg$batch_size, 32L)
  expect_true(is.character(cfg$loss_config))
  expect_true(!is.null(cfg$device))
})
