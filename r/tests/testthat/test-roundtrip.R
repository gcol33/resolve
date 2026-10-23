# End-to-end round trip through the C ABI: train -> save -> load -> predict on a
# synthetic signal, asserting the reloaded model produces finite predictions
# correlated with the truth, and the checkpoint carries the training recipe.

test_that("train -> save -> load -> predict round trip recovers a signal", {
  skip_if_no_backend()
  skip_on_cran()

  n <- 400L
  tables <- signal_tables(n, n_species = 8L)
  model_file <- local_model_file()

  dataset <- load_tables(
    tables,
    roles = signal_roles(longitude = "lon", latitude = "lat"),
    targets = regression_target()
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
  expect_s4_class(fit$trainer, "Rcpp_Trainer")

  # Reload the saved checkpoint and predict on the full dataset.
  predictor <- resolve.load(model_file, device = "cpu")
  preds <- resolve.predict.dataset(predictor, dataset)

  # predict_dataset returns the documented C-ABI contract: a list with
  # `predictions` / `targets` / `plot_ids`, where `predictions` is keyed by
  # target name (resolve_capi.h resolve_predictor_predict_dataset).
  expect_true("y" %in% names(preds$predictions))
  pred_y <- as.numeric(preds$predictions$y)
  expect_equal(length(pred_y), n)
  expect_true(all(is.finite(pred_y)))
  # The reloaded model recovers the signal (an untrained/broken load would be ~0).
  expect_gt(cor(pred_y, tables$header$y), 0.9)
})

# A classification target's `probabilities` entry is the softmax row the
# predicted code was taken from: n_plots x n_classes, rows summing to one,
# columns named by the class labels in code order, and the row-wise argmax
# (0-based) equal to `predictions`. A regression target has no entry.
test_that("predict_dataset returns named class probabilities for a classification target", {
  skip_if_no_backend()
  skip_on_cran()

  n <- 240L
  labels <- c("forest", "grass", "scrub")
  tables <- signal_tables(n)
  tables$header$hab <- labels[1L + (seq_len(n) %% 3L)]
  model_file <- local_model_file()

  dataset <- load_tables(
    tables,
    roles = signal_roles(covariates = "cov1"),
    targets = list(
      y = list(column = "y", task = "regression"),
      hab = list(column = "hab", task = "classification", num_classes = 3)
    )
  )

  resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 5L, patience = 5L,
    lr = 1e-2, batchSize = 32L, testSize = 0.25, seed = 3L,
    savePath = model_file, verbose = FALSE
  )
  predictor <- resolve.load(model_file, device = "cpu")

  for (batch_size in c(-1L, 32L)) {
    preds <- resolve.predict.dataset(predictor, dataset, batchSize = batch_size)

    expect_equal(names(preds$probabilities), "hab")
    probs <- preds$probabilities$hab
    expect_true(is.matrix(probs))
    expect_equal(dim(probs), c(n, 3L))
    # The loader factorizes sorted-unique labels, so code k is labels[k + 1].
    expect_equal(colnames(probs), labels)
    expect_true(all(probs >= 0 & probs <= 1))
    expect_equal(rowSums(probs), rep(1, n), tolerance = 1e-5)

    codes <- as.integer(preds$predictions$hab)
    expect_equal(apply(probs, 1L, which.max) - 1L, codes)
  }

  one_shot <- resolve.predict.dataset(predictor, dataset, batchSize = -1L)
  chunked <- resolve.predict.dataset(predictor, dataset, batchSize = 32L)
  expect_equal(chunked$probabilities$hab, one_shot$probabilities$hab, tolerance = 1e-5)
})

test_that("load_train_config round-trips the training hyperparameters", {
  skip_if_no_backend()
  skip_on_cran()

  model_file <- local_model_file()
  dataset <- load_tables(signal_tables(120L), signal_roles(covariates = "cov1"),
                         regression_target())

  resolve.train.dataset(dataset, hiddenDims = c(16L), maxEpochs = 30L,
                        patience = 10L, batchSize = 32L, testSize = 0.25,
                        seed = 3L, lossConfig = "smape", savePath = model_file,
                        verbose = FALSE)

  cfg <- resolve.load_train_config(model_file)
  # Enums come back as their names and the requested batch size is
  # recoverable: the config list round-trips rather than carrying integer enum
  # codes or dropping the batch size.
  expect_equal(cfg$batch_size, 32L)
  expect_equal(cfg$max_epochs, 30L)
  expect_equal(cfg$patience, 10L)
  expect_equal(cfg$loss_config, "smape")
  expect_equal(cfg$device, "cpu")
})

test_that("fixedEpochs refits on every plot for a fixed duration", {
  skip_if_no_backend()
  skip_on_cran()

  model_file <- local_model_file()
  dataset <- load_tables(signal_tables(60L), signal_roles(covariates = "cov1"),
                         regression_target())

  fit <- resolve.train.dataset(dataset, hiddenDims = c(16L), maxEpochs = 20L,
                               fixedEpochs = 4L, batchSize = 16L, testSize = 0,
                               seed = 3L, savePath = model_file, verbose = FALSE)
  expect_length(fit$result$train_loss, 4L)
  expect_length(fit$result$test_loss, 0L)
  expect_equal(fit$result$best_epoch, 3L)
  expect_equal(resolve.load_train_config(model_file)$fixed_epochs, 4L)

  expect_error(resolve.train.dataset(dataset, maxEpochs = 20L, testSize = 0,
                                     verbose = FALSE),
               "fixedEpochs")
  expect_error(resolve.train.dataset(dataset, maxEpochs = 20L, fixedEpochs = 21L,
                                     verbose = FALSE),
               "maxEpochs")
})
