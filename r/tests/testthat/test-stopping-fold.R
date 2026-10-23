# An explicit stopping fold: resolve.train.dataset(stoppingPlots = ...) holds
# out the named plots for early stopping instead of a random testSize fraction.

test_that("the named plots are the held-out fold and the checkpoint says so", {
  skip_if_no_backend()
  skip_on_cran()

  ds <- load_tables(signal_tables(40L), signal_roles(), regression_target())
  fold <- paste0("P", c(3L, 17L, 8L, 30L, 22L))
  path <- local_model_file()

  fit <- resolve.train.dataset(ds, hiddenDims = c(8L, 4L), maxEpochs = 3L,
                               patience = 3L, batchSize = 16L,
                               stoppingPlots = fold, seed = 5L,
                               savePath = path, verbose = FALSE)

  expect_identical(fit$trainer$test_plot_ids(), fold)
  expect_length(fit$trainer$train_plot_ids(), 35L)
  expect_length(intersect(fit$trainer$train_plot_ids(), fold), 0L)

  meta <- fit$trainer$run_metadata()
  expect_identical(meta$hold_out, "explicit")
  expect_identical(meta$split_seed, 5L)
  expect_identical(meta$n_plots_test, 5L)

  saved <- resolve.load_run_metadata(path)
  expect_identical(saved$hold_out, "explicit")
  expect_identical(saved$n_plots_test, 5L)
  expect_identical(saved$best_epoch, fit$result$best_epoch)
})

test_that("a random split records its fraction and seed", {
  skip_if_no_backend()
  skip_on_cran()

  ds <- load_tables(signal_tables(40L), signal_roles(), regression_target())
  path <- local_model_file()
  fit <- resolve.train.dataset(ds, hiddenDims = c(8L, 4L), maxEpochs = 2L,
                               patience = 2L, batchSize = 16L, testSize = 0.25,
                               seed = 9L, savePath = path, verbose = FALSE)
  saved <- resolve.load_run_metadata(path)
  expect_identical(saved$hold_out, "random")
  expect_equal(saved$test_size, 0.25)
  expect_identical(saved$split_seed, 9L)
})

test_that("an unusable stopping fold is refused", {
  skip_if_no_backend()
  skip_on_cran()

  ds <- load_tables(signal_tables(20L), signal_roles(), regression_target())
  train <- function(...) {
    resolve.train.dataset(ds, hiddenDims = c(8L, 4L), maxEpochs = 1L,
                          patience = 1L, verbose = FALSE, ...)
  }
  expect_error(train(stoppingPlots = c("P1", "no_such_plot")), "no_such_plot")
  expect_error(train(stoppingPlots = c("P2", "P2")), "more than once")
  expect_error(train(stoppingPlots = paste0("P", 1:20)), "none to fit")
  expect_error(train(stoppingPlots = character()), "non-empty character")
  expect_error(train(stoppingPlots = c("P1", NA)), "non-empty character")
  expect_error(train(stoppingPlots = "P1", testSize = 0.3), "not both")
})
