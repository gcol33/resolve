# Per-target missing-label masking: the `missing` key of a target
# specification. "drop" (default) leaves a plot without that label out of the
# fit; "mask" keeps it, and it trains and is scored on its other targets.

# The signal tables with every third `y` missing and a two-class target `cls`
# missing on a different, overlapping set of plots. The CSVs live as long as
# `env`; the returned indices (0-based) are the plots labelled for each target.
masking_files <- function(n = 160L, env = parent.frame()) {
  tables <- signal_tables(n)
  i <- seq_len(n) - 1L
  header <- tables$header
  header$y[i %% 3L == 0L] <- NA
  header$cls <- ifelse(header$cov1 + header$cov2 > 0, "pos", "neg")
  header$cls[i %% 4L == 1L | i %% 12L == 0L] <- NA
  files <- local_csv_pair(header, tables$species, env = env)
  c(files, list(labelled_y = i[i %% 3L != 0L],
                labelled_cls = i[!(i %% 4L == 1L | i %% 12L == 0L)]))
}

masking_dataset <- function(files, missing) {
  resolve.dataset.csv(
    header = files$header, species = files$species,
    roles = signal_roles(),
    targets = list(
      y = list(column = "y", task = "regression", missing = missing),
      cls = list(column = "cls", task = "classification", missing = missing)
    ),
    config = small_hash_config(track_unknown_fraction = FALSE)
  )
}

test_that("missing is an accepted target key and its value is checked", {
  expect_silent(resolveR:::.resolve_normalize_roles_targets(
    list(plot_id = "plot_id"),
    list(y = list(column = "y", task = "regression", missing = "mask"))))
})

test_that("mask keeps partly labelled plots with the sentinels", {
  skip_if_no_backend()
  skip_on_cran()
  files <- masking_files()

  drop <- masking_dataset(files, "drop")
  mask <- masking_dataset(files, "mask")
  expect_equal(drop$n_plots(),
               length(intersect(files$labelled_y, files$labelled_cls)))
  expect_equal(mask$n_plots(),
               length(union(files$labelled_y, files$labelled_cls)))

  schema <- mask$schema()
  expect_equal(schema$targets$y$missing, "mask")
  expect_equal(schema$targets$cls$missing, "mask")
  expect_equal(drop$schema()$targets$y$missing, "drop")

  targets <- mask$targets()
  expect_equal(sum(is.finite(targets$y)), length(files$labelled_y))
  expect_equal(sum(targets$cls >= 0), length(files$labelled_cls))
  expect_setequal(unique(targets$cls[targets$cls < 0]), -1)

  expect_error(masking_dataset(files, "sometimes"), "missing-label policy")
})

test_that("a masked fit reports per-target counts and keeps the policy in the checkpoint", {
  skip_if_no_backend()
  skip_on_cran()
  files <- masking_files()
  model_file <- local_model_file()

  dataset <- masking_dataset(files, "mask")
  fit <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 4L, patience = 4L,
    lr = 1e-2, batchSize = 16L, testSize = 0.25, seed = 3L,
    savePath = model_file, verbose = FALSE
  )
  trainer <- fit$trainer
  test_idx <- trainer$test_indices() + 1L
  y_test <- dataset$targets()$y[test_idx]
  cls_test <- dataset$targets()$cls[test_idx]
  expect_equal(fit$result$final_metrics$y$n_labelled, sum(is.finite(y_test)))
  expect_equal(fit$result$final_metrics$cls$n_labelled, sum(cls_test >= 0))

  classified <- trainer$compute_classification_predictions("cls")
  expect_equal(length(classified$actuals), sum(cls_test >= 0))
  expect_equal(as.numeric(classified$actuals),
               as.numeric(cls_test[classified$test_rows + 1L]))

  residuals <- trainer$compute_residuals("y")
  expect_equal(length(residuals$test_rows), sum(is.finite(y_test)))

  predictor <- resolve.load(model_file, device = "cpu")
  expect_equal(predictor$schema()$targets$y$missing, "mask")
  preds <- resolve.predict.dataset(predictor, dataset)
  expect_equal(length(preds$predictions$y), dataset$n_plots())
  expect_true(all(is.finite(preds$predictions$y)))
})

expect_decomposes <- function(byTarget, total, targets, aux = list()) {
  expect_setequal(names(byTarget), targets)
  for (target in targets) expect_length(byTarget[[target]], length(total))
  terms <- Reduce(`+`, c(byTarget[targets], unname(aux)))
  expect_equal(terms, total, tolerance = 1e-5)
}

test_that("the loss histories split by target and add up to the totals", {
  skip_if_no_backend()
  skip_on_cran()
  files <- masking_files()

  dataset <- masking_dataset(files, "mask")
  fit <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 4L, patience = 4L,
    lr = 1e-2, batchSize = 4L, testSize = 0.25, seed = 3L, verbose = FALSE
  )
  result <- fit$result
  expect_length(result$train_loss, 4L)
  expect_decomposes(result$train_loss_by_target, result$train_loss, c("y", "cls"))
  expect_decomposes(result$test_loss_by_target, result$test_loss, c("y", "cls"))
  expect_length(result$train_aux_loss, 0L)

  moe <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 3L, patience = 3L,
    lr = 1e-2, batchSize = 16L, testSize = 0.25, seed = 3L,
    moeRouting = "soft", verbose = FALSE
  )$result
  expect_named(moe$train_aux_loss, "moe")
  expect_length(moe$train_aux_loss$moe, 3L)
  expect_decomposes(moe$train_loss_by_target, moe$train_loss, c("y", "cls"),
                    aux = moe$train_aux_loss)
  expect_decomposes(moe$test_loss_by_target, moe$test_loss, c("y", "cls"))

  refit <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 3L, fixedEpochs = 3L,
    lr = 1e-2, batchSize = 16L, testSize = 0, seed = 3L, verbose = FALSE
  )$result
  expect_length(refit$test_loss, 0L)
  expect_length(refit$test_loss_by_target, 0L)
  expect_decomposes(refit$train_loss_by_target, refit$train_loss, c("y", "cls"))
})
