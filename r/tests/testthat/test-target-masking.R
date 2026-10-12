# Per-target missing-label masking (gcol33/resolve#119): the `missing` key of a
# target specification. "drop" (default) leaves a plot without that label out
# of the fit; "mask" keeps it, and it trains and is scored on its other targets.

masking_files <- function(n = 160L) {
  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")
  i <- seq_len(n) - 1L
  c1 <- sin(i * 0.13)
  c2 <- cos(i * 0.07)
  y <- 2.0 * c1 - 1.5 * c2 + 3.0
  y[i %% 3L == 0L] <- NA
  cls <- ifelse(c1 + c2 > 0, "pos", "neg")
  cls[i %% 4L == 1L | i %% 12L == 0L] <- NA
  write.csv(data.frame(plot_id = paste0("P", i), c1 = c1, c2 = c2, y = y,
                       cls = cls, stringsAsFactors = FALSE),
            header_file, row.names = FALSE, na = "")
  write.csv(data.frame(plot_id = paste0("P", i), species_id = paste0("sp", i %% 6L),
                       cover = 1.0, stringsAsFactors = FALSE),
            species_file, row.names = FALSE)
  list(header = header_file, species = species_file,
       labelled_y = i[i %% 3L != 0L],
       labelled_cls = i[!(i %% 4L == 1L | i %% 12L == 0L)])
}

masking_dataset <- function(files, missing) {
  resolve.dataset.csv(
    header = files$header, species = files$species,
    roles = list(plot_id = "plot_id", species_id = "species_id",
                 abundance = "cover", covariates = c("c1", "c2")),
    targets = list(
      y = list(column = "y", task = "regression", missing = missing),
      cls = list(column = "cls", task = "classification", missing = missing)
    ),
    config = list(species_encoding = "hash", hash_dim = 4L, top_k = 2L,
                  track_unknown_fraction = FALSE)
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
  on.exit(unlink(c(files$header, files$species)), add = TRUE)

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
  model_file <- tempfile(fileext = ".pt")
  on.exit({
    unlink(c(files$header, files$species, model_file))
    unlink(sub("\\.pt$", ".json", model_file))
  }, add = TRUE)

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

expect_decomposes <- function(byTarget, total, targets, moeAux = NULL) {
  expect_setequal(names(byTarget), targets)
  for (target in targets) expect_length(byTarget[[target]], length(total))
  terms <- Reduce(`+`, byTarget[targets])
  if (!is.null(moeAux)) terms <- terms + moeAux
  expect_equal(terms, total, tolerance = 1e-5)
}

test_that("the loss histories split by target and add up to the totals", {
  skip_if_no_backend()
  skip_on_cran()
  files <- masking_files()
  on.exit(unlink(c(files$header, files$species)), add = TRUE)

  dataset <- masking_dataset(files, "mask")
  fit <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 4L, patience = 4L,
    lr = 1e-2, batchSize = 4L, testSize = 0.25, seed = 3L, verbose = FALSE
  )
  result <- fit$result
  expect_length(result$train_loss, 4L)
  expect_decomposes(result$train_loss_by_target, result$train_loss, c("y", "cls"))
  expect_decomposes(result$test_loss_by_target, result$test_loss, c("y", "cls"))
  expect_length(result$train_moe_aux_loss, 0L)

  moe <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 3L, patience = 3L,
    lr = 1e-2, batchSize = 16L, testSize = 0.25, seed = 3L,
    moeRouting = "soft", verbose = FALSE
  )$result
  expect_length(moe$train_moe_aux_loss, 3L)
  expect_decomposes(moe$train_loss_by_target, moe$train_loss, c("y", "cls"),
                    moeAux = moe$train_moe_aux_loss)
  expect_decomposes(moe$test_loss_by_target, moe$test_loss, c("y", "cls"))

  refit <- resolve.train.dataset(
    dataset, hiddenDims = c(16L, 8L), maxEpochs = 3L, fixedEpochs = 3L,
    lr = 1e-2, batchSize = 16L, testSize = 0, seed = 3L, verbose = FALSE
  )$result
  expect_length(refit$test_loss, 0L)
  expect_length(refit$test_loss_by_target, 0L)
  expect_decomposes(refit$train_loss_by_target, refit$train_loss, c("y", "cls"))
})
