# Categorical covariates through R: a string-valued header column named in
# `roles$categoricals` is factorized against a sorted vocabulary (codes 1..K,
# 0 reserved for an unseen level or a missing cell), embedded by the model, and
# its vocabulary travels with the checkpoint so new plots are coded the same way.

bedrock_levels <- c("granite", "limestone", "schist")

bedrock_tables <- function(n) {
  tables <- signal_tables(n)
  tables$header$bedrock <- bedrock_levels[seq_len(n) %% 3L + 1L]
  tables
}

bedrock_roles <- function() signal_roles(categoricals = "bedrock")

sorted_vocab <- function(vocab) vocab[order(names(vocab))]

test_that("a categorical column is coded against its sorted levels", {
  skip_if_no_backend()
  skip_on_cran()

  tables <- bedrock_tables(30L)
  dataset <- load_tables(tables, bedrock_roles(), regression_target())

  schema <- dataset$schema()
  expect_equal(schema$categorical_names, "bedrock")
  # Three levels plus the reserved unknown code.
  expect_equal(schema$categorical_vocab_sizes, 4L)

  vocab <- dataset$categorical_vocab()
  expect_equal(names(vocab), "bedrock")
  expect_equal(sorted_vocab(vocab$bedrock),
               stats::setNames(1:3, bedrock_levels))

  codes <- dataset$categorical_ids()
  expect_equal(dim(codes), c(30L, 1L))
  expect_equal(as.integer(codes[, 1]), match(tables$header$bedrock, bedrock_levels))
})

test_that("a categorical reaches the model and its vocabulary travels with the checkpoint", {
  skip_if_no_backend()
  skip_on_cran()

  dataset <- load_tables(bedrock_tables(60L), bedrock_roles(), regression_target())
  model_file <- local_model_file()
  fit <- resolve.train.dataset(dataset, hiddenDims = c(16L, 8L), maxEpochs = 3L,
                               batchSize = 16L, testSize = 0.25, seed = 3L,
                               savePath = model_file, verbose = FALSE)
  expect_equal(sorted_vocab(fit$trainer$categorical_vocab()$bedrock),
               sorted_vocab(dataset$categorical_vocab()$bedrock))

  predictor <- resolve.load(model_file, device = "cpu")
  expect_equal(predictor$schema()$categorical_names, "bedrock")
  expect_equal(sorted_vocab(predictor$categorical_vocab()$bedrock),
               stats::setNames(1:3, bedrock_levels))

  # Four plots identical but for their bedrock: a known level, an unseen one, a
  # missing cell, and another known level.
  files <- local_csv_pair(
    data.frame(plot_id = c("R1", "R2", "R3", "R4"), cov1 = 0.1, cov2 = 0.2,
               bedrock = c("granite", "basalt", NA, "schist")),
    data.frame(plot_id = c("R1", "R2", "R3", "R4"), species_id = "sp1", cover = 1)
  )
  scoring <- resolve.dataset.csv(files$header, files$species, roles = bedrock_roles(),
                                 targets = list(), vocabs = predictor$vocabs(),
                                 config = predictor$dataset_config())
  expect_equal(as.integer(scoring$categorical_ids()[, 1]), c(1L, 0L, 0L, 3L))

  y <- resolve.predict.dataset(predictor, scoring)$predictions$y
  expect_true(all(is.finite(y)))
  # The unseen level and the missing cell share the reserved code, so they
  # predict alike; two different known levels do not.
  expect_equal(y[2], y[3])
  expect_false(isTRUE(all.equal(y[1], y[4])))
})
