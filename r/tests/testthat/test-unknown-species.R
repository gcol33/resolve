# Species a checkpoint never saw. A dataset built against the checkpoint's
# vocabularies counts them per plot (the fraction of a plot's species that are
# unknown, and how many), encodes them with the reserved code 0 in a pooled
# encoding, and stays scoreable even when a plot holds nothing else.

# Training plots each recording three of the species sp0..sp5.
three_species_tables <- function(n) {
  tables <- signal_tables(n)
  i <- rep(seq_len(n), each = 3L)
  tables$species <- data.frame(
    plot_id = paste0("P", i),
    species_id = paste0("sp", (i + rep(0:2, n)) %% 6L),
    cover = 1.0,
    stringsAsFactors = FALSE
  )
  tables
}

# Q1 holds two known species and one unknown, Q2 only known species, Q3 only
# unknown ones.
scoring_files <- function(env = parent.frame()) {
  local_csv_pair(
    data.frame(plot_id = c("Q1", "Q2", "Q3"), cov1 = 0.1, cov2 = 0.2),
    data.frame(plot_id = c("Q1", "Q1", "Q1", "Q2", "Q2", "Q3", "Q3"),
               species_id = c("sp1", "sp2", "sp_new", "sp3", "sp4", "sp_x", "sp_y"),
               cover = 1),
    env = env
  )
}

train_and_load <- function(config, env = parent.frame()) {
  dataset <- load_tables(three_species_tables(60L), signal_roles(), regression_target(),
                         config = config, env = env)
  model_file <- local_model_file(env)
  resolve.train.dataset(dataset, hiddenDims = c(16L, 8L), maxEpochs = 2L,
                        batchSize = 16L, testSize = 0.25, seed = 3L,
                        savePath = model_file, verbose = FALSE)
  resolve.load(model_file, device = "cpu")
}

score_with <- function(predictor, files) {
  resolve.dataset.csv(files$header, files$species, roles = signal_roles(),
                      targets = list(), vocabs = predictor$vocabs(),
                      config = predictor$dataset_config())
}

test_that("unknown species are counted per plot and the plots stay scoreable", {
  skip_if_no_backend()
  skip_on_cran()

  predictor <- train_and_load(small_hash_config(track_unknown_fraction = TRUE,
                                                track_unknown_count = TRUE))
  scoring <- score_with(predictor, scoring_files())

  expect_equal(scoring$plot_ids(), c("Q1", "Q2", "Q3"))
  expect_equal(as.numeric(scoring$unknown_fraction()), c(1 / 3, 0, 1), tolerance = 1e-6)
  expect_equal(as.numeric(scoring$unknown_count()), c(1, 0, 2))

  preds <- resolve.predict.dataset(predictor, scoring)
  expect_length(preds$predictions$y, 3L)
  expect_true(all(is.finite(preds$predictions$y)))
})

test_that("a pooled encoding gives an unknown species the reserved code 0", {
  skip_if_no_backend()
  skip_on_cran()

  predictor <- train_and_load(list(species_encoding = "rank_pool",
                                   track_unknown_fraction = TRUE,
                                   track_unknown_count = TRUE))
  vocab <- predictor$species_vocab()
  expect_equal(vocab[1], "<UNK>")
  expect_setequal(vocab[-1], paste0("sp", 0:5))

  scoring <- score_with(predictor, scoring_files())
  ids <- scoring$species_ids()
  mask <- scoring$pool_mask()
  # Each real species keeps its vocabulary position (counting <UNK> as 0);
  # every unknown one is 0 while still marked present in the pool mask.
  q1 <- ids[1, mask[1, ] == 1]
  expect_setequal(q1, c(match("sp1", vocab) - 1L, match("sp2", vocab) - 1L, 0L))
  q2 <- ids[2, mask[2, ] == 1]
  expect_setequal(q2, c(match("sp3", vocab) - 1L, match("sp4", vocab) - 1L))
  expect_equal(unname(ids[3, mask[3, ] == 1]), c(0L, 0L))
  expect_equal(as.numeric(scoring$unknown_count()), c(1, 0, 2))

  preds <- resolve.predict.dataset(predictor, scoring)
  expect_true(all(is.finite(preds$predictions$y)))
})
