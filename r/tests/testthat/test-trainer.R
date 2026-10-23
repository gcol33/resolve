# Tests for RESOLVE Trainer

# Builds a model on a small hashed dataset, trains it for `train_config`'s
# epochs, saves it, and checks that both the live trainer and the checkpoint
# report every field of `train_config` as given (to float32 precision, the
# width the engine stores learning rates at).
expect_config_round_trip <- function(train_config) {
  tables <- signal_tables(60L)
  dataset <- load_tables(tables, signal_roles(), regression_target())
  model <- new(.resolve_module()$ResolveModel, dataset$schema(),
               list(species_encoding = "hash", hash_dim = 4L, top_k = 2L,
                    hidden_dims = c(16L, 8L)))
  trainer <- new(.resolve_module()$Trainer, model, train_config)

  live <- trainer$get_config()
  for (field in names(train_config)) {
    expect_equal(live[[field]], train_config[[field]], tolerance = 1e-6, info = field)
  }

  trainer$prepare_data_from_dataset(dataset, 0.25, 3L)
  result <- trainer$fit()
  expect_length(result$train_loss, train_config$max_epochs)

  model_file <- local_model_file()
  trainer$save(model_file, NULL)
  saved <- resolve.load_train_config(model_file)
  for (field in names(train_config)) {
    expect_equal(saved[[field]], train_config[[field]], tolerance = 1e-6, info = field)
  }
}

test_that("a trainer keeps its configuration and the checkpoint carries it", {
  skip_if_no_backend()
  skip_on_cran()

  expect_config_round_trip(list(
    batch_size = 16L,
    max_epochs = 2L,
    patience = 5L,
    lr = 0.001,
    lr_scheduler = "none"
  ))
})

test_that("a cosine learning-rate schedule round-trips through the checkpoint", {
  skip_if_no_backend()
  skip_on_cran()

  expect_config_round_trip(list(
    batch_size = 16L,
    max_epochs = 3L,
    patience = 5L,
    lr = 0.01,
    lr_scheduler = "cosine",
    lr_min = 0.0001
  ))
})

test_that("a step learning-rate schedule round-trips through the checkpoint", {
  skip_if_no_backend()
  skip_on_cran()

  expect_config_round_trip(list(
    batch_size = 16L,
    max_epochs = 5L,
    patience = 10L,
    lr = 0.01,
    lr_scheduler = "step",
    lr_step_size = 2L,
    lr_gamma = 0.5
  ))
})

test_that("A mixture of experts builds at either placement", {
  skip_if_no_backend()
  skip_on_cran()

  schema <- fixture_schema(n_plots = 60L)

  # Four widths so the tail split is visible: the backbone keeps the first two
  # and the mixture produces the last, which stays the latent width.
  base_config <- list(
    species_encoding = "hash",
    hash_dim = 16L,
    hidden_dims = c(32L, 24L, 16L, 12L),
    moe_routing = "soft",
    n_experts = 3L,
    expert_hidden_dims = c(10L),
    moe_top_k = 2L,
    moe_noise_std = 0.0
  )

  for (placement in c("tail", "post")) {
    model_config <- base_config
    model_config$moe_placement <- placement
    model <- new(.resolve_module()$ResolveModel, schema, model_config)
    expect_equal(model$latent_dim(), 12L)
  }
})

test_that("A tail mixture is refused where there is no MLP tail", {
  skip_if_no_backend()
  skip_on_cran()

  schema <- fixture_schema(n_plots = 60L, n_species_vocab = 21L)

  model_config <- list(
    species_encoding = "hash",
    hash_dim = 16L,
    hidden_dims = c(32L, 16L),
    encoder_architecture = "tabnet",
    moe_routing = "soft",
    moe_placement = "tail",
    n_experts = 3L
  )

  # The message has to name the placement that works.
  expect_error(
    new(.resolve_module()$ResolveModel, schema, model_config),
    "moe_placement=post",
    fixed = TRUE
  )

  model_config$moe_placement <- "post"
  expect_silent(new(.resolve_module()$ResolveModel, schema, model_config))
})


test_that("freeze_composition fixes the composition tables and is refused without them", {
  skip_if_no_backend()
  skip_on_cran()

  schema <- fixture_schema(
    n_plots = 60L,
    has_abundance = TRUE,
    has_taxonomy = TRUE,
    n_genera = 5L,
    n_families = 3L,
    n_species_vocab = 21L,
    n_genera_vocab = 5L,
    n_families_vocab = 3L
  )
  model_config <- list(
    species_encoding = "rank_pool",
    hidden_dims = c(16L, 8L),
    species_embed_dim = 8L,
    freeze_composition = TRUE
  )

  model <- new(.resolve_module()$ResolveModel, schema, model_config)
  tables <- model$composition_parameters()
  expect_length(tables, 3L)
  expect_equal(dim(tables[[1]]), c(21L, 8L))

  model_config$encoder_architecture <- "tabnet"
  expect_error(
    new(.resolve_module()$ResolveModel, schema, model_config),
    "freeze_composition",
    fixed = TRUE
  )
})

test_that("a run resumed from a periodic checkpoint is the uninterrupted run", {
  skip_if_no_backend()
  skip_on_cran()

  tables <- signal_tables(60L)
  dataset <- load_tables(tables, signal_roles(), regression_target())
  dir <- local_dir()
  train <- function(...) {
    resolve.train.dataset(dataset, hiddenDims = c(16L, 8L), maxEpochs = 6L,
                          patience = 50L, batchSize = 8L, seed = 17L,
                          checkpointDir = dir, checkpointEvery = 2L,
                          verbose = FALSE, ...)
  }

  full <- train()$result
  resumed <- train(resumeFrom = file.path(dir, "checkpoint_4.pt"))$result

  expect_equal(full$resumed_from_epoch, 0L)
  expect_equal(resumed$resumed_from_epoch, 4L)
  expect_equal(resumed$train_loss, full$train_loss)
  expect_equal(resumed$test_loss, full$test_loss)
  expect_equal(resumed$best_epoch, full$best_epoch)
  expect_equal(resumed$final_metrics, full$final_metrics)
})

test_that("resume refuses a checkpoint without run progress", {
  skip_if_no_backend()
  skip_on_cran()

  tables <- signal_tables(40L)
  dataset <- load_tables(tables, signal_roles(), regression_target())
  dir <- local_dir()
  resolve.train.dataset(dataset, hiddenDims = c(8L), maxEpochs = 2L, batchSize = 8L,
                        checkpointDir = dir, verbose = FALSE)
  expect_error(
    resolve.train.dataset(dataset, hiddenDims = c(8L), maxEpochs = 2L, batchSize = 8L,
                          resumeFrom = file.path(dir, "checkpoint.pt"), verbose = FALSE),
    "no run progress"
  )
  expect_error(resolve.train.dataset(dataset, checkpointEvery = 2L), "checkpointDir")
})
