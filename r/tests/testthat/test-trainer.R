# Tests for RESOLVE Trainer

test_that("Trainer can be created with config", {
  skip_on_cran()

  # Create minimal schema
  schema <- list(
    n_plots = 100L,
    n_species = 50L,
    has_coordinates = TRUE,
    has_abundance = FALSE,
    has_taxonomy = FALSE,
    n_genera = 0L,
    n_families = 0L,
    track_unknown_fraction = FALSE,
    track_unknown_count = FALSE,
    targets = list(
      area = list(
        task = "regression",
        transform = "none"
      )
    )
  )

  model_config <- list(
    species_encoding = "hash",
    hash_dim = 16L,
    hidden_dims = c(32L, 16L)
  )

  train_config <- list(
    batch_size = 16L,
    max_epochs = 2L,
    patience = 5L,
    lr = 0.001
  )

  model <- new(.resolve_module$ResolveModel, schema, model_config)
  trainer <- new(.resolve_module$Trainer, model, train_config)

  # Should not raise
  expect_true(!is.null(trainer))
})

test_that("LR scheduler options are accepted", {
  skip_on_cran()

  schema <- list(
    n_plots = 50L,
    n_species = 20L,
    has_coordinates = TRUE,
    has_abundance = FALSE,
    has_taxonomy = FALSE,
    n_genera = 0L,
    n_families = 0L,
    track_unknown_fraction = FALSE,
    track_unknown_count = FALSE,
    targets = list(
      area = list(task = "regression", transform = "none")
    )
  )

  model_config <- list(
    species_encoding = "hash",
    hash_dim = 16L,
    hidden_dims = c(32L, 16L)
  )

  # Config with LR scheduler

  train_config <- list(
    batch_size = 16L,
    max_epochs = 3L,
    lr = 0.01,
    lr_scheduler = "cosine",
    lr_min = 0.0001
  )

  model <- new(.resolve_module$ResolveModel, schema, model_config)
  trainer <- new(.resolve_module$Trainer, model, train_config)

  expect_true(!is.null(trainer))
})

test_that("Step LR scheduler config is accepted", {
  skip_on_cran()

  schema <- list(
    n_plots = 50L,
    n_species = 20L,
    has_coordinates = TRUE,
    has_abundance = FALSE,
    has_taxonomy = FALSE,
    n_genera = 0L,
    n_families = 0L,
    track_unknown_fraction = FALSE,
    track_unknown_count = FALSE,
    targets = list(
      area = list(task = "regression", transform = "none")
    )
  )

  model_config <- list(
    species_encoding = "hash",
    hash_dim = 16L,
    hidden_dims = c(32L, 16L)
  )

  train_config <- list(
    batch_size = 16L,
    max_epochs = 5L,
    lr = 0.01,
    lr_scheduler = "step",
    lr_step_size = 2L,
    lr_gamma = 0.5
  )

  model <- new(.resolve_module$ResolveModel, schema, model_config)
  trainer <- new(.resolve_module$Trainer, model, train_config)

  expect_true(!is.null(trainer))
})
