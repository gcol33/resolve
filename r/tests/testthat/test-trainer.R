# Tests for RESOLVE Trainer

test_that("Trainer can be created with config", {
  skip_on_cran()

  # Create minimal schema
  schema <- list(
    nPlots = 100,
    nSpecies = 50,
    hasCoordinates = TRUE,
    hasTaxonomy = FALSE,
    trackUnknownFraction = FALSE,
    targets = list(
      list(
        name = "area",
        task = "regression",
        transform = "none"
      )
    )
  )

  model_config <- list(
    speciesEncoding = "hash",
    hashDim = 16L,
    hiddenDims = c(32L, 16L)
  )

  train_config <- list(
    batchSize = 16L,
    maxEpochs = 2L,
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
    nPlots = 50,
    nSpecies = 20,
    hasCoordinates = TRUE,
    hasTaxonomy = FALSE,
    trackUnknownFraction = FALSE,
    targets = list(
      list(name = "area", task = "regression", transform = "none")
    )
  )

  model_config <- list(
    speciesEncoding = "hash",
    hashDim = 16L,
    hiddenDims = c(32L, 16L)
  )

  # Config with LR scheduler

  train_config <- list(
    batchSize = 16L,
    maxEpochs = 3L,
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
    nPlots = 50,
    nSpecies = 20,
    hasCoordinates = TRUE,
    hasTaxonomy = FALSE,
    trackUnknownFraction = FALSE,
    targets = list(
      list(name = "area", task = "regression", transform = "none")
    )
  )

  model_config <- list(
    speciesEncoding = "hash",
    hashDim = 16L,
    hiddenDims = c(32L, 16L)
  )

  train_config <- list(
    batchSize = 16L,
    maxEpochs = 5L,
    lr = 0.01,
    lr_scheduler = "step",
    lr_step_size = 2L,
    lr_gamma = 0.5
  )

  model <- new(.resolve_module$ResolveModel, schema, model_config)
  trainer <- new(.resolve_module$Trainer, model, train_config)

  expect_true(!is.null(trainer))
})
