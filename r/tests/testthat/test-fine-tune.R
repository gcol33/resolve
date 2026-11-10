# Fine-tuning from a trained model: resolve.parent(), resolve.train.dataset(
# initFrom = ...) and resolve.derive_suite(), over the habitat suite the
# package's example builds.

.fine_tune_suite <- function(env = parent.frame()) {
  example <- new.env(parent = globalenv())
  sys.source(system.file("examples", "habitat-suite.R", package = "resolveR"), envir = example)
  resolve.seal_suite(example$suite_dir)
  example
}

.elevation_dataset <- function(example, parent) {
  resolve.dataset.csv(example$plots, example$species, roles = example$roles,
                      targets = list(elevation = list(column = "elevation",
                                                      task = "regression")),
                      vocabs = parent$vocabs(), config = parent$dataset_config())
}

test_that("a fine-tuned model starts from its parent and records it", {
  skip_if_no_backend()
  skip_on_cran()

  example <- .fine_tune_suite()
  parent <- resolve.parent(example$suite_dir, target = "habitat", seed = 2L)
  record <- parent$record()
  expect_identical(record$suite, "habitat_example")
  expect_identical(record$suite_target, "habitat")
  expect_identical(record$suite_seed, 2L)
  manifest <- jsonlite::read_json(file.path(example$suite_dir, "manifest.json"))
  expect_identical(record$sha256, manifest$targets[[1]]$members[[2]]$sha256)

  ds <- .elevation_dataset(example, parent)
  path <- local_model_file()
  fit <- resolve.train.dataset(ds, initFrom = parent, freezeEncoder = TRUE, maxEpochs = 5L,
                               savePath = path, verbose = FALSE)
  expect_identical(fit$trainer$run_metadata()$parent$sha256, record$sha256)
  expect_equal(fit$trainer$get_scalers()$continuous_mean, parent$scalers()$continuous_mean)

  saved <- resolve.load_run_metadata(path)
  expect_identical(saved$parent$sha256, record$sha256)
  expect_identical(saved$parent$reuse, "encoder")
  expect_true(saved$parent$n_tensors > 0)
  expect_true(resolve.load_train_config(path)$batch_size > 0)
  expect_true(resolve.load(path)$model_config()$freeze_encoder)
})

test_that("a checkpoint path is a parent too, and the head settings may differ", {
  skip_if_no_backend()
  skip_on_cran()

  example <- .fine_tune_suite()
  checkpoint <- file.path(example$suite_dir, "seed_1.pt")
  parent <- resolve.parent(checkpoint)
  expect_identical(parent$record()$suite, "")
  ds <- .elevation_dataset(example, parent)
  fit <- resolve.train.dataset(ds, initFrom = checkpoint, maxEpochs = 2L, verbose = FALSE,
                               modelConfig = list(head_hidden_dims = list(8L)))
  expect_identical(fit$trainer$run_metadata()$parent$sha256, parent$record()$sha256)
})

test_that("a dataset or model that departs from the parent is refused", {
  skip_if_no_backend()
  skip_on_cran()

  example <- .fine_tune_suite()
  parent <- resolve.parent(example$suite_dir, target = "habitat")
  # The first 40 plots record fewer species, so a vocabulary fitted on them
  # differs from the parent's.
  plots <- read.csv(example$plots)[1:40, ]
  species <- read.csv(example$species)
  species <- species[species$plot_id %in% plots$plot_id, ]
  own <- resolve.dataset.frame(plots, species, roles = example$roles,
                               targets = list(elevation = list(column = "elevation",
                                                               task = "regression")),
                               config = parent$dataset_config())
  expect_error(resolve.train.dataset(own, initFrom = parent, maxEpochs = 1L, verbose = FALSE),
               "species vocabularies")
  ds <- .elevation_dataset(example, parent)
  expect_error(resolve.train.dataset(ds, initFrom = parent, hiddenDims = c(7L, 3L),
                                     maxEpochs = 1L, verbose = FALSE),
               "hidden_dims")
  expect_error(resolve.train.dataset(ds, initReuse = "encoder_heads", maxEpochs = 1L,
                                     verbose = FALSE),
               "initReuse needs initFrom")
  expect_error(resolve.parent(example$suite_dir, target = "altitude"), "altitude")
})

test_that("fine-tuned members are released as a derived suite and scored", {
  skip_if_no_backend()
  skip_on_cran()

  example <- .fine_tune_suite()
  members <- character()
  for (seed in 1:2) {
    parent <- resolve.parent(example$suite_dir, target = "habitat", seed = seed)
    ds <- .elevation_dataset(example, parent)
    path <- file.path(tempdir(), sprintf("elevation_tuned_%d.pt", seed))
    resolve.train.dataset(ds, initFrom = parent, freezeEncoder = TRUE, maxEpochs = 5L,
                          seed = seed, savePath = path, verbose = FALSE)
    members[as.character(seed)] <- path
  }
  derived <- file.path(tempdir(), "elevation_derived")
  unlink(derived, recursive = TRUE)
  manifest <- resolve.derive_suite(derived, parent = example$suite_dir,
                                   name = "elevation_example", licence = "CC-BY-4.0",
                                   trainingScope = "150 synthetic Alpine plots",
                                   targets = list(elevation = list(members = members,
                                                                   units = "m")))
  expect_identical(manifest$format_version, 5)
  expect_identical(manifest$targets[[1]]$derived_from$suite, "habitat_example")
  expect_identical(manifest$targets[[1]]$derived_from$target, "habitat")
  parent_manifest <- jsonlite::read_json(file.path(example$suite_dir, "manifest.json"))
  expect_identical(
    vapply(manifest$targets[[1]]$members, function(m) m$parent_sha256, ""),
    vapply(parent_manifest$targets[[1]]$members, function(m) m$sha256, "")
  )

  suite <- resolve.load_suite(derived)
  pred <- as.data.frame(resolve.predict.suite(suite, example$species, header = example$plots))
  expect_true("elevation" %in% names(pred))
  expect_true("elevation_sd" %in% names(pred))
  expect_error(resolve.derive_suite(derived, parent = example$suite_dir,
                                    name = "again", licence = "CC-BY-4.0",
                                    trainingScope = "x",
                                    targets = list(elevation = list(members = members))),
               "already holds a suite manifest")
})
