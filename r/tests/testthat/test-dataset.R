# Tests for ResolveDataset C++ binding

test_that("resolve.dataset.csv loads data correctly", {
  skip_if_no_backend()
  skip_on_cran()

  dataset <- load_tables(
    taxonomy_tables(),
    roles = taxonomy_roles(),
    targets = regression_target("area"),
    config = list(species_encoding = "hash", hash_dim = 16, top_k = 2)
  )

  # Check dataset properties
  expect_equal(dataset$n_plots(), 3)

  # Check schema
  schema <- dataset$schema()
  expect_equal(schema$n_plots, 3)
  expect_true(schema$has_coordinates)
  expect_true(schema$has_taxonomy)

  # Check coordinates
  coords <- dataset$coordinates()
  expect_equal(nrow(coords), 3)
  expect_equal(ncol(coords), 2)

  # Check hash embedding
  hash_emb <- dataset$hash_embedding()
  expect_equal(nrow(hash_emb), 3)
  expect_equal(ncol(hash_emb), 16)

  # Check targets
  targets <- dataset$targets()
  expect_true("area" %in% names(targets))
  expect_equal(length(targets$area), 3)

  # Check plot IDs
  plot_ids <- dataset$plot_ids()
  expect_equal(length(plot_ids), 3)
  expect_true("p1" %in% plot_ids)
})


test_that("resolve.dataset.csv works without taxonomy", {
  skip_if_no_backend()
  skip_on_cran()

  # Minimal header and species tables, no taxonomy.
  files <- local_csv_pair(
    data.frame(plot_id = c("p1", "p2"), target_val = c(1.0, 2.0)),
    data.frame(plot_id = c("p1", "p1", "p2"), species_id = c("sp1", "sp2", "sp1"))
  )

  dataset <- resolve.dataset.csv(
    header = files$header,
    species = files$species,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id"
    ),
    targets = regression_target("target_val")
  )

  expect_equal(dataset$n_plots(), 2)
  schema <- dataset$schema()
  expect_false(schema$has_taxonomy)
})


test_that("resolve.dataset.csv supports classification targets", {
  skip_if_no_backend()
  skip_on_cran()

  files <- local_csv_pair(
    data.frame(plot_id = c("p1", "p2", "p3"), habitat = c(0, 1, 2)),
    data.frame(plot_id = c("p1", "p2", "p3"), species_id = c("sp1", "sp2", "sp3"))
  )

  dataset <- resolve.dataset.csv(
    header = files$header,
    species = files$species,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id"
    ),
    targets = list(
      habitat = list(column = "habitat", task = "classification", num_classes = 5)
    )
  )

  expect_equal(dataset$n_plots(), 3)
  targets <- dataset$targets()
  expect_true("habitat" %in% names(targets))
})


test_that("resolve.dataset.csv config options work", {
  skip_if_no_backend()
  skip_on_cran()

  files <- local_csv_pair(
    data.frame(plot_id = c("p1", "p2"), area = c(100.0, 200.0)),
    data.frame(plot_id = c("p1", "p1", "p2", "p2"),
               species_id = c("sp1", "sp2", "sp1", "sp3"),
               cover = c(0.5, 0.5, 0.8, 0.2))
  )

  dataset <- resolve.dataset.csv(
    header = files$header,
    species = files$species,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id",
      abundance = "cover"
    ),
    targets = list(
      area = list(column = "area", task = "regression", transform = "log1p")
    ),
    config = list(
      species_encoding = "hash",
      hash_dim = 64,
      top_k = 3,
      track_unknown_fraction = TRUE,
      track_unknown_count = TRUE
    )
  )

  config <- dataset$config()
  expect_equal(config$hash_dim, 64)
  expect_equal(config$top_k, 3)

  hash_emb <- dataset$hash_embedding()
  expect_equal(ncol(hash_emb), 64)
})

test_that("an empty-string role is unset, a typo is still an error", {
  skip_if_no_backend()
  skip_on_cran()

  tables <- taxonomy_tables()
  files <- local_csv_pair(tables$header, tables$species)

  load_with <- function(roles) {
    resolve.dataset.csv(
      header = files$header,
      species = files$species,
      roles = roles,
      targets = regression_target("area"),
      config = list(species_encoding = "hash", hash_dim = 16)
    )
  }

  base_roles <- taxonomy_roles()

  cleared <- base_roles
  cleared$longitude <- ""
  cleared$latitude <- ""
  ds_cleared <- load_with(cleared)

  omitted <- base_roles
  omitted$longitude <- NULL
  omitted$latitude <- NULL
  ds_omitted <- load_with(omitted)

  expect_equal(ds_cleared$n_plots(), ds_omitted$n_plots())
  expect_false(ds_cleared$schema()$has_coordinates)
  expect_false(ds_omitted$schema()$has_coordinates)

  typo <- base_roles
  typo$latitude <- "lattitude"
  expect_error(load_with(typo))
})

test_that("species_budget narrows the pooled encoding", {
  skip_if_no_backend()
  skip_on_cran()

  n_plots <- 6L
  n_species <- 6L
  # Every plot records every species with rotating covers, so which species is
  # most abundant moves with the plot and a selection that silently did nothing
  # is visible.
  grid <- expand.grid(plot = seq_len(n_plots), sp = seq_len(n_species))
  files <- local_csv_pair(
    data.frame(plot_id = paste0("p", seq_len(n_plots)), area = seq_len(n_plots) * 10),
    data.frame(plot_id = paste0("p", grid$plot),
               species_id = paste0("sp", grid$sp),
               cover = ((grid$sp + grid$plot) %% n_species) + 1)
  )

  load_with <- function(budget, selection) {
    resolve.dataset.csv(
      header = files$header,
      species = files$species,
      roles = list(plot_id = "plot_id", species_id = "species_id", abundance = "cover"),
      targets = regression_target("area"),
      config = list(
        species_encoding = "rank_pool",
        selection = selection,
        species_budget = budget
      )
    )
  }

  full <- load_with(0L, "top")
  expect_equal(ncol(full$species_ids()), n_species)
  # With no budget the encoding really did select nothing, and says so.
  expect_equal(full$schema()$dataset_config$selection, "all")
  expect_equal(full$schema()$dataset_config$species_budget, 0L)

  narrowed <- load_with(2L, "bottom")
  expect_equal(ncol(narrowed$species_ids()), 2L)
  expect_equal(narrowed$schema()$dataset_config$selection, "bottom")
  expect_equal(narrowed$schema()$dataset_config$species_budget, 2L)
})

# --- targets / roles / config specification validation ---------------------
#
# A target's NAME is the key the engine reads, and the C ABI carries a named
# list as a keyed map. The engine reads every structured argument strictly: an
# unnamed list, an unknown key or a repeated name is refused where the caller
# can still see what they typed, never dropped.

valid_targets <- function() {
  list(area = list(column = "area", task = "regression"))
}

# A twelve-plot dataset loaded with the given roles, targets and config.
spec_load <- function(roles = list(abundance = "cover"), targets = valid_targets(),
                      config = list(species_encoding = "hash")) {
  n <- 12L
  files <- local_csv_pair(
    data.frame(plot_id = paste0("p", seq_len(n)), area = seq_len(n) * 1.0,
               habitat = rep(c("forest", "grass", "bog"), 4L)),
    data.frame(plot_id = paste0("p", seq_len(n)),
               species_id = paste0("sp", seq_len(n) %% 4L),
               cover = 1.0)
  )
  resolve.dataset.csv(header = files$header, species = files$species,
                      roles = roles, targets = targets, config = config)
}

spec_error <- function(...) {
  tryCatch({
    spec_load(...)
    ""
  }, error = function(e) conditionMessage(e))
}

test_that("an unnamed or partially named target list is rejected, not dropped", {
  skip_if_no_backend()
  expect_match(spec_error(targets = list(list(column = "area", task = "regression"))),
               "NAMED list")
  expect_match(spec_error(targets = list(area = list(column = "area"),
                                         list(column = "habitat"))),
               "empty name")
})

test_that("an unknown key in a target specification is rejected by name", {
  skip_if_no_backend()
  err <- spec_error(targets = list(area = list(name = "area", type = "regression")))
  expect_match(err, "target 'area'")
  expect_match(err, "'name'")
  # The message hands back the accepted spellings.
  expect_match(err, "column")
  expect_match(err, "task")
})

test_that("a target named twice is rejected", {
  skip_if_no_backend()
  expect_match(spec_error(targets = list(area = list(column = "area"),
                                         area = list(column = "habitat"))),
               "given twice")
})

test_that("an unknown role key or an unnamed roles list is rejected", {
  skip_if_no_backend()
  err <- spec_error(roles = list(abundance = "cover", speciesId = "species"))
  expect_match(err, "roles")
  expect_match(err, "'speciesId'")
  expect_match(err, "species_id")
  expect_match(spec_error(roles = list("plot_id")), "roles: unknown key")
})

test_that("an unknown config key is rejected, not trained with the default", {
  skip_if_no_backend()
  err <- spec_error(config = list(species_encodng = "embed"))
  expect_match(err, "dataset config")
  expect_match(err, "'species_encodng'")
  expect_match(err, "species_encoding")
  expect_match(spec_error(config = list(hash_dim = "64")), "hash_dim")
  expect_match(spec_error(config = list(species_encoding = "hashing")), "species_encoding")
})

test_that("a target without a column reads the column of its name", {
  skip_if_no_backend()
  ds <- spec_load(targets = list(area = list(task = "regression")))
  expect_equal(names(ds$schema()$targets), "area")
  expect_equal(ds$targets()$area, seq_len(12L) * 1.0)
})

test_that("an invalid target task names the field", {
  skip_if_no_backend()
  expect_match(spec_error(targets = list(area = list(task = "segmentation"))), "task")
})

test_that("every documented target and role key is accepted, and the role defaults apply", {
  skip_if_no_backend()
  ds <- spec_load(
    roles = list(abundance = "cover", longitude = "", latitude = "", genus = "",
                 family = "", covariates = character(0), categoricals = character(0)),
    targets = list(habitat = list(column = "habitat", task = "classification",
                                  transform = "none", num_classes = 3L, weight = 1.0,
                                  class_mapping = c(forest = 0L, grass = 1L, bog = 2L),
                                  missing = "mask")))
  expect_equal(ds$n_plots(), 12L)
  expect_equal(ds$schema()$targets$habitat$class_names, c("forest", "grass", "bog"))
})

test_that("a dataset without targets loads for scoring and is refused for training", {
  skip_if_no_backend()
  ds <- spec_load(targets = list())
  expect_equal(ds$n_plots(), 12L)
  expect_error(resolve.train.dataset(ds, maxEpochs = 1L, verbose = FALSE), "target")
})

test_that("a missing covariate or coordinate is NaN and the policy is recorded", {
  skip_if_no_backend()
  skip_on_cran()

  files <- local_csv_pair(
    c("plot_id,lat,lon,elev,area",
      "p1,45.1,10.2,300,100",
      "p2,,,0,200",
      "p3,45.3,10.4,,150"),
    c("plot_id,species_id,cover",
      "p1,sp1,0.5", "p2,sp1,0.8", "p3,sp2,0.4")
  )

  load <- function(policy) {
    resolve.dataset.csv(
      header = files$header,
      species = files$species,
      roles = list(plot_id = "plot_id", species_id = "species_id",
                   abundance = "cover", latitude = "lat", longitude = "lon",
                   covariates = "elev"),
      targets = regression_target("area"),
      config = list(species_encoding = "hash", hash_dim = 8,
                    missing_values = policy)
    )
  }

  indicated <- load("indicate")
  expect_equal(indicated$config()$missing_values, "indicate")
  expect_equal(indicated$schema()$dataset_config$missing_values, "indicate")
  expect_true(all(is.nan(indicated$coordinates()[2, ])))
  covariates <- indicated$covariates()
  expect_equal(covariates[2, 1], 0)
  expect_true(is.nan(covariates[3, 1]))

  expect_equal(load("zero")$schema()$dataset_config$missing_values, "zero")
  expect_error(load("impute"), "missing value policy")
})
