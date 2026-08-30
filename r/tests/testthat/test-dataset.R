# Tests for ResolveDataset C++ binding

test_that("resolve.dataset.csv loads data correctly", {
  skip_if_no_backend()
  skip_on_cran()

  # Create temporary test CSV files
  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")

  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  # Create header data (plot-level)
  header_data <- data.frame(
    plot_id = c("p1", "p2", "p3"),
    lon = c(10.0, 11.0, 12.0),
    lat = c(50.0, 51.0, 52.0),
    area = c(100.0, 200.0, 150.0),
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)

  # Create species data (species-plot occurrences)
  species_data <- data.frame(
    plot_id = c("p1", "p1", "p2", "p2", "p3", "p3"),
    species_id = c("sp1", "sp2", "sp1", "sp3", "sp2", "sp3"),
    cover = c(0.5, 0.5, 0.8, 0.2, 0.6, 0.4),
    genus = c("Quercus", "Fagus", "Quercus", "Pinus", "Fagus", "Pinus"),
    family = c("Fagaceae", "Fagaceae", "Fagaceae", "Pinaceae", "Fagaceae", "Pinaceae"),
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  # Load dataset
  dataset <- resolve.dataset.csv(
    header = header_file,
    species = species_file,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id",
      abundance = "cover",
      longitude = "lon",
      latitude = "lat",
      genus = "genus",
      family = "family"
    ),
    targets = list(
      area = list(column = "area", task = "regression")
    ),
    config = list(
      species_encoding = "hash",
      hash_dim = 16,
      top_k = 2
    )
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

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")

  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  # Minimal header
  header_data <- data.frame(
    plot_id = c("p1", "p2"),
    target_val = c(1.0, 2.0),
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)

  # Minimal species (no taxonomy)
  species_data <- data.frame(
    plot_id = c("p1", "p1", "p2"),
    species_id = c("sp1", "sp2", "sp1"),
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  dataset <- resolve.dataset.csv(
    header = header_file,
    species = species_file,
    roles = list(
      plot_id = "plot_id",
      species_id = "species_id"
    ),
    targets = list(
      target_val = list(column = "target_val", task = "regression")
    )
  )

  expect_equal(dataset$n_plots(), 2)
  schema <- dataset$schema()
  expect_false(schema$has_taxonomy)
})


test_that("resolve.dataset.csv supports classification targets", {
  skip_if_no_backend()
  skip_on_cran()

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")

  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  header_data <- data.frame(
    plot_id = c("p1", "p2", "p3"),
    habitat = c(0, 1, 2),  # classification target
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)

  species_data <- data.frame(
    plot_id = c("p1", "p2", "p3"),
    species_id = c("sp1", "sp2", "sp3"),
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  dataset <- resolve.dataset.csv(
    header = header_file,
    species = species_file,
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

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")

  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  header_data <- data.frame(
    plot_id = c("p1", "p2"),
    area = c(100.0, 200.0),
    stringsAsFactors = FALSE
  )
  write.csv(header_data, header_file, row.names = FALSE)

  species_data <- data.frame(
    plot_id = c("p1", "p1", "p2", "p2"),
    species_id = c("sp1", "sp2", "sp1", "sp3"),
    cover = c(0.5, 0.5, 0.8, 0.2),
    stringsAsFactors = FALSE
  )
  write.csv(species_data, species_file, row.names = FALSE)

  # Test with custom config
  dataset <- resolve.dataset.csv(
    header = header_file,
    species = species_file,
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

test_that("an empty-string role is unset, a typo is still an error (#111)", {
  skip_if_no_backend()
  skip_on_cran()

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")
  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  write.csv(data.frame(
    plot_id = c("p1", "p2", "p3"),
    lon = c(10.0, 11.0, 12.0),
    lat = c(50.0, 51.0, 52.0),
    area = c(100.0, 200.0, 150.0),
    stringsAsFactors = FALSE
  ), header_file, row.names = FALSE)

  write.csv(data.frame(
    plot_id = c("p1", "p1", "p2", "p2", "p3", "p3"),
    species_id = c("sp1", "sp2", "sp1", "sp3", "sp2", "sp3"),
    cover = c(0.5, 0.5, 0.8, 0.2, 0.6, 0.4),
    genus = c("Quercus", "Fagus", "Quercus", "Pinus", "Fagus", "Pinus"),
    family = c("Fagaceae", "Fagaceae", "Fagaceae", "Pinaceae", "Fagaceae", "Pinaceae"),
    stringsAsFactors = FALSE
  ), species_file, row.names = FALSE)

  load_with <- function(roles) {
    resolve.dataset.csv(
      header = header_file,
      species = species_file,
      roles = roles,
      targets = list(area = list(column = "area", task = "regression")),
      config = list(species_encoding = "hash", hash_dim = 16)
    )
  }

  base_roles <- list(
    plot_id = "plot_id", species_id = "species_id", abundance = "cover",
    longitude = "lon", latitude = "lat", genus = "genus", family = "family"
  )

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

test_that("species_budget narrows the pooled encoding (#113)", {
  skip_if_no_backend()
  skip_on_cran()

  header_file <- tempfile(fileext = ".csv")
  species_file <- tempfile(fileext = ".csv")
  on.exit({
    unlink(header_file)
    unlink(species_file)
  }, add = TRUE)

  n_plots <- 6L
  n_species <- 6L
  write.csv(data.frame(
    plot_id = paste0("p", seq_len(n_plots)),
    area = seq_len(n_plots) * 10,
    stringsAsFactors = FALSE
  ), header_file, row.names = FALSE)

  # Every plot records every species with rotating covers, so which species is
  # most abundant moves with the plot and a selection that silently did nothing
  # is visible.
  grid <- expand.grid(plot = seq_len(n_plots), sp = seq_len(n_species))
  write.csv(data.frame(
    plot_id = paste0("p", grid$plot),
    species_id = paste0("sp", grid$sp),
    cover = ((grid$sp + grid$plot) %% n_species) + 1,
    stringsAsFactors = FALSE
  ), species_file, row.names = FALSE)

  load_with <- function(budget, selection) {
    resolve.dataset.csv(
      header = header_file,
      species = species_file,
      roles = list(plot_id = "plot_id", species_id = "species_id", abundance = "cover"),
      targets = list(area = list(column = "area", task = "regression")),
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
  expect_equal(full$schema()$selection, "all")
  expect_equal(full$schema()$species_budget, 0L)

  narrowed <- load_with(2L, "bottom")
  expect_equal(ncol(narrowed$species_ids()), 2L)
  expect_equal(narrowed$schema()$selection, "bottom")
  expect_equal(narrowed$schema()$species_budget, 2L)
})
