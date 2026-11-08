# Terrain correction and the terrain field through the R front door: elevation
# grids written as the engine reads them (a tiles.json and float32 .npy tiles)
# over a surface linear in latitude and longitude, whose descriptors have
# closed forms; a correction fitted, saved and applied; a field fitted from the
# grids; and a suite target corrected after combining.

.terrain_plane <- function(lat, lon) 100 + 1000 * (lat - 46) + 500 * (lon - 10)

# A float32 C-order .npy file of `grid`.
.write_npy <- function(grid, path) {
  header <- sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", nrow(grid), ncol(grid))
  pad <- (64 - (10 + nchar(header) + 1) %% 64) %% 64
  header <- paste0(header, strrep(" ", pad), "\n")
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(as.raw(c(0x93, charToRaw("NUMPY"), 0x01, 0x00)), con)
  writeBin(as.integer(nchar(header)), con, size = 2, endian = "little")
  writeBin(charToRaw(header), con)
  writeBin(as.numeric(t(grid)), con, size = 4, endian = "little")
}

# Plane tiles on the degree squares (south, west), pixel is point, `cells` per
# degree, under `dir`.
.write_plane_terrain <- function(dir, squares, cells = 24L) {
  d <- 1 / cells
  tiles <- lapply(squares, function(sq) {
    name <- sprintf("T%d_%d", sq[1], sq[2])
    lat <- sq[1] + 1 - (seq_len(cells) - 1) * d
    lon <- sq[2] + (seq_len(cells) - 1) * d
    .write_npy(outer(lat, lon, .terrain_plane), file.path(dir, paste0(name, ".npy")))
    list(name = name, file = paste0(name, ".npy"), rows = cells, cols = cells,
         lat_first = sq[1] + 1, lon_first = sq[2], dlat = d, dlon = d)
  })
  jsonlite::write_json(list(format = "resolve-terrain-grids", version = 1L, tiles = unname(tiles)),
                       file.path(dir, "tiles.json"), auto_unbox = TRUE, digits = NA)
}

.terrain_squares <- lapply(as.list(as.data.frame(t(expand.grid(45:47, 8:10)))), as.integer)

test_that("descriptors read from grids have the plane's closed forms", {
  skip_if_no_backend()
  grids <- local_dir("terrain_grids_")
  .write_plane_terrain(grids, .terrain_squares)
  lat <- c(46.31, 45.77, 46.4)
  lon <- c(9.42, 9.95, 10.2)
  d <- resolve.terrain_descriptors(lat, lon, grids = grids)
  expect_equal(nrow(d), 3L)
  expect_equal(d$elevation, .terrain_plane(lat, lon), tolerance = 1e-6)
  expect_equal(d$elevation_1km, .terrain_plane(lat, lon), tolerance = 1e-6)
  expect_true(all(d$slope > 0 & d$slope < 90))
  expect_equal(d$slope, d$slope_1km, tolerance = 1e-3)
  expect_true(is.na(resolve.terrain_descriptors(50.5, 9.5, grids = grids)$elevation))
  expect_error(resolve.terrain_descriptors(46, 9), "exactly one")
})

test_that("a terrain correction learns to trust the elevation model where the coordinate is precise", {
  skip_if_no_backend()
  skip_on_cran()
  set.seed(4)
  corpus <- function(n) {
    truth <- runif(n, 200, 1700)
    log_u <- runif(n, 0, 3.5)
    wrong <- runif(n) < 0.05 + 0.15 * log_u
    dem <- truth + rnorm(n, 0, 2) + ifelse(wrong, rnorm(n, 0, 300), 0)
    spread <- runif(n, 5, 45)
    list(truth = truth, prediction = truth + rnorm(n, 0, spread), dem = dem, dem1km = dem + rnorm(n, 0, 15),
         uncertainty = 10^log_u, spread = spread)
  }
  train <- corpus(6000)
  test <- corpus(2000)
  dir <- local_dir("terrain_corr_")
  path <- file.path(dir, "altitude.terrain")
  report <- resolve.fit_terrain_correction(
    "altitude", "elevation", train$prediction, train$dem, train$truth, path, dem1km = train$dem1km,
    uncertainty = train$uncertainty, spread = train$spread,
    config = list(scale = 500, epochs = 80L, batch_size = 256L, hidden_dim = 64L))
  expect_equal(report$n_train, 6000)
  r <- resolve.terrain_correction(path, test$prediction, test$dem, dem1km = test$dem1km,
                                  uncertainty = test$uncertainty, spread = test$spread)
  mae <- function(x) mean(abs(x - test$truth))
  expect_lt(mae(r$corrected), 0.5 * min(mae(test$prediction), mae(test$dem)))
  expect_equal(colnames(r$weights), c("resolve", "dem", "dem_1km"))
  expect_equal(unname(rowSums(r$weights)), rep(1, 2000), tolerance = 1e-6)
  expect_gt(mean(r$weights[test$uncertainty < 3, "dem"]), 0.6)
  expect_equal(r$target, "altitude")

  # Point values alone, and a plot without a descriptor keeps its prediction.
  dem <- test$dem
  dem[5] <- NA
  p <- resolve.terrain_correction(path, test$prediction, dem)
  expect_false(p$applied[5])
  expect_equal(p$corrected[5], test$prediction[5])
  expect_true(all(p$applied[-5]))
})

test_that("a terrain field fitted from grids reproduces their descriptors", {
  skip_if_no_backend()
  skip_on_cran()
  grids <- local_dir("terrain_field_grids_")
  .write_plane_terrain(grids, list(c(46L, 9L)), cells = 60L)
  out <- local_dir("terrain_field_")
  config <- list(levels = 6L, log2_table_size = 10L, coarsest_resolution = 8L, hidden_dim = 16L,
                 base_cells = 8L, steps = 600L, batch_size = 4096L)
  m <- resolve.fit_terrain_tile(grids, "T46_9", file.path(out, "T46_9.tile"), config = config)
  expect_lt(m$rmse, 5)
  field <- resolve.assemble_terrain_field(file.path(out, "T46_9.tile"), file.path(out, "field.bin"))
  lat <- runif(50, 46.05, 46.9)
  lon <- runif(50, 9.05, 9.9)
  f <- resolve.terrain_descriptors(lat, lon, field = field)
  g <- resolve.terrain_descriptors(lat, lon, grids = grids)
  expect_lt(mean(abs(f$elevation - g$elevation)), 5)

  # The tile whole, from the grids, a coded field and the neural field.
  stored <- resolve.terrain_grid("T46_9", grids = grids)
  expect_equal(stored$tile$rows, 60)
  expect_equal(dim(stored$elevation), c(60L, 60L))
  r <- 3
  c <- 7
  expect_equal(stored$elevation[r, c],
               .terrain_plane(stored$tile$lat_first - (r - 1) * stored$tile$dlat,
                              stored$tile$lon_first + (c - 1) * stored$tile$dlon), tolerance = 1e-5)
  coded <- file.path(out, "coded.bin")
  resolve.encode_terrain_grids(grids, coded, maxError = 0.5)
  expect_lte(max(abs(resolve.terrain_grid("T46_9", field = coded)$elevation - stored$elevation)), 0.5 + 1e-3)
  expect_lt(mean(abs(resolve.terrain_grid("T46_9", field = field)$elevation - stored$elevation)), 5)
  expect_error(resolve.terrain_grid("T10_10", grids = grids), "T10_10")
})

test_that("a suite target is corrected after combining, from columns or grids", {
  skip_if_no_backend()
  skip_on_cran()
  root <- local_dir("suite_terrain_")
  .build_suite(root)
  set.seed(2)
  n <- 400
  truth <- runif(n, 0, 10)
  resolve.fit_terrain_correction("y", "elevation", truth + rnorm(n), truth + rnorm(n, 0, 0.1), truth,
                                 file.path(root, "y_terrain.pack"),
                                 config = list(scale = 10, epochs = 3L, batch_size = 64L, hidden_dim = 16L))
  manifest <- jsonlite::read_json(file.path(root, "manifest.json"))
  manifest$targets[[2]]$terrain_correction <- list(file = "y_terrain.pack")
  manifest$format_version <- 4L
  jsonlite::write_json(manifest, file.path(root, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  sealed <- resolve.seal_suite(root)
  expect_equal(sealed$format_version, 4)
  suite <- resolve.load_suite(root)

  data <- .suite_plots(40L, with_targets = FALSE)
  header <- data$header
  header$dem <- .terrain_plane(header$lat, header$lon)
  header$unc <- ifelse(seq_len(nrow(header)) %% 2L == 0L, NA, 30)
  preds <- suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header,
    columns = list(terrain = list(elevation = "dem", uncertainty = "unc")), terrain = "columns"))
  y <- preds$targets$y$terrain
  expect_equal(y$source, "columns")
  expect_equal(y$dem, header$dem, tolerance = 1e-6)
  expect_true(all(y$applied))
  expect_null(preds$targets$hab$terrain)
  frame <- as.data.frame(preds)
  expect_true(all(c("y_terrain", "y_terrain_w_resolve", "y_terrain_w_dem", "y_terrain_w_dem_1km", "y_dem",
                    "y_dem_1km") %in% names(frame)))

  grids <- local_dir("suite_terrain_grids_")
  .write_plane_terrain(grids, .terrain_squares)
  from_grids <- suppressMessages(resolve.predict.suite(
    suite, species = data$species, header = header, terrain = list(source = "grids", grids = grids)))
  expect_equal(from_grids$targets$y$terrain$dem, .terrain_plane(header$lat, header$lon), tolerance = 1e-5)
  expect_error(suppressMessages(resolve.predict.suite(suite, species = data$species, header = header,
                                                      terrain = TRUE)), "ships no terrain field")
})
