# Terrain correction: a prediction of altitude or slope combined with an
# elevation model read at the plot's coordinates, and the compressed elevation
# model (terrain field) a suite ships for it. Thin wrappers over the engine's
# terrain_source / terrain_correction / terrain_field (resolve_terrain_* in
# the C ABI).

#' Read Elevation-Model Descriptors at Plot Coordinates
#'
#' Reads four descriptors per plot from an elevation model: the elevation at
#' the coordinate (bilinear between cell centres), the mean elevation over the
#' 1 km square centred on it, the slope of the cell nearest the coordinate
#' from its four edge neighbours (as Earth Engine's `ee.Terrain.slope`), and
#' the inclination of the least-squares plane through the 1 km square. The
#' elevation model is either a directory of elevation grids (a `tiles.json`
#' and its float32 `.npy` tiles) or a terrain field file, the same tiles
#' compressed into neural fields; both are read through one definition, so
#' the two differ only by the field's reconstruction error.
#'
#' @param latitude,longitude Plot coordinates in degrees (WGS84).
#' @param grids Directory of elevation grids, or `NULL`.
#' @param field Path of a terrain field file (neural or coded), or `NULL`.
#'   Give exactly one of `grids` and `field`.
#' @param device With `field`: `"cpu"`, `"cuda"` or `"cuda:N"`.
#' @return A data.frame with one row per plot: `elevation` and
#'   `elevation_1km` (m), `slope` and `slope_1km` (degrees); `NA` where the
#'   elevation model holds nothing.
#' @seealso [resolve.predict.suite()] with `terrain`.
#' @export
resolve.terrain_descriptors <- function(latitude, longitude, grids = NULL, field = NULL, device = "cpu") {
  if (is.null(grids) == is.null(field)) stop("give exactly one of grids and field", call. = FALSE)
  if (length(latitude) != length(longitude)) stop("latitude and longitude differ in length", call. = FALSE)
  .resolve_require_backend()
  source <- if (!is.null(grids)) list(grids = normalizePath(grids, mustWork = TRUE))
            else list(field = normalizePath(field, mustWork = TRUE), device = device)
  out <- .resolve_module()$terrain_descriptors(source, .resolve_na_real(latitude), .resolve_na_real(longitude))
  as.data.frame(lapply(out, .resolve_na_real), optional = TRUE)
}


#' Read One Tile of an Elevation Model
#'
#' Returns one tile of an elevation model whole: the stored cells of a
#' directory of elevation grids, or the cells of a terrain field file decoded
#' (a coded field) or evaluated (a neural field), so a field can be compared
#' with the grids it was made from cell by cell.
#'
#' @param tile The tile's name, as `tiles.json` and the field file record it
#'   (the Copernicus GLO-30 tiles are named like `"GLO30_N46_E009"`).
#' @inheritParams resolve.terrain_descriptors
#' @return A list: `tile`, the tile's geometry (`name`, `rows`, `cols`,
#'   `lat_first`, `lon_first`, `dlat`, `dlon`; cell `[r, c]` is centred at
#'   `lat_first - (r - 1) * dlat`, `lon_first + (c - 1) * dlon`), and
#'   `elevation`, a `rows` x `cols` matrix in metres, its first row the
#'   northern edge.
#' @seealso [resolve.terrain_descriptors()], [resolve.encode_terrain_grids()].
#' @export
resolve.terrain_grid <- function(tile, grids = NULL, field = NULL, device = "cpu") {
  if (is.null(grids) == is.null(field)) stop("give exactly one of grids and field", call. = FALSE)
  if (!is.character(tile) || length(tile) != 1L || is.na(tile)) stop("tile must be one name", call. = FALSE)
  .resolve_require_backend()
  source <- if (!is.null(grids)) list(grids = normalizePath(grids, mustWork = TRUE))
            else list(field = normalizePath(field, mustWork = TRUE), device = device)
  out <- .resolve_module()$terrain_grid(source, tile)
  geometry <- out$tile
  list(tile = geometry,
       elevation = matrix(out$elevation, nrow = geometry$rows, ncol = geometry$cols, byrow = TRUE))
}


#' Fit a Terrain Correction
#'
#' Learns, per plot, how much to trust a prediction of altitude or slope and
#' an elevation model read at the plot's coordinates. An elevation model read
#' at a precise coordinate is close to the recorded altitude and far off where
#' the coordinate is wrong or coarse; a prediction from species composition
#' errs the other way round. A small network reads each plot's features (the
#' prediction, the descriptors and their differences, the recorded location
#' uncertainty, the members' spread, species recognition) and returns
#' softmax weights over three sources -- the prediction, the descriptor at the
#' coordinate, the descriptor over 1 km -- plus a bounded offset, so that a
#' weight of one returns a source exactly. It is fitted with the mean absolute
#' error.
#'
#' @param target The suite target corrected (as stored in the file).
#' @param descriptor `"elevation"` (an altitude target) or `"slope"`.
#' @param prediction,truth The prediction and the recorded value per plot
#'   (`NA` in `truth`: unrecorded, left out).
#' @param dem The descriptor at each plot's coordinate (`elevation` or
#'   `slope` of [resolve.terrain_descriptors()]).
#' @param dem1km,uncertainty,spread,recognisedShare,nSpecies Optional per-plot
#'   values: the descriptor over 1 km, the recorded location uncertainty in m,
#'   the suite members' standard deviation, the share of species the
#'   vocabulary recognises, the number of species.
#' @param path Where the fitted correction is written.
#' @param config `NULL` for the defaults, or a list of settings (`scale`,
#'   `offset_fraction`, `hidden_dim`, `n_layers`, `epochs`, `batch_size`,
#'   `lr`, `weight_decay`, `dem_1km_dropout`, `seed`, `device`;
#'   `resolve.default_config("terrain_correction")` lists their defaults).
#' @return The fit's report (invisibly): training plots and the mean absolute
#'   error of the prediction, the descriptor and the corrected value on them.
#' @export
resolve.fit_terrain_correction <- function(target, descriptor, prediction, dem, truth, path, dem1km = NULL,
                                           uncertainty = NULL, spread = NULL, recognisedShare = NULL,
                                           nSpecies = NULL, config = NULL) {
  .resolve_require_backend()
  inputs <- .resolve_terrain_inputs(prediction, dem, dem1km, uncertainty, spread, recognisedShare, nSpecies)
  report <- .resolve_module()$fit_terrain_correction(as.character(target), as.character(descriptor), inputs,
                                                     .resolve_na_real(truth), if (is.null(config)) list() else config,
                                                     path.expand(path))
  invisible(report)
}


#' Apply a Saved Terrain Correction
#'
#' @inheritParams resolve.fit_terrain_correction
#' @param path The correction file written by [resolve.fit_terrain_correction()]
#'   (or shipped in a suite).
#' @param device `"cpu"`, `"cuda"` or `"cuda:N"`.
#' @return A list: `corrected` (the raw prediction where the correction does
#'   not apply: no prediction or no descriptor), `weights` (plots x 3 matrix,
#'   columns `resolve`, `dem`, `dem_1km`), `offset`, `applied` (logical), and
#'   the correction's `target`, `descriptor` and `report`.
#' @export
resolve.terrain_correction <- function(path, prediction, dem, dem1km = NULL, uncertainty = NULL, spread = NULL,
                                       recognisedShare = NULL, nSpecies = NULL, device = "cpu") {
  .resolve_require_backend()
  inputs <- .resolve_terrain_inputs(prediction, dem, dem1km, uncertainty, spread, recognisedShare, nSpecies)
  .resolve_terrain_result(.resolve_module()$terrain_correction_apply(normalizePath(path, mustWork = TRUE),
                                                                     inputs, device))
}


#' Fit and Assemble a Terrain Field
#'
#' A terrain field is an elevation model compressed tile by tile: each tile is
#' a coarse base plus a multiresolution hash encoding (Mueller et al. 2022)
#' and a small network for the detail, fitted to the tile's cells with the
#' mean absolute error of the elevation and of its four-neighbour differences
#' (so the slope survives compression), and stored in float16.
#' `resolve.fit_terrain_tile()` fits one tile into its own file;
#' `resolve.assemble_terrain_field()` merges the tile files into one field
#' file that [resolve.terrain_descriptors()] and suites read.
#'
#' @param grids Directory of elevation grids (a `tiles.json` and its tiles).
#' @param tile The tile's name in `tiles.json`.
#' @param path Where the tile (or the assembled field) is written.
#' @param config `NULL` for the defaults, or a list of settings (`levels`,
#'   `log2_table_size`, `features`, `coarsest_resolution`, `finest_fraction`,
#'   `hidden_dim`, `n_layers`, `base_cells`, `steps`, `batch_size`, `lr`,
#'   `gradient_weight`, `seed`, `device`;
#'   `resolve.default_config("terrain_field")` lists their defaults).
#' @param tiles Paths of the tile files to merge.
#' @return `resolve.fit_terrain_tile()`: the tile's metrics (RMSE, mean, 99th
#'   percentile and largest absolute error in m, slope error in degrees,
#'   seconds, bytes). `resolve.assemble_terrain_field()`: `path`, invisibly.
#' @export
resolve.fit_terrain_tile <- function(grids, tile, path, config = NULL) {
  .resolve_require_backend()
  .resolve_module()$fit_terrain_tile(normalizePath(grids, mustWork = TRUE), as.character(tile),
                                     if (is.null(config)) list() else config, path.expand(path))
}

#' @rdname resolve.fit_terrain_tile
#' @export
resolve.assemble_terrain_field <- function(tiles, path) {
  .resolve_require_backend()
  .resolve_module()$assemble_terrain_field(normalizePath(tiles, mustWork = TRUE), path.expand(path))
  invisible(path)
}


#' Code Elevation Grids Into a Coded Terrain Field
#'
#' Compresses elevation grids with a bounded error: every cell is quantised to
#' a multiple of `2 * maxError`, so none moves by more than `maxError`, and
#' each block of cells is coded on its own (Lorenzo prediction, as in the SZ
#' error-bounded compressor, and Golomb-Rice codes with a run mode for flat
#' ground). The result is a terrain field file that
#' [resolve.terrain_descriptors()] and suites read like a neural field.
#'
#' @param grids Directory of elevation grids (a `tiles.json` and its tiles).
#' @param path Where the coded field is written.
#' @param maxError Largest error a cell may carry, in m (default 2).
#' @param block Cells per side of an independently coded block (default 64).
#' @param tiles Names of the tiles to code; `NULL` codes every tile.
#' @return A named list of tile -> metrics (RMSE, mean, 99th percentile and
#'   largest error in m, slope error in degrees, seconds, bytes), invisibly.
#' @export
resolve.encode_terrain_grids <- function(grids, path, maxError = 2, block = 64L, tiles = NULL) {
  .resolve_require_backend()
  invisible(.resolve_module()$encode_terrain_grids(normalizePath(grids, mustWork = TRUE),
                                                   list(max_error = as.numeric(maxError), block = as.integer(block)),
                                                   path.expand(path), as.character(tiles)))
}


.resolve_terrain_inputs <- function(prediction, dem, dem1km, uncertainty, spread, recognisedShare, nSpecies) {
  n <- length(prediction)
  check <- function(x, what) {
    if (is.null(x)) return(NULL)
    if (length(x) != n) stop(sprintf("%s must have one value per plot (%d)", what, n), call. = FALSE)
    .resolve_na_real(x)
  }
  inputs <- list(prediction = .resolve_na_real(prediction), dem = check(dem, "dem"),
                 dem_1km = check(dem1km, "dem1km"), uncertainty_m = check(uncertainty, "uncertainty"),
                 spread = check(spread, "spread"), recognised_share = check(recognisedShare, "recognisedShare"),
                 n_species = check(nSpecies, "nSpecies"))
  inputs[!vapply(inputs, is.null, logical(1))]
}

.resolve_terrain_result <- function(raw) {
  weights <- raw$weights
  if (is.matrix(weights)) colnames(weights) <- c("resolve", "dem", "dem_1km")
  out <- list(corrected = .resolve_na_real(raw$corrected), weights = weights,
              offset = .resolve_na_real(raw$offset), applied = as.logical(unlist(raw$applied)))
  if (!is.null(raw$target)) out$target <- raw$target
  if (!is.null(raw$descriptor)) out$descriptor <- raw$descriptor
  if (!is.null(raw$report)) out$report <- raw$report
  out
}

# The terrain option of resolve.predict.suite(): NULL / FALSE, TRUE (the
# suite's field), a source name, or a list of source, grids, device.
.resolve_terrain_option <- function(terrain) {
  if (is.null(terrain) || isFALSE(terrain)) return(NULL)
  if (isTRUE(terrain)) return(list(source = "field"))
  if (is.character(terrain) && length(terrain) == 1) terrain <- list(source = terrain)
  if (!is.list(terrain)) stop("terrain must be NULL, TRUE, a source name or a list", call. = FALSE)
  if (!is.null(terrain$grids)) terrain$grids <- normalizePath(terrain$grids, mustWork = TRUE)
  terrain
}
