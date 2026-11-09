# Code Elevation Grids Into a Coded Terrain Field

Compresses elevation grids with a bounded error: every cell is quantised
to a multiple of `2 * maxError`, so none moves by more than `maxError`,
and each block of cells is coded on its own (Lorenzo prediction, as in
the SZ error-bounded compressor, and Golomb-Rice codes with a run mode
for flat ground). The result is a terrain field file that
[`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md)
and suites read like a neural field.

## Usage

``` r
resolve.encode_terrain_grids(
  grids,
  path,
  maxError = 2,
  block = 64L,
  tiles = NULL
)
```

## Arguments

- grids:

  Directory of elevation grids (a `tiles.json` and its tiles).

- path:

  Where the coded field is written.

- maxError:

  Largest error a cell may carry, in m (default 2).

- block:

  Cells per side of an independently coded block (default 64).

- tiles:

  Names of the tiles to code; `NULL` codes every tile.

## Value

A named list of tile -\> metrics (RMSE, mean, 99th percentile and
largest error in m, slope error in degrees, seconds, bytes), invisibly.
