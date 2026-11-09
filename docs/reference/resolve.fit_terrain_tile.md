# Fit and Assemble a Terrain Field

A terrain field is an elevation model compressed tile by tile: each tile
is a coarse base plus a multiresolution hash encoding (Mueller et al.
2022) and a small network for the detail, fitted to the tile's cells
with the mean absolute error of the elevation and of its four-neighbour
differences (so the slope survives compression), and stored in float16.
`resolve.fit_terrain_tile()` fits one tile into its own file;
`resolve.assemble_terrain_field()` merges the tile files into one field
file that
[`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md)
and suites read.

## Usage

``` r
resolve.fit_terrain_tile(grids, tile, path, config = NULL)

resolve.assemble_terrain_field(tiles, path)
```

## Arguments

- grids:

  Directory of elevation grids (a `tiles.json` and its tiles).

- tile:

  The tile's name in `tiles.json`.

- path:

  Where the tile (or the assembled field) is written.

- config:

  `NULL` for the defaults, or a list of settings (`levels`,
  `log2_table_size`, `features`, `coarsest_resolution`,
  `finest_fraction`, `hidden_dim`, `n_layers`, `base_cells`, `steps`,
  `batch_size`, `lr`, `gradient_weight`, `seed`, `device`;
  `resolve.default_config("terrain_field")` lists their defaults).

- tiles:

  Paths of the tile files to merge.

## Value

`resolve.fit_terrain_tile()`: the tile's metrics (RMSE, mean, 99th
percentile and largest absolute error in m, slope error in degrees,
seconds, bytes). `resolve.assemble_terrain_field()`: `path`, invisibly.
