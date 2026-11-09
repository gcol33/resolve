# Read One Tile of an Elevation Model

Returns one tile of an elevation model whole: the stored cells of a
directory of elevation grids, or the cells of a terrain field file
decoded (a coded field) or evaluated (a neural field), so a field can be
compared with the grids it was made from cell by cell.

## Usage

``` r
resolve.terrain_grid(tile, grids = NULL, field = NULL, device = "cpu")
```

## Arguments

- tile:

  The tile's name, as `tiles.json` and the field file record it (the
  Copernicus GLO-30 tiles are named like `"GLO30_N46_E009"`).

- grids:

  Directory of elevation grids, or `NULL`.

- field:

  Path of a terrain field file (neural or coded), or `NULL`. Give
  exactly one of `grids` and `field`.

- device:

  With `field`: `"cpu"`, `"cuda"` or `"cuda:N"`.

## Value

A list: `tile`, the tile's geometry (`name`, `rows`, `cols`,
`lat_first`, `lon_first`, `dlat`, `dlon`; cell `[r, c]` is centred at
`lat_first - (r - 1) * dlat`, `lon_first + (c - 1) * dlon`), and
`elevation`, a `rows` x `cols` matrix in metres, its first row the
northern edge.

## See also

[`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md),
[`resolve.encode_terrain_grids()`](https://gillescolling.com/resolve/reference/resolve.encode_terrain_grids.md).
