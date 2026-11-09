# Read Elevation-Model Descriptors at Plot Coordinates

Reads four descriptors per plot from an elevation model: the elevation
at the coordinate (bilinear between cell centres), the mean elevation
over the 1 km square centred on it, the slope of the cell nearest the
coordinate from its four edge neighbours (as Earth Engine's
`ee.Terrain.slope`), and the inclination of the least-squares plane
through the 1 km square. The elevation model is either a directory of
elevation grids (a `tiles.json` and its float32 `.npy` tiles) or a
terrain field file, the same tiles compressed into neural fields; both
are read through one definition, so the two differ only by the field's
reconstruction error.

## Usage

``` r
resolve.terrain_descriptors(
  latitude,
  longitude,
  grids = NULL,
  field = NULL,
  device = "cpu"
)
```

## Arguments

- latitude, longitude:

  Plot coordinates in degrees (WGS84).

- grids:

  Directory of elevation grids, or `NULL`.

- field:

  Path of a terrain field file (neural or coded), or `NULL`. Give
  exactly one of `grids` and `field`.

- device:

  With `field`: `"cpu"`, `"cuda"` or `"cuda:N"`.

## Value

A data.frame with one row per plot: `elevation` and `elevation_1km` (m),
`slope` and `slope_1km` (degrees); `NA` where the elevation model holds
nothing.

## See also

[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
with `terrain`.
