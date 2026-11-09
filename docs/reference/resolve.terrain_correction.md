# Apply a Saved Terrain Correction

Apply a Saved Terrain Correction

## Usage

``` r
resolve.terrain_correction(
  path,
  prediction,
  dem,
  dem1km = NULL,
  uncertainty = NULL,
  spread = NULL,
  recognisedShare = NULL,
  nSpecies = NULL,
  device = "cpu"
)
```

## Arguments

- path:

  The correction file written by
  [`resolve.fit_terrain_correction()`](https://gillescolling.com/resolve/reference/resolve.fit_terrain_correction.md)
  (or shipped in a suite).

- prediction:

  The prediction per plot.

- dem:

  The descriptor at each plot's coordinate (`elevation` or `slope` of
  [`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md)).

- dem1km, uncertainty, spread, recognisedShare, nSpecies:

  Optional per-plot values: the descriptor over 1 km, the recorded
  location uncertainty in m, the suite members' standard deviation, the
  share of species the vocabulary recognises, the number of species.

- device:

  `"cpu"`, `"cuda"` or `"cuda:N"`.

## Value

A list: `corrected` (the raw prediction where the correction does not
apply: no prediction or no descriptor), `weights` (plots x 3 matrix,
columns `resolve`, `dem`, `dem_1km`), `offset`, `applied` (logical), and
the correction's `target`, `descriptor` and `report`.
