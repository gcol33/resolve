# Fit a Terrain Correction

Learns, per plot, how much to trust a prediction of altitude or slope
and an elevation model read at the plot's coordinates. An elevation
model read at a precise coordinate is close to the recorded altitude and
far off where the coordinate is wrong or coarse; a prediction from
species composition errs the other way round. A small network reads each
plot's features (the prediction, the descriptors and their differences,
the recorded location uncertainty, the members' spread, species
recognition) and returns softmax weights over three sources – the
prediction, the descriptor at the coordinate, the descriptor over 1 km –
plus a bounded offset, so that a weight of one returns a source exactly.
It is fitted with the mean absolute error.

## Usage

``` r
resolve.fit_terrain_correction(
  target,
  descriptor,
  prediction,
  dem,
  truth,
  path,
  dem1km = NULL,
  uncertainty = NULL,
  spread = NULL,
  recognisedShare = NULL,
  nSpecies = NULL,
  config = NULL
)
```

## Arguments

- target:

  The suite target corrected (as stored in the file).

- descriptor:

  `"elevation"` (an altitude target) or `"slope"`.

- prediction, truth:

  The prediction and the recorded value per plot (`NA` in `truth`:
  unrecorded, left out).

- dem:

  The descriptor at each plot's coordinate (`elevation` or `slope` of
  [`resolve.terrain_descriptors()`](https://gillescolling.com/resolve/reference/resolve.terrain_descriptors.md)).

- path:

  Where the fitted correction is written.

- dem1km, uncertainty, spread, recognisedShare, nSpecies:

  Optional per-plot values: the descriptor over 1 km, the recorded
  location uncertainty in m, the suite members' standard deviation, the
  share of species the vocabulary recognises, the number of species.

- config:

  `NULL` for the defaults, or a list of settings (`scale`,
  `offset_fraction`, `hidden_dim`, `n_layers`, `epochs`, `batch_size`,
  `lr`, `weight_decay`, `dem_1km_dropout`, `seed`, `device`;
  `resolve.default_config("terrain_correction")` lists their defaults).

## Value

The fit's report (invisibly): training plots and the mean absolute error
of the prediction, the descriptor and the corrected value on them.
