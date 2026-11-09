# Flatten Suite Predictions to One Row per Plot

The RESOLVE engine's prediction table, the columns
`resolve predict --suite` writes: `plot_id`, then per target the
prediction (`<target>`; for a vote also `<target>_code` and
`<target>_agreement`; for a spherical mean `<target>_latitude` and
`<target>_longitude`), its dispersion (`<target>_sd` for a mean,
`<target>_circular_sd` for a bearing, `<target>_spread_km` for a
spherical mean), the four recognition columns, and, when present,
`<target>_seed<N>` for each member (`<target>_seed<N>_latitude` and
`_longitude` for a spherical mean); a gap-fitted target adds
`<target>_fitted` (`<target>_fitted_latitude` / `_fitted_longitude`) and
`<target>_fit_method`, and a target with recorded values but no gap fit
`<target>_recorded` (logical) there instead; a terrain-corrected target
`<target>_terrain`, `<target>_terrain_w_resolve`, `_w_dem`,
`_w_dem_1km`, `<target>_dem` and `<target>_dem_1km`.

## Usage

``` r
# S3 method for class 'resolve_suite_predictions'
as.data.frame(
  x,
  row.names = NULL,
  optional = FALSE,
  probabilities = FALSE,
  ...
)
```

## Arguments

- x:

  A `resolve_suite_predictions` object from
  [`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md).

- row.names, optional:

  Ignored; present for the generic's signature.

- probabilities:

  Also include each vote target's class probabilities,
  `<target>_prob_<class>` (default `FALSE`).

- ...:

  Unused.

## Value

A data.frame with one row per plot.
