# Fit a Single Model's Gaps From Its Predictions and the Dataset's Tables

[`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md)
for a single checkpoint: the species composition (every named species,
whatever the model's vocabulary holds), positions, survey years and
recorded values are read from the dataset's own tables, aligned to the
plots the model predicted.

## Usage

``` r
resolve.fit_prediction_gaps(
  predictions,
  species,
  header = NULL,
  columns,
  targets,
  gapFitting = NULL
)
```

## Arguments

- predictions:

  The output of
  [`resolve.predict.dataset()`](https://gillescolling.com/resolve/reference/resolve.predict.dataset.md).

- species:

  The species records, one row per record (a data.frame).

- header:

  The plots, one row per plot (a data.frame), or `NULL` to read
  plot-level columns from each plot's first species row.

- columns:

  Named list: `plot_id`, `species_id`, optionally `latitude`,
  `longitude`, `survey_year`, and `recorded`, a named list of target -\>
  recorded column (`c(latitude, longitude)` for a location).

- targets:

  Named list of target -\> list(`scale`, `period`, `outputs`): the scale
  as in
  [`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md)
  and the prediction head(s) read (default the target's name; a bearing
  may read its sine and cosine heads, a location reads its latitude and
  longitude heads).

- gapFitting:

  As in
  [`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md).

## Value

A named list of `resolve_gap_fit`, one per target.
