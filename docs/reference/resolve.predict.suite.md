# Predict With a Model Suite

Score plots with every loaded target of a suite. The plots are read with
the columns the manifest's input contract names; `columns` renames them
for a table that spells them differently. No target column is read: the
plots to score carry no answer.

## Usage

``` r
resolve.predict.suite(
  suite,
  species,
  header = NULL,
  columns = NULL,
  batchSize = 4096L,
  keepMembers = FALSE,
  gapFitting = NULL,
  names = NULL,
  maxEditDistance = 1L,
  terrain = NULL
)
```

## Arguments

- suite:

  A `Suite` from
  [`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md).

- species:

  The species records, one row per record: a data.frame or the path of a
  CSV file.

- header:

  The plots, one row per plot: a data.frame, a CSV path, or `NULL`.
  Required when the suite reads plot-level columns (coordinates or
  covariates); when given it decides which plots are scored and their
  order. Must be the same kind (data.frame or path) as `species`.

- columns:

  Named list renaming the manifest's input columns, with keys among
  `plot_id`, `species_id`, `abundance`, `latitude`, `longitude`,
  `genus`, `family`, `covariates`, `categoricals` (another key is an
  error), `recorded` (a named list of regression target -\> the column
  holding its recorded value, `c(latitude, longitude)` for a location;
  adds `<target>_recorded` to
  [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) and
  `recorded` to the target, or with `gapFitting` the gap fit), for gap
  fitting `survey_year`, and for terrain correction `terrain` (a named
  list of `elevation`, `elevation_1km`, `slope`, `slope_1km` and
  `uncertainty` column names).

- batchSize:

  Forward-pass chunk size per member (default 4096).

- keepMembers:

  Also return every member's own prediction (default `FALSE`).

- gapFitting:

  `NULL` (default), `TRUE`, or a list of gap fitting settings
  (`candidates`, `k_grid`, `spatial`, `matching_seed`, `n_folds`,
  `max_fold_share`, `matching_sample`; see
  [`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md)):
  fit each target `columns$recorded` names from the dataset's own
  complete plots. Adds `<target>_fitted` and `<target>_fit_method` to
  [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) and a
  `gap_fit` (`fitted`, `method`, `report`) to each fitted target.

- names:

  Resolve the species names against each target's vocabulary before
  encoding: the path of the release's `names.json`, which maps entered
  names and synonyms to the names the models learnt, or `TRUE` for the
  vocabularies alone. Each target then carries `species_names`, a
  data.frame of the names not found as written and the route each took
  (see
  [`resolve.names()`](https://gillescolling.com/resolve/reference/resolve.names.md)).
  `NULL` (default) reads every name as written.

- maxEditDistance:

  With `names`: the edits a misspelled name may be away from a known
  name (default 1; 0 turns the spelling step off; 2 can also rename
  species the models never saw).

- terrain:

  `NULL` (default), or correct every target the suite gives a terrain
  correction (altitude, slope) by an elevation model read at the plots:
  `"field"` (or `TRUE`) reads the suite's terrain field at the
  coordinates, `list(source = "grids", grids = dir)` elevation grids,
  `"columns"` your own values named in `columns$terrain`. Adds
  `<target>_terrain`, the weight of each source and the descriptors read
  to [`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html), and
  a `terrain` result to each corrected target (see
  [`resolve.terrain_correction()`](https://gillescolling.com/resolve/reference/resolve.terrain_correction.md)).

## Value

A `resolve_suite_predictions` list: `plot_ids`, and `targets`, a named
list with one entry per target holding `value` (the combined prediction;
class codes for a vote, with the class labels in `label`; a plots x 2
matrix of `latitude` and `longitude` in degrees for a spherical mean),
`agreement`, `probabilities`, `dispersion`, `imputed` (logical, `FALSE`
where the plot is not imputed), the recognition columns above,
`recorded` (for a target `columns$recorded` names: logical, `TRUE` where
the plot records a value, so `value` is imputed where it is `FALSE`),
`members` (plots x members, with `keepMembers`; for a spherical mean a
list of `latitude` and `longitude`, each plots x members), and the
target's `units`, `dispersion_units`, `status` and `limit`.
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html) flattens
it to one row per plot.

## Details

Per plot and target the result holds the combined prediction and, each
on its own:

- for a class vote: the share of members naming the winning class
  (`agreement`) and the members' mean class probabilities (a tie in
  votes goes to the tied class with the higher mean probability);

- for a mean: the members' standard deviation (`dispersion`), in the
  target's units;

- for a circular mean (a bearing): the members' circular standard
  deviation (`dispersion`), in the target's units;

- for a spherical mean (a location): the median great-circle distance of
  the members from the combined point (`dispersion`), in km;

- how much of the plot's species list the target's vocabulary
  recognises: `n_species`, `n_recognised`, `recognised_share` (by
  distinct species) and `recognised_abundance_share` (by abundance).

Agreement and dispersion measure how far the members disagree; they are
not calibrated intervals. `NA` marks a value that is undefined for the
plot (a plot with no recorded species has no recognised share).

A plot none of whose species the target's vocabulary recognises (or with
no species at all) is not imputed: the models were trained on plots
holding species they know. `imputed` is `FALSE` for it, and its
prediction, agreement, probabilities, dispersion, members, gap fit and
terrain-corrected value are `NA` (a class code `-1`).

## See also

[`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md)

## Examples

``` r
source(system.file("examples", "habitat-suite.R", package = "resolveR"))
resolve.seal_suite(suite_dir)
suite <- resolve.load_suite(suite_dir)
preds <- resolve.predict.suite(suite, species = read.csv(species), header = read.csv(plots))
head(preds$targets$habitat$label)
head(as.data.frame(preds))
```
