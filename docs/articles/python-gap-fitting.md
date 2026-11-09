# Python: gap fitting

Imputing a dataset’s missing values from its own complete plots: the
functions that fit one target or a single model’s targets, their
settings, the result and its report, and the neighbour searches the
choice is tested with. [Imputing a dataset’s
gaps](https://gillescolling.com/resolve/articles/gap-fitting.md) walks
through a fit; a model suite fits its targets through
[`SuitePredictor.predict(gap_fitting=...)`](https://gillescolling.com/resolve/articles/python-suites.html#SuitePredictor.predict).

[All of the Python
reference](https://gillescolling.com/resolve/articles/python-reference.md)

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The RESOLVE engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

## `fit_gaps()`

``` python
fit_gaps(target, raw, recorded, species, coordinates=None, years=None,
         config=GapFittingConfig(), folds=None)
```

Fits one target’s gaps. `target` is a `GapTarget`. `raw` and `recorded`
are the model’s imputed value and the dataset’s recorded value per plot
in the target’s units, `(n,)` tensors, or `(n, 2)` latitude and
longitude for a geographic target; a `NaN` recorded value marks a gap.
`species` is the `(n, n_species)` composition, dense or sparse CSR, an
entry other than 0 a species present. `coordinates` is `(n, 2)` latitude
and longitude (`NaN` where unknown) and `years` `(n,)` survey years;
either may be `None`. `folds`, an int64 tensor with the fold of each
complete plot in plot order (numbered from 0), replaces the matched
layout, so a fit reproduces another implementation’s layout. Returns a
`GapFit`. With no gap the recorded values come back; with fewer than two
complete plots the gaps keep the raw imputed value and the report says
why.

``` python
import numpy as np
import torch
import resolve_core as rc

rng = np.random.default_rng(1)
raw = torch.from_numpy(rng.uniform(1950, 2020, 300))
recorded = 1990 + 0.5 * (raw - raw.mean()) + torch.from_numpy(rng.normal(0, 2, 300))
recorded[::5] = float("nan")
species = torch.from_numpy((rng.random((300, 20)) < 0.2).astype("float32"))
fit = rc.fit_gaps(rc.GapTarget("year"), raw, recorded, species)
fit.report["groups"]["unlocated"]["method"]
```

## `fit_prediction_gaps()`

``` python
fit_prediction_gaps(predictions, species, header=None, columns=None, targets=None,
                    gap_fitting=None)
```

Fits a single model’s targets from its imputed values
(`Predictor.predict_dataset`) and the dataset’s tables, read in the
order of `predictions.plot_ids` by `read_gap_tables`. `targets` maps
each target to `{"scale": ..., "period": ..., "outputs": [...]}`: the
scale by name (`"linear"`, `"log"`, `"circular"`, `"geographic"`) and
the prediction head(s) read, by default the target’s name; a bearing may
read its sine and cosine heads, a location reads its latitude and
longitude heads. `gap_fitting` is a `GapFittingConfig`, a `dict` of its
fields or `None` for the defaults. Returns a `dict` of target to
`GapFit`.
`fit_prediction_gaps_tables(predictions, tables, requests, config)` is
the form over a `GapTables` and a list of `GapFitRequest`.

## `read_gap_tables()`

``` python
read_gap_tables(species, header=None, columns=None, plot_ids=None)
```

Reads what gap fitting needs about the plots `plot_ids`, in that order:
the presence of every species the records name, whatever a model’s
vocabulary holds, the positions, the survey years and the recorded
values. `species` and `header` are data frames or CSV paths of the same
kind; without `header`, plot-level columns come from each plot’s first
species row. `columns` is a `GapColumns` or a `dict` of `plot_id`,
`species_id`, `latitude`, `longitude`, `survey_year` and `recorded`
(`{target: column}`, `{target: [latitude, longitude]}` for a location).
Returns a `GapTables`. `read_gap_tables_csv` and
`read_gap_tables_columns` are the forms over CSV paths and string
columns.

## `GapFittingConfig`

``` python
GapFittingConfig()
```

- `candidates` (`list[GapMethod]`, default every method): the methods
  the choice is made among. A method that reads spatial neighbours is a
  candidate only where the gaps carry coordinates, so the gaps without
  coordinates need one that reads no position.
- `k_grid` (`list[int]`, default `[1, 3, 5, 10, 20, 50]`): the neighbour
  counts `k` is chosen from.
- `spatial` (`bool`, default `True`): spatial neighbours as candidates;
  `False` reads composition only.
- `matching_seed` (`int`, default `20261106`): the seed of the order the
  fold layout deals its groups in, the buffers it draws and the complete
  plots it is compared on.
- `n_folds` (`int`, default `10`): the folds of the cross-validation;
  the estimated error needs at least 3.
- `max_fold_share` (`float`, default `0.5`): the largest share of the
  complete plots one fold of a layout may hold.
- `matching_sample` (`int`, default `1000`): the complete plots whose
  distances the layouts are compared on.

`validate()` raises for an empty grid, a `k` below 1, fewer than 2 folds
or a `max_fold_share` outside (1 / `n_folds`, 1\].

## `GapTarget`

``` python
GapTarget(name="", scale=GapScale.Linear, period=360.0)
```

The target a fit is for: its name (as the report states it), its
`GapScale` and, for a bearing, the length of a full turn in the target’s
units.

## `GapScale`

How a target’s values are laid out for fitting: `Linear` (the value),
`Log` (log1p of the value, for a target whose error is measured on that
scale; its local constant is the median), `Circular` (a bearing as its
sine and cosine; a least-squares coefficient is a complex number that
turns and shrinks the imputed value) and `Geographic` (latitude and
longitude projected to km of a Lambert azimuthal equal-area plane
centred on the complete plots’ geographic median, one coefficient for
both axes).

## `GapMethod`

The methods, in the order a tie between equal errors is broken:
`LocalConstant`, `Resolve`, `OffsetSlope`, `KnnSpecies`, `StackSpecies`,
`KnnSpatial`, `StackSpeciesSpatial`. Their names in a report and in
`GapFit.method` are `local_constant`, `resolve`, `offset_slope`,
`knn_species`, `stack_species`, `knn_spatial` and
`stack_species_spatial`.

## `GapFit`

- `fitted`: per plot in the target’s units, shaped as `raw`: a complete
  plot’s recorded value, a gap’s fitted value.
- `method` (`list[str]`): `"recorded"` for a complete plot, the method’s
  name for a gap; `method_codes` is the same as an int64 tensor, -1 for
  a complete plot.
- `report` (`dict`): `target`, `scale`, `n_complete`, `n_gaps`; `axes`,
  per axis (`similarity`, `km`, `years`) whether the fold layout was
  matched on it and the quantiles of the gaps’ distance to their nearest
  complete plot; `folds`, the layout (`n_folds`, `block_km`,
  `block_years`, `buffer_km`, `buffer_years`, `mismatch` per axis,
  `total_mismatch`, `n_layouts`, `n_matching_plots`, `injected`);
  `neighbours`, per kind the `k` chosen and the cross-validated error of
  each `k`; `groups`, per group of gaps (`located`, `unlocated`) the
  method chosen, every candidate’s cross-validated error (`cv_error`),
  the error estimated for the whole choice by nested cross-validation
  (`estimated_error`, over `n_estimated` complete plots), and for the
  method chosen its `coefficients`, `n_fitted` and `k`. `skipped` says
  why nothing was chosen when that happened.

## `gap_folds()`

``` python
gap_folds(target, recorded, species, coordinates=None, years=None,
          config=GapFittingConfig())
```

The fold layout the choice is judged on, for inspecting a fit: a `dict`
of `complete` and `gaps` (the plots’ rows), `folds` (the fold of each
complete plot, an int64 tensor) and `layout` (as the report’s `folds`).

## `GapColumns`

The columns `read_gap_tables` reads: `plot_id`, `species_id`,
`latitude`, `longitude`, `survey_year` (each `str` or `None`) and
`recorded` (`list[tuple[str, list[str]]]`).

## `GapTables`

What `read_gap_tables` returns: `species` (sparse CSR presence, one row
per plot), `coordinates` (`(n, 2)` or `None`), `years` (`(n,)` or
`None`) and `recorded` (`dict` of target to its recorded values, `NaN`
where missing).

## `GapFitRequest`

``` python
GapFitRequest(target, outputs)
```

One target of a single model for `fit_prediction_gaps_tables`: its
`GapTarget` and the prediction head(s) it reads.

## `nearest_by_great_circle()`

``` python
nearest_by_great_circle(queries, reference, k, exclude_self=False)
```

For each row of `queries`, the `k` rows of `reference` nearest to it by
great-circle distance, nearest first. Both are `(rows, 2)` latitude and
longitude tensors in degrees. A row with a missing coordinate is never a
neighbour and has none. The search is exact and computed in blocks, as
`nearest_by_cosine` is. Returns `(indices, km)`, -1 and `NaN` where a
row has fewer than `k`.
