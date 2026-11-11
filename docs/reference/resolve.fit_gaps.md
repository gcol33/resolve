# Fit a Dataset's Gaps From Its Own Complete Plots

A dataset usually records a variable on most of its plots and lacks it
on some. Gap fitting chooses, on the plots that carry it, how to combine
the model's raw prediction with the dataset's own neighbours, and
applies that choice to the plots that lack it (the gaps). The candidates
are the complete plots' centre (`local_constant`), the raw prediction
(`resolve`), the raw prediction calibrated by least squares
(`offset_slope`), the median of the k most similar complete plots in
species composition (`knn_species`) or the k nearest by great-circle
distance (`knn_spatial`), and the raw prediction combined with those
medians by least squares (`stack_species`, `stack_species_spatial`).

## Usage

``` r
resolve.fit_gaps(
  raw,
  recorded,
  species,
  scale = "linear",
  period = 360,
  coordinates = NULL,
  years = NULL,
  gapFitting = NULL,
  name = "",
  folds = NULL
)
```

## Arguments

- raw:

  The model's prediction per plot in the target's units: a numeric
  vector, or for `scale = "geographic"` a plots x 2 matrix of latitude
  and longitude. `NA` marks a plot the model did not impute: it is
  neither a gap nor a complete plot.

- recorded:

  The dataset's recorded value per plot, shaped as `raw`; `NA` marks a
  gap.

- species:

  A plots x species numeric matrix; an entry other than 0 marks a
  species present.

- scale:

  How the values lie: `"linear"`, `"log"` (fitted on log1p),
  `"circular"` (a bearing with full turn `period`) or `"geographic"`.

- period:

  A bearing's full turn, in the target's units (`"circular"`).

- coordinates:

  `NULL` or a plots x 2 matrix of latitude and longitude (`NA` where
  unknown): the spatial axis.

- years:

  `NULL` or each plot's survey year (`NA` where unknown).

- gapFitting:

  `NULL` for the defaults, or a list of settings: `candidates` (method
  names), `k_grid` (the neighbour counts k is chosen from, default 1, 3,
  5, 10, 20, 50), `spatial` (`FALSE` reads composition only),
  `matching_seed`, `n_folds` (default 10), `max_fold_share` (the largest
  share of the complete plots one fold may hold, default 0.5) and
  `matching_sample` (complete plots the layouts are compared on, default
  1000).

- name:

  The target's name, as the report states it.

- folds:

  `NULL`, or an integer fold per complete plot (a plot with both
  `recorded` and `raw`; in plot order, numbered from 0) replacing the
  matched layout (to reproduce another implementation's layout).

## Value

A `resolve_gap_fit` list: `fitted` (as `raw`; a plot recording the value
keeps it, a plot lacking it that was not imputed is `NA`), `method` (per
plot, `"recorded"` for a plot recording the value, `"not_imputed"` for
one lacking it whose `raw` is `NA`) and `report` (`n_complete`,
`n_gaps`, `n_not_imputed`, the gaps' distance per axis, the fold layout,
the k chosen per neighbour kind, and per group of gaps every candidate's
cross-validated error, the estimated error of the whole choice, the same
error per held-out fold (`fold_error`, over `fold_n_estimated` complete
plots) and the chosen method's coefficients).

## Details

Each candidate is judged by cross-validation over the complete plots,
with folds laid out to hold plots out the way the gaps are held out
(k-fold nearest-neighbour distance matching, Linnenbrink et al. 2024):
squares of an equal-area plane, survey-year periods, both, or plot by
plot, the last also with buffers drawn from the gaps' distances,
whichever leaves the held-out plots as far from the rest, in species
similarity, in km and in survey years, as the gaps are from the complete
plots. Every fitted part (k, least-squares coefficients) is fitted on
the other folds, and the candidate with the lowest out-of-fold error, by
the target's own error (mean absolute error; on a log scale that of
log1p; the circular error of a bearing; the median km of a location),
predicts the gaps. The whole choice, repeated with each fold held out,
gives the error reported for it (`estimated_error`, nested
cross-validation). Gaps with and without coordinates choose apart. The
raw prediction is never replaced: the fitted values come back beside it.

Undated plots that are probably older than the dated ones, and plots
without coordinates that probably come from an area with no
georeferenced plots, are better served by the raw prediction: the
complete plots cannot show how far off such gaps are.

## See also

[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
with `gapFitting`, which fits a suite's targets after combining them.

## Examples

``` r
set.seed(1)
n <- 300
raw <- runif(n, 1950, 2020)
recorded <- 1990 + 0.5 * (raw - mean(raw)) + rnorm(n, 0, 2)
recorded[sample(n, 60)] <- NA
species <- matrix(rbinom(n * 20, 1, 0.2), n, 20)
fit <- resolve.fit_gaps(raw, recorded, species, name = "year")
fit$report$groups$unlocated$method
table(fit$method)
```
