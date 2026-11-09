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
  radii = NULL
)
```

## Arguments

- raw:

  The model's prediction per plot in the target's units: a numeric
  vector, or for `scale = "geographic"` a plots x 2 matrix of latitude
  and longitude.

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
  5, 10, 20, 50), `spatial` (`FALSE` reads composition only) and
  `matching_seed`.

- name:

  The target's name, as the report states it.

- radii:

  `NULL`, or a list of `similarity`, `km`, `years`: one hiding distance
  per complete plot replacing the drawn ones (to reproduce another
  implementation's draw).

## Value

A `resolve_gap_fit` list: `fitted` (as `raw`; a complete plot keeps its
recorded value), `method` (per plot, `"recorded"` for a complete plot)
and `report` (the gaps' distance per axis, the k chosen per neighbour
kind, every candidate's matched error and the chosen method's
coefficients, per group of gaps).

## Details

Each candidate is judged on the complete plots at the distance the gaps
actually are: every complete plot is predicted with the complete plots
nearer to it than a gap is to its nearest complete plot hidden, in
species similarity, in km and in survey years (nearest-neighbour
distance matching, Mila et al. 2022), by the target's own error (mean
absolute error; on a log scale that of log1p; the circular error of a
bearing; the median km of a location). Gaps with and without coordinates
choose apart. The raw prediction is never replaced: the fitted values
come back beside it.

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
