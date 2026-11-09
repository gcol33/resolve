# Cross-Validate a Model on a Dataset

Train and score the same model recipe on k folds of a dataset's plots:
each fold is held out once while the model, restarted from the same
initial weights, trains on the others. Folds are drawn at random with
`seed`, or, with `spatial`, as blocks of a latitude / longitude grid, so
a held-out fold is a region the model never saw.

## Usage

``` r
resolve.cross_validate(dataset, nFolds = 5L, seed = 42L, spatial = NULL, ...)
```

## Arguments

- dataset:

  A ResolveDataset object from
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md).

- nFolds:

  Number of folds (default 5).

- seed:

  Random seed of the folds (default 42).

- spatial:

  `NULL` (default) for random folds, or a named list of the spatial
  block settings: `lat_size` and `lon_size` (block size in degrees) and
  `balance` (`TRUE` assigns blocks to folds by greedy bin-packing for
  folds of similar size, `FALSE` round-robin); a field left out takes
  the engine's default
  ([`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
  `"spatial_block"`). Needs coordinates.

- ...:

  The model and training arguments of
  [`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
  (`hiddenDims`, `maxEpochs`, `lr`, `modelConfig`, `trainConfig`,
  `pretrain`, ...).

## Value

A list with `n_folds`, `mean_metrics` and `std_metrics` (per target and
metric, over the folds), `fold_results` (each fold's training result, as
`result` of
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md))
and `total_time_seconds`.

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets)
cv <- resolve.cross_validate(dataset, nFolds = 3L, maxEpochs = 10L)
cv$mean_metrics$elevation$r2

blocks <- resolve.cross_validate(dataset, nFolds = 3L, maxEpochs = 10L,
                                 spatial = list(lat_size = 0.2, lon_size = 0.3))
```
