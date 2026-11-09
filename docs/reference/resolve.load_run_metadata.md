# Recover the Run Metadata From a Checkpoint

Reads back the run metadata persisted in a RESOLVE checkpoint: training
time, train/test plot counts, how the held-out fold was chosen
(`hold_out`: `"random"`, a `test_size` fraction drawn with `split_seed`,
or `"explicit"`, the plots named by `stoppingPlots`), best and total
epochs, the RESOLVE version and timestamps, and the per-target
final-metric tree.

## Usage

``` r
resolve.load_run_metadata(path)
```

## Arguments

- path:

  Path to a `.pt` checkpoint.

## Value

A named list of run-metadata fields, including `final_metrics`, a nested
list keyed by target then metric (e.g. `$final_metrics$area$rmse`).

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets)
path <- file.path(tempdir(), "model.pt")
fit <- resolve.train.dataset(dataset, maxEpochs = 20L, savePath = path, verbose = FALSE)

meta <- resolve.load_run_metadata(path)
meta$best_epoch
meta$final_metrics$elevation$rmse
```
