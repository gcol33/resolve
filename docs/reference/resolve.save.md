# Save a Trained RESOLVE Model

Write a trained model, its scalers and its vocabularies to a checkpoint
file that
[`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md)
reads back.

## Usage

``` r
resolve.save(trainer, path)
```

## Arguments

- trainer:

  The list returned by
  [`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md),
  or the `Trainer` object it carries.

- path:

  Path of the checkpoint file to write.

## Value

No return value, called for its side effect of writing the checkpoint to
`path`.

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets)
fit <- resolve.train.dataset(dataset, maxEpochs = 20L, verbose = FALSE)
resolve.save(fit, file.path(tempdir(), "model.pt"))
```
