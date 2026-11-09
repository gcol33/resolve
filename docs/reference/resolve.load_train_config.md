# Recover the Training Configuration From a Checkpoint

Reads back the training hyperparameters that were persisted in a RESOLVE
checkpoint (batch size, learning rate, weight decay, phase boundaries,
learning-rate schedule, band thresholds, VRAM fraction, ...) without
loading the model weights. Fields the checkpoint does not persist
(device, AMP and cuDNN flags) come back at their defaults.

## Usage

``` r
resolve.load_train_config(path)
```

## Arguments

- path:

  Path to a `.pt` checkpoint written by
  [`resolve.save()`](https://gillescolling.com/resolve/reference/resolve.save.md)
  or a training run.

## Value

A named list of training-configuration fields, the keys and spellings
`trainConfig` in
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
takes (enums such as `loss_config` and `lr_scheduler` by name).

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

cfg <- resolve.load_train_config(path)
cfg$max_epochs
cfg$batch_size
```
