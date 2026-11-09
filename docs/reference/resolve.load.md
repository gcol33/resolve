# Load a Trained RESOLVE Model

Load a model from a saved checkpoint.

## Usage

``` r
resolve.load(path, device = "cpu", vram_fraction = 1)
```

## Arguments

- path:

  Path to checkpoint file

- device:

  Device to load the model onto: `"cpu"` (default), `"cuda"` (the
  current CUDA device) or `"cuda:N"` (CUDA device N).

- vram_fraction:

  Fraction of GPU VRAM the caching allocator may use when loading onto
  CUDA (default 1.0). Pass a lower value (e.g. 0.80) when sharing the
  GPU with a desktop / GUI to leave headroom.

## Value

A `Predictor` object (an Rcpp module class) exposing the loaded model.
Pass it to
[`resolve.predict.dataset()`](https://gillescolling.com/resolve/reference/resolve.predict.dataset.md)
to score a dataset built from its vocabularies (`predictor$vocabs()`),
or read its `schema()` / `dataset_config()` to see what the checkpoint
expects.

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

predictor <- resolve.load(path)
new_plots <- resolve.dataset.csv(plots, species, roles = roles,
                                 vocabs = predictor$vocabs(),
                                 config = predictor$dataset_config())
preds <- resolve.predict.dataset(predictor, new_plots)
head(preds$predictions$elevation)
```
