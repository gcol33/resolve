# Blend Predictions With Their Nearest Reference Plots

Retrieval augmentation (Gorishniy et al. 2024, TabR): from now on every
prediction of `predictor` is blended with the labels of the plot's k
nearest reference plots in the model's latent space (cosine similarity,
weighted by a softmax of similarity over `temperature`):
`(1 - alpha) * prediction + alpha * retrieved`, a regression target in
original units, a classification target as class probabilities (the
neighbours' label distribution), its class the most probable one. Only
the neighbours labelled for a target count for it; a plot with none
keeps the model's prediction.

## Usage

``` r
resolve.retrieval(
  predictor,
  reference,
  kNeighbors = NULL,
  alpha = NULL,
  temperature = NULL,
  batchSize = 4096L
)
```

## Arguments

- predictor:

  A `Predictor` from
  [`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md).

- reference:

  The reference plots, normally the training plots: a ResolveDataset
  built in the model's namespace (`vocabs = predictor$vocabs()`) that
  carries every target of the model. `NULL` removes a retrieval set
  before.

- kNeighbors:

  Reference plots each prediction reads.

- alpha:

  Share of the neighbours' labels in the blend, in \[0, 1\].

- temperature:

  Softmax temperature over the neighbours' cosine similarities (smaller
  puts the weight on the nearest).

- batchSize:

  Chunk size of the forward pass that encodes the reference plots
  (default 4096).

## Value

The predictor, invisibly.

## Details

Settings left `NULL` take the RESOLVE engine's defaults
([`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
`"retrieval"`). A reference plot that is also scored retrieves itself
first.

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
reference <- resolve.dataset.csv(plots, species, roles = roles, targets = targets,
                                 vocabs = predictor$vocabs(),
                                 config = predictor$dataset_config())
resolve.retrieval(predictor, reference, kNeighbors = 8L, alpha = 0.3)
preds <- resolve.predict.dataset(predictor, new_plots)
head(preds$predictions$elevation)
```
