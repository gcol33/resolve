# Predict with Dataset (C++ API)

Make predictions on a ResolveDataset using a trained model.

## Usage

``` r
resolve.predict.dataset(
  predictor,
  dataset,
  returnLatent = FALSE,
  batchSize = 4096L
)
```

## Arguments

- predictor:

  A Predictor object from resolve.load()

- dataset:

  A ResolveDataset object from resolve.dataset.csv()

- returnLatent:

  Return latent representations (default FALSE)

- batchSize:

  Forward-pass chunk size along dim 0 (default 4096). Use -1 for a
  single forward over the whole dataset (can OOM on large inputs); a
  positive value chunks the forward and concatenates results on CPU to
  bound peak VRAM. Matches the Python `batch_size` argument.

## Value

A list with `predictions` (one numeric vector per target: a regression
value on the original scale, or a classification target's predicted
class code), `probabilities` (one numeric matrix per classification
target, `n_plots x n_classes`, each row the softmax over the classes and
summing to one; the columns are named by the class labels the checkpoint
carries, in code order), `targets` (the dataset's own target values,
when it carried them), `plot_ids`, and `latent` when `returnLatent` is
`TRUE`. With a retrieval set
([`resolve.retrieval()`](https://gillescolling.com/resolve/reference/resolve.retrieval.md))
the predictions and probabilities are the blend.

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
head(preds$predictions$habitat)   # predicted class code per plot
preds$probabilities$habitat[1, ]  # the first plot's probability over classes
```
