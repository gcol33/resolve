# Python: saving and predicting

Checkpoints, what they record about the run, and predictions for new
plots: the predictor, its predictions, retrieval of the nearest training
plots, the scalers, the run record, and the helpers that rebuild a
checkpoint’s loader settings and targets.

[All of the Python
reference](https://gillescolling.com/resolve/articles/python-reference.md)

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

The examples on this page run in order on the survey that ships with the
R package: 150 plots in the Alps, with one table of plots and one of the
species recorded on them. They start from a short fit saved as a
checkpoint; [Predicting new
plots](https://gillescolling.com/resolve/articles/prediction.md) walks
through the same steps with their output.

``` python
import os
import tempfile

import pandas as pd
import resolve_core as rc

data = "https://raw.githubusercontent.com/gcol33/resolve/master/r/inst/extdata/"
plots = pd.read_csv(data + "plots.csv")
species = pd.read_csv(data + "species.csv")

roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"
targets = [rc.TargetSpec.regression("elevation"),
           rc.TargetSpec.classification("habitat", 4)]
dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets)

config = rc.TrainConfig()
config.max_epochs = 100
trainer = rc.Trainer(rc.ResolveModel(dataset.schema), config)
trainer.prepare_data(dataset)
trainer.fit()

folder = tempfile.mkdtemp()
path = os.path.join(folder, "model.pt")
trainer.save(path, metadata=trainer.run_metadata)
```

## `Predictor`

``` python
Predictor.load(path, device="cpu", vram_fraction=1.0)
Predictor(model, scalers, device="cpu")
```

Predictions from a trained model. `load` reads a checkpoint
`Trainer.save` wrote; `device` is `"cpu"`, `"cuda"` (the current GPU) or
`"cuda:N"`, and anything else raises `ValueError`. The checkpoint is
read onto that device, so one trained on a GPU loads on a machine
without one. `vram_fraction` caps the share of GPU memory the predictor
may use. A file that cannot be read raises `StorageError` after a few
retries; an unreadable archive raises `RuntimeError`. The constructor
assembles a predictor from a model and its `Scalers`.

``` python
predictor = rc.Predictor.load(path)
predictor.schema.n_species_vocab
```

### `predict_dataset()`

``` python
predict_dataset(dataset, return_latent=False, batch_size=4096)
```

The predictions for every plot of a dataset, as `ResolvePredictions`.
The dataset has to be coded with the checkpoint’s vocabularies, or its
integer codes would index the wrong rows of the embedding tables: load
new plots with `ResolveDataset.from_csv_with_vocabs` (or another
`*_with_vocabs` loader), passing `predictor.external_vocabs` and
`predictor.dataset_config`. A dataset coded any other way is refused.
`return_latent` also returns each plot’s latent representation.
`batch_size` is the number of plots per forward pass, `-1` for one pass
over the whole dataset; the outputs are moved to the CPU as each chunk
is done, and the result does not depend on the chunk size.

``` python
plots.head(20).to_csv(os.path.join(folder, "new_plots.csv"), index=False)
species.to_csv(os.path.join(folder, "new_species.csv"), index=False)
new_plots = rc.ResolveDataset.from_csv_with_vocabs(
    os.path.join(folder, "new_plots.csv"),
    os.path.join(folder, "new_species.csv"),
    roles, [], predictor.external_vocabs, predictor.dataset_config)
preds = predictor.predict_dataset(new_plots)
preds.predictions["elevation"][:5]
```

### `predict()`

``` python
predict(*, coordinates=None, covariates=None, unknown_fraction=None,
        unknown_count=None, hash_embedding=None, genus_ids=None,
        family_ids=None, species_ids=None, species_vector=None,
        pool_genus_ids=None, pool_family_ids=None, pool_weights=None,
        pool_mask=None, pool_has_cover=None, categorical_ids=None,
        return_latent=False, batch_size=4096)
```

The predictions from tensors, every argument by keyword; leave out what
the encoding does not use. The first five are the parts of the
continuous block, assembled and standardised as in training;
`unknown_fraction` and `unknown_count` are required where the model
reads them. The other inputs are the `ResolveDataset` tensors of the
same names. `batch_size` chunks the forward as in `predict_dataset`.

``` python
from_tensors = predictor.predict(
    unknown_fraction=new_plots.unknown_fraction,
    hash_embedding=new_plots.hash_embedding,
    genus_ids=new_plots.genus_ids, family_ids=new_plots.family_ids)
```

### Embeddings

- `get_embeddings(*, ..., batch_size=4096)`: each plot’s latent
  representation, `(n_plots, latent_dim)`, from the keyword inputs of
  `predict`.
- `get_species_embeddings()`: the species table,
  `(n_species_vocab, species_embed_dim)`, or `None` for the hash and
  sparse encodings.
- `get_genus_embeddings()`, `get_family_embeddings()`: the genus and
  family tables, or `None` without taxonomy.

The tables come back detached, on the CPU. [Latent and species
embeddings](https://gillescolling.com/resolve/articles/embeddings.md)
reads them.

``` python
latent = predictor.get_embeddings(
    unknown_fraction=new_plots.unknown_fraction,
    hash_embedding=new_plots.hash_embedding,
    genus_ids=new_plots.genus_ids, family_ids=new_plots.family_ids)
latent.shape
```

### `set_retrieval()`

``` python
set_retrieval(reference, config=RetrievalConfig(), batch_size=4096)
clear_retrieval()
```

Retrieval augmentation (Gorishniy and others 2024, TabR): every later
prediction is blended with the labels of the plot’s nearest reference
plots in the latent space. `reference` is a dataset coded with the
checkpoint’s vocabularies and carrying every target’s labels, normally
the training plots; `target_specs_from_schema` gives its targets. For
each plot the `k_neighbors` reference plots of highest cosine similarity
of their latents are weighted by a softmax of similarity over
`temperature`. Only neighbours labelled for a target count for it, and a
plot with none keeps the model’s prediction. A regression target becomes
`(1 - alpha) * prediction + alpha * the neighbours' weighted label`, in
original units; a classification target blends its probabilities with
the neighbours’ label distribution and predicts the largest. The latent
and `get_embeddings` are not blended. The search is exact. A reference
plot that is also scored finds itself first. `clear_retrieval` returns
to the model’s own predictions.

``` python
reference = rc.ResolveDataset.from_pandas(plots, species, roles,
                                          rc.target_specs_from_schema(predictor.schema),
                                          predictor.dataset_config,
                                          schema_source=dataset)
retrieval = rc.RetrievalConfig()
retrieval.k_neighbors = 16
predictor.set_retrieval(reference, retrieval)
blended = predictor.predict_dataset(new_plots)
predictor.clear_retrieval()
```

### `optimize_for_inference()`

``` python
optimize_for_inference()
```

Folds the batch-normalisation layers into the linear layers before them.
Call it after loading and before a large prediction run.

### Attributes

- `model` (`ResolveModel`), `scalers` (`Scalers`), `schema`
  (`ResolveSchema`) and `device` (`str`).
- `external_vocabs` (`ExternalVocabs`): every training vocabulary, the
  categorical codes included, for the `*_with_vocabs` loaders.
- `dataset_config` (`DatasetConfig`): the loader settings the checkpoint
  was trained under (`dataset_config_from_checkpoint`).
- `categorical_vocab` (`CategoricalVocab`), `species_vocab`,
  `genus_vocab`, `family_vocab` (`list[str]`, `"<UNK>"` first).
- `retrieval` (`RetrievalConfig` or `None`) and `n_retrieval_reference`
  (`int`): the active retrieval and the size of its reference set.

## `ResolvePredictions`

What `predict_dataset` and `predict` return.

- `predictions` (`dict[str, Tensor]`): one entry per target. A
  regression target is in its original units, a `Log1p` target inverted.
  A classification target is an int64 class code, indexing the target’s
  `class_names` in `predictor.schema.targets`.
- `probabilities` (`dict[str, Tensor]`): one `(n_plots, n_classes)`
  entry per classification target, column `j` the probability of code
  `j`; each row sums to one and its largest entry is the predicted code.
  Regression targets have no entry.
- `targets` (`dict[str, Tensor]`): the recorded values, where the
  dataset carried them.
- `plot_ids` (`list[str]`): the plot identifiers, in row order.
- `latent` (`Tensor` or `None`): `(n_plots, latent_dim)` with
  `return_latent=True`.

A per-plot confidence comes from the probabilities alone: the largest
probability, the margin between the two largest, or the entropy.

``` python
probs = preds.probabilities["habitat"]
confidence = probs.max(dim=1).values
top2 = probs.topk(2, dim=1).values
margin = top2[:, 0] - top2[:, 1]

habitat = next(t for t in predictor.schema.targets if t.name == "habitat")
table = pd.DataFrame({"plot_id": preds.plot_ids,
                      "elevation": preds.predictions["elevation"].numpy()})
table["habitat"] = [habitat.class_names[c] for c in preds.predictions["habitat"]]
for j, name in enumerate(habitat.class_names):
    table[f"habitat_prob_{name}"] = probs[:, j].numpy()
```

## `RetrievalConfig`

``` python
RetrievalConfig()
```

The settings of `Predictor.set_retrieval`: `k_neighbors` (`int`, default
`96`), the reference plots each prediction reads; `alpha` (`float`,
default `0.3`), the neighbours’ share of the blend; `temperature`
(`float`, default `1.0`), of the softmax over cosine similarity.
`validate()` raises for a value out of range.

## `nearest_by_cosine()`

``` python
nearest_by_cosine(queries, reference, k, exclude_self=False)
```

The search `set_retrieval` runs, on any rows: for each row of `queries`,
the `k` rows of `reference` most similar to it by cosine similarity,
most similar first. Both are dense `(rows, dim)` tensors of one `dim`; a
numpy array goes through `torch.from_numpy`. On rows of species presence
(1 where a plot records a species) the cosine similarity is the Ochiai
index of two species lists. The search is exact and computed in blocks,
so memory stays bounded whatever the size of either set. It runs on the
device `reference` is on; `queries` may stay in host memory and is moved
over a block of rows at a time. With `exclude_self=True`, `queries` and
`reference` are the same plots in the same order (pass the same tensor
twice): plot `i` is never returned as its own neighbour, while another
plot with the same species list still is. `k` is clamped to the
reference size, one less with `exclude_self`. Returns
`(indices, similarities)`: `(n_queries, k)` int64 rows of `reference`
and their float32 similarities, on the CPU.

``` python
rows = torch.from_numpy(presence.astype("float32")).cuda()
indices, similarities = rc.nearest_by_cosine(rows, rows, 50, exclude_self=True)
```

## `Scalers`

The scaling fitted on the fitting plots, which a predictor applies to
new plots: `continuous_mean` and `continuous_scale` (one entry per
column of the continuous block), `continuous_fill` (the value a missing
cell of each column reads as before scaling; `None` for a checkpoint
written before missing cells were filled) and `target_scalers` (`dict`
of regression target to its `(mean, scale)`).

``` python
predictor.scalers.target_scalers
```

## `RunMetadata`

The run as a checkpoint records it, from `Trainer.run_metadata` or
`Trainer.load_run_metadata(path)`.

- `resolve_version` (`str`), `created_at`, `completed_at` (`str`) and
  `train_time_seconds` (`float`).
- `n_plots_train`, `n_plots_test` (`int`): the fitting and held-out
  plots.
- `hold_out` (`HoldOutMode`), `test_size` (`float`, 0 for an explicit
  fold) and `split_seed` (`int`, the `prepare_data` seed, which also
  seeds the epoch shuffle).
- `best_epoch`, `total_epochs` (`int`) and `final_metrics`
  (`dict[str, dict[str, float]]`).

``` python
record = rc.Trainer.load_run_metadata(path)
record.hold_out, record.n_plots_test
```

## `HoldOutMode`

How a run’s held-out fold was drawn: `Random` (a `test_size` share of
the plots, drawn with `split_seed`) or `Explicit` (the plots passed as
`stopping_indices`).

## `dataset_config_from_checkpoint()`

``` python
dataset_config_from_checkpoint(schema, model_config)
```

The `DatasetConfig` a checkpoint was loaded with, rebuilt from its
schema and model settings: `species_encoding`, `hash_dim` and `top_k`
from the `ModelConfig`, because they size the model, and every other
loader setting from the schema. `use_cuda_hash` is not restored.
`Predictor.dataset_config` returns the same.

``` python
rc.dataset_config_from_checkpoint(predictor.schema, predictor.model.config).species_encoding
```

## `target_specs_from_schema()`

``` python
target_specs_from_schema(schema)
```

The targets a checkpoint was trained on, as `TargetSpec`s each reading
the column of its own name: the targets a retrieval reference set is
loaded with. The `*_with_vocabs` loaders replay the class codes.
