# Python: training

Fitting a model, cross-validating it and pretraining its encoder without
labels: the model and its architecture settings, the trainer and its
settings, what a fit returns, the held-out evaluations, and the
pretrainers.

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
species recorded on them. [Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) walks
through the same steps with their output.

## `ResolveModel`

``` python
ResolveModel(schema, config=ModelConfig())
```

The network: a species encoder, a shared encoder that turns a plot’s
inputs into its *latent representation* (a vector of `latent_dim`
numbers), and one output head per target. The schema fixes the input
widths and the heads. The encoding settings that also size the model
(the species encoding, `hash_dim`, `top_k`, `top_k_species` and the
taxonomy slot count) come from the dataset’s loader settings through
`model_config_for_dataset`.

``` python
import os
import tempfile

import pandas as pd
import torch
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
roles.latitude = "lat"
roles.longitude = "lon"
targets = [rc.TargetSpec.regression("elevation"),
           rc.TargetSpec.classification("habitat", 4)]
dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets)

model_config = rc.ModelConfig()
model_config.hidden_dims = [128, 64]
model = rc.ResolveModel(dataset.schema,
                        rc.model_config_for_dataset(dataset, model_config))
model.latent_dim
```

### Forward passes

``` python
forward(continuous, genus_ids=None, family_ids=None, species_ids=None,
        species_vector=None, pool_genus_ids=None, pool_family_ids=None,
        pool_weights=None, pool_mask=None, pool_has_cover=None,
        categorical_ids=None)
```

The model’s outputs for a batch of plots, one tensor per target: a
regression target in standardised units, a classification target as
logits. `continuous` is the continuous block
(`ResolveDataset.continuous_block`), standardised; the other inputs are
the dataset tensors of the same names, and the encoding decides which
are needed. A model with categorical columns given no `categorical_ids`
reads every one as unknown. Calling the model, `model(...)`, is the same
as `forward`.

The other forward methods take the same inputs, positionally in this
order or by keyword:

- `get_latent(...)`: the latent representation, `(batch, latent_dim)`.
- `forward_with_aux(...)`: a `ModelForwardResult`, the outputs with the
  auxiliary loss terms.
- `forward_single(target, ...)`: the output of one target’s head.
- `encode_with_activations(...)`: the latent and a list with the
  activations of every hidden layer of the species encoder’s final
  stage, each `(rows, neurons)`; empty for the tabular architectures and
  TraitNet.
- `get_gate_probs(...)`: the mixture-of-experts gate probabilities,
  `(batch, n_experts)`; `None` without a mixture.

``` python
inputs = dict(genus_ids=dataset.genus_ids, family_ids=dataset.family_ids)
model.eval()
outputs = model(dataset.continuous_block(True), **inputs)
{name: out.shape for name, out in outputs.items()}
```

### Attributes

- `schema` (`ResolveSchema`) and `config` (`ModelConfig`): what the
  model was built from.
- `latent_dim` (`int`): the width of the latent representation.
- `species_encoding` (`SpeciesEncodingMode`) and `uses_explicit_vector`
  (`bool`): the species encoder, and whether it reads `species_vector`.
- `uses_moe` (`bool`) and `n_experts` (`int`): whether a mixture of
  experts routes the plots, and over how many experts.
- `aux_loss_names` (`list[str]`): the auxiliary loss terms the model
  reports: `moe` with a mixture of experts, `tabnet_sparsity` for
  TabNet.
- `row_reference_rows` (`int`): the training rows SAINT’s row attention
  keeps as the reference set inference attends over; 0 for other
  architectures.
- `requires_species_graph`, `has_species_graph` (`bool`),
  `species_graph_edge_index` and `species_graph_edge_type`: the species
  graph of a heterogeneous GNN.

### Methods

- `parameters()` (`list[Tensor]`, ready for a torch optimiser),
  `named_parameters()` (`dict`), `state_dict()` (`dict` of cloned
  parameters and buffers), `load_state_dict(state_dict, strict=True)`
  (returns the missing and the unexpected keys) and `n_parameters()`.
- `train(mode=True)`, [`eval()`](https://rdrr.io/r/base/eval.html),
  `zero_grad()`, `requires_grad_(requires_grad=True)` and `to(device)`,
  where `device` is `"cpu"`, `"cuda"` or `"cuda:N"`.
- `get_species_weights()`, `get_genus_weights()`,
  `get_family_weights()`: the embedding tables,
  `(n_species_vocab, species_embed_dim)` and the like, or `None` where
  the encoding has no such table.
- `composition_parameters()`: the species, genus and family tables the
  species encoder reads a plot’s composition through, as the parameters
  themselves.
- `set_traits(traits)`: attach the species trait matrix of the TraitNet
  architecture; the model keeps it in a buffer and saves it in the
  checkpoint.
- `set_species_graph(edge_index, edge_type)`: replace the species graph
  of a heterogeneous GNN (see `build_species_graph`).

## `ModelForwardResult`

What `forward_with_aux` returns: `outputs` (`dict[str, Tensor]`, one
entry per target) and `aux_losses` (`dict[str, Tensor]`), the auxiliary
terms the forward adds to the training loss, each already weighted:
`moe` (the mixture-of-experts load balancing, times
`moe_aux_loss_weight`) and `tabnet_sparsity` (TabNet’s mask entropy,
times `tabnet.sparsity_coefficient`). Empty for a model with neither.

## `model_config_for_dataset()`

``` python
model_config_for_dataset(dataset, config=ModelConfig())
```

`config` with the fields that size a model to its data taken from
`dataset`: `species_encoding`, `hash_dim`, `top_k` and `top_k_species`
from its loader settings, and `n_taxonomy_slots` from the width of its
genus slots. Build a model for a dataset as
`ResolveModel(dataset.schema, model_config_for_dataset(dataset, config))`.

## `ModelConfig`

``` python
ModelConfig()
```

The architecture of a model.

``` python
model_config = rc.ModelConfig()
model_config.hidden_dims = [256, 128, 64]
model_config.dropout = 0.2
model_config.activation = rc.ActivationType.SiLU
```

Species encoding:

- `species_encoding` (`SpeciesEncodingMode`, default `Hash`), `hash_dim`
  (`int`, default `32`), `top_k` (`int`, default `3`), `top_k_species`
  (`int`, default `10`): as on the dataset; `model_config_for_dataset`
  copies them.
- `n_taxonomy_slots` (`int`, default `3`): the genus and family slots
  per plot, twice `top_k` under `TopBottom`.
- `uses_explicit_vector` (`bool`, default `False`): read the sparse
  species vector, which turns `Hash` into `Sparse`; refused with
  `Embed`, `RankPool` and `Transformer`.
- `species_embed_dim` (`int`, default `32`), `genus_emb_dim` and
  `family_emb_dim` (`int`, default `8`), `categorical_embed_dim` (`int`,
  default `8`): embedding widths.

Shared encoder:

- `hidden_dims` (`list[int]`, default
  `[2048, 1024, 512, 256, 128, 64]`): the layer widths; the last is the
  latent width.
- `dropout` (`float`, default `0.3`), `activation` (`ActivationType`,
  default `GELU`), `normalization` (`NormLayerType`, default
  `BatchNorm`), `norm_groups` (`int`, default `32`, for `GroupNorm`),
  `use_residual` (`bool`, default `False`), `leaky_relu_slope` (`float`,
  default `0.01`), `elu_alpha` (`float`, default `1.0`).

Heads:

- `head_hidden_dims` (`list[int]`, default `[]`, one linear layer per
  target), `head_activation` (`ActivationType`, default `GELU`),
  `head_dropout` (`float`, default `0.0`).

Pooled and transformer encodings:

- `cover_dropout` (`float`, default `0.0`): the probability of replacing
  a plot’s pool weights with its presence mask during training.
- `freeze_composition` (`bool`, default `False`): keep the species,
  genus and family tables at their initial values. Needs the MLP
  architecture and an encoding with such a table.
- `d_model` (`int`, default `128`), `n_heads` (`int`, default `4`),
  `n_attention_layers` (`int`, default `0`, pooling alone),
  `transformer_ff_dim` (`int`, default `256`), `transformer_dropout`
  (`float`, default `0.1`), `transformer_pooling` (`TransformerPooling`,
  default `Attention`).

Mixture of experts:

- `moe_routing` (`MoERoutingType`, default `None_`): no mixture, `Soft`
  or `TopK` routing.
- `moe_placement` (`MoEPlacement`, default `Tail`), `n_experts` (`int`,
  default `4`, at least 2), `expert_hidden_dims` (`list[int]`, default
  `[256, 128]`), `moe_top_k` (`int`, default `2`), `moe_noise_std`
  (`float`, default `0.1`, routing noise during training),
  `moe_aux_loss_weight` (`float`, default `0.01`, the weight of the
  load-balancing loss).

Architecture:

- `encoder_architecture` (`EncoderArchitecture`, default `MLP`):
  replaces the shared MLP; each architecture reads its settings from its
  own sub-configuration: `ft_transformer`, `tabnet`, `saint`,
  `excelformer`, `trait_net`, `gnn`, `heterogeneous_gnn`.
- `parallel_layers` (`ParallelLayersConfig`) and `tabm` (`TabMConfig`):
  alternative final stages of the MLP encoder.

The final stage of the MLP encoder is one of four: the plain MLP, a
tail-placed mixture of experts, TabM, or a block of parallel branches.
At most one of the last three may be enabled;
`moe_placement = MoEPlacement.Post` moves the mixture after the encoder,
where it combines with either of the others.

Every species encoding reaches the tabular architectures as numeric
features beside the continuous block, and the genus and family slots of
`Hash`, `Embed` and `Sparse` as categorical inputs. `HeterogeneousGNN`
and the `CoOccurrence` GNN graph read the species vector and need
`Sparse`; the `Taxonomic` GNN graph reads the genus and family slots and
does not take `RankPool` or `Transformer`. Building the model refuses an
unsupported pair with a message naming what would work.

## `FTTransformerConfig`

The settings of `EncoderArchitecture.FTTransformer`,
`ModelConfig.ft_transformer`: a transformer over one token per input
feature. `d_model` (`192`), `n_heads` (`8`), `n_layers` (`3`),
`attention_dropout` (`0.2`, on the attention weights), `ffn_dropout`
(`0.1`, inside the feed-forward layer and on each sublayer’s residual
branch), `ffn_multiplier` (`4`) and `pre_norm` (`True`).

## `TabNetConfig`

The settings of `EncoderArchitecture.TabNet`, `ModelConfig.tabnet`:
`n_steps` (`3`), `n_d` and `n_a` (`64`), `relaxation_factor` (`1.5`),
`sparsity_coefficient` (`0.001`), `virtual_batch_size` (`128`) and
`use_sparsemax` (`True`). `use_sparsemax = False` selects 1.5-entmax
(Peters, Niculae and Martins 2019), which keeps more features per step
than sparsemax. `virtual_batch_size` is the slice ghost batch
normalisation runs at; 0 normalises the batch in one piece. TabNet adds
`sparsity_coefficient` times the entropy of its feature masks to the
training loss, recorded as `tabnet_sparsity`.

## `SAINTConfig`

The settings of `EncoderArchitecture.SAINT`, `ModelConfig.saint`:
`d_model` (`128`), `n_heads` (`8`), `n_layers` (`6`),
`attention_dropout` (`0.1`), `use_row_attention` (`True`) and
`row_attention_rows` (`256`). SAINT attends across rows as well as
features. In training a batch is cut into groups of `row_attention_rows`
rows; at inference each plot attends over a fixed reference set of
`row_attention_rows - 1` training rows plus itself, saved with the
weights, so a plot’s prediction does not depend on which plots share its
batch.

## `ExcelFormerConfig`

The settings of `EncoderArchitecture.ExcelFormer`,
`ModelConfig.excelformer`: `d_model` (`192`), `n_heads` (`8`),
`n_layers` (`3`), `attention_dropout` (`0.2`), `ffn_multiplier` (`4`),
`importance_threshold` (`0.5`) and `pre_norm` (`True`).

## `GNNConfig`

The settings of `EncoderArchitecture.GNN`, `ModelConfig.gnn`: a graph
neural network over a k-nearest-neighbour graph of the plots in a
forward pass. `gnn_type` (`GNNType`, `GAT`), `n_layers` (`3`),
`hidden_dim` (`256`), `n_heads` (`4`), `k_neighbors` (`10`),
`graph_mode` (`GraphConstructionMode`, `Spatial`), `edge_dropout`
(`0.1`) and `use_edge_features` (`False`). `graph_mode` is what
neighbourhood is measured on: the coordinates, the genus and family
composition, or the species vector. A spatial graph trains on the whole
training set in one batch, so the graph spans every plot.
`use_edge_features` weights each edge by its similarity instead of 1.

## `HeterogeneousGNNConfig`

The settings of `EncoderArchitecture.HeterogeneousGNN`,
`ModelConfig.heterogeneous_gnn`: message passing on a graph over the
species vocabulary (`build_species_graph`). `hidden_dim` (`128`),
`output_dim` (`64`), `n_layers` (`3`), `n_edge_types` (`3`), `n_heads`
(`4`), `dropout` (`0.1`), `k_cooccurrence` (`20`),
`cooccurrence_threshold` (`0.01`), `use_taxonomic_edges` (`True`) and
`use_cooccurrence_edges` (`True`). `Trainer.prepare_data` builds the
graph from the dataset and the checkpoint carries it.

## `TraitNetConfig`

The settings of `EncoderArchitecture.TraitNet`, `ModelConfig.trait_net`:
`env_dim` (`128`), `trait_dim` (`64`), `interaction_dim` (`256`),
`interaction` (`TraitInteractionMode`, `Bilinear`) and
`shared_trait_encoder` (`True`). `interaction` is how the environment
and a species’ traits combine and `interaction_dim` the width of the
result; `shared_trait_encoder = False` gives every species its own trait
encoder. The trait matrix comes in through `ResolveModel.set_traits`.

## `TabMConfig`

TabM, an ensemble of MLPs sharing most of their weights, as the
encoder’s final stage, `ModelConfig.tabm`: `enabled` (`False`),
`n_ensembles` (`16`) and `aggregation` (`TabMAggregation`, `Mean`).

## `ParallelLayersConfig`

A block of parallel branches as the encoder’s final stage,
`ModelConfig.parallel_layers`: `enabled` (`False`), `branches`
(`list[ParallelBranchConfig]`), `aggregation` (`ParallelAggregation`,
`Concat`), `attention_heads` (`4`) and `use_residual` (`True`). Enabled,
the block replaces the plain MLP, so the branches carry the encoder’s
final capacity.

## `ParallelBranchConfig`

One branch of a parallel block: `hidden_dims` (`[]`), `activation`
(`GELU`), `normalization` (`BatchNorm`), `dropout` (`0.3`) and
`branch_weight` (`1.0`), the factor on the branch’s output in the
aggregation; 0 takes it out of the sum and keeps its parameters.

``` python
branch = rc.ParallelBranchConfig()
branch.hidden_dims = [64, 32]
parallel = rc.ModelConfig()
parallel.parallel_layers.enabled = True
parallel.parallel_layers.branches = [branch, branch]
```

## `Trainer`

``` python
Trainer(model, config)
```

Fits a model: it holds the split into fitting and held-out plots, the
optimiser, the training loop and the evaluations on the held-out plots.

``` python
config = rc.TrainConfig()
config.max_epochs = 100
trainer = rc.Trainer(model, config)
trainer.prepare_data(dataset, test_size=0.2, seed=42)
result = trainer.fit()
result.final_metrics["elevation"]["mae"]
```

### `prepare_data()`

``` python
prepare_data(dataset, test_size=0.2, seed=42)
prepare_data(dataset, stopping_indices, seed=42)
```

Splits the plots, fits the input and target scalers on the fitting
plots, and records the categorical codes. The *held-out fold* is the set
of plots training does not fit on: early stopping, the best epoch, the
final metrics and the evaluations read them. The first form holds out a
random `test_size` share of the plots, drawn with `seed`; `test_size=0`
holds out nothing and needs `fixed_epochs`. The second holds out the
plots at the positions `stopping_indices` (from
`ResolveDataset.plot_indices`), such as a spatial block; `seed` then
orders the fitting plots and seeds the epoch shuffle.

Each call replaces everything an earlier one set. `prepare_data` raises
when the dataset and model hash widths disagree, when the split leaves
no plot to fit, and, for an explicit fold, for an empty fold, a position
out of range or listed twice, a fold of every plot, and a fold with no
labelled plot for a target whose `missing` policy is `Mask`. For a
heterogeneous GNN it also builds the species graph.

`seed` fixes the split. The starting weights come from torch’s global
random number generator when the model is built, so seed that generator
before building the model to repeat a fit.

``` python
fold = dataset.plot_indices(["A001", "A002", "A003", "A004", "A005"])
trainer.prepare_data(dataset, stopping_indices=fold, seed=42)
trainer.test_plot_ids()
```

`prepare_data_raw(*, targets, coordinates=None, covariates=None, ..., categorical_ids=None, test_size=0.2, seed=42)`
takes the inputs as tensors by keyword instead: the parts of the
continuous block and the other model inputs as in `Predictor.predict`,
and `targets`, a dict of target name to `(n_plots,)` tensor.

### `fit()`

``` python
fit()
```

Trains and returns a `TrainResult`. The run stops at `max_epochs`, or
earlier once `patience` epochs pass without improvement on the held-out
fold, and keeps the weights of the best epoch. With `fixed_epochs` it
runs that many epochs of the `max_epochs` schedule and keeps the final
weights. On a CUDA out-of-memory error it halves `batch_size` and
restarts, down to `batch_size_floor`; `effective_batch_size` is the
batch size it trained at. The held-out fold is scored in chunks; an
out-of-memory error there halves the chunk.

### `resume()`

``` python
resume(checkpoint_path)
```

Continues a run from a checkpoint `fit` wrote while training, `best.pt`
or `checkpoint_<N>.pt` under `TrainConfig.checkpoint_dir`, at the epoch
after it, with its optimiser, early-stopping, mixed-precision and
random-number state, and returns the result of the whole run. The
trainer needs the run’s model architecture and settings
(`Trainer.load_train_config`) and must be prepared on the run’s data
with its seed. Raises `ValueError` for a checkpoint without run progress
or a split that differs from the run’s.

``` python
run_dir = tempfile.mkdtemp()
config = rc.TrainConfig()
config.max_epochs = 60
config.fixed_epochs = 60
config.checkpoint_dir = run_dir
config.checkpoint_every = 20
run = rc.Trainer(rc.ResolveModel(dataset.schema, model.config), config)
run.prepare_data(dataset, seed=42)
run.fit()

again = rc.Trainer(rc.ResolveModel(dataset.schema, model.config),
                   rc.Trainer.load_train_config(
                       os.path.join(run_dir, "checkpoint_40.pt")))
again.prepare_data(dataset, seed=42)
resumed = again.resume(os.path.join(run_dir, "checkpoint_40.pt"))
resumed.resumed_from_epoch
```

### Evaluations of the held-out fold

- `compute_residuals(target_name)`: a `ResidualAnalysis` of a regression
  target.
- `compute_classification_predictions(target_name)`: a
  `ClassificationPredictions` of a classification target.
- `compute_calibration(target_name, n_bins=10)`: a `CalibrationResult`
  of a classification target; no bins for a regression target.
- `compute_diagnostics()`: a `NetworkDiagnostics` of the species
  encoder’s final stage, under every species encoding; not for the
  tabular architectures or TraitNet.
- `test_plot_ids()`, `train_plot_ids()`: the plot identifiers of the
  held-out and the fitting plots, after `prepare_data(dataset)`;
  `test_indices()`, `train_indices()`: their positions in the dataset,
  as int64 tensors.

``` python
trainer.prepare_data(dataset, test_size=0.2, seed=42)
result = trainer.fit()
residuals = trainer.compute_residuals("elevation")
classes = trainer.compute_classification_predictions("habitat")
calibration = trainer.compute_calibration("habitat", n_bins=5)
calibration.expected_calibration_error
```

### `cross_validate()`

``` python
cross_validate(n_folds=5, seed=42)
cross_validate_spatial(spatial_config, n_folds=5, seed=42)
```

*Cross-validation* fits the model once per fold, each time holding out
one of `n_folds` parts of the plots, and scores each fit on the part it
held out. `cross_validate` draws the folds at random with `seed`;
`cross_validate_spatial` cuts the plots into blocks of latitude and
longitude (`SpatialBlockConfig`) and assigns whole blocks to folds, so a
fold’s plots lie apart from the plots it was fitted on. Both return a
`CrossValidationResult`, start each fold from the model’s initial
weights, and restore the trainer’s own split when they finish. Spatial
blocks need coordinates, and fewer blocks than folds is an error.

``` python
folds = trainer.cross_validate(n_folds=3, seed=1)
blocks = rc.SpatialBlockConfig()
blocks.lat_size = 0.5
blocks.lon_size = 0.5
spatial = trainer.cross_validate_spatial(blocks, n_folds=3, seed=1)
spatial.mean_metrics["elevation"]["mae"]
```

### `predict()`

``` python
predict(continuous, genus_ids=None, family_ids=None, ...)
```

The model’s outputs in evaluation mode, from the unstandardised
continuous block, standardised here with the fitted scalers, and the
other inputs of `ResolveModel.forward`: a regression target in its
original units, a classification target as logits. For a whole dataset,
save a checkpoint and use `Predictor`.

### Checkpoints

``` python
save(path, metadata=None)
load_state(path, device="cpu", vram_fraction=1.0)
Trainer.load(path, device="cpu", vram_fraction=1.0)
Trainer.load_train_config(path)
Trainer.load_run_metadata(path)
```

`save` writes the model, the scalers and the vocabularies to a
checkpoint, with the run’s `RunMetadata` when `metadata` is given
(`trainer.run_metadata`). `load_state` loads a checkpoint’s weights,
scalers and categorical codes into this trainer, which then evaluates as
the original run did when it is built and prepared the same way; a
missing parameter or buffer raises. The static `load` returns the model,
its `Scalers` and its `CategoricalVocab`. `load_train_config` returns
the `TrainConfig` the checkpoint records; settings it does not record
(device, checkpoint directory, mixed-precision and cuDNN flags) keep
their defaults. `load_run_metadata` returns its `RunMetadata`. A file
that cannot be read or written raises `StorageError`.

``` python
path = os.path.join(tempfile.mkdtemp(), "model.pt")
trainer.save(path, metadata=trainer.run_metadata)
rc.Trainer.load_run_metadata(path).n_plots_train
```

### Attributes

- `model` (`ResolveModel`) and `config` (`TrainConfig`).
- `scalers` (`Scalers`): the scaling fitted on the fitting plots.
- `categorical_vocab` (`CategoricalVocab`): the categorical codes
  recorded by `prepare_data`.
- `run_metadata` (`RunMetadata`): the split `prepare_data` made and,
  after `fit`, the run’s timing, epochs and final metrics.
- `effective_batch_size` (`int`): the batch size the last `fit` trained
  at; 0 before the first.

## `TrainConfig`

``` python
TrainConfig()
```

The settings of a training run.

Run length and selection:

- `max_epochs` (`int`, default `500`), `patience` (`int`, default `50`):
  the epoch limit, and the epochs without improvement that stop the run.
- `fixed_epochs` (`int`, default `0`): run this many epochs of the
  `max_epochs` schedule and keep the final weights; 0 stops early.
- `selection_metric` (`SelectionMetric`, default `Loss`): the held-out
  value early stopping, the best epoch and `ReduceOnPlateau` follow.
- `batch_size` (`int`, default `4096`), `batch_size_floor` (`int`,
  default `1024`): the plots per optimiser step, and the smallest batch
  the out-of-memory retry drops to.

Optimiser and learning rate:

- `lr` (`float`, default `0.001`), `weight_decay` (`float`, default
  `0.0001`): the AdamW learning rate and weight decay.
- `lr_scheduler` (`LRSchedulerType`, default `None_`): how the rate
  changes over the run.
- `lr_step_size` (`int`, default `100`), `lr_gamma` (`float`, default
  `0.1`): the period and factor of `StepLR`.
- `lr_min` (`float`, default `1e-6`): the end point of `CosineAnnealing`
  and `LinearDecay`, and the floor of `ReduceOnPlateau`.
- `lr_plateau_factor` (`float`, default `0.1`), `lr_plateau_patience`
  (`int`, default `10`), `lr_plateau_threshold` (`float`, default
  `0.0`), `lr_plateau_cooldown` (`int`, default `0`): `ReduceOnPlateau`,
  PyTorch’s `ReduceLROnPlateau` on the held-out value `selection_metric`
  names. With threshold 0 it counts improvement by early stopping’s
  rule.
- `warmup_steps` (`int`, default `0`) or `warmup_epochs` (`int`, default
  `0`), not both: a linear warmup over that many optimiser steps or
  epochs, which scales every schedule.

Loss:

- `loss_config` (`LossConfigMode`, default `Combined`): the regression
  loss.
- `phase_boundaries` (`(int, int)`, default `(100, 300)`): the epochs at
  which the `Combined` regression loss changes phase: mean absolute
  error, then with the symmetric mean absolute percentage error added,
  then with a penalty outside the tolerance band added.
- `band_threshold` (`float`, default `0.25`): the tolerance band of that
  penalty. `band_thresholds` (`list[float]`, default
  `[0.1, 0.25, 0.5]`): the relative bands with a reported accuracy
  (`band_10`, `band_25`, `band_50`).
- `nca_temperature` (`float`, default `0.1`), `nca_neighbors` (`int`,
  default `32`), `nca_weight` (`float`, default `0.1`): the
  neighbourhood components analysis term of `LossConfigMode.NCA`.

Device and precision:

- `device` (`str`, default `"cpu"`): `"cpu"`, `"cuda"` or `"cuda:N"`.
- `use_amp` (`bool`, default `False`): mixed precision on CUDA, with
  `amp_init_scale` (`65536.0`), `amp_growth_factor` (`2.0`),
  `amp_backoff_factor` (`0.5`) and `amp_growth_interval` (`2000`) for
  the gradient scale.
- `cudnn_benchmark` (`bool`, default `True`), `allow_tf32` (`bool`,
  default `True`), `vram_fraction` (`float`, default `1.0`, the share of
  GPU memory the run may use).

Checkpoints:

- `checkpoint_dir` (`str`, default `""`): the directory for `best.pt`,
  `checkpoint_<N>.pt`, `progress.json` and the final `checkpoint.pt`;
  empty writes none.
- `checkpoint_every` (`int`, default `0`): write `checkpoint_<N>.pt`
  every N epochs; 0 writes only `best.pt`.

## `TrainResult`

What `fit` and `resume` return.

Attributes:

- `final_metrics` (`dict[str, dict[str, float]]`): per target, the
  metrics on the held-out fold. A regression target reports `mae`,
  `rmse`, `r2`, `smape` (range 0 to 2) and `band_10`, `band_25`,
  `band_50` (the share of plots within that relative band), in original
  units. A classification target reports `accuracy`, `macro_f1`,
  `weighted_f1` and `precision_<c>`, `recall_<c>`, `f1_<c>` per class. A
  target with `missing = Mask` also reports `n_labelled`.
- `best_epoch` (`int`): the epoch with the best held-out value of
  `selection_metric`.
- `train_loss_history`, `test_loss_history` (`list[float]`): one entry
  per epoch; the second is empty when nothing is held out.
- `train_loss_history_by_target`, `test_loss_history_by_target`
  (`dict[str, list[float]]`): those losses split into each target’s
  weighted term.
- `train_aux_loss_history` (`dict[str, list[float]]`): the auxiliary
  terms of the training loss, `moe` and `tabnet_sparsity`.
- `lr_history` (`list[float]`): the learning rate of each epoch’s last
  step.
- `selection_history` (`list[float]`): the held-out value of
  `selection_metric` at each epoch.
- `baselines` (`dict[str, BaselineMetrics]`): a naive baseline per
  target.
- `diagnostics` (`NetworkDiagnostics`): the network’s state at the end
  of the fit.
- `train_time_seconds` (`float`), `effective_batch_size` (`int`),
  `resumed_from_epoch` (`int`, 0 for a run trained from the start).

## `BaselineMetrics`

A target’s naive baseline beside the model: `baseline_mae`,
`baseline_mse` (predicting the training mean), `model_mae`, `model_mse`,
`skill_score` (`1 - model_mse / baseline_mse`), `r_squared`,
`training_mean`; for a classification target `baseline_accuracy`
(predicting the most frequent training class), `model_accuracy`,
`accuracy_lift` and `training_mode`. Regression values are in original
units and match `final_metrics`.

## `CrossValidationResult`

What `cross_validate` and `cross_validate_spatial` return: `n_folds`,
`mean_metrics` and `std_metrics` (the mean and standard deviation of
each metric across folds, by target), `fold_results` (one `TrainResult`
per fold) and `total_time_seconds`.

## `SpatialBlockConfig`

The blocks of `cross_validate_spatial`: `lat_size` and `lon_size`
(`float`, default `1.0`), the block height and width in degrees, and
`balance` (`bool`, default `False`), which assigns blocks to folds by
greedy bin-packing to even out the fold sizes instead of in turn.

## `ResidualAnalysis`

What `compute_residuals` returns, in original units, over the held-out
plots labelled for the target: `target_name`, `predictions`, `actuals`,
`residuals`, `test_rows` (each row’s position in `test_plot_ids()`),
`mean_residual`, `std_residual`, `skewness`, `kurtosis` and the residual
quantiles `q05`, `q25`, `q50`, `q75`, `q95`.

## `ClassificationPredictions`

What `compute_classification_predictions` returns, over the held-out
plots labelled for the target: `target_name`, `predicted_classes` and
`actuals` (int64 codes), `probabilities` (`(n, n_classes)`), `test_rows`
(each row’s position in `test_plot_ids()`) and `class_names` (the labels
in code order).

## `CalibrationResult`

What `compute_calibration` returns: `target_name`, `class_idx`, `bins`
(`list[CalibrationBin]`), `expected_calibration_error` and
`max_calibration_error`. A classifier is *calibrated* when the plots it
gives a probability p are right a share p of the time.

## `CalibrationBin`

One bin of predicted probability: `bin_start`, `bin_end`, `count` (the
plots in it), `mean_predicted_prob` and `actual_frequency` (the share of
them predicted correctly).

## `NetworkDiagnostics`

The activations of the species encoder’s hidden layers over the held-out
fold: `layers` (`list[LayerDiagnostics]`), `total_neurons`,
`total_dead`, `total_saturated`, `overall_dead_fraction`,
`overall_saturated_fraction`, `has_issues` and `summary`, a sentence
describing them.

## `LayerDiagnostics`

One hidden layer: `name`, `n_neurons`, `n_dead` and `dead_fraction`
(neurons that never activate), `n_saturated` and `saturated_fraction`,
`mean_activation`, `std_activation` and `sparsity`.

## `PretrainConfig`

The settings of `JEPAPretrainer` and `SCARFPretrainer`. *Pretraining*
trains the encoder on a task built from the inputs alone, with no
target, before the model is fitted.

- Shared: `pretrain_epochs` (`100`), `pretrain_lr` (`1e-4`),
  `pretrain_weight_decay` (`1e-4`), `batch_size` (`4096`),
  `batch_size_floor` (`1024`), `warmup_steps` (`0`), `device` (`"cpu"`),
  `vram_fraction` (`1.0`) and `seed` (`42`), which seeds every draw the
  pretraining makes.
- Masking, for JEPA: `mask_ratio` (`0.3`, the share of features masked)
  and `mask_strategy` (`MaskStrategy`, `Random`).
- JEPA: `ema_decay` (`0.996`) and `ema_decay_end` (`1.0`), the decay of
  the slowly updated target encoder; `predictor_hidden_dim` (`256`),
  `predictor_n_layers` (`2`) and `predictor_dropout` (`0.1`).
- SCARF: `corruption_rate` (`0.6`, the share of features replaced),
  `temperature` (`0.1`), `projection_dim` (`128`) and `mixup_alpha`
  (`0.0`), the strength of a mixup of the corrupted view with another
  plot of the batch; 0 switches it off.

## `JEPAPretrainer`

``` python
JEPAPretrainer(model, config=PretrainConfig())
pretrain(continuous, genus_ids=None, family_ids=None, ...)
```

Pretrains a model’s encoder in place by joint-embedding prediction
(T-JEPA): the encoder reads a masked view of each plot and learns to
predict the latent a slowly updated copy of itself gives the whole plot.
`pretrain` takes the model inputs of `ResolveModel.forward`,
`continuous` unstandardised and required, and returns a
`PretrainResult`.

## `SCARFPretrainer`

``` python
SCARFPretrainer(model, config=PretrainConfig())
pretrain(continuous, genus_ids=None, family_ids=None, ...)
```

Pretrains a model’s encoder in place by contrastive learning (SCARF):
each plot and a corrupted copy of it, with a share of its inputs
replaced by values from other plots, form the pair the encoder learns to
match. The inputs are those of `JEPAPretrainer.pretrain`.

``` python
pretrain_config = rc.PretrainConfig()
pretrain_config.pretrain_epochs = 5
fresh = rc.ResolveModel(dataset.schema, model.config)
scarf = rc.SCARFPretrainer(fresh, pretrain_config)
pretrained = scarf.pretrain(dataset.continuous_block(True), **inputs)
pretrained.loss_history
```

## `PretrainResult`

What the JEPA, SCARF and masked-species pretrainers return:
`loss_history` (one entry per epoch), `epochs_completed`,
`total_time_seconds` and `effective_batch_size`.

## `MaskStrategy`

Which features JEPA masks: `Random` (each feature on its own), `Block`
(one contiguous range per plot) or `Structured` (whole contiguous groups
of features).

## `MLMPretrainConfig`

The settings of `MaskedSpeciesPretrainer`: `mask_prob` (`0.15`, the
share of a plot’s species masked), `pretrain_epochs` (`50`),
`pretrain_lr` (`1e-4`), `pretrain_weight_decay` (`1e-4`), `batch_size`
(`4096`), `batch_size_floor` (`1024`), `warmup_steps` (`0`), `device`
(`"cpu"`), `vram_fraction` (`1.0`) and `seed` (`42`).

## `MaskedSpeciesPretrainer`

``` python
MaskedSpeciesPretrainer(model, config=MLMPretrainConfig())
pretrain(*, species_ids, pool_mask, pool_genus_ids=None, pool_family_ids=None,
         pool_weights=None, ...)
```

Pretrains the species encoder of a `Transformer`-encoding model in place
by masked-species prediction: a share of each plot’s species is hidden
and the encoder learns to name them from the rest. `pretrain` takes the
model inputs by keyword, `species_ids` and `pool_mask` required, and
returns a `PretrainResult`.

``` python
transformer = rc.DatasetConfig()
transformer.species_encoding = rc.SpeciesEncodingMode.Transformer
tokens = rc.ResolveDataset.from_pandas(plots, species, roles, targets, transformer)
token_model = rc.ResolveModel(
    tokens.schema, rc.model_config_for_dataset(tokens, rc.ModelConfig()))
masked = rc.MLMPretrainConfig()
masked.pretrain_epochs = 5
rc.MaskedSpeciesPretrainer(token_model, masked).pretrain(
    species_ids=tokens.species_ids, pool_mask=tokens.pool_mask,
    pool_genus_ids=tokens.pool_genus_ids,
    pool_family_ids=tokens.pool_family_ids,
    pool_weights=tokens.pool_weights)
```

## `VAEConfig`

The settings of `VAEPretrainer`: `latent_dim` (`64`), `encoder_dims`
(`[512, 256, 128]`), `decoder_dims` (`[128, 256, 512]`), `dropout`
(`0.1`), `kl_weight` (`1.0`, the weight of the divergence term),
`kl_anneal_epochs` (`20`, the epochs over which that weight rises from
0), `pretrain_epochs` (`100`), `pretrain_lr` (`0.001`),
`pretrain_weight_decay` (`0.01`), `batch_size` (`4096`),
`batch_size_floor` (`1024`), `warmup_steps` (`0`), `device` (`"cpu"`),
`vram_fraction` (`1.0`) and `seed` (`42`).

## `VAEPretrainer`

``` python
VAEPretrainer(n_species, config=VAEConfig())
pretrain(species_vector)
```

A variational autoencoder, a network that compresses each plot’s species
vector to `latent_dim` numbers and reconstructs it, trained on the plots
alone. `n_species` is the vocabulary size
(`ResolveSchema.n_species_vocab`); `pretrain` reads the sparse
encoding’s `species_vector` and returns a `VAEPretrainResult`.
`get_latent_dim()` returns the latent width and
`get_projection_weights()` the encoder’s first layer, one row per unit
of its first hidden layer and one column per species.

``` python
sparse = rc.DatasetConfig()
sparse.species_encoding = rc.SpeciesEncodingMode.Sparse
vectors = rc.ResolveDataset.from_pandas(plots, species, roles, targets, sparse)
vae_config = rc.VAEConfig()
vae_config.latent_dim = 8
vae_config.pretrain_epochs = 20
vae = rc.VAEPretrainer(vectors.schema.n_species_vocab, vae_config)
vae.pretrain(vectors.species_vector).loss_history[-1]
```

## `VAEPretrainResult`

What `VAEPretrainer.pretrain` returns: `loss_history`,
`recon_loss_history` and `kl_loss_history` (one entry per epoch),
`epochs_completed`, `total_time_seconds` and `effective_batch_size`.

## `LossConfigMode`

The regression loss: `MAE` (mean absolute error alone), `SMAPE`
(symmetric mean absolute percentage error), `Combined` (the three phases
of `phase_boundaries`, the default) or `NCA` (`Combined`, with a
neighbourhood components analysis term added for every classification
target).

## `LRSchedulerType`

How the learning rate changes over a run: `None_` (constant), `StepLR`
(times `lr_gamma` every `lr_step_size` epochs), `CosineAnnealing` (down
to `lr_min`), `ReduceOnPlateau` (times `lr_plateau_factor` when the
held-out value stalls) or `LinearDecay` (down to `lr_min` in a straight
line, step by step).

## `SelectionMetric`

The held-out value a run selects by: `Loss` (the lowest wins) or
`Accuracy` (the highest held-out accuracy of the one classification
target; refused without a held-out fold or without exactly one
classification target).

## `ActivationType`

`ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Tanh`, `Mish`, `ELU`, `SELU`,
`Softplus` or `PReLU`.

## `NormLayerType`

`BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNorm` or `None_`.

## `EncoderArchitecture`

The shared encoder: `MLP`, `FTTransformer`, `TabNet`, `SAINT`,
`TraitNet`, `GNN`, `ExcelFormer` or `HeterogeneousGNN`.

## `MoERoutingType`

How a mixture of experts routes a plot: `None_` (no mixture), `Soft`
(every expert, weighted) or `TopK` (the `moe_top_k` highest-weighted
experts).

## `MoEPlacement`

Where the mixture sits. Under `Tail` the experts are the encoder’s final
stage: `hidden_dims` without its last two widths is the backbone, and
the mixture maps it to `hidden_dims[-1]`, the latent. It is available to
every species encoding of the MLP encoder. Under `Post` the encoder
makes its latent as usual and the mixture maps it to one of the same
width; any architecture takes it.

## `GNNType`

The graph layer of `EncoderArchitecture.GNN`: `GCN`, `GAT` or
`GraphSAGE`.

## `GraphConstructionMode`

What the plot graph of `EncoderArchitecture.GNN` measures neighbourhood
on: `Spatial` (the coordinates), `Taxonomic` (the genus and family
slots) or `CoOccurrence` (the species vector).

## `TraitInteractionMode`

How TraitNet combines the environment and a species’ traits: `Bilinear`,
`MLP` or `Attention`.

## `TabMAggregation`

How TabM combines its ensemble: `Mean` or `Median`.

## `TransformerPooling`

How the transformer encoding pools its species tokens into one vector:
`Attention` or `Cls` (a learned summary token, which needs at least one
attention layer).

## `ParallelAggregation`

How a parallel block combines its branches: `Concat`, `Sum`, `Mean`,
`Attention` or `Gated`.
