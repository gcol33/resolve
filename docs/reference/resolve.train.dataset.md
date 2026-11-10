# Train a Model on a Dataset

Build a model for a dataset loaded with
[`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md),
hold out a fold of its plots for early stopping, and train.

## Usage

``` r
resolve.train.dataset(
  dataset,
  hiddenDims = NULL,
  maxEpochs = NULL,
  patience = NULL,
  fixedEpochs = NULL,
  selectionMetric = NULL,
  lr = NULL,
  lrScheduler = NULL,
  lrStepSize = NULL,
  lrGamma = NULL,
  lrMin = NULL,
  lrPlateauFactor = NULL,
  lrPlateauPatience = NULL,
  lrPlateauThreshold = NULL,
  lrPlateauCooldown = NULL,
  warmupSteps = NULL,
  warmupEpochs = NULL,
  batchSize = NULL,
  device = NULL,
  testSize = 0.2,
  stoppingPlots = NULL,
  seed = 42L,
  savePath = NULL,
  checkpointDir = NULL,
  checkpointEvery = NULL,
  resumeFrom = NULL,
  lossConfig = NULL,
  ncaTemperature = NULL,
  ncaNeighbors = NULL,
  ncaWeight = NULL,
  moeRouting = NULL,
  moePlacement = NULL,
  nExperts = NULL,
  expertHiddenDims = NULL,
  moeTopK = NULL,
  moeNoiseStd = NULL,
  moeAuxLossWeight = NULL,
  coverDropout = NULL,
  plotTotals = NULL,
  plotTotalsDropout = NULL,
  freezeComposition = NULL,
  freezeEncoder = NULL,
  dModel = NULL,
  nHeads = NULL,
  nAttentionLayers = NULL,
  transformerFfDim = NULL,
  transformerPooling = NULL,
  transformerDropout = NULL,
  modelConfig = list(),
  trainConfig = list(),
  pretrain = NULL,
  initFrom = NULL,
  initReuse = NULL,
  verbose = TRUE
)
```

## Arguments

- dataset:

  A ResolveDataset object from resolve.dataset.csv()

- hiddenDims:

  Hidden layer widths of the encoder MLP.

- maxEpochs:

  Maximum training epochs.

- patience:

  Early stopping patience in epochs.

- fixedEpochs:

  Train exactly this many epochs of the `maxEpochs` schedule, with no
  early stopping, and keep the final weights (0, the default, is early
  stopping). The learning-rate schedule is still laid out over
  `maxEpochs`. The only mode that accepts `testSize = 0`, for a refit on
  every plot.

- selectionMetric:

  The held-out value early stopping, `best_epoch` (the returned weights)
  and a `"plateau"` schedule follow: `"loss"` (the default, the lowest
  selection loss wins) or `"accuracy"` (the highest held-out accuracy of
  the dataset's one classification target wins, and a plateau follows
  PyTorch's `ReduceLROnPlateau` in mode `"max"`). `"accuracy"` needs a
  held-out fold and exactly one classification target.

- lr:

  Learning rate.

- lrScheduler:

  Learning-rate schedule: `"none"` (constant), `"step"` multiplies the
  rate by `lrGamma` every `lrStepSize` epochs, `"cosine"` anneals it
  from `lr` to `lrMin` over `maxEpochs`, `"linear"` lowers it linearly
  at every optimizer step, from `lr` at the end of the warmup to `lrMin`
  at the last step of `maxEpochs`, and `"plateau"` multiplies it by
  `lrPlateauFactor` when the held-out value that drives early stopping
  has not improved for more than `lrPlateauPatience` epochs (PyTorch's
  `ReduceLROnPlateau`, in mode `"min"` on the loss and `"max"` on the
  accuracy, see `selectionMetric`). `"plateau"` needs a held-out fold,
  so it is refused with `testSize = 0`.

- lrStepSize:

  Epochs between decays under `lrScheduler = "step"`.

- lrGamma:

  Factor of each decay under `lrScheduler = "step"`.

- lrMin:

  The rate the cosine and linear schedules end at, and the floor a
  plateau never decays below.

- lrPlateauFactor:

  Factor a plateau multiplies the rate by, in (0, 1).

- lrPlateauPatience:

  Epochs without improvement of the held-out value (`selectionMetric`) a
  plateau allows; the next one decays the rate. Like `patience`, an
  epoch counts only once the loss objective has reached its final phase.

- lrPlateauThreshold:

  Relative improvement on the best held-out value an epoch must make to
  count as one, in \[0, 1) (the default 0 is any improvement, the rule
  early stopping and `best_epoch` use; PyTorch's default is 1e-4).

- lrPlateauCooldown:

  Epochs after a decay during which no epoch counts against the rate.

- warmupSteps:

  Optimizer steps over which the rate climbs linearly, from
  `lr / warmupSteps` at the first step to the schedule's rate (0, the
  default, is no warmup). The schedule carries on unchanged after it. An
  epoch that begins inside the warmup counts against neither `patience`
  nor a `"plateau"` schedule; its held-out value can still be the best.

- warmupEpochs:

  The warmup in epochs instead of steps (each epoch the
  `ceiling(nTrain / batchSize)` steps it takes). Give `warmupSteps` or
  `warmupEpochs`, not both.

- batchSize:

  Batch size.

- device:

  Device to train on: `"cpu"`, `"cuda"` (the current CUDA device) or
  `"cuda:N"`.

- testSize:

  Fraction of the data held out for early stopping (default 0.2), drawn
  at random with `seed`. 0 holds nothing out and needs `fixedEpochs`.

- stoppingPlots:

  Plot IDs to hold out for early stopping instead of a random `testSize`
  fraction (default `NULL`): a withheld spatial block or database, or
  the fold a published protocol selects its epoch on. Early stopping,
  `best_epoch`, `final_metrics` and the baselines read exactly these
  plots; every other plot trains. Each ID must be a plot of `dataset`,
  listed once, and at least one plot must be left to train. With a
  masked target, the fold needs at least one plot labelled for it. Not
  combined with `testSize`.

- seed:

  Random seed of the split and the epoch shuffle (default 42).

- savePath:

  Path to save model checkpoint (optional)

- checkpointDir:

  Directory the run writes checkpoints to while it trains (optional):
  `best.pt` at every new best epoch, `checkpoint_<N>.pt` every
  `checkpointEvery` epochs, `progress.json` after every epoch (see
  [`resolve.progress()`](https://gillescolling.com/resolve/reference/resolve.progress.md))
  and `checkpoint.pt` at the end. `best.pt` and `checkpoint_<N>.pt`
  carry the run's progress, so `resumeFrom` can continue from them.

- checkpointEvery:

  Also write `checkpoint_<N>.pt` every this many epochs (0, the default,
  writes `best.pt` only). Needs `checkpointDir`.

- resumeFrom:

  A `best.pt` or `checkpoint_<N>.pt` an interrupted run wrote
  (optional). The run continues at the epoch after it, with its
  optimizer, early-stopping and random-number state, as if it had never
  stopped; `result$resumed_from_epoch` names that epoch. Pass the
  interrupted run's dataset, model and training arguments and `seed`, so
  the split and recipe are the ones it trained on.

- lossConfig:

  Loss configuration: "mae", "smape", "combined" (the default: MAE, then
  MAE and SMAPE, then the band penalty, by phase), or "nca", which takes
  the "combined" regression schedule and adds the Neighbourhood
  Components Analysis term to classification targets.

- ncaTemperature:

  Scale of the stochastic-neighbour softmax in the NCA term, i.e. the
  effective number of neighbours each sample spreads over. Acts only
  when `lossConfig = "nca"`.

- ncaNeighbors:

  Neighbours per sample the NCA term sums over, its most similar
  in-batch samples. Zero or less keeps the whole batch. Acts only when
  `lossConfig = "nca"`.

- ncaWeight:

  Weight of the NCA term against the cross-entropy it is added to. Acts
  only when `lossConfig = "nca"`.

- moeRouting:

  Mixture-of-experts routing: `"none"` (a single dense encoder),
  `"soft"` (every expert contributes, weighted by the gate) or `"topk"`
  (only the `moeTopK` highest-gated experts run per plot).

- moePlacement:

  Where the mixture sits: `"tail"` makes it the encoder's final stage,
  so `hiddenDims` minus its last two widths becomes the backbone and the
  experts produce the latent; `"post"` runs it over the finished latent
  instead, preserving the width. `"post"` is the only placement open to
  an encoder with no MLP tail. Acts only when `moeRouting` is not
  `"none"`.

- nExperts:

  Expert networks in the mixture (at least 2).

- expertHiddenDims:

  Hidden widths inside each expert.

- moeTopK:

  Experts activated per plot under `"topk"` routing.

- moeNoiseStd:

  Standard deviation of the noise added to the gate logits during
  training, which spreads load across experts.

- moeAuxLossWeight:

  Weight on the mixture's load-balancing loss, added to the task loss.

- coverDropout:

  Cover-dropout rate applied to species cover values in rank-pool /
  transformer encoding modes.

- plotTotals:

  Add each plot's combined cover and species count to the rank-pool /
  transformer features, with a flag saying they are present.

- plotTotalsDropout:

  With `plotTotals`, the probability that a training plot's totals are
  withheld, so the model also predicts plots given without them
  ([`resolve.plot_totals`](https://gillescolling.com/resolve/reference/resolve.plot_totals.md)).

- freezeComposition:

  Keep the species, genus and family embedding tables at their
  initialisation instead of learning them.

- freezeEncoder:

  Keep every parameter outside the task heads at its value, and run that
  part in evaluation mode while the heads train. With `initFrom`, the
  heads learn on the encoder of the trained parent.

- dModel:

  Model dimension for the transformer / rank-pool encoder.

- nHeads:

  Number of attention heads in the transformer encoder. Ignored unless
  `nAttentionLayers > 0`.

- nAttentionLayers:

  Number of self-attention layers in the transformer encoder (0 pools
  the token embeddings without attention).

- transformerFfDim:

  Feed-forward hidden dimension inside each transformer block.

- transformerPooling:

  Pooling strategy used to collapse the per- species token sequence into
  a plot embedding: `"attention"` or `"cls"`.

- transformerDropout:

  Dropout rate applied inside transformer blocks.

- modelConfig:

  Named list of any further `ModelConfig` fields, for example
  `list(encoder_architecture = "ft_transformer", dropout = 0.1)` or
  `list(encoder_architecture = "heterogeneous_gnn", heterogeneous_gnn = list(k_cooccurrence = 10))`
  (the species graph is built from the dataset).

- trainConfig:

  Named list of any further `TrainConfig` fields, for example
  `list(weight_decay = 1e-4, use_amp = TRUE)`.

- pretrain:

  Self-supervised pretraining of the encoder on the dataset's plots
  before training (optional): a list with `method`, `"jepa"`, `"scarf"`
  or `"masked_species"` (a transformer-encoded model), and any of that
  method's settings
  ([`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
  `"pretrain"`, or `"masked_species"`), e.g.
  `list(method = "scarf", pretrain_epochs = 20)`.

- initFrom:

  Fine-tune: start from a trained model, a parent from
  [`resolve.parent()`](https://gillescolling.com/resolve/reference/resolve.parent.md)
  or the path of a checkpoint. `dataset` must be loaded with the
  parent's `vocabs()` and `dataset_config()`; the model takes the
  parent's configuration, and only the head settings
  (`head_hidden_dims`, `head_activation`, `head_dropout`), the dropout
  rates and the freeze settings may differ. The encoder starts from the
  parent's weights, the coordinates and covariates are scaled as the
  parent scaled them, and the checkpoint records the parent
  ([`resolve.load_run_metadata()`](https://gillescolling.com/resolve/reference/resolve.load_run_metadata.md)
  `parent`). Not combined with `pretrain`.

- initReuse:

  With `initFrom`: what is carried over besides the encoder, `"encoder"`
  (the default: every head starts fresh) or `"encoder_heads"` (also the
  head of each target the parent shares under the same name, task,
  transform and classes, with its scaling).

- verbose:

  Print training progress (default TRUE)

## Value

A list with `trainer`, `result`, `dataset` and, with `pretrain`,
`pretrain` (the pretraining's `loss_history`, `total_time_seconds` and
`epochs_completed`). `result` carries `best_epoch`,
`train_time_seconds`, `final_metrics`, the loss histories `train_loss`
and `test_loss` (one value per epoch), the learning rate each epoch
trained at `lr` (the rate of its last optimizer step, which warmup and
`"linear"` move within an epoch), the held-out value of
`selectionMetric` each epoch scored `selection` (the held-out loss, or
the accuracy; empty when nothing is held out), the loss histories' split
by target `train_loss_by_target` and `test_loss_by_target` (named lists
of per-epoch vectors: each target's weighted term of the total, so per
epoch they add up to it), `train_aux_loss` (a named list of the
auxiliary terms of `train_loss`, which belong to no target, each already
weighted: `moe`, the mixture-of-experts load balancing, and
`tabnet_sparsity`, TabNet's feature-mask sparsity; empty for a model
with neither), the per-target `baselines` (the model against the
training mean or mode, regression values in original units),
`diagnostics`, `effective_batch_size` – the batch size the run actually
trained at, which is smaller than `batchSize` when the CUDA
auto-halve-on-OOM retry fired – and `resumed_from_epoch` (0 unless
`resumeFrom` continued the run).

## Details

Every model and training setting has the RESOLVE engine's default
([`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
`"model"` / `"train"`). The named arguments below set the common ones;
`modelConfig` and `trainConfig` reach every field of the RESOLVE
engine's `ModelConfig` and `TrainConfig`, nested architecture configs
included (`encoder_architecture`, `ft_transformer`, `tabnet`, `saint`,
`gnn`, `trait_net`, `excelformer`, `heterogeneous_gnn`, `tabm`,
`parallel_layers`, and the activation, normalization and head settings).
A key the RESOLVE engine does not know is an error, and so is a field
set both by a named argument and in the list. The fields that size a
model to its data (the species encoding, hash width, `top_k` and the
taxonomy slots) come from the dataset.

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
fit$result$final_metrics$elevation$r2

ft <- resolve.train.dataset(dataset, maxEpochs = 5L, verbose = FALSE,
                            modelConfig = list(encoder_architecture = "ft_transformer"))
```
