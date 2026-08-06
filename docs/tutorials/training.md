# Training Models

This guide covers RESOLVE's training configuration, evaluation surface, and
cross-validation.

## Three configuration structs

A run is described by three objects, each owning one stage:

| Struct | Decides |
|--------|---------|
| `DatasetConfig` | how the species set is encoded when the data is loaded |
| `ModelConfig` | the architecture built on top of that encoding |
| `TrainConfig` | the optimizer, schedule, precision, and device |

`DatasetConfig.species_encoding` and `ModelConfig.species_encoding` have to
agree, and so do their `hash_dim` values: the first decides which tensors the
loader fills, the second sizes the layers that read them. `prepare_data` names
both values when they disagree.

## Basic training

=== "Python"

    ```python
    import resolve_core as rc

    model   = rc.ResolveModel(dataset.schema, rc.ModelConfig())
    trainer = rc.Trainer(model, rc.TrainConfig())
    trainer.prepare_data(dataset, test_size=0.2, seed=42)

    result = trainer.fit()
    print(result.final_metrics)
    ```

=== "R"

    ```r
    trainer <- resolve.train.dataset(dataset, maxEpochs = 200L, patience = 30L,
                                     testSize = 0.2, seed = 42L)
    ```

=== "Command line"

    ```bash
    resolve train --header plots.csv --species species.csv \
      --plot-id plot_id --species-id species --abundance cover \
      --target area:regression --target habitat:classification:3 \
      --max-epochs 200 --patience 30 --test-size 0.2 \
      --output model.pt
    ```

The model is built from the dataset's schema, so its input widths and its output
heads follow the data. `prepare_data` splits the plots into a training and a
held-out test fold, fits the feature and target scalers on the training fold
alone, and keeps the split reachable afterwards.

## TrainConfig

```python
cfg = rc.TrainConfig()
cfg.batch_size   = 4096
cfg.max_epochs   = 500
cfg.patience     = 50
cfg.lr           = 1e-3
cfg.weight_decay = 1e-4
cfg.device       = "cuda"
```

| Field | Default | Meaning |
|-------|---------|---------|
| `batch_size` | `4096` | Samples per optimizer step |
| `batch_size_floor` | `1024` | Smallest batch the OOM retry drops to |
| `max_epochs` | `500` | Hard upper limit on epochs |
| `patience` | `50` | Epochs without improvement before stopping |
| `lr` | `1e-3` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `loss_config` | `Combined` | Loss preset, see below |
| `phase_boundaries` | `(100, 300)` | Epochs at which the regression loss changes phase |
| `band_threshold` | `0.25` | Tolerance band the phase-3 penalty optimizes toward |
| `band_thresholds` | `[0.1, 0.25, 0.5]` | Bands that get a reported accuracy |
| `nca_temperature` | `0.1` | Scale of the NCA stochastic-neighbour softmax |
| `nca_neighbors` | `32` | Neighbours per sample the NCA term sums over |
| `nca_weight` | `0.1` | Weight of the NCA term against cross-entropy |
| `lr_scheduler` | `None_` | `StepLR` or `CosineAnnealing` |
| `use_amp` | `False` | Mixed precision on CUDA |
| `cudnn_benchmark` | `True` | Auto-tune cuDNN algorithms |
| `allow_tf32` | `True` | TF32 matmuls on Ampere and later |
| `vram_fraction` | `1.0` | Cap on the CUDA caching allocator |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `checkpoint_dir` | `""` | Directory for periodic checkpoints; empty disables them |
| `checkpoint_every` | `0` | Write every N epochs; `0` writes only the best |

The best epoch's weights are restored when early stopping fires, so a larger
`patience` never yields a worse model. It costs compute.

### Loss presets

`loss_config` selects how the regression objective is composed:

| Preset | Composition |
|--------|-------------|
| `LossConfigMode.MAE` | Mean absolute error for the whole run; phases never advance |
| `LossConfigMode.SMAPE` | MAE plus symmetric MAPE at full weight from epoch 0 |
| `LossConfigMode.Combined` | The three phases below (default) |
| `LossConfigMode.NCA` | The three phases for regression, plus the NCA term on classification |

Classification targets use cross-entropy, weighted per class when
`TargetConfig.class_weights` is set.

### The NCA term

`LossConfigMode.NCA` adds the Neighbourhood Components Analysis objective of
Goldberger, Roweis, Hinton and Salakhutdinov (NIPS 2004) to every classification
target's cross-entropy. Each sample draws a stochastic neighbour from the rest of
the batch with probability

$$p_{ij} = \frac{\exp(u_i \cdot u_j / T)}{\sum_{k \neq i} \exp(u_i \cdot u_k / T)}, \qquad p_{ii} = 0$$

over the L2-normalized head outputs $u$, and the term is
$-\log \sum_{j \in C_i} p_{ij}$ averaged over the samples that have a same-class
neighbour in the batch. It pulls same-class samples together and pushes
different-class samples apart on the unit sphere, which is a pressure
cross-entropy does not apply.

Cross-entropy stays in the sum because the NCA objective is invariant under a
permutation of the head's output coordinates: it constrains the neighbourhood
structure and says nothing about which coordinate belongs to which class, so on
its own it would leave `argmax` over the head output, the rule every prediction
and accuracy path uses, without meaning.

Three `TrainConfig` fields tune it. $T$ is `nca_temperature` (default 0.1) and
sets the effective number of neighbours each sample spreads over; the neighbour
set is each sample's `nca_neighbors` (default 32) nearest in the batch, with 0 or
less keeping the whole batch; and `nca_weight` (default 0.1) scales the term
against the cross-entropy it is added to. They travel in the checkpoint
(`train_nca_temperature`, `train_nca_neighbors`, `train_nca_weight`) and are
inert under the other three presets.

```python
cfg.loss_config     = rc.LossConfigMode.NCA
cfg.nca_temperature = 0.2
cfg.nca_neighbors   = 16
cfg.nca_weight      = 0.3
```

### Phased regression loss

Regression training moves through three phases at `phase_boundaries`:

1. **Phase 1** (up to epoch 100): mean absolute error, for robust early learning

    $$\mathcal{L}_\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

2. **Phase 2** (to epoch 300): symmetric MAPE joins it, for scale-invariant refinement

    $$\mathcal{L}_\text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2 + \epsilon}$$

3. **Phase 3**: a differentiable hinge penalizes predictions whose ratio to the
   target leaves the tolerance band $[1 - \tau, 1 + \tau]$, with
   $\tau$ = `band_threshold`

    $$\mathcal{L}_\text{band} = \frac{1}{n} \sum_{i=1}^{n} \max\left(0,\ \left|\frac{\hat{y}_i}{y_i} - 1\right| - \tau\right)$$

`band_threshold` is what training optimizes. `band_thresholds` only selects
which band accuracies get reported.

```python
cfg.phase_boundaries = (50, 150)
cfg.band_threshold   = 0.25
```

## ModelConfig

```python
cfg = rc.ModelConfig()
cfg.hidden_dims = [512, 256, 128]
cfg.dropout     = 0.3
cfg.activation  = rc.ActivationType.GELU
```

| Field | Default | Meaning |
|-------|---------|---------|
| `hidden_dims` | `[2048, 1024, 512, 256, 128, 64]` | Shared encoder layer widths |
| `dropout` | `0.3` | Dropout between encoder layers |
| `activation` | `GELU` | `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Tanh`, `Mish`, `ELU`, `SELU`, `Softplus`, `PReLU` |
| `normalization` | `BatchNorm` | `BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `None_` |
| `use_residual` | `False` | Residual connections between encoder blocks |
| `head_hidden_dims` | `[]` | Per-target head layers; empty gives a single linear head |
| `head_dropout` | `0.0` | Dropout inside the heads |
| `genus_emb_dim` / `family_emb_dim` | `8` | Taxonomy embedding widths |
| `categorical_embed_dim` | `8` | Embedding width per categorical column |
| `species_embed_dim` | `32` | Species embedding width (embed, rank-pool, and transformer modes) |
| `encoder_architecture` | `MLP` | The encoder above the species encoding, see below |

### Encoder architectures

`encoder_architecture` swaps the shared encoder itself, each with its own
sub-config on `ModelConfig`:

```python
cfg.encoder_architecture = rc.EncoderArchitecture.FTTransformer
cfg.ft_transformer.d_model = 192
cfg.ft_transformer.n_layers = 3
```

| Value | Sub-config | Notes |
|-------|------------|-------|
| `MLP` | `hidden_dims` and friends | Default |
| `FTTransformer` | `ft_transformer` | Feature tokenization plus self-attention over features |
| `TabNet` | `tabnet` | Sequential attentive feature selection; `use_sparsemax=False` selects 1.5-entmax |
| `SAINT` | `saint` | Row and column attention |
| `ExcelFormer` | `excelformer` | Semi-permeable attention driven by learned feature importance |
| `TraitNet` | `trait_net` | Environment and trait interaction; needs traits set on the model |
| `GNN` | `gnn` | Spatial, taxonomic, or co-occurrence graph; the spatial mode needs coordinates and trains full-batch |
| `HeterogeneousGNN` | `heterogeneous_gnn` | Taxonomic and co-occurrence edges together |

### Mixture of experts

Expert routing over the shared encoder, available in hash mode:

```python
cfg.moe_routing        = rc.MoERoutingType.TopK   # or Soft
cfg.n_experts          = 4
cfg.moe_top_k          = 2
cfg.expert_hidden_dims = [256, 128]
```

## Training results

`fit` returns a `TrainResult`:

```python
result = trainer.fit()

print(f"Best epoch:  {result.best_epoch}")
print(f"Epochs run:  {len(result.train_loss_history)}")
print(f"Wall time:   {result.train_time_seconds:.1f}s")

for target, metrics in result.final_metrics.items():
    print(f"\n{target}:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")
```

### Metrics

Regression targets are scored in their original units; a `Log1p` target is
inverted first.

| Key | Meaning |
|-----|---------|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `r2` | Coefficient of determination |
| `smape` | Symmetric mean absolute percentage error, range `[0, 2]` |
| `band_10`, `band_25`, `band_50` | Fraction of predictions within that relative band, one key per entry in `band_thresholds` |

Classification targets:

| Key | Meaning |
|-----|---------|
| `accuracy` | Overall accuracy |
| `macro_f1` | Unweighted mean of per-class F1 |
| `weighted_f1` | Support-weighted mean of per-class F1 |
| `precision_<c>`, `recall_<c>`, `f1_<c>` | Per class |

### Baselines

`fit` also scores a naive baseline per target, so a metric can be read against
something: the training mean for regression, the training mode for
classification.

```python
for target, baseline in result.baselines.items():
    print(target, baseline.skill_score, baseline.r_squared, baseline.accuracy_lift)
```

## Reading the held-out fold

The trainer keeps its test fold reachable after fitting.

```python
trainer.test_plot_ids()    # plot IDs of the held-out fold
trainer.train_plot_ids()
trainer.test_indices()     # int64 positions into the dataset
trainer.train_indices()
```

### Regression residuals

```python
residuals = trainer.compute_residuals("area")

residuals.predictions      # per-plot, original units
residuals.actuals
residuals.residuals
residuals.mean_residual, residuals.std_residual
residuals.skewness, residuals.kurtosis
residuals.q05, residuals.q25, residuals.q50, residuals.q75, residuals.q95
```

### Classification predictions

```python
classified = trainer.compute_classification_predictions("habitat")

classified.predicted_classes   # int64 [n_test]
classified.probabilities       # float [n_test, n_classes]
classified.actuals             # int64 [n_test]
classified.class_names
```

### Calibration

```python
calibration = trainer.compute_calibration("habitat", n_bins=10)

print(calibration.expected_calibration_error)
print(calibration.max_calibration_error)
for b in calibration.bins:
    print(b.bin_start, b.bin_end, b.mean_predicted_prob, b.actual_frequency, b.count)
```

A regression target returns no bins.

### Network diagnostics

```python
diagnostics = trainer.compute_diagnostics()

print(diagnostics.summary)
print(diagnostics.overall_dead_fraction, diagnostics.overall_saturated_fraction)
for layer in diagnostics.layers:
    print(layer.name, layer.n_dead, layer.n_saturated, layer.mean_activation)
```

Diagnostics read the encoder's per-layer activations, which the hash encoder
exposes. Other species encoders report that diagnostics are unavailable.

## Cross-validation

```python
cv = trainer.cross_validate(n_folds=5, seed=42)

print(cv.n_folds, cv.total_time_seconds)
print(cv.mean_metrics)      # {target: {metric: mean over folds}}
print(cv.std_metrics)
print(cv.fold_results)      # per-fold TrainResult
```

Every fold restarts from the model's as-constructed weights, so a
`cross_validate` call after a `fit` does not warm-start folds from weights that
already saw their held-out rows. The trainer's own split is restored afterwards,
so the evaluators above keep scoring the fold they scored before.

### Spatial blocks

Random folds leak between neighbouring plots. Spatial block cross-validation
assigns whole latitude-longitude blocks to folds instead:

```python
from resolve_core import SpatialBlockConfig

blocks = SpatialBlockConfig()
blocks.lat_size = 1.0     # degrees
blocks.lon_size = 1.0
blocks.balance  = True    # greedy bin-packing, rather than round-robin

cv = trainer.cross_validate_spatial(blocks, n_folds=5, seed=42)
```

Blocks smaller than the fold count leave folds empty, which is rejected rather
than silently reduced. Coordinates are required.

## Saving and loading

```python
trainer.save("model.pt")

predictor = rc.Predictor.load("model.pt", device="cpu")
```

A checkpoint carries the weights, the scalers, the schema with its fitted
species, taxonomy, and categorical vocabularies, the model config, and the
training config. Three readers get at it without a full load:

```python
rc.Trainer.load_train_config("model.pt")    # the hyperparameters it was fit with
rc.Trainer.load_run_metadata("model.pt")    # timing, plot counts, best epoch, metrics
trainer.load_state("model.pt")              # weights and scalers into an existing trainer
```

`load_state` is the way to score a checkpoint from a freshly built trainer: build
the same model and dataset, call `prepare_data` with the same `test_size` and
`seed`, then `load_state`, and the test-fold evaluators above work as they did in
the original run.

## GPU training

```python
cfg = rc.TrainConfig()
cfg.device        = "cuda"
cfg.use_amp       = True
cfg.vram_fraction = 0.80    # leave headroom when sharing the GPU with a desktop
```

If the device runs out of memory, the trainer releases its optimizer, AMP, and
allocator caches, halves `batch_size`, and restarts from epoch 0, down to
`batch_size_floor`. `trainer.config.batch_size` afterwards is the batch size the
run actually used. See [Performance Tuning](performance-tuning.md).

## Next steps

- [Encoding Modes](encoding-modes.md): choosing how the species set is encoded
- [Performance Tuning](performance-tuning.md): speed, memory, and accuracy levers
- [Making Predictions](prediction.md): scoring new data
- [Understanding Embeddings](embeddings.md): reading the learned representation
