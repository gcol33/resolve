# Trainer

Everything in this page lives in `resolve_core`.

```python
import resolve_core as rc
```

---

## Trainer

Owns the split, the optimizer, the training loop, and the held-out evaluation
surface.

```python
trainer = rc.Trainer(model, config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ResolveModel` | The model to fit |
| `config` | `TrainConfig` | Optimizer, schedule, precision, device |

### Preparing data

```python
trainer.prepare_data(dataset, test_size=0.2, seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | The loaded dataset |
| `test_size` | `float` | `0.2` | Fraction held out; `0` holds nothing out and needs `fixed_epochs` |
| `seed` | `int` | `42` | Split seed |

`prepare_data` splits the plots, fits the feature and target scalers on the
training fold alone, records the categorical vocabulary, and keeps the split
reachable. Each call replaces everything an earlier one set (the split, the
scalers, the plot IDs, the vocabulary and the GPU copies), so a later `fit`
trains on the latest split alone. It raises when `DatasetConfig.hash_dim` and
`ModelConfig.hash_dim` disagree, and when the split leaves no plot to fit.

To hold out a fold of your choosing instead of a random fraction, such as a
withheld spatial block or database, or the fold a published protocol selects
its epoch on, pass the plots' positions:

```python
fold = dataset.plot_indices(["plot_17", "plot_204", "plot_311"])
trainer.prepare_data(dataset, fold, seed=42)
# or by keyword
trainer.prepare_data(dataset, stopping_indices=fold, seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | The loaded dataset |
| `stopping_indices` | `list[int]` | required | 0-based positions in `dataset.plot_ids` of the held-out plots |
| `seed` | `int` | `42` | Orders the fitting rows and seeds the epoch shuffle |

Early stopping, `best_epoch`, `final_metrics`, the baselines, the evaluators
below and `test_indices()` / `test_plot_ids()` then read exactly these plots,
in the order given; every other plot is a fitting row. The fitting rows keep
the order the random split would give them, so naming the fold a random split
drew reproduces that run exactly. `dataset.plot_indices(ids)` maps plot IDs to
positions and raises `ValueError` for an ID the dataset does not hold.
`prepare_data` raises `ValueError` for an empty fold, a position out of range,
a position listed twice, a fold of every plot, and a fold with no labelled plot
for a target whose `missing` policy is `Mask`.

For a `heterogeneous_gnn` model it also builds the species graph the encoder
passes messages on, from the dataset's taxonomy and co-occurrence as
`heterogeneous_gnn` asks; `Trainer.save` writes it into the checkpoint, so
scoring reads the graph the weights were trained on. A graph set by hand
through `model.set_species_graph` is left alone.

`seed` governs the SPLIT. It does not cover model weight initialisation, which
draws from the process-global torch RNG the way any PyTorch module does, so two
runs with the same `seed` still start from different weights. To reproduce a fit
end to end, seed that RNG before constructing the model -- which is what the
CLI's `--seed` does:

```python
import torch

torch.manual_seed(7)                       # weight initialisation
model = rc.ResolveModel(dataset.schema, model_config)
trainer = rc.Trainer(model, train_config)
trainer.prepare_data(dataset, test_size=0.2, seed=7)   # the split
```

The same applies to `cross_validate` and `cross_validate_spatial`: their `seed`
argument selects the folds, and every fold restarts from the trainer's
as-constructed weights, so seeding before construction fixes those too.

`prepare_data_raw` takes the tensors directly, for callers assembling their own
inputs. Every argument is a keyword; `targets` maps each target name to its
`(n_plots,)` tensor, and the tensor keywords are those of
[`Predictor.predict`](predictor.md#predicting-on-tensors):

```
prepare_data_raw(*, targets, coordinates=None, covariates=None, ...,
                 categorical_ids=None, test_size=0.2, seed=42)
```

### Fitting

```python
result = trainer.fit()
```

Returns a `TrainResult`. Runs to `max_epochs` or until `patience` epochs pass
without improvement, restoring the best epoch's weights either way. With
`fixed_epochs > 0` it runs exactly that many epochs of the `max_epochs`
schedule instead and keeps the final weights; `best_epoch` then names the last
epoch, and a fit prepared with `test_size=0` returns no held-out losses or
metrics. On a CUDA
out-of-memory error it releases its caches, halves `batch_size`, and restarts
from epoch 0, down to `batch_size_floor`. The held-out fold is evaluated in
chunks of at least 4096 plots (the whole fold at once for a `gnn` encoder,
whose graph spans it); an out-of-memory error there halves the chunk instead
and never the training batch. The GIL is released for the whole fit.

### Resuming a run

```python
config = rc.Trainer.load_train_config("ckpt/checkpoint_200.pt")
config.checkpoint_dir = "ckpt"
config.checkpoint_every = 50
trainer = rc.Trainer(rc.ResolveModel(dataset.schema, model_config), config)
trainer.prepare_data(dataset, test_size=0.2, seed=42)   # the run's own split
result = trainer.resume("ckpt/checkpoint_200.pt")
```

With a `checkpoint_dir`, `fit` writes `best.pt` at every new best epoch and
`checkpoint_<N>.pt` every `checkpoint_every` epochs. Both carry the run's
progress: the epochs done, the loss histories, the best epoch's loss, weights
and metrics, the early-stopping patience, the optimizer and mixed-precision
state, the batch size the run had reached and the random-number streams.
`resume` picks the run up at the epoch after the checkpoint's, as if it had
never stopped, and returns the result of the whole run; on the CPU it is the
uninterrupted run exactly. `resumed_from_epoch` names the epoch it continued
from. The trainer needs the run's model architecture and recipe and the run's
split: prepare the same data with the same hold-out and seed. `resume` raises
`ValueError` for a checkpoint without run progress (the final `checkpoint.pt`,
or one `save` wrote) and for a split or scalers that differ from the run's.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `ResolveModel` | The model being fit |
| `config` | `TrainConfig` | The config in force; `batch_size` reflects any OOM halving |
| `scalers` | `Scalers` | Feature and per-target scaling fitted on the training fold |
| `categorical_vocab` | `CategoricalVocab` | Captured at `prepare_data` time |
| `run_metadata` | `RunMetadata` | The split `prepare_data` made and, after `fit`, the run's timing, epochs and final metrics |

### The held-out fold

| Method | Returns |
|--------|---------|
| `test_plot_ids()` | `list[str]`; needs `prepare_data(dataset)` |
| `train_plot_ids()` | `list[str]` |
| `test_indices()` | int64 `Tensor` of positions into the dataset |
| `train_indices()` | int64 `Tensor` |

### Evaluators

All three score the trainer's own held-out fold.

```python
residuals = trainer.compute_residuals("area")
classified = trainer.compute_classification_predictions("habitat")
calibration = trainer.compute_calibration("habitat", n_bins=10)
diagnostics = trainer.compute_diagnostics()
```

| Method | Returns |
|--------|---------|
| `compute_residuals(target_name)` | `ResidualAnalysis` |
| `compute_classification_predictions(target_name)` | `ClassificationPredictions` |
| `compute_calibration(target_name, n_bins=10)` | `CalibrationResult`; no bins for a regression target |
| `compute_diagnostics()` | `NetworkDiagnostics`; per-layer activations are available for the hash encoder |

### Cross-validation

```python
cv = trainer.cross_validate(n_folds=5, seed=42)

from resolve_core import SpatialBlockConfig
blocks = SpatialBlockConfig()
blocks.lat_size = 1.0
blocks.lon_size = 1.0
blocks.balance  = True
cv = trainer.cross_validate_spatial(blocks, n_folds=5, seed=42)
```

Both return a `CrossValidationResult`, release the GIL, reset each fold to the
model's as-constructed weights, and restore the trainer's own split when they
finish. Spatial blocks need coordinates, and a block count below the fold count
is rejected.

### Prediction

```python
outputs = trainer.predict(continuous, genus_ids=..., family_ids=...)
```

Runs the model in eval mode on raw tensors and returns `dict[str, Tensor]`:
regression targets in original units, classification targets as logits.
`continuous` is the unstandardised block, standardised here with the fitted
scalers; the other arguments match `ResolveModel.forward`. For a whole dataset,
save a checkpoint and use [`Predictor`](predictor.md).

### Checkpoints

```python
trainer.save("model.pt")
trainer.save("model.pt", metadata=trainer.run_metadata)

trainer.load_state("model.pt", device="cpu", vram_fraction=1.0)

rc.Trainer.load_train_config("model.pt")     # -> TrainConfig
rc.Trainer.load_run_metadata("model.pt")     # -> RunMetadata
```

`load_state` loads weights, scalers, and the categorical vocabulary into an
existing trainer in place, which is how a checkpoint is scored from a freshly
built trainer: build the same model and dataset, `prepare_data` with the same
`test_size` and `seed`, then `load_state`, and the evaluators above work as they
did in the original run. `load_state` raises when a parameter or buffer is
missing rather than leaving it at random init.

`load_train_config` recovers the persisted hyperparameters. Fields that are not
persisted (device, checkpoint directory, AMP and cuDNN flags, the log callback)
keep their `TrainConfig` defaults.

Checkpoint I/O retries on a transient filesystem error, with exponential
backoff, tunable through `RESOLVE_IO_RETRY_ATTEMPTS` (default 3) and
`RESOLVE_IO_RETRY_BACKOFF_MS` (default 100).

---

## TrainConfig

```python
config = rc.TrainConfig()
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `4096` | Samples per optimizer step |
| `batch_size_floor` | `int` | `1024` | Smallest batch the OOM retry drops to |
| `max_epochs` | `int` | `500` | Hard upper limit |
| `patience` | `int` | `50` | Epochs without improvement before stopping |
| `fixed_epochs` | `int` | `0` | Run exactly N epochs of the `max_epochs` schedule and keep the final weights; `0` stops early |
| `lr` | `float` | `1e-3` | AdamW learning rate |
| `weight_decay` | `float` | `1e-4` | AdamW weight decay |
| `loss_config` | `LossConfigMode` | `Combined` | `MAE`, `SMAPE`, `Combined`, `NCA` |
| `phase_boundaries` | `(int, int)` | `(100, 300)` | Epochs at which the regression loss changes phase |
| `band_threshold` | `float` | `0.25` | Tolerance band the phase-3 penalty optimizes |
| `band_thresholds` | `list[float]` | `[0.1, 0.25, 0.5]` | Bands that get a reported accuracy |
| `nca_temperature` | `float` | `0.1` | Scale of the NCA stochastic-neighbour softmax |
| `nca_neighbors` | `int` | `32` | Neighbours per sample the NCA term sums over |
| `nca_weight` | `float` | `0.1` | Weight of the NCA term against cross-entropy |
| `lr_scheduler` | `LRSchedulerType` | `None_` | `None_`, `StepLR`, `CosineAnnealing` |
| `lr_step_size` | `int` | `100` | `StepLR` period |
| `lr_gamma` | `float` | `0.1` | `StepLR` factor |
| `lr_min` | `float` | `1e-6` | `CosineAnnealing` endpoint |
| `use_amp` | `bool` | `False` | Mixed precision on CUDA |
| `amp_init_scale` | `float` | `65536.0` | Initial gradient scale |
| `amp_growth_factor` | `float` | `2.0` | Scale growth |
| `amp_backoff_factor` | `float` | `0.5` | Scale reduction on overflow |
| `amp_growth_interval` | `int` | `2000` | Steps between scale increases |
| `cudnn_benchmark` | `bool` | `True` | Auto-tune cuDNN algorithms |
| `allow_tf32` | `bool` | `True` | TF32 matmuls on Ampere and later |
| `vram_fraction` | `float` | `1.0` | Cap on the CUDA caching allocator |
| `checkpoint_dir` | `str` | `""` | Directory for `best.pt`, `checkpoint_<N>.pt`, `progress.json` and the final `checkpoint.pt`; empty disables |
| `checkpoint_every` | `int` | `0` | Write `checkpoint_<N>.pt` every N epochs; `0` writes only `best.pt` |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` |

`LossConfigMode.NCA` keeps the `Combined` regression schedule and adds the
Neighbourhood Components Analysis term to every classification target. See
[Loss presets](../tutorials/training.md#loss-presets).

---

## TrainResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_epoch` | `int` | Epoch with the best test-fold loss |
| `final_metrics` | `dict[str, dict[str, float]]` | Per target, per metric; a target with `missing = Mask` also reports `n_labelled`, the held-out plots it was scored on |
| `train_loss_history` | `list[float]` | One entry per epoch run |
| `test_loss_history` | `list[float]` | Same length; empty when nothing is held out |
| `train_loss_history_by_target` | `dict[str, list[float]]` | `train_loss_history` split by target: per epoch, the term each target adds to the total (its weighted loss, averaged over the epoch's batches like the total; a batch with no labelled row for a masked target adds 0) |
| `test_loss_history_by_target` | `dict[str, list[float]]` | `test_loss_history` split by target: each target's weighted term of the phase-invariant selection loss; empty when nothing is held out |
| `train_aux_loss_history` | `dict[str, list[float]]` | The auxiliary terms of `train_loss_history`, which belong to no target, keyed by name: `moe` (mixture-of-experts load balancing) and `tabnet_sparsity` (TabNet's mask sparsity), each already weighted; empty for a model with neither. Per epoch the train terms plus these add up to the total, and the test terms add up to the test total, within float rounding |
| `train_time_seconds` | `float` | Wall time of the whole run, a resumed run's earlier epochs included |
| `resumed_from_epoch` | `int` | The epoch `resume` continued the run from; `0` for a run `fit` trained from the start |
| `baselines` | `dict[str, BaselineMetrics]` | Naive baseline per target |
| `diagnostics` | `NetworkDiagnostics` | Network health at the end of the fit |

### Metric keys

Regression targets are reported in their original units; a `Log1p` target is
inverted first.

| Key | Meaning |
|-----|---------|
| `mae` | Mean absolute error |
| `rmse` | Root mean squared error |
| `r2` | Coefficient of determination |
| `smape` | Symmetric mean absolute percentage error, range `[0, 2]` |
| `band_10`, `band_25`, `band_50` | Fraction within that relative band, one key per `band_thresholds` entry |

| Key | Meaning |
|-----|---------|
| `accuracy` | Overall accuracy |
| `macro_f1` | Unweighted mean of per-class F1 |
| `weighted_f1` | Support-weighted mean of per-class F1 |
| `precision_<c>`, `recall_<c>`, `f1_<c>` | Per class |

### BaselineMetrics

Regression values are in the target's original units, like the metric keys,
and the model's scores are the ones `final_metrics` reports for the returned
weights: `model_mae` is `final_metrics[t]["mae"]`, `r_squared` is
`final_metrics[t]["r2"]`, `model_accuracy` is `final_metrics[t]["accuracy"]`.

| Attribute | Description |
|-----------|-------------|
| `baseline_mse`, `baseline_mae` | Predicting the training mean |
| `model_mse`, `model_mae` | The fitted model |
| `skill_score` | `1 - model_mse / baseline_mse` |
| `r_squared` | Coefficient of determination |
| `baseline_accuracy`, `model_accuracy`, `accuracy_lift` | Classification, against the training mode |
| `training_mean`, `training_mode` | The baseline values themselves |

---

## Evaluation result types

### ResidualAnalysis

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | |
| `predictions`, `actuals`, `residuals` | `list[float]` | Per held-out plot labelled for the target, original units |
| `test_rows` | `list[int]` | Position of each row in the test fold (`test_plot_ids()`) |
| `mean_residual`, `std_residual` | `float` | |
| `skewness`, `kurtosis` | `float` | |
| `q05`, `q25`, `q50`, `q75`, `q95` | `float` | Residual quantiles |

### ClassificationPredictions

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | |
| `predicted_classes` | int64 `Tensor` `[n_labelled]` | |
| `probabilities` | float `Tensor` `[n_labelled, n_classes]` | |
| `actuals` | int64 `Tensor` `[n_labelled]` | |
| `test_rows` | int64 `Tensor` `[n_labelled]` | Position of each row in the test fold (`test_plot_ids()`) |
| `class_names` | `list[str]` | Index equals code |

### CalibrationResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | |
| `class_idx` | `int` | |
| `bins` | `list[CalibrationBin]` | Empty for a regression target |
| `expected_calibration_error` | `float` | |
| `max_calibration_error` | `float` | |

`CalibrationBin` carries `bin_start`, `bin_end`, `mean_predicted_prob`,
`actual_frequency`, and `count`.

### NetworkDiagnostics

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | `list[LayerDiagnostics]` | Per layer |
| `total_neurons`, `total_dead`, `total_saturated` | `int` | |
| `overall_dead_fraction`, `overall_saturated_fraction` | `float` | |
| `has_issues` | `bool` | |
| `summary` | `str` | Human-readable summary |

`LayerDiagnostics` carries `name`, `n_neurons`, `n_dead`, `n_saturated`,
`dead_fraction`, `saturated_fraction`, `mean_activation`, `std_activation`, and
`sparsity`.

### CrossValidationResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_folds` | `int` | |
| `mean_metrics` | `dict[str, dict[str, float]]` | Mean across folds |
| `std_metrics` | `dict[str, dict[str, float]]` | Standard deviation across folds |
| `fold_results` | `list[TrainResult]` | One per fold |
| `total_time_seconds` | `float` | |

### SpatialBlockConfig

```python
from resolve_core import SpatialBlockConfig
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `lat_size` | `float` | `1.0` | Block height in degrees |
| `lon_size` | `float` | `1.0` | Block width in degrees |
| `balance` | `bool` | `False` | Greedy bin-packing rather than round-robin assignment |

### RunMetadata

| Attribute | Type |
|-----------|------|
| `resolve_version` | `str` |
| `created_at`, `completed_at` | `str` |
| `train_time_seconds` | `float` |
| `n_plots_train`, `n_plots_test` | `int` |
| `hold_out` | `HoldOutMode`: `Random` (a `test_size` fraction drawn with `split_seed`) or `Explicit` (the plots passed as `stopping_indices`) |
| `test_size` | `float`; `0` under `Explicit` |
| `split_seed` | `int`; the `prepare_data` seed, which also seeds the epoch shuffle |
| `best_epoch`, `total_epochs` | `int` |
| `final_metrics` | `dict[str, dict[str, float]]` |

---

## Loss

Regression training moves through three phases at `phase_boundaries`.

**Phase 1**, mean absolute error:

$$\mathcal{L}_\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Phase 2** adds symmetric MAPE:

$$\mathcal{L}_\text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2 + \epsilon}$$

**Phase 3** adds a differentiable hinge on the tolerance band $\tau$ =
`band_threshold`:

$$\mathcal{L}_\text{band} = \frac{1}{n} \sum_{i=1}^{n} \max\left(0,\ \left|\frac{\hat{y}_i}{y_i} - 1\right| - \tau\right)$$

Classification targets use cross-entropy throughout, weighted per class when
`TargetConfig.class_weights` is set.

### Metrics

`rc.Metrics` exposes the same functions as static methods, for scoring tensors
outside a trainer:

| Method | Signature |
|--------|-----------|
| `mae` | `mae(pred, target)` |
| `rmse` | `rmse(pred, target)` |
| `r_squared` | `r_squared(pred, target)` |
| `smape` | `smape(pred, target, eps=1e-8)` |
| `band_accuracy` | `band_accuracy(pred, target, threshold=0.25)` |
| `accuracy` | `accuracy(pred, target)` |
| `confusion_matrix` | `confusion_matrix(pred, target, num_classes)` |
| `classification_metrics` | `classification_metrics(pred, target, num_classes)` |
| `accuracy_at_threshold` | `accuracy_at_threshold(pred, target, confidence, threshold)` |
| `accuracy_coverage_curve` | `accuracy_coverage_curve(pred, target, confidence, thresholds=[0.0, 0.5, 0.8, 0.9, 0.95])` |
| `compute` | `compute(pred, target, task, transform=TransformType.None_, band_thresholds=[0.25, 0.5, 0.75], num_classes=0)` |

---

## See also

- [Training Models](../tutorials/training.md)
- [Performance Tuning](../tutorials/performance-tuning.md)
- [Predictor](predictor.md)
