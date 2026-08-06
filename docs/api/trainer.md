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
| `test_size` | `float` | `0.2` | Fraction held out |
| `seed` | `int` | `42` | Split seed |

`prepare_data` splits the plots, fits the feature and target scalers on the
training fold alone, records the categorical vocabulary, and keeps the split
reachable. It raises when `DatasetConfig.hash_dim` and `ModelConfig.hash_dim`
disagree.

`prepare_data_raw` takes the tensors directly, for callers assembling their own
inputs:

```
prepare_data_raw(coordinates, covariates, hash_embedding, species_ids,
                 species_vector, genus_ids, family_ids,
                 unknown_fraction, unknown_count, targets,
                 categorical_ids=None, test_size=0.2, seed=42,
                 pool_genus_ids=None, pool_family_ids=None, pool_weights=None,
                 pool_mask=None, pool_has_cover=None)
```

### Fitting

```python
result = trainer.fit()
```

Returns a `TrainResult`. Runs to `max_epochs` or until `patience` epochs pass
without improvement, restoring the best epoch's weights either way. On a CUDA
out-of-memory error it releases its caches, halves `batch_size`, and restarts
from epoch 0, down to `batch_size_floor`. The GIL is released for the whole fit.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `ResolveModel` | The model being fit |
| `config` | `TrainConfig` | The config in force; `batch_size` reflects any OOM halving |
| `scalers` | `Scalers` | Feature and per-target scaling fitted on the training fold |
| `categorical_vocab` | `CategoricalVocab` | Captured at `prepare_data` time |

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

Runs the model in eval mode on raw tensors and returns `dict[str, Tensor]`.
The signature matches `ResolveModel.forward`. For a whole dataset, save a
checkpoint and use [`Predictor`](predictor.md).

### Checkpoints

```python
trainer.save("model.pt")
trainer.save("model.pt", metadata=run_metadata)

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
| `checkpoint_dir` | `str` | `""` | Periodic checkpoint directory; empty disables |
| `checkpoint_every` | `int` | `0` | Write every N epochs; `0` writes only the best |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` |

`LossConfigMode.NCA` keeps the `Combined` regression schedule and adds the
Neighbourhood Components Analysis term to every classification target. See
[Loss presets](../tutorials/training.md#loss-presets).

---

## TrainResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_epoch` | `int` | Epoch with the best test-fold loss |
| `final_metrics` | `dict[str, dict[str, float]]` | Per target, per metric |
| `train_loss_history` | `list[float]` | One entry per epoch run |
| `test_loss_history` | `list[float]` | Same length |
| `train_time_seconds` | `float` | Wall time |
| `resumed_from_epoch` | `int` | Non-zero when the fit resumed from a checkpoint |
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

| Attribute | Description |
|-----------|-------------|
| `baseline_mse`, `baseline_mae` | Predicting the training mean |
| `model_mse`, `model_mae` | The fitted model |
| `skill_score` | Improvement over the mean baseline |
| `r_squared` | Coefficient of determination |
| `baseline_accuracy`, `model_accuracy`, `accuracy_lift` | Classification, against the training mode |
| `training_mean`, `training_mode` | The baseline values themselves |

---

## Evaluation result types

### ResidualAnalysis

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | |
| `predictions`, `actuals`, `residuals` | `list[float]` | Per held-out plot, original units |
| `mean_residual`, `std_residual` | `float` | |
| `skewness`, `kurtosis` | `float` | |
| `q05`, `q25`, `q50`, `q75`, `q95` | `float` | Residual quantiles |

### ClassificationPredictions

| Attribute | Type | Description |
|-----------|------|-------------|
| `target_name` | `str` | |
| `predicted_classes` | int64 `Tensor` `[n_test]` | |
| `probabilities` | float `Tensor` `[n_test, n_classes]` | |
| `actuals` | int64 `Tensor` `[n_test]` | |
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
