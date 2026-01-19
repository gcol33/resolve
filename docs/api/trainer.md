# Trainer

The `Trainer` class handles model training with phased optimization.

## Class Definition

```python
class Trainer:
    """
    Training orchestrator for RESOLVE models.

    Implements phased training, early stopping, and checkpoint management.
    """
```

## Constructor

### `Trainer(model, dataset, **kwargs)`

Create a trainer instance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ResolveModel` | required | Model to train |
| `dataset` | `ResolveDataset` | required | Training dataset |
| `max_epochs` | `int` | `200` | Maximum training epochs |
| `patience` | `int` | `30` | Early stopping patience |
| `batch_size` | `int` | `256` | Batch size |
| `lr` | `float` | `1e-3` | Learning rate |
| `val_fraction` | `float` | `0.2` | Validation set fraction |
| `phase_boundaries` | `tuple` | `(50, 150)` | Epoch boundaries for phase transitions |
| `device` | `str` | `"auto"` | Device (`"auto"`, `"cpu"`, `"cuda"`) |
| `species_normalization` | `str` | `"relative_plot"` | Abundance normalization mode |
| `track_unknown_count` | `bool` | `False` | Track unknown species count |

**Example:**

```python
trainer = Trainer(
    model=model,
    dataset=dataset,
    max_epochs=200,
    patience=30,
    batch_size=256,
    phase_boundaries=(50, 150),
)
```

## Methods

### `fit()`

Train the model.

**Returns:** `TrainingResult`

```python
result = trainer.fit()
print(f"Best epoch: {result.best_epoch}")
```

### `save(path)`

Save trained model checkpoint.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Output file path |

```python
trainer.save("model.pt")
```

### `load(path, device="auto")` (class method)

Load model from checkpoint.

**Returns:** Tuple of `(model, species_encoder, scalers)`

```python
model, encoder, scalers = Trainer.load("model.pt")
```

## Training Phases

RESOLVE uses phased training for regression targets:

### Phase 1: MAE Loss (epochs 0 to phase_boundaries[0])

Mean Absolute Error for robust initial learning:

$$\mathcal{L}_\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Phase 2: SMAPE Loss (epochs phase_boundaries[0] to phase_boundaries[1])

Symmetric Mean Absolute Percentage Error for scale-invariant refinement:

$$\mathcal{L}_\text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon}$$

### Phase 3: Band Loss (epochs > phase_boundaries[1])

Soft band accuracy for calibrated predictions:

$$\mathcal{L}_\text{band} = 1 - \sigma\left(\frac{\tau - |y_i - \hat{y}_i| / |y_i|}{\beta}\right)$$

---

# TrainingResult

The `TrainingResult` dataclass contains training outcomes.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_epoch` | `int` | Epoch with best validation loss |
| `stopped_epoch` | `int` | Epoch when training stopped |
| `final_metrics` | `dict` | Final validation metrics per target |
| `history` | `dict` | Training history (losses, metrics) |

## Metrics

**Regression targets:**
- `mae`: Mean Absolute Error
- `smape`: Symmetric MAPE
- `band_5pct`, `band_10pct`, `band_20pct`: Band accuracy

**Classification targets:**
- `accuracy`: Overall accuracy
- `f1_macro`: Macro F1 score
