# Trainer

The `Trainer` class handles model training with automatic model construction and phased optimization.

## Class Definition

```python
class Trainer:
    """
    Training orchestrator for RESOLVE models.

    Automatically builds model from dataset schema, implements training
    with early stopping, and provides prediction with confidence filtering.
    """
```

## Constructor

### `Trainer(dataset, **kwargs)`

Create a trainer instance. The model is automatically constructed from the dataset schema.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | Training dataset |
| `species_encoding` | `str` | `"hash"` | Species encoding mode: `"hash"`, `"embed"`, `"rank_pool"`, or `"transformer"` |
| `hash_dim` | `int` | `32` | Dimension of feature hash (hash mode) |
| `species_embed_dim` | `int` | `32` | Embedding dimension per species (embed mode) |
| `top_k` | `int` | `5` | Number of top species for taxonomy embeddings |
| `top_k_species` | `int` | `10` | Number of top species (embed mode) |
| `hidden_dims` | `list[int]` | `[2048, 1024, 512, 256, 128, 64]` | Hidden layer dimensions |
| `dropout` | `float` | `0.3` | Dropout rate |
| `batch_size` | `int` | `32768` | Batch size |
| `max_epochs` | `int` | `500` | Maximum training epochs |
| `patience` | `int` | `50` | Early stopping patience |
| `lr` | `float` | `1e-3` | Learning rate |
| `loss_config` | `str` | `"mae"` | Loss preset: `"mae"`, `"combined"`, or `"smape"` |
| `checkpoint_dir` | `str` | `None` | Directory for checkpoints |
| `n_heads` | `int` | `4` | Number of attention heads (transformer mode) |
| `transformer_ff_dim` | `int` | `256` | Feed-forward dimension (transformer mode) |
| `transformer_pooling` | `str` | `"attention"` | Pooling strategy: `"attention"` or `"mean"` (transformer mode) |
| `transformer_dropout` | `float` | `0.1` | Dropout rate in transformer layers (transformer mode) |
| `n_attention_layers` | `int` | `0` | Number of transformer encoder layers |
| `device` | `str` | `"auto"` | Device (`"auto"`, `"cpu"`, `"cuda"`) |

**Example:**

```python
trainer = Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=64,
    top_k=10,
    max_epochs=200,
    patience=30,
    loss_config="mae",
)
```

## Encoding Modes

The `species_encoding` parameter controls how species composition is represented as model input.

| Mode | Description | Best for |
|------|-------------|----------|
| `"hash"` | Feature hashing into a fixed-dimension vector. Fast and memory-efficient. | Default choice, large species pools |
| `"embed"` | Learned embeddings per species. More expressive but requires a fixed vocabulary. | Moderate species pools with enough training data |
| `"rank_pool"` | Variable-length species lists with weighted pooling. Handles plots with widely varying species counts without padding waste. | Datasets where species richness varies widely across plots |
| `"transformer"` | Transformer-based species encoder with attention pooling. Captures inter-species interactions via self-attention. | Large datasets where species co-occurrence patterns matter |

### Hash mode

```python
trainer = Trainer(dataset, species_encoding="hash", hash_dim=64)
```

### Embed mode

```python
trainer = Trainer(dataset, species_encoding="embed", species_embed_dim=32, top_k_species=10)
```

### Rank-pool mode

```python
trainer = Trainer(dataset, species_encoding="rank_pool", max_epochs=100)
```

### Transformer mode

```python
trainer = Trainer(
    dataset,
    species_encoding="transformer",
    n_heads=4,
    transformer_ff_dim=256,
    n_attention_layers=2,
    transformer_pooling="attention",
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

### `predict(dataset, output_space="raw", confidence_threshold=0.0)`

Predict on a dataset.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | Dataset to predict on |
| `output_space` | `str` | `"raw"` | `"raw"` (original scale) or `"transformed"` |
| `confidence_threshold` | `float` | `0.0` | Minimum confidence (0-1), predictions below set to NaN |

**Returns:** `dict[str, np.ndarray]` mapping target names to predictions

**Confidence semantics:**
- **Regression**: confidence = 1 - unknown_fraction, where unknown_fraction is the proportion of species abundance not seen during training
- **Classification**: confidence = max softmax probability across classes

```python
# Get all predictions
preds = trainer.predict(dataset)

# Filter to confident predictions only
preds = trainer.predict(dataset, confidence_threshold=0.8)
```

## Loss Presets

The `loss_config` parameter controls the loss function:

| Preset | Description |
|--------|-------------|
| `"mae"` | Pure MAE loss (default, most stable) |
| `"combined"` | 80% MAE + 15% SMAPE + 5% band accuracy |
| `"smape"` | 50% MAE + 50% SMAPE |

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
