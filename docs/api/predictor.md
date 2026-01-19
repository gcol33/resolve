# Predictor

The `Predictor` class provides the inference interface for trained models.

## Class Definition

```python
class Predictor:
    """
    Inference interface for trained RESOLVE models.

    Loads saved checkpoints and produces predictions on new data.
    """
```

## Constructor

### `Predictor(model, species_encoder, scalers, device="auto")`

Create a predictor from model components.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `ResolveModel` | Trained model |
| `species_encoder` | `SpeciesEncoder` | Fitted species encoder |
| `scalers` | `dict` | Fitted scalers |
| `device` | `str` | Device for inference |

## Class Methods

### `load(path, device="auto")`

Load predictor from saved checkpoint.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Path to checkpoint file |
| `device` | `str` | `"auto"` | Device for inference |

**Returns:** `Predictor`

**Example:**

```python
predictor = Predictor.load("model.pt", device="cpu")
```

## Methods

### `predict(dataset, return_latent=False, output_space="raw")`

Generate predictions for a dataset.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | Dataset to predict on |
| `return_latent` | `bool` | `False` | Include latent representations |
| `output_space` | `str` | `"raw"` | Output space for regression (`"raw"` or `"transformed"`) |

**Returns:** `ResolvePredictions`

**Example:**

```python
predictions = predictor.predict(dataset, return_latent=True)
```

### `get_embeddings(dataset)`

Get latent embeddings for all plots.

**Returns:** `np.ndarray` of shape `(n_plots, latent_dim)`

```python
embeddings = predictor.get_embeddings(dataset)
```

### `get_genus_embeddings()`

Get learned genus embedding weights.

**Returns:** `np.ndarray` of shape `(n_genera, genus_emb_dim)`

```python
genus_emb = predictor.get_genus_embeddings()
```

### `get_family_embeddings()`

Get learned family embedding weights.

**Returns:** `np.ndarray` of shape `(n_families, family_emb_dim)`

```python
family_emb = predictor.get_family_embeddings()
```

---

# ResolvePredictions

Container for model predictions.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `predictions` | `dict[str, np.ndarray]` | Predictions per target |
| `plot_ids` | `np.ndarray` | Plot identifiers |
| `latent` | `np.ndarray \| None` | Latent representations (if requested) |

## Methods

### `__getitem__(target)`

Get predictions for a specific target.

```python
area_preds = predictions["area"]
```

### `to_dataframe()`

Convert predictions to pandas DataFrame.

**Returns:** `pd.DataFrame`

```python
df = predictions.to_dataframe()
```

### `to_csv(path)`

Save predictions to CSV file.

```python
predictions.to_csv("predictions.csv")
```
