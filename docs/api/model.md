# ResolveModel

The `ResolveModel` class defines the neural network architecture.

## Class Definition

```python
class ResolveModel(nn.Module):
    """
    RESOLVE neural network model.

    Combines species encoding, shared encoder, and task-specific heads
    for multi-target prediction from species composition data.
    """
```

## Constructor

### `ResolveModel(schema, targets, **kwargs)`

Create a RESOLVE model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `ResolveSchema` | required | Dataset schema |
| `targets` | `dict` | required | Target configurations |
| `hash_dim` | `int` | `32` | Feature hashing dimension |
| `hidden_dims` | `list[int]` | `[256, 128, 64]` | Hidden layer sizes |
| `dropout` | `float` | `0.1` | Dropout rate |
| `track_unknown_count` | `bool` | `False` | Include unknown species count |

**Example:**

```python
model = ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[256, 128, 64],
    hash_dim=32,
)
```

## Properties

### `latent_dim`

Returns the dimension of the latent representation.

```python
print(f"Latent dim: {model.latent_dim}")
```

### `target_configs`

Returns target configuration dictionary.

```python
configs = model.target_configs
```

## Methods

### `forward(continuous, genus_ids, family_ids)`

Forward pass through the model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `continuous` | `Tensor` | Continuous features `(batch, n_features)` |
| `genus_ids` | `Tensor \| None` | Genus indices `(batch, top_k)` |
| `family_ids` | `Tensor \| None` | Family indices `(batch, top_k)` |

**Returns:** `dict[str, Tensor]` - Predictions for each target

### `get_latent(continuous, genus_ids, family_ids)`

Get latent representations without task head predictions.

**Returns:** `Tensor` of shape `(batch, latent_dim)`

## Architecture

```
Input Features:
├── Coordinates (2)        ─┐
├── Covariates (n)          │
├── Hash embedding (32)     ├──→ Concatenate ──→ PlotEncoder ──→ Latent
├── Unknown fraction (1)    │                          │
└── Unknown count (1)*     ─┘                          │
                                                       ↓
Taxonomy Features:                              ┌─────────────────┐
├── Genus embeddings       ──→ Aggregate ──→    │   Task Heads    │
└── Family embeddings                           └─────────────────┘
                                                       │
                                               ┌───────┴───────┐
                                               ↓               ↓
                                          Regression    Classification
```

*If `track_unknown_count=True`
