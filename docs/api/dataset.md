# ResolveDataset

The `ResolveDataset` class handles data loading and preprocessing for RESOLVE.

## Class Definition

```python
class ResolveDataset:
    """
    Dataset container for RESOLVE models.

    Handles loading, validation, and preprocessing of plot-level
    and species composition data.
    """
```

## Constructor

### `ResolveDataset(header, species, roles, targets)`

Create a dataset from pandas DataFrames.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `header` | `pd.DataFrame` | Plot-level data (one row per plot) |
| `species` | `pd.DataFrame` | Species occurrences (one row per species-plot) |
| `roles` | `dict` | Column name mappings |
| `targets` | `dict` | Target variable configurations |

**Example:**

```python
dataset = ResolveDataset(
    header=header_df,
    species=species_df,
    roles=roles,
    targets=targets,
)
```

## Class Methods

### `from_csv(header, species, roles, targets)`

Load dataset from CSV files.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `header` | `str \| Path` | Path to header CSV file |
| `species` | `str \| Path` | Path to species CSV file |
| `roles` | `dict` | Column name mappings |
| `targets` | `dict` | Target variable configurations |

**Returns:** `ResolveDataset`

**Example:**

```python
dataset = ResolveDataset.from_csv(
    header="data/plots.csv",
    species="data/species.csv",
    roles=roles,
    targets=targets,
)
```

## Properties

### `schema`

Returns the `ResolveSchema` describing the dataset structure.

```python
schema = dataset.schema
print(f"Plots: {schema.n_plots}")
print(f"Species: {schema.n_species}")
```

### `plot_ids`

Returns array of plot identifiers.

```python
plot_ids = dataset.plot_ids
```

## Methods

### `get_coordinates()`

Returns coordinates array or `None` if not available.

```python
coords = dataset.get_coordinates()
# Shape: (n_plots, 2) or None
```

### `get_covariates()`

Returns covariates array or `None` if not available.

```python
covariates = dataset.get_covariates()
# Shape: (n_plots, n_covariates) or None
```

### `get_targets()`

Returns dictionary of target arrays.

```python
targets = dataset.get_targets()
# {"area": array(...), "habitat": array(...)}
```

---

# ResolveSchema

The `ResolveSchema` dataclass describes dataset structure.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_plots` | `int` | Number of plots |
| `n_species` | `int` | Number of unique species |
| `n_genera` | `int` | Number of unique genera |
| `n_families` | `int` | Number of unique families |
| `covariate_names` | `list[str]` | Names of covariate columns |
| `targets` | `dict` | Target configurations |
| `has_coordinates` | `bool` | Whether coordinates are available |
| `has_taxonomy` | `bool` | Whether taxonomy is available |
