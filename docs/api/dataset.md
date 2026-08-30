# Dataset

Everything in this page lives in `resolve_core`.

```python
import resolve_core as rc
```

---

## RoleMapping

Maps your column names onto RESOLVE's semantic roles.

```python
roles = rc.RoleMapping()
```

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `plot_id` | `str` | Yes | Plot identifier, present in both tables |
| `species_id` | `str` | Yes | Species identifier in the species table |
| `abundance` | `str` | No | Abundance / cover column in the species table |
| `longitude` | `str` | No | Longitude column in the header table |
| `latitude` | `str` | No | Latitude column in the header table |
| `genus` | `str` | No | Genus column in the species table |
| `family` | `str` | No | Family column in the species table |
| `covariates` | `list[str]` | No | Numeric header columns, standardized at fit time |
| `categoricals` | `list[str]` | No | String header columns, factorized at load time |

`covariates` and `categoricals` must be disjoint.

**Methods**

| Method | Returns |
|--------|---------|
| `has_coordinates()` | `bool`, true when both `longitude` and `latitude` are set |
| `has_taxonomy()` | `bool`, true when `genus` or `family` is set |
| `has_abundance()` | `bool` |
| `has_categoricals()` | `bool` |

---

## TargetSpec

Declares one prediction target and how its column is read.

```python
rc.TargetSpec.regression("area")                            # untransformed
rc.TargetSpec.regression("area", rc.TransformType.Log1p)    # log1p on load
rc.TargetSpec.classification("habitat", 3)
rc.TargetSpec.classification_with_mapping("habitat", {"forest": 0, "grassland": 1})
```

| Static method | Signature |
|---------------|-----------|
| `regression` | `regression(column, transform=TransformType.None_)` |
| `classification` | `classification(column, num_classes)` |
| `classification_with_mapping` | `classification_with_mapping(column, mapping)` |

| Attribute | Type | Description |
|-----------|------|-------------|
| `column_name` | `str` | Column read from the header table |
| `target_name` | `str` | Name the target is reported under |
| `task` | `TaskType` | `Regression` or `Classification` |
| `transform` | `TransformType` | `None_` or `Log1p` |
| `num_classes` | `int` | Classification only |
| `weight` | `float` | Weight of this target in the multi-task loss |
| `class_mapping` | `dict[str, int]` | Explicit label to code map; empty means auto-fit alphabetically |

---

## DatasetConfig

Decides how the species set becomes model input.

```python
config = rc.DatasetConfig()
```

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `species_encoding` | `SpeciesEncodingMode` | `Hash` | `Hash`, `Embed`, `Sparse`, `RankPool`, `Transformer` |
| `hash_dim` | `int` | `32` | Width of the hashed vector |
| `top_k` | `int` | `3` | Species kept per plot before hashing, and the number of fixed taxonomy slots |
| `top_k_species` | `int` | `10` | Species IDs per plot in embed mode |
| `selection` | `SelectionMode` | `Top` | `Top`, `Bottom`, `TopBottom`, `All`. Which species survive a plot's budget |
| `species_budget` | `int` | `0` | Species kept per plot by `selection` under `RankPool` / `Transformer` / `Sparse`; `0` = no budget, encode every species |
| `representation` | `RepresentationMode` | `Abundance` | `Abundance` or `PresenceAbsence`, used by sparse mode |
| `normalization` | `NormalizationMode` | `Raw` | `Raw`, `Norm`, `Log1p`, applied before hashing |
| `aggregation` | `AggregationMode` | `Abundance` | `Abundance` or `Count` |
| `track_unknown_fraction` | `bool` | `True` | Add an input column holding each plot's share of abundance from species outside the vocabulary |
| `track_unknown_count` | `bool` | `False` | Add an input column holding each plot's number of records naming a species outside the vocabulary |
| `use_taxonomy` | `bool` | `True` | Use genus and family when present |
| `use_cuda_hash` | `bool` | `False` | Compute hash embeddings on the GPU per batch |
| `pool_weighting` | `PoolWeighting` | `Log1p` | `Binary`, `Abundance`, `Log1p`, `Norm`, `Rank`, used by pool modes |
| `pool_species_cap` | `int` | `0` | `0` no cap, `-1` auto p99, `>0` manual cap |

!!! note "Unknown-mass columns"
    Both columns are measured against the species vocabulary the dataset
    encodes with, over each plot's full record list (before any top-k
    selection or pool cap). `from_csv` fits that vocabulary from the file it
    is reading, which covers every name in it, so a plain training load reads
    0.0 for every plot. The columns become informative on the vocabulary-reusing
    loaders (`from_csv_with_schema`, `from_*_with_vocabs`), which encode new
    data against a trained checkpoint's vocabulary: there, a species the
    checkpoint never saw counts toward both.

---

## ResolveDataset

The loaded, encoded dataset. Built with one of the static loaders; there is no
public constructor.

### Loading from files

```python
dataset = rc.ResolveDataset.from_csv(
    "plots.csv", "species.csv", roles, targets, config,
)
```

| Loader | Signature |
|--------|-----------|
| `from_csv` | `(header_path, species_path, roles, targets, config=DatasetConfig())` |
| `from_species_csv` | `(species_path, roles, targets, config=DatasetConfig())` |
| `from_csv_with_schema` | `(header_path, species_path, roles, targets, schema_source, config=DatasetConfig())` |
| `from_csv_with_vocabs` | `(header_path, species_path, roles, targets, vocabs, config=DatasetConfig())` |
| `from_species_csv_with_schema` | `(species_path, roles, targets, schema, config=DatasetConfig())` |
| `from_species_csv_with_vocabs` | `(species_path, roles, targets, vocabs, config=DatasetConfig())` |

`from_csv_with_schema` takes either a `ResolveDataset` or a `ResolveSchema` as
its vocabulary source; the schema overload lets a checkpoint stand in for
training CSVs that no longer exist. `from_csv_with_vocabs` takes an
`ExternalVocabs`, which is the complete form because the categorical maps live
on the `Predictor` rather than the schema.

`from_species_csv` reads plot-level values from each plot's first occurrence and
carries no `covariates` or `categoricals`.

### Loading from DataFrames

```python
dataset = rc.ResolveDataset.from_pandas(header_df, species_df, roles, targets)
```

```
from_pandas(header, species=None, roles=None, targets=None,
            config=None, schema_source=None)
```

| `species` | Behaviour |
|-----------|-----------|
| `DataFrame` | Both tables in memory |
| `str` | Header in memory, species table streamed from that CSV path |
| `None` | Single-table mode; `header` is the long-format species frame |

`from_dataframe` is an alias. Cells are stringified the way `DataFrame.to_csv`
writes them, so the result equals `from_csv` on the CSV that frame would
serialize to.

Lower-level column entry points take `(names, columns)` string tables directly
and release the GIL around the build: `from_columns`, `from_columns_header`,
`from_species_columns`, `from_columns_with_schema`, `from_columns_with_vocabs`,
`from_species_columns_with_vocabs`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `schema` | `ResolveSchema` | Structure of the loaded dataset |
| `config` | `DatasetConfig` | The config the loader ran with |
| `n_plots` | `int` | Number of plots after target-driven drops |
| `plot_ids` | `list[str]` | Plot identifiers, in row order |
| `species_vocab` | `list[str]` | Ordered species vocabulary; index 0 is `"<UNK>"` |
| `targets` | `dict[str, Tensor]` | Target tensors, in the space the loader stored them |
| `categorical_vocab` | `CategoricalVocab` | Fitted string-to-code maps |
| `taxonomy_vocab` | `TaxonomyVocab` | Fitted genus and family maps |

### Tensor accessors

Every accessor returns a torch tensor, or `None` when that encoding does not
produce it.

| Accessor | Shape | Populated for |
|----------|-------|---------------|
| `coordinates` | `(n_plots, 2)` | Coordinates declared |
| `covariates` | `(n_plots, n_covariates)` | Covariates declared |
| `categorical_ids` | `(n_plots, n_categoricals)` | Categoricals declared |
| `hash_embedding` | `(n_plots, hash_dim)` | `Hash`, with `use_cuda_hash=False` |
| `species_ids` | `(n_plots, top_k_species)` or `(n_plots, max_species)` | `Embed`, `RankPool`, `Transformer` |
| `species_vector` | `(n_plots, n_species_vocab)` | `Sparse` |
| `genus_ids`, `family_ids` | `(n_plots, n_taxonomy_slots)` | `Hash`, `Sparse`, `Embed`, with taxonomy |
| `pool_genus_ids`, `pool_family_ids` | `(n_plots, max_species)` | `RankPool`, `Transformer` |
| `pool_weights` | `(n_plots, max_species)` | `RankPool`, `Transformer` |
| `pool_mask` | `(n_plots, max_species)` | `RankPool`, `Transformer` |
| `pool_has_cover` | `(n_plots,)` | `RankPool`, `Transformer` |
| `unknown_fraction`, `unknown_count` | `(n_plots,)` | Tracking enabled |
| `raw_species_ids`, `raw_weights`, `plot_offsets` | flat COO | `use_cuda_hash=True` |

### Methods

| Method | Returns |
|--------|---------|
| `has_pool_data()` | `bool`, true for the pool encoders |
| `has_raw_species_data()` | `bool`, true under `use_cuda_hash` |
| `external_vocabs()` | `ExternalVocabs` carrying every vocabulary this dataset fitted |

---

## ResolveSchema

Describes a loaded dataset and travels into the checkpoint.

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_plots` | `int` | Plots after target-driven drops |
| `n_species` | `int` | Distinct species, excluding the unknown slot |
| `n_species_vocab` | `int` | Species vocabulary size, including `"<UNK>"` |
| `n_genera`, `n_families` | `int` | Taxonomy vocabulary sizes |
| `n_genera_vocab`, `n_families_vocab` | `int` | Same, as embedded table heights |
| `covariate_names` | `list[str]` | Numeric covariate columns, in order |
| `categorical_names` | `list[str]` | Categorical columns, in order |
| `categorical_vocab_sizes` | `list[int]` | Code count per categorical column |
| `categorical_embed_dim` | `int` | Embedding width per categorical column |
| `targets` | `list[TargetConfig]` | One per declared target |
| `has_coordinates`, `has_abundance`, `has_taxonomy` | `bool` | What the data carried |
| `track_unknown_fraction`, `track_unknown_count` | `bool` | Loader settings |
| `species_vocab`, `genus_vocab`, `family_vocab` | `list[str]` | Fitted vocabularies, index equals code, `[0]` is `"<UNK>"` |
| `top_k_species`, `selection`, `species_budget`, `representation`, `normalization`, `aggregation`, `use_taxonomy` | | The `DatasetConfig` knobs the loader consumed. `selection` is the one the load APPLIED: a pooled or sparse dataset with no `species_budget` encodes every record, so it reports `All` |
| `pool_weighting`, `pool_species_cap` | | Pool settings, so inference recomputes the same weights |

| Method | Returns |
|--------|---------|
| `has_categoricals()` | `bool` |
| `n_categoricals()` | `int` |
| `has_species_vocab()` | `bool`, false for a checkpoint written before vocabularies travelled |
| `has_taxonomy_vocab()` | `bool` |

### TargetConfig

One entry of `ResolveSchema.targets`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Target name |
| `task` | `TaskType` | `Regression` or `Classification` |
| `transform` | `TransformType` | `None_` or `Log1p` |
| `num_classes` | `int` | Classification only |
| `weight` | `float` | Weight in the multi-task loss |
| `class_weights` | `list[float]` | Per-class cross-entropy weights; empty means unweighted |
| `class_names` | `list[str]` | Class vocabulary, index equals code; empty for a regression or already-integer target |

---

## CategoricalVocab

The string-to-code maps fitted for each categorical column. Codes run `1..K`,
with `0` reserved for unknown and missing.

| Member | Description |
|--------|-------------|
| `fit_column(column_name, raw_values)` | Fit one column |
| `encode(column_name, raw_value)` | Code for one value |
| `encode_batch(column_names, raw_values_per_column)` | Codes for several columns |
| `vocab_size(column_name)` | Code count, including the unknown slot |
| `has_column(column_name)` | `bool` |
| `column_map(column_name)` | `dict[str, int]` |
| `column_names` | `list[str]` |
| `vocab_sizes` | `list[int]` |

---

## ExternalVocabs

Every vocabulary fitted at training time, in the form the `*_with_vocabs`
loaders take.

| Attribute | Type |
|-----------|------|
| `species_vocab` | `list[str]` |
| `taxonomy` | `TaxonomyVocab` |
| `categorical` | `CategoricalVocab` |
| `targets` | `list[TargetConfig]` |

Get one from `predictor.external_vocabs` (complete, including the categorical
maps) or `dataset.external_vocabs()`.

```python
rc.external_vocabs_from_schema(schema)
```

rebuilds what a schema carries. The categorical maps are not on the schema, so
take them from `Predictor.categorical_vocab` or use
`Predictor.external_vocabs`, which folds them in.

```python
rc.dataset_config_from_checkpoint(schema, model_config)
```

reassembles the loading-side `DatasetConfig` a checkpoint was built with.
`species_encoding`, `hash_dim`, and `top_k` come from the `ModelConfig` because
they size the model; everything else the loader consumed comes from the schema.
`use_cuda_hash` is deliberately not restored, being a training-time compute
path.

---

## Enums

| Enum | Values |
|------|--------|
| `TaskType` | `Regression`, `Classification` |
| `TransformType` | `None_`, `Log1p` |
| `SpeciesEncodingMode` | `Hash`, `Embed`, `Sparse`, `RankPool`, `Transformer` |
| `SelectionMode` | `Top`, `Bottom`, `TopBottom`, `All` |
| `RepresentationMode` | `Abundance`, `PresenceAbsence` |
| `NormalizationMode` | `Raw`, `Norm`, `Log1p` |
| `AggregationMode` | `Abundance`, `Count` |
| `PoolWeighting` | `Binary`, `Abundance`, `Log1p`, `Norm`, `Rank` |

---

## See also

- [Data Preparation](../tutorials/data-preparation.md)
- [Encoding Modes](../tutorials/encoding-modes.md)
- [Model](model.md)
