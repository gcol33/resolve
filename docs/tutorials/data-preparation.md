# Data Preparation

This guide covers how to shape your data for RESOLVE and what the loader does
with it.

## Data structure

RESOLVE reads two tables.

### Header table

One row per plot, carrying targets, coordinates, and covariates:

```csv
plot_id,latitude,longitude,area,elevation,bedrock,habitat
P001,45.2,6.1,100,1200,limestone,forest
P002,46.1,7.2,250,800,granite,grassland
P003,44.8,5.9,50,1500,limestone,forest
```

### Species table

One row per species occurrence:

```csv
plot_id,species,genus,family,cover
P001,Quercus robur,Quercus,Fagaceae,25
P001,Fagus sylvatica,Fagus,Fagaceae,40
P001,Pinus sylvestris,Pinus,Pinaceae,10
P002,Festuca rubra,Festuca,Poaceae,60
P002,Trifolium repens,Trifolium,Fabaceae,15
```

The two are joined on the plot ID. Species rows whose plot is absent from the
header are dropped, so filtering the header is enough to subset a dataset.

### One long table

When the targets travel with the species rows, a single table works:

```python
dataset = rc.ResolveDataset.from_species_csv("survey.csv", roles, targets, config)
```

The single-table loader reads plot-level values from each plot's first
occurrence. It carries no `covariates` or `categoricals` and warns when either
is declared; use the two-table loader for plot-level predictors.

## Required fields

At minimum:

1. A **plot identifier**, present in both tables
2. A **species identifier** in the species table
3. At least one **target column** in the header table

## Roles

`RoleMapping` maps your column names onto RESOLVE's semantic roles:

=== "Python"

    ```python
    import resolve_core as rc

    roles = rc.RoleMapping()
    roles.plot_id      = "plot_id"
    roles.species_id   = "species"
    roles.abundance    = "cover"
    roles.latitude     = "latitude"
    roles.longitude    = "longitude"
    roles.genus        = "genus"
    roles.family       = "family"
    roles.covariates   = ["elevation"]
    roles.categoricals = ["bedrock"]
    ```

=== "R"

    ```r
    roles <- list(
      plot_id      = "plot_id",
      species_id   = "species",
      abundance    = "cover",
      latitude     = "latitude",
      longitude    = "longitude",
      genus        = "genus",
      family       = "family",
      covariates   = list("elevation"),
      categoricals = list("bedrock")
    )
    ```

    R takes roles, targets, and config as named lists whose keys are the
    snake_case names above. R function arguments (`maxEpochs`, `testSize`) are
    camelCase.

| Role | Required | Effect when present |
|------|----------|---------------------|
| `plot_id` | Yes | Joins the two tables and labels every prediction |
| `species_id` | Yes | The entity whose set forms the composition |
| `abundance` | No | Weights each species; absent means presence at weight 1.0 |
| `latitude` / `longitude` | No | Two continuous inputs, and the key for spatial block cross-validation |
| `genus` / `family` | No | Taxonomy embedding tables, giving rare and unseen species a fallback identity |
| `covariates` | No | Numeric header columns, standardized at fit time |
| `categoricals` | No | String header columns, factorized on load into their own embedding tables |

`covariates` and `categoricals` have to be disjoint; a column in both is
rejected.

## Targets

Targets are declared separately from roles, because each one carries a task and
a transform:

=== "Python"

    ```python
    targets = [
        rc.TargetSpec.regression("area"),                          # untransformed
        rc.TargetSpec.regression("area", rc.TransformType.Log1p),  # log1p
        rc.TargetSpec.classification("habitat", 3),
        rc.TargetSpec.classification_with_mapping(
            "habitat", {"forest": 0, "grassland": 1, "wetland": 2}
        ),
    ]
    ```

=== "R"

    ```r
    targets <- list(
      area    = list(column = "area", task = "regression", transform = "log1p"),
      habitat = list(column = "habitat", task = "classification", num_classes = 3L)
    )
    ```

    The list is keyed by target name; `column` names the column it reads. Drop
    `column` and the key names the column directly.

A classification column of strings is factorized alphabetically when no mapping
is given, and the resulting vocabulary lands on `schema.targets[i].class_names`
so predictions can be turned back into labels. `Log1p` is applied on load and
inverted before predictions and regression metrics are reported, so everything
you read is in the original units.

## What the loader does with missing values

| Situation | Result |
|-----------|--------|
| Missing or unparseable **target** value | The plot is dropped, and the count is reported |
| Classification label outside an explicit `class_mapping` | The plot is dropped, and the count is reported |
| Missing **covariate** cell | Coerced to `0.0`, the plot is kept, a per-column warning names the count |
| Missing **coordinate** | Coerced to `(0, 0)`, the plot is kept, a warning names the count |
| Missing **categorical** cell | Encoded as code `0`, the reserved unknown slot |
| Missing **abundance** cell | Weight `1.0`, with a counted warning |

Covariate coercion to `0.0` puts a real, extreme value into the standardization,
so treat that warning as an instruction to handle the column upstream:

```python
import pandas as pd

header = pd.read_csv("plots.csv")
header = header.dropna(subset=["elevation"])
```

Species records belonging to dropped plots are removed before the species and
taxonomy vocabularies are fitted, so a dropped plot cannot leave unreferenced
rows in an embedding table.

## Loading from DataFrames

`from_pandas` builds a dataset from frames already in memory, which avoids a
write-to-CSV and read-back round trip whenever the header is filtered per fit:

```python
import pandas as pd

header = pd.read_csv("plots.csv").query("elevation > 500")
species = pd.read_csv("species.csv")

dataset = rc.ResolveDataset.from_pandas(header, species, roles, targets, config=cfg)
```

Cells are stringified the way `DataFrame.to_csv` writes them, and the engine
runs the identical loader body, so the result equals `from_csv` on the CSV that
frame would serialize to.

Passing a path for `species` keeps the large table on disk and streams it once
while the header stays in memory:

```python
dataset = rc.ResolveDataset.from_pandas(header, "species.csv", roles, targets)
```

R has the same verb:

```r
dataset <- resolve.dataset.frame(header = header_df, species = species_df,
                                 roles = roles, targets = targets)
```

## Checking what was loaded

The schema describes the dataset the loader actually produced:

```python
schema = dataset.schema

print(f"Plots:      {schema.n_plots}")
print(f"Species:    {schema.n_species} ({schema.n_species_vocab} vocabulary slots)")
print(f"Genera:     {schema.n_genera}")
print(f"Families:   {schema.n_families}")
print(f"Covariates: {schema.covariate_names}")
print(f"Categoricals: {schema.categorical_names} sizes {schema.categorical_vocab_sizes}")

for target in schema.targets:
    print(f"{target.name}: {target.task}, classes={target.class_names}")
```

`n_plots` is the count after target-driven drops, so comparing it against the
row count of your header file tells you how many plots the loader removed.
Vocabulary counts include the reserved `<UNK>` slot at index 0.

## Reusing a training vocabulary

Every non-hash encoder indexes an embedding table by an integer code fitted from
the data it was built on. New data therefore has to be encoded against the
training vocabulary, or the same species name lands on a different row:

```python
test_dataset = rc.ResolveDataset.from_csv_with_schema(
    "test_plots.csv", "test_species.csv", roles, targets, train_dataset,
)
```

From a checkpoint alone, without the training CSVs:

```python
predictor = rc.Predictor.load("model.pt")

test_dataset = rc.ResolveDataset.from_csv_with_vocabs(
    "test_plots.csv", "test_species.csv", roles, targets,
    predictor.external_vocabs, predictor.dataset_config,
)
```

`Predictor.predict_dataset` rejects a dataset built any other way. See
[Making Predictions](prediction.md) for the full inference path.

## Next steps

- [Encoding Modes](encoding-modes.md): how the species set becomes model input
- [Training Models](training.md): configuration and cross-validation
- [Quick Start](quickstart.md): the complete workflow in one page
