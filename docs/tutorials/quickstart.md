# Quick Start

This guide walks through a complete RESOLVE workflow: loading data, training a
model, and making predictions.

## Installation

See [Installation](installation.md); the engine builds from source against your
installed PyTorch.

## Data Requirements

RESOLVE expects two data sources:

1. **Header file**: One row per plot with plot-level attributes
2. **Species file**: One row per species occurrence (plot x species)

A single long table also works, with the targets carried inline; see
`ResolveDataset.from_species_csv`.

## Example Workflow

=== "Python"

    ```python
    import resolve_core as rc

    # 1. Map your column names onto RESOLVE's semantic roles
    roles = rc.RoleMapping()
    roles.plot_id     = "PlotObservationID"
    roles.species_id  = "Species"
    roles.latitude    = "Latitude"
    roles.longitude   = "Longitude"
    roles.abundance   = "Cover"      # optional
    roles.genus       = "Genus"      # optional
    roles.family      = "Family"     # optional
    roles.covariates  = ["Elevation"]

    # 2. Declare the targets
    targets = [
        rc.TargetSpec.regression("Area", rc.TransformType.Log1p),
        rc.TargetSpec.classification("Habitat", 5),
    ]

    # 3. Choose how the species set is encoded
    data_config = rc.DatasetConfig()
    data_config.species_encoding = rc.SpeciesEncodingMode.RankPool
    data_config.pool_weighting   = rc.PoolWeighting.Log1p

    # 4. Load
    dataset = rc.ResolveDataset.from_csv(
        "plots.csv", "species.csv", roles, targets, data_config,
    )
    print(f"Plots: {dataset.schema.n_plots}")
    print(f"Species vocabulary: {dataset.schema.n_species_vocab}")

    # 5. Build the model from the dataset's schema
    model_config = rc.ModelConfig()
    model_config.species_encoding = rc.SpeciesEncodingMode.RankPool
    model_config.hidden_dims      = [256, 128]
    model = rc.ResolveModel(dataset.schema, model_config)

    # 6. Train
    train_config = rc.TrainConfig()
    train_config.max_epochs = 200
    train_config.patience   = 30
    trainer = rc.Trainer(model, train_config)
    trainer.prepare_data(dataset, test_size=0.2, seed=42)
    result = trainer.fit()
    print(result.final_metrics)

    # 7. Save
    trainer.save("model.pt")

    # 8. Predict
    predictor  = rc.Predictor.load("model.pt")
    predictions = predictor.predict_dataset(dataset)
    print(predictions.predictions["Area"][:5])
    ```

=== "R"

    ```r
    library(resolve)

    dataset <- resolve.dataset.csv(
      header  = "plots.csv",
      species = "species.csv",
      roles   = list(
        plot_id    = "PlotObservationID",
        species_id = "Species",
        latitude   = "Latitude",
        longitude  = "Longitude",
        abundance  = "Cover",
        genus      = "Genus",
        family     = "Family"
      ),
      targets = list(
        Area    = list(column = "Area",    task = "regression",     transform = "log1p"),
        Habitat = list(column = "Habitat", task = "classification", num_classes = 5L)
      ),
      config  = list(species_encoding = "rank_pool", pool_weighting = "log1p")
    )

    trainer <- resolve.train.dataset(dataset, maxEpochs = 200L, patience = 30L)
    resolve.save(trainer, "model.pt")

    predictor <- resolve.load("model.pt")
    preds     <- resolve.predict.dataset(predictor, dataset)
    ```

=== "Command line"

    ```bash
    resolve train \
      --header plots.csv --species species.csv \
      --plot-id PlotObservationID --species-id Species \
      --lat Latitude --lon Longitude --abundance Cover \
      --genus Genus --family Family \
      --target Area:regression --target Habitat:classification:5 \
      --encoding rank_pool --pool-weighting log1p \
      --output model.pt

    resolve predict \
      --model model.pt --header plots.csv --species species.csv \
      --plot-id PlotObservationID --species-id Species \
      --output predictions.csv
    ```

## Role Mapping

`RoleMapping` maps your column names onto RESOLVE's semantic roles:

| Role | Required | Description |
|------|----------|-------------|
| `plot_id` | Yes | Plot identifier, present in both files |
| `species_id` | Yes | Species identifier in the species file |
| `latitude` | No | Latitude coordinate |
| `longitude` | No | Longitude coordinate |
| `abundance` | No | Species abundance / cover value |
| `genus` | No | Genus name for taxonomy embeddings |
| `family` | No | Family name for taxonomy embeddings |
| `covariates` | No | Numeric predictor columns |
| `categoricals` | No | String predictor columns, factorized on load |

## Target Configuration

Targets are built with the `TargetSpec` constructors:

```python
rc.TargetSpec.regression("Area")                            # untransformed
rc.TargetSpec.regression("Area", rc.TransformType.Log1p)    # log1p target
rc.TargetSpec.classification("Habitat", 5)                  # 5 classes
rc.TargetSpec.classification_with_mapping("Habitat", {"M": 0, "N": 1})
```

A classification column of strings is factorized alphabetically when no explicit
mapping is given, and the class names are recorded on the schema.

## Next Steps

- [Data Preparation](data-preparation.md): Detailed data formatting guide
- [Training Models](training.md): Advanced training options
- [Understanding Embeddings](embeddings.md): Interpreting learned representations
