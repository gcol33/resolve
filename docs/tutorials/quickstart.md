# Quick Start

This guide walks through a complete RESOLVE workflow: loading data, training a model, and making predictions.

## Installation

=== "Python"

    ```bash
    pip install resolve
    ```

=== "R"

    ```r
    # Install from GitHub (CRAN submission pending)
    remotes::install_github("gcol33/resolve", subdir = "r")
    ```

## Data Requirements

RESOLVE expects two data sources:

1. **Header file**: One row per plot with plot-level attributes
2. **Species file**: One row per species occurrence (plot × species)

## Example Workflow

=== "Python"

    ```python
    import resolve

    # 1. Define semantic roles (map your column names)
    roles = {
        "plot_id": "PlotObservationID",
        "species_id": "Species",
        "species_plot_id": "PlotObservationID",
        "coords_lat": "Latitude",
        "coords_lon": "Longitude",
        "abundance": "Cover",           # optional
        "taxonomy_genus": "Genus",      # optional
        "taxonomy_family": "Family",    # optional
    }

    # 2. Define targets
    targets = {
        "area": {
            "column": "Area",
            "task": "regression",
            "transform": "log1p",
        },
        "habitat": {
            "column": "Habitat",
            "task": "classification",
            "num_classes": 5,
        },
    }

    # 3. Load data
    dataset = resolve.ResolveDataset.from_csv(
        header="plots.csv",
        species="species.csv",
        roles=roles,
        targets=targets,
        species_normalization="relative",
    )

    # 4. Check schema
    print(f"Plots: {dataset.schema.n_plots}")
    print(f"Species: {dataset.schema.n_species}")

    # 5. Train (model built automatically from dataset)
    trainer = resolve.Trainer(
        dataset,
        species_encoding="hash",
        hash_dim=32,
        max_epochs=200,
        patience=30,
    )
    result = trainer.fit()

    # 6. Save model
    trainer.save("model.pt")

    # 7. Predict with confidence filtering
    predictions = trainer.predict(new_dataset, confidence_threshold=0.8)
    ```

=== "R"

    ```r
    library(resolve)

    # 1. Load and prepare data
    header <- read.csv("plots.csv")
    species <- read.csv("species.csv")

    # 2. Create encoder and fit on species data
    encoder <- resolve.encoder(hashDim = 32L, topK = 5L)
    encoder$fit(species)

    # 3. Define schema
    schema <- list(
      nPlots = nrow(header),
      nSpecies = encoder$n_species(),
      hasCoordinates = TRUE,
      hasTaxonomy = TRUE,
      targets = list(
        list(name = "area", task = "regression", transform = "log1p"),
        list(name = "habitat", task = "classification", numClasses = 5L)
      )
    )

    # 4. Create model and trainer
    model_config <- list(
      speciesEncoding = "hash",
      hashDim = 32L,
      hiddenDims = c(128L, 64L)
    )

    train_config <- list(
      batchSize = 64L,
      maxEpochs = 200L,
      patience = 30L,
      lr = 0.001
    )

    model <- new(.resolve_module$ResolveModel, schema, model_config)
    trainer <- new(.resolve_module$Trainer, model, train_config)

    # 5. Train
    # trainer$fit(train_data)

    # 6. Save model
    resolve.save(trainer, "model.pt")

    # 7. Predict
    # predictions <- resolve.predict(trainer, new_data)
    ```

## Role Mapping

The `roles` dictionary maps your column names to RESOLVE's semantic roles:

| Role | Required | Description |
|------|----------|-------------|
| `plot_id` | Yes | Unique plot identifier in header |
| `species_id` | Yes | Species identifier in species file |
| `species_plot_id` | Yes | Plot identifier in species file (for joining) |
| `coords_lat` | No | Latitude coordinate |
| `coords_lon` | No | Longitude coordinate |
| `abundance` | No | Species abundance/cover value |
| `taxonomy_genus` | No | Genus name for taxonomy embeddings |
| `taxonomy_family` | No | Family name for taxonomy embeddings |
| `covariates` | No | List of additional predictor columns |

## Target Configuration

Each target specifies:

- `column`: Column name in header file
- `task`: `"regression"` or `"classification"`
- `transform`: Optional transform (`"log1p"` for regression)
- `num_classes`: Required for classification tasks
- `weight`: Optional loss weighting (default: 1.0)

## Next Steps

- [Data Preparation](data-preparation.md): Detailed data formatting guide
- [Training Models](training.md): Advanced training options
- [Understanding Embeddings](embeddings.md): Interpreting learned representations
