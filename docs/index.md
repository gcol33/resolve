# RESOLVE

**Representation Encoding for Structured Observation Learning with Vector Embeddings**

An opinionated torch-based package for predicting sample attributes from compositional data — sets of entities with optional abundances, covariates, and coordinates.

## Overview

RESOLVE treats compositional data as *contextual signal* — a rich, structured representation that encodes information about sample-level attributes. Given a set of entities (species in a plot, symptoms in a patient, products in a basket), RESOLVE learns to predict properties of the sample.

**Core idea**: Compositional data encodes a shared latent representation that simultaneously informs multiple sample attributes.

## Key Features

- **Multiple species encodings**: Feature hashing, learned embeddings, rank-pooling for variable-length lists, and transformer-based attention over species
- **Multi-target prediction**: Single shared encoder, multiple task heads
- **Phased training**: MAE → SMAPE → band accuracy optimization
- **Semantic role mapping**: Flexible column naming, strict structure
- **Categorical covariates**: String columns factorized on load into their own embedding tables
- **Vocabularies in the checkpoint**: New data is encoded against the codes the model trained on
- **Unknown species tracking**: Each sample carries the share of its abundance from species outside the training vocabulary, and how many there are, as encoder inputs
- **Abundance normalization**: Raw, relative (per-plot), or log-scaled modes
- **CPU-first**: Works without GPU, scales with CUDA when available

## Architecture

```
Species data ─────┐
                  ├──→ SpeciesEncoder ──→ hash embedding + taxonomy IDs
Coordinates ──────┤                       + unknown mass features
                  ├──→ PlotEncoder (shared) ──→ latent representation
Covariates ───────┘
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                              TaskHead(area)   TaskHead(elev)   TaskHead(habitat)
                                    │                 │                 │
                                    ↓                 ↓                 ↓
                              regression       regression       classification
```

## Quick Start

=== "Python"

    ```python
    import resolve_core as rc

    # Map your column names onto RESOLVE's semantic roles
    roles = rc.RoleMapping()
    roles.plot_id    = "PlotObservationID"
    roles.species_id = "Species"
    roles.abundance  = "Cover"

    dataset = rc.ResolveDataset.from_csv(
        "plots.csv",
        "species.csv",
        roles,
        [rc.TargetSpec.regression("Area", rc.TransformType.Log1p),
         rc.TargetSpec.classification("Habitat", 5)],
    )

    # Train
    model   = rc.ResolveModel(dataset.schema, rc.ModelConfig())
    trainer = rc.Trainer(model, rc.TrainConfig())
    trainer.prepare_data(dataset, test_size=0.2, seed=42)
    trainer.fit()
    trainer.save("model.pt")

    # Predict
    predictor  = rc.Predictor.load("model.pt")
    predictions = predictor.predict_dataset(new_dataset)
    ```

=== "R"

    ```r
    library(resolve)

    dataset <- resolve.dataset.csv(
      header  = "plots.csv",
      species = "species.csv",
      roles   = list(plot_id = "PlotObservationID", species_id = "Species",
                     abundance = "Cover"),
      targets = list(
        Area    = list(column = "Area", task = "regression", transform = "log1p"),
        Habitat = list(column = "Habitat", task = "classification", num_classes = 5L)
      )
    )

    trainer   <- resolve.train.dataset(dataset, maxEpochs = 200L)
    resolve.save(trainer, "model.pt")

    predictor <- resolve.load("model.pt")
    preds     <- resolve.predict.dataset(predictor, dataset)
    ```

## Installation

=== "Python"

    The engine and its bindings build from source with CMake and an installed
    PyTorch:

    ```bash
    git clone https://github.com/gcol33/resolve.git
    cd resolve/src/core/python
    pip install .
    ```

=== "R"

    ```r
    install.packages("pak")
    pak::pak("gcol33/resolve/r")
    ```

## License

MIT License - see [LICENSE](https://github.com/gcol33/resolve/blob/main/LICENSE.md) for details.
