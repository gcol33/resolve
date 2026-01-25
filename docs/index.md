# RESOLVE

**Representation Encoding of Species Outcomes via Linear Vector Embeddings**

An opinionated torch-based package for predicting plot-level attributes from species composition, environment, and space.

## Overview

RESOLVE treats species composition as *biotic context* — a rich, structured signal that encodes information about plot attributes. Rather than predicting species from environment (as in SDMs), RESOLVE predicts plot properties from species.

**Core claim**: Species composition encodes a shared latent representation that simultaneously informs multiple plot attributes (area, elevation, climate, habitat class).

## Key Features

- **Hybrid species encoding**: Feature hashing for full species lists + learned embeddings for dominant taxa
- **Multi-target prediction**: Single shared encoder, multiple task heads
- **Phased training**: MAE → SMAPE → band accuracy optimization
- **Semantic role mapping**: Flexible column naming, strict structure
- **Unknown species tracking**: Detects and quantifies novel species at inference time
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
    import resolve

    # Load data with semantic role mapping
    dataset = resolve.ResolveDataset.from_csv(
        header="plots.csv",
        species="species.csv",
        roles={
            "plot_id": "PlotObservationID",
            "species_id": "Species",
            "species_plot_id": "PlotObservationID",
        },
        targets={
            "area": {"column": "Area", "task": "regression", "transform": "log1p"},
            "habitat": {"column": "Habitat", "task": "classification", "num_classes": 5},
        },
    )

    # Train (model built automatically)
    trainer = resolve.Trainer(dataset)
    trainer.fit()
    trainer.save("model.pt")

    # Predict with confidence filtering
    predictions = trainer.predict(new_dataset, confidence_threshold=0.8)
    ```

=== "R"

    ```r
    library(resolve)

    # Create and fit encoder
    encoder <- resolve.encoder(hashDim = 32L)
    encoder$fit(species_data)

    # Configure and train
    schema <- list(nPlots = 100, nSpecies = 50, ...)
    model <- new(.resolve_module$ResolveModel, schema, model_config)
    trainer <- new(.resolve_module$Trainer, model, train_config)

    # Save and predict
    resolve.save(trainer, "model.pt")
    ```

## Installation

=== "Python"

    ```bash
    pip install resolve
    ```

    Or from source:

    ```bash
    git clone https://github.com/gcol33/resolve.git
    cd resolve
    pip install -e .
    ```

=== "R"

    ```r
    # Install from GitHub (CRAN submission pending)
    remotes::install_github("gcol33/resolve", subdir = "r")
    ```

## License

MIT License - see [LICENSE](https://github.com/gcol33/resolve/blob/main/LICENSE.md) for details.
