# Spacc

**Species Presence and Abundance for Coordinate-linked Characteristics**

An opinionated torch-based package for predicting plot-level attributes from species composition, environment, and space.

## Overview

Spacc treats species composition as *biotic context* — a rich, structured signal that encodes information about plot attributes. Rather than predicting species from environment (as in SDMs), Spacc predicts plot properties from species.

**Core claim**: Species composition encodes a shared latent representation that simultaneously informs multiple plot attributes (area, elevation, climate, habitat class).

## Installation

```bash
pip install spacc
```

Or from source:

```bash
git clone https://github.com/gcol33/spacc.git
cd spacc
pip install -e .
```

## Quick Start

```python
import spacc

# Define semantic roles (map your column names)
roles = {
    "plot_id": "PlotObservationID",
    "species_id": "Species",
    "species_plot_id": "PlotObservationID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover",  # optional
    "taxonomy_genus": "Genus",  # optional
    "taxonomy_family": "Family",  # optional
    "covariates": ["Temperature", "Precipitation"],
}

# Define targets
targets = {
    "area": {"column": "Area", "task": "regression", "transform": "log1p"},
    "elevation": {"column": "Elevation", "task": "regression"},
    "habitat": {"column": "Habitat", "task": "classification", "num_classes": 5},
}

# Load data
dataset = spacc.SpaccDataset.from_csv(
    header="plots.csv",
    species="species.csv",
    roles=roles,
    targets=targets,
)

# Build and train model
model = spacc.SpaccModel(schema=dataset.schema, targets=targets)
trainer = spacc.Trainer(model, dataset)
result = trainer.fit()
trainer.save("model.pt")

# Predict
predictor = spacc.Predictor.load("model.pt")
predictions = predictor.predict(new_dataset)
predictions.to_csv("predictions.csv")
```

## Features

- **Hybrid species encoding**: Feature hashing for full species lists + learned embeddings for dominant taxa
- **Multi-target prediction**: Single shared encoder, multiple task heads
- **Phased training**: MAE → SMAPE → band accuracy optimization
- **Semantic role mapping**: Flexible column naming, strict structure
- **CPU-first**: Works without GPU, scales with CUDA when available

## Architecture

```
Species data ─────┐
                  ├──→ SpeciesEncoder ──→ hash embedding + taxonomy IDs
Coordinates ──────┤
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

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- pandas ≥ 2.0
- scikit-learn ≥ 1.3

## License

MIT
