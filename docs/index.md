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

```python
import resolve

# Define semantic roles (map your column names)
roles = {
    "plot_id": "PlotObservationID",
    "species_id": "Species",
    "species_plot_id": "PlotObservationID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover",
    "taxonomy_genus": "Genus",
    "taxonomy_family": "Family",
}

# Define targets
targets = {
    "area": {"column": "Area", "task": "regression", "transform": "log1p"},
    "habitat": {"column": "Habitat", "task": "classification", "num_classes": 5},
}

# Load data
dataset = resolve.ResolveDataset.from_csv(
    header="plots.csv",
    species="species.csv",
    roles=roles,
    targets=targets,
)

# Build and train model
model = resolve.ResolveModel(schema=dataset.schema, targets=targets)
trainer = resolve.Trainer(model, dataset)
result = trainer.fit()
trainer.save("model.pt")

# Predict
predictor = resolve.Predictor.load("model.pt")
predictions = predictor.predict(new_dataset)
```

## Installation

```bash
pip install resolve
```

Or from source:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve
pip install -e .
```

## License

MIT License - see [LICENSE](https://github.com/gcol33/resolve/blob/main/LICENSE.md) for details.
