# RESOLVE

[![Tests](https://github.com/gcol33/resolve/actions/workflows/tests.yml/badge.svg)](https://github.com/gcol33/resolve/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://gillescolling.com/resolve)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Representation Encoding of Species Outcomes via Linear Vector Embeddings**

An opinionated torch-based package for predicting plot-level attributes from species composition, environment, and space.

## Overview

RESOLVE treats species composition as *biotic context* — a rich, structured signal that encodes information about plot attributes. Rather than predicting species from environment (as in SDMs), RESOLVE predicts plot properties from species.

**Core claim**: Species composition encodes a shared latent representation that simultaneously informs multiple plot attributes (area, elevation, climate, habitat class).

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
predictions.to_csv("predictions.csv")
```

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid species encoding** | Feature hashing for full species lists + learned embeddings for dominant taxa |
| **Multi-target prediction** | Single shared encoder, multiple task heads (regression & classification) |
| **Phased training** | MAE → SMAPE → band accuracy optimization |
| **Semantic role mapping** | Flexible column naming, strict structure |
| **Unknown species tracking** | Detects and quantifies novel species at inference time |
| **Abundance normalization** | Raw, relative (per-plot), or log-scaled modes |
| **CPU-first** | Works without GPU, scales with CUDA when available |

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

### Linear Compositional Pooling

Species effects are aggregated linearly (abundance-weighted sum) before nonlinear mixing in PlotEncoder. This preserves interpretability: each species contributes additively to the latent signal before the network learns complex interactions.

## Documentation

- **[Getting Started](https://gillescolling.com/resolve/tutorials/quickstart/)**: Complete workflow walkthrough
- **[Data Preparation](https://gillescolling.com/resolve/tutorials/data-preparation/)**: Data formatting guide
- **[Training](https://gillescolling.com/resolve/tutorials/training/)**: Advanced training options
- **[API Reference](https://gillescolling.com/resolve/api/dataset/)**: Full API documentation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- pandas ≥ 2.0
- scikit-learn ≥ 1.3

## Support

If you find RESOLVE useful for your research, consider supporting development:

<a href="https://www.buymeacoffee.com/gillescolling" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 150px !important;" ></a>

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Citation

If you use RESOLVE in your research, please cite:

```bibtex
@software{resolve,
  author = {Colling, Gilles},
  title = {RESOLVE: Representation Encoding of Species Outcomes via Linear Vector Embeddings},
  year = {2025},
  url = {https://github.com/gcol33/resolve}
}
```
