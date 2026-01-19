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
from resolve import ResolveDataset, Trainer

# Load data
dataset = ResolveDataset.from_csv(
    header="plots.csv",
    species="species.csv",
    roles={
        "plot_id": "plot_id",
        "species_id": "species",
        "species_plot_id": "plot_id",
    },
    targets={"y": {"column": "response", "task": "regression"}},
)

# Train
trainer = Trainer(dataset)
trainer.fit()

# Predict
preds = trainer.predict(dataset)
preds = trainer.predict(dataset, confidence_threshold=0.8)  # only confident predictions
```

For more complex use cases with taxonomy, coordinates, and multiple targets:

```python
dataset = ResolveDataset.from_csv(
    header="plots.csv",
    species="species.csv",
    roles={
        "plot_id": "PlotID",
        "species_id": "Species",
        "species_plot_id": "PlotID",
        "abundance": "Cover",
        "taxonomy_genus": "Genus",
        "taxonomy_family": "Family",
        "coords_lat": "Latitude",
        "coords_lon": "Longitude",
    },
    targets={
        "area": {"column": "Area", "task": "regression", "transform": "log1p"},
        "habitat": {"column": "Habitat", "task": "classification", "num_classes": 5},
    },
)

trainer = Trainer(dataset, hash_dim=64, top_k=10)
trainer.fit()
trainer.save("model.pt")
```

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid species encoding** | Feature hashing for full species lists + learned embeddings for dominant taxa |
| **Multi-target prediction** | Single shared encoder, multiple task heads (regression & classification) |
| **Phased training** | MAE → SMAPE → band accuracy optimization |
| **Semantic role mapping** | Flexible column naming, strict structure |
| **Unknown species tracking** | Detects and quantifies novel species at inference time |
| **Abundance normalization** | Raw, normalized (sum-to-one), or log1p modes |
| **Confidence filtering** | Set threshold to filter uncertain predictions |
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

## What RESOLVE Assumes

RESOLVE models plot-level attributes as a function of species composition using linear aggregation of species representations. Individual species contributions are summed or weighted before any nonlinear transformation, meaning that species–species interactions are not modeled directly at the species level but may emerge at the plot level through the encoder. Taxonomic information (e.g. genus or family) is optional and, when provided, acts as a structured prior that can improve generalization, especially for rare or sparsely observed species. Species not seen during training are handled explicitly and reduce prediction confidence, reflecting extrapolation beyond the learned species space rather than model uncertainty in a statistical sense.

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
