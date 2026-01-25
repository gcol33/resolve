# RESOLVE

[![Tests](https://github.com/gcol33/resolve/actions/workflows/tests.yml/badge.svg)](https://github.com/gcol33/resolve/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://gillescolling.com/resolve)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Representation Encoding for Structured Observation Learning with Vector Embeddings**

A torch-based package for predicting sample attributes from compositional data—sets of entities with optional abundances or weights.

## Overview

RESOLVE treats compositional data as *contextual signal*—a rich, structured representation that encodes information about sample-level attributes. Given a set of entities (species in a plot, symptoms in a patient, products in a basket), RESOLVE learns to predict properties of the sample.

**Core idea**: Compositional data encodes a shared latent representation that simultaneously informs multiple sample attributes.

### Example Domains

| Domain | Entities | Sample | Predictions |
|--------|----------|--------|-------------|
| **Ecology** | Plant species | Vegetation plot | Plot area, habitat type, elevation |
| **Medicine** | Symptoms, conditions | Patient | Diagnosis, severity, treatment response |
| **Retail** | Products | Shopping basket | Customer segment, churn risk |
| **Genomics** | Genes, variants | Sample | Phenotype, disease risk |
| **Text** | Words, n-grams | Document | Topic, sentiment, author |

## Quick Start

```python
from resolve import ResolveDataset, Trainer

# Load data
dataset = ResolveDataset.from_csv(
    header="samples.csv",       # one row per sample
    species="entities.csv",     # entity-sample associations
    roles={
        "plot_id": "sample_id",
        "species_id": "entity_id",
        "species_plot_id": "sample_id",
    },
    targets={"y": {"column": "response", "task": "regression"}},
)

# Train
trainer = Trainer(dataset)
trainer.fit()

# Predict with confidence filtering
preds = trainer.predict(dataset)
preds = trainer.predict(dataset, confidence_threshold=0.8)
```

### Ecology Example

Predict vegetation plot area and habitat from species composition:

```python
from resolve import ResolveDataset, Trainer, RoleMapping, TargetConfig, TrainerConfig

dataset = ResolveDataset.from_csv(
    header="plots.csv",
    species="species_records.csv",
    roles=RoleMapping(
        plot_id="PlotID",
        species_id="Species",
        species_plot_id="PlotID",
        abundance="Cover",
        taxonomy_genus="Genus",
        taxonomy_family="Family",
        coords_lat="Latitude",
        coords_lon="Longitude",
    ),
    targets={
        "area": TargetConfig(column="Area", task="regression", transform="log1p"),
        "habitat": TargetConfig(column="Habitat", task="classification", num_classes=5),
    },
)

config = TrainerConfig(hash_dim=64, top_k=10, hidden_dims=[512, 256, 128])
trainer = Trainer(**config.to_trainer_kwargs(dataset))
trainer.fit()
```

### Medical Example

Predict diagnosis from patient symptoms:

```python
dataset = ResolveDataset.from_csv(
    header="patients.csv",
    species="symptoms.csv",
    roles=RoleMapping(
        plot_id="patient_id",
        species_id="symptom_code",
        species_plot_id="patient_id",
        abundance="severity",  # optional: symptom intensity
    ),
    targets={
        "diagnosis": TargetConfig(column="icd_code", task="classification", num_classes=50),
        "severity": TargetConfig(column="severity_score", task="regression"),
    },
)
```

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid entity encoding** | Feature hashing for full entity lists + learned embeddings for dominant entities |
| **Multi-target prediction** | Single shared encoder, multiple task heads (regression & classification) |
| **Phased training** | MAE → SMAPE → band accuracy optimization |
| **Semantic role mapping** | Flexible column naming via `RoleMapping` dataclass |
| **Unknown entity tracking** | Detects and quantifies novel entities at inference time |
| **Abundance normalization** | Raw, normalized (sum-to-one), or log1p modes |
| **Confidence filtering** | Set threshold to filter uncertain predictions |
| **Typed configuration** | `TrainerConfig` dataclass with presets (TINY_MODEL → MAX_MODEL) |
| **CPU-first** | Works without GPU, scales with CUDA when available |

## Performance

Optimized CUDA kernels for GPU acceleration. Benchmarks on RTX 4090:

| Operation | Dataset Size | CPU | GPU | Speedup |
|-----------|-------------|-----|-----|---------|
| Hash Embedding | 10K records | 0.08 ms | 0.02 ms | 5x |
| Hash Embedding | 100K records | 1.3 ms | 0.04 ms | **35x** |
| Hash Embedding | 1M records | 32 ms | 0.08 ms | **400x** |

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
Entity data ──────┐
                  ├──→ EntityEncoder ──→ hash embedding + hierarchy IDs
Coordinates ──────┤                      + unknown mass features
                  ├──→ SampleEncoder (shared) ──→ latent representation
Covariates ───────┘
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                              TaskHead(y1)     TaskHead(y2)     TaskHead(y3)
                                    │                 │                 │
                                    ↓                 ↓                 ↓
                              regression       regression       classification
```

### Linear Compositional Pooling

Entity effects are aggregated linearly (abundance-weighted sum) before nonlinear mixing in the encoder. This preserves interpretability: each entity contributes additively to the latent signal before the network learns complex interactions.

## Configuration

Use `TrainerConfig` for clean, reusable training setups:

```python
from resolve import TrainerConfig

# Custom config
config = TrainerConfig(
    hash_dim=128,
    top_k=20,
    hidden_dims=[1024, 512, 256],
    max_epochs=500,
    patience=30,
)

# Or use presets
from resolve.config import LARGE_MODEL, MEDIUM_MODEL
trainer = Trainer(**LARGE_MODEL.to_trainer_kwargs(dataset))
```

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

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Citation

If you use RESOLVE in your research, please cite:

```bibtex
@software{resolve,
  author = {Colling, Gilles},
  title = {RESOLVE: Representation Encoding for Structured Observation Learning with Vector Embeddings},
  year = {2025},
  url = {https://github.com/gcol33/resolve}
}
```
