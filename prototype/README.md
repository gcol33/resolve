# RESOLVE C++ Core Prototype

**R**elational **E**ncoding via **S**tructured **O**bservation **L**earning with **V**ector **E**mbeddings

This folder contains the prototype for migrating RESOLVE to a true C++ core with Python and R bindings.

## Vision

**Option B Architecture**: C++ is the single source of truth. Python and R are thin binding layers.

- All core logic (encoding, model, training, inference) lives in C++
- Python uses pybind11 to call C++ directly
- R uses Rcpp to call C++ directly
- No feature drift between implementations
- Maximum performance

## Current Structure

```
prototype/
├── core/                         # C++ core library (THE implementation)
│   ├── include/resolve/
│   │   ├── types.hpp            ✅ Enums, configs, batch types
│   │   ├── vocab.hpp            ✅ SpeciesVocab, TaxonomyVocab
│   │   ├── dataset.hpp          ✅ ResolveDataset with CSV loading
│   │   ├── loss.hpp             ✅ N-phase flexible loss, Metrics
│   │   ├── trainer.hpp          ✅ Full-featured trainer
│   │   ├── encoder.hpp          ✅ PlotEncoder, PlotEncoderEmbed, PlotEncoderSparse
│   │   ├── model.hpp            ✅ ResolveModel
│   │   ├── plot_encoder.hpp     ✅ Generalized PlotEncoder
│   │   └── predictor.hpp        ✅ Predictor for inference
│   ├── src/
│   │   ├── encoder.cpp          ✅ Encoder implementations
│   │   ├── model.cpp            ✅ Model implementation
│   │   ├── plot_encoder.cpp     ✅ Plot encoder implementation
│   │   └── trainer.cpp          ✅ Trainer training loop implementation
│   ├── tests/
│   │   ├── test_plot_encoder.cpp  ✅ Plot encoder tests
│   │   ├── test_model.cpp       ✅ Model tests
│   │   ├── test_loss.cpp        ✅ Loss/metrics tests
│   │   └── CMakeLists.txt       ✅ Test build config
│   └── CMakeLists.txt           ✅ Build configuration
│
├── bindings/
│   ├── python/                  # pybind11 bindings
│   │   ├── src/bindings.cpp     ✅ Full C++ binding code
│   │   ├── resolve/__init__.py  ✅ Pythonic wrapper
│   │   ├── tests/
│   │   │   └── test_model.py    ✅ Model tests
│   │   ├── CMakeLists.txt       ✅ Build configuration
│   │   └── pyproject.toml       ✅ Package configuration
│   │
│   └── r/                       # Rcpp bindings
│       ├── src/bindings.cpp     ✅ Full R binding code (model, predictor, metrics)
│       ├── R/
│       │   ├── resolve.R        ✅ Formula API (data/obs pattern)
│       │   ├── model.R          ✅ ResolveModel R6 class
│       │   ├── predictor.R      ✅ ResolvePredictor R6 class
│       │   └── metrics.R        ✅ Metrics functions
│       ├── tests/testthat/
│       │   └── test-metrics.R   ✅ Metrics tests
│       └── DESCRIPTION          ✅ Package metadata
│
├── examples/
│   ├── python_example.py       ✅ Full Python training workflow
│   ├── r_example.R             ✅ Full R usage example
│   └── inference_example.py    ✅ Inference with uncertainty
│
└── README.md                    ✅ This file
```

## What's Implemented

### C++ Core (✅ Complete)

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Types | `types.hpp` | ✅ | Enums, TargetConfig, RoleMapping, ResolveSchema, ModelConfig, TrainConfig, ResolveBatch, TrainResult |
| Vocabularies | `vocab.hpp` | ✅ | SpeciesVocab, TaxonomyVocab with JSON serialization |
| Dataset | `dataset.hpp` | ✅ | ResolveDataset with CSV loading, splitting, filtering |
| Loss | `loss.hpp` | ✅ | PhaseConfig, PhasedLoss (N-phase), MultiTaskLoss, Metrics |
| Trainer | `trainer.hpp` + `trainer.cpp` | ✅ | StandardScaler, OneCycleLR, Trainer with full training loop |
| Encoders | `encoder.hpp` | ✅ | PlotEncoder (hash), PlotEncoderEmbed, PlotEncoderSparse, TaskHead |
| Model | `model.hpp` | ✅ | ResolveModel with all 3 encoding modes |
| Plot Encoder | `plot_encoder.hpp` | ✅ | Generalized encoder: hash/embed/onehot/numeric/raw |
| Predictor | `predictor.hpp` | ✅ | Inference with trained models, MC dropout uncertainty |

### Python Bindings (✅ Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| `bindings.cpp` | ✅ | Exposes all C++ types, enums, configs, encoders, model |
| `__init__.py` | ✅ | Pythonic wrappers with fallback to pure Python |
| `pyproject.toml` | ✅ | scikit-build-core based packaging |
| `tests/` | ✅ | pytest tests for model |

### R Bindings (✅ Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| `bindings.cpp` | ✅ | Full Rcpp bindings for PlotEncoder, ResolveModel, Predictor, metrics |
| `resolve.R` | ✅ | Formula API with data/obs pattern |
| `model.R` | ✅ | ResolveModel R6 class with forward, get_latent, save/load |
| `predictor.R` | ✅ | ResolvePredictor R6 class for inference |
| `metrics.R` | ✅ | band_accuracy, mae, rmse, smape functions |
| `DESCRIPTION` | ✅ | Package metadata |
| `tests/` | ✅ | testthat tests for metrics |

## Still Needed

### Finalization

1. Documentation generation (pkgdown for R, sphinx for Python)
2. CI/CD setup for automated builds and testing

## Building

### Prerequisites

- CMake >= 3.18
- C++17 compiler
- LibTorch (PyTorch C++ distribution)
- nlohmann/json
- pybind11 (for Python bindings)
- Rcpp + RcppTorch (for R bindings)

### Build C++ Core

```bash
cd prototype/core
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build .
```

### Build Python Package

```bash
cd prototype/bindings/python
pip install -e .
```

### Build R Package

```bash
cd prototype/bindings/r
R CMD build .
R CMD INSTALL resolveR_0.1.0.tar.gz
```

## Design Principles

1. **C++ is the source of truth** - All core logic lives in C++
2. **Bindings are thin** - Minimal logic in Python/R wrappers
3. **No R/Python dependencies in core** - Pure C++ with LibTorch
4. **Modern C++17** - Use std::optional, std::variant, etc.
5. **Consistent API** - Same interface in Python and R

## API Concept: data/obs Pattern

RESOLVE handles **many-to-one relationships**: multiple observations per unit.

| Domain | data (one per unit) | obs (many per unit) | by |
|--------|---------------------|---------------------|-----|
| Ecology | plots (with targets) | species occurrences | plot_id |
| E-commerce | customers | purchase history | customer_id |
| Healthcare | patients | diagnoses | patient_id |
| NLP | documents | words | doc_id |

## Quick Start

### R (Formula API)

```r
library(resolveR)

# Ecology: predict soil pH from species composition
fit <- resolve(
  ph ~ latitude + elevation + hash(species_id, top = 5, by = "cover"),
  data = plots,
  obs = species_df,
  by = "plot_id"
)

# E-commerce: predict customer LTV from purchase history
fit <- resolve(
  ltv ~ age + tenure + hash(product_id, top = 10, by = "amount"),
  data = customers,
  obs = purchases,
  by = "customer_id"
)

# Top 3 + bottom 3 (dominant + rare)
fit <- resolve(
  ph ~ latitude + hash(species_id, top = 3, bottom = 3, by = "abundance"),
  data = plots,
  obs = species_df,
  by = "plot_id"
)

# Add taxonomic embeddings (learned vector representations)
fit <- resolve(
  ph ~ latitude + hash(species_id, top = 5, by = "cover") + embed(genus, family),
  data = plots,
  obs = species_df,
  by = "plot_id"
)

# Predictions
predictions <- predict(fit, newdata = test_plots, obs = test_species)
```

### Python

```python
from resolve import PlotEncoder, ResolveModel, EncodingType, DataSource

# Create encoder with formula-like specification
encoder = PlotEncoder()
encoder.add_numeric("coords", ["latitude", "longitude"], source=DataSource.Plot)
encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")

# Fit and transform
encoded = encoder.fit_transform(plot_data, obs_data, plot_ids)

# Create and train model
model = ResolveModel(schema, config)
outputs = model.forward(encoded.continuous_features())
```

See the `examples/` folder for complete working examples.
