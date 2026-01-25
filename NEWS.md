# RESOLVE Changelog

## v0.4.0 (2025-01-25)

### New Features

- **R² metric**: Coefficient of determination for regression evaluation (computed on original scale)
- **Class weights**: Support for imbalanced classification via `class_weights` in target config
- **LR scheduling**: StepLR and CosineAnnealing scheduler options

### Testing & CI

- Comprehensive test suite: Catch2 (C++), pytest (Python), testthat (R)
- GitHub Actions workflows for automated testing and releases

### Packaging

- Python: `resolve-core` (C++ bindings) + `resolve` (high-level wrapper)
- R: Full testthat integration, CRAN-ready structure

## v0.1.0 (2025-01-19)

Initial release of RESOLVE (Representation Encoding for Structured Observation Learning with Vector Embeddings).

### Features

- **Hybrid species encoding**: Feature hashing for full species lists + learned embeddings for dominant taxa
- **Multi-target prediction**: Single shared encoder, multiple task heads (regression and classification)
- **Phased training**: MAE -> SMAPE -> band accuracy optimization for regression targets
- **Semantic role mapping**: Flexible column naming with strict structural requirements
- **Unknown species tracking**: Detects and quantifies novel species at inference time
- **Abundance normalization**: Raw, relative (per-plot), or log-scaled modes
- **CPU-first design**: Works without GPU, scales with CUDA when available

### Core Components

- `ResolveDataset`: Data loading with semantic role mapping
- `ResolveModel`: Neural network architecture with shared encoder and task-specific heads
- `Trainer`: Training loop with phased optimization and early stopping
- `Predictor`: Inference interface with embedding extraction

### Architecture

- Linear compositional pooling: Species effects aggregated linearly before nonlinear mixing
- Taxonomy-aware embeddings: Learned representations for genera and families
- Feature hashing: Scalable species encoding via locality-sensitive hashing
