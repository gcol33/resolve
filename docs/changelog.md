# Changelog

All notable changes to RESOLVE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-01-25

### Added

- **R² metric**: Coefficient of determination for regression evaluation (computed on original scale)
- **Class weights**: Support for imbalanced classification via `class_weights` in target config
- **LR scheduling**: StepLR and CosineAnnealing scheduler options
- **Comprehensive test suite**: Catch2 tests for C++, pytest for Python, testthat for R
- **CI/CD workflows**: GitHub Actions for tests, PyPI release, R-CMD-check

### Changed

- **Package split**: C++ bindings now distributed as `resolve-core`, high-level Python wrapper as `resolve`
- **R package version** synced to 0.4.0

### Fixed

- R² now computed on original scale (before inverse transform)
- NAMESPACE exports for R metric functions

## [0.2.0] - 2025-01-24

### Changed

- **Simplified API**: `Trainer` now takes only `dataset` as required argument, builds model automatically
- **Renamed normalization modes**: `relative_plot` → `relative`, `log_scaled` → `log`
- **Loss presets**: New `loss_config` parameter with presets `"mae"`, `"combined"`, `"smape"`
- **Confidence filtering**: `trainer.predict(dataset, confidence_threshold=0.8)` to filter uncertain predictions

### Added

- **Species encoding modes**: `species_encoding="hash"` (default) or `"embed"` for learned species embeddings
- **Tensor caching**: `cache_dir` parameter for faster restarts with large datasets
- **Gradient checkpointing**: Reduced memory usage for large models

### Fixed

- Fixed `expm1` overflow in metrics computation for extreme predictions
- Improved R bindings with formula API support

## [0.1.0] - 2025-01-19

### Added

- Initial release of RESOLVE (Representation Encoding of Species Outcomes via Linear Vector Embeddings)
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
