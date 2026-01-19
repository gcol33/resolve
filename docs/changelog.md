# Changelog

All notable changes to RESOLVE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
