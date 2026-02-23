# RESOLVE - Claude Code Context

## Architecture

RESOLVE has a **dual-backend architecture**: a primary standalone C++ engine (libtorch) with thin nanobind/Rcpp language bindings, plus a complete fallback pure-Python/PyTorch implementation. The backend is automatically selected at runtime via `src/resolve/backend/`: C++ if `_resolve_core` is installed, otherwise Python.

```
+------------------------------------------------------------------+
|                  RESOLVE C++ Engine (libtorch)                   |
|                                                                  |
|  Data Layer:                                                     |
|  - CSV loading (fast-cpp-csv-parser)                             |
|  - Role mapping (plot_id, species_id, coords, taxonomy, etc.)    |
|  - ResolveDataset.from_csv() - high-level API                    |
|                                                                  |
|  Encoding Layer:                                                 |
|  - Feature hashing (species -> hash vector)                      |
|  - Taxonomy encoding (genus/family -> embeddings)                |
|  - TaxonomyVocab                                                 |
|                                                                  |
|  Model Layer:                                                    |
|  - ResolveModel (MLP with multi-head output)                     |
|  - CUDA kernels (hash embedding, etc.)                           |
|                                                                  |
|  Training Layer:                                                 |
|  - Trainer (dataset-first API)                                   |
|  - Loss functions (PhasedLoss, MultiTaskLoss)                    |
|  - Metrics (band accuracy, MAE, RMSE, SMAPE)                     |
|                                                                  |
|  Inference Layer:                                                |
|  - Predictor with confidence thresholds                          |
|  - Embedding extraction                                          |
+------------------------------------------------------------------+
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
+----------------+  +----------------+  +----------------+
|  R bindings    |  | Python bindings|  |     CLI        |
|   (Rcpp)       |  |  (nanobind)    |  |  (standalone)  |
+----------------+  +----------------+  +----------------+
```

## Design Goals

1. **Standalone C++ engine** - Complete functionality without any language runtime
2. **CLI tool** - Train and predict from command line
3. **Thin bindings** - R/Python wrappers around C++ are just API translations, no logic
4. **Python fallback** - Full Python/PyTorch implementation for development and environments without C++ build

## Paper Project

The research paper using RESOLVE is located at:
- **Path**: J:/Phd Local/Gilles_paper_resolve/
- **Title**: Species composition as biotic context: predicting plot area and habitat from assemblages
- **Data**: ASAAS dataset (~1.9M vegetation plots)
- **Targets**: Area (regression) + EUNIS habitat (9-class classification)

## Key Directories

- src/core/ - C++ libtorch implementation (primary backend)
  - cpp_src/ - Implementation files
  - include/resolve/ - Headers
  - cuda/ - CUDA kernels (hash embedding, benchmarks)
  - python/ - Python bindings (nanobind → `_resolve_core`)
  - cli/ - CLI application (train, predict, info commands)
  - tests/ - Catch2 unit tests and benchmarks
- src/resolve/ - Pure Python/PyTorch implementation (fallback backend)
  - model/ - Encoder classes, ResolveModel
  - encode/ - Species encoding (hash, embed, rank_pool, bag, WFO taxonomy)
  - train/ - Trainer with mixin architecture
  - data/ - Dataset utilities
  - backend/ - Runtime backend selection (C++ vs Python)
- r/ - R package with Rcpp bindings

## Tech Stack Preferences

- Python bindings: nanobind (not pybind11)
- R bindings: Rcpp
- Build system: CMake + scikit-build-core (Python), devtools (R)
- CSV parsing: fast-cpp-csv-parser
- CLI parsing: CLI11

## Development Philosophy

- Prefer newest tools over safest — use modern, actively developed libraries
- C++ is the primary backend; Python is a development/fallback backend
- Backend auto-selection at runtime (no user configuration needed)
- All data processing available in C++ for standalone use (CLI, R, Python bindings)

## Completed Infrastructure (all phases done)

- **CSV Loading & Role Mapping**: `ResolveDataset::from_csv()`, fast-cpp-csv-parser via FetchContent
- **Dataset-First Trainer**: `Trainer(ResolveModel, TrainConfig)` with `prepare_data(ResolveDataset)`, cross-validation, calibration, residual analysis
- **CLI**: `resolve train/predict/info` commands via CLI11
- **Bindings**: nanobind (`_resolve_core`) with dataset/model/trainer bindings; Rcpp with `resolve_load_dataset()`
- **Testing**: Catch2 unit tests (dataset, loss, model, new modules) + CUDA benchmarks

---

## Performance Optimization Status

| Phase | Status | Description |
|-------|--------|-------------|
| A: Fused Embeddings | **DONE** | `FusedPositionalEmbedding` in C++ encoder — single lookup with offset indexing |
| B: CUDA Kernels | Partial | Hash embedding kernel in `cuda/kernels.cu`; fused embed+linear not yet done |
| C: torch.compile | Not started | TorchScript/JIT optimization for C++ inference |
| D: Async Pipeline | **DONE** | Double-buffered GPU prefetch in Trainer (`prefetch_hash_[2]`) |

Remaining performance work (if needed for paper experiments):
- Phase B: Fused embedding + concat + linear CUDA kernel
- Phase C: `torch::jit::optimize_for_inference()` for C++ inference path
