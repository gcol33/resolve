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
| B: CUDA Kernels | **DONE** | Hash kernels (5 variants + auto-select in `cuda/kernels.cu`); Triton fused linear+CE (`csrc/fused_linear_ce.py`); fused embed+concat+linear (`csrc/fused_embed_linear.py`, integrated in `PlotEncoder`) |
| C: JIT Inference | **DONE** | BN fusion in `Predictor.optimize_for_inference()` (C++) and `ResolveModel.optimize_for_inference()` (Python) |
| D: Async Pipeline | **DONE** | Double-buffered GPU prefetch in Trainer via `CUDAPrefetcher`; auto-enables at batch_size >= 16384 |

### Additional done optimizations

- **Numba parallel kernels**: `@njit(parallel=True)` hash aggregation + taxonomy top-k (`encode/species_fast.py`)
- **torch.compile()**: Integrated in Trainer (`compile_model=True`, mode `reduce-overhead`)
- **GPU-resident data**: Full dataset on GPU via `GPUTensorLoader` (auto-enabled on CUDA)

---

## Remaining Work

All major optimization phases (A-D) and architecture features are complete.

### Benchmark note: Fused embed+concat+linear Triton kernel

The Triton kernel in `csrc/fused_embed_linear.py` is **correct** but ~90x slower than PyTorch's cuBLAS path for typical RESOLVE dimensions (D_in=83, D_out=2048). The intermediate concat tensor (~1.3 MB at B=4096) is too small to justify the fusion overhead. The kernel is disabled by default (`force_triton=False`); the PyTorch fallback path is always used. The Triton kernel could be revisited for architectures with much larger intermediate tensors.

### MoE for embed and sparse encoding modes

**Goal**: Extend Mixture of Experts routing (currently hash-only) to embed and sparse modes.

**Current limitation** (`src/core/cpp_src/model.cpp:79,108`):
```cpp
if (use_moe) {
    throw std::runtime_error("MoE is currently only supported for Hash encoding mode");
}
```

**What exists**: `PlotEncoderMoE` with gating network and expert routing for hash mode.

**What's needed**:
- Adapter layers for embed mode (input = concatenated embedding IDs → expert input)
- Adapter layers for sparse mode (input = explicit species vector → expert input)
- Update model construction in both C++ and Python backends

**Files to modify**:
- `src/core/cpp_src/model.cpp` — Remove restriction, add embed/sparse MoE paths
- `src/core/include/resolve/model.hpp` — New MoE encoder variants
- `src/resolve/model/resolve.py` — Python fallback MoE for embed/sparse

### Embedding weight extraction API

**Goal**: Extract learned genus/family embedding weights for downstream analysis and export.

**Current state** (`src/core/cpp_src/predictor.cpp:178-186`): Two stubs returning empty tensors:
```cpp
torch::Tensor Predictor::get_genus_embeddings() const { return torch::Tensor(); }
torch::Tensor Predictor::get_family_embeddings() const { return torch::Tensor(); }
```

**What's needed**:
- Access embedding tables from the encoder (varies by type: `PlotEncoder`, `PlotEncoderEmbed`, etc.)
- Clone and return weight tensors
- Handle encoder types that don't have taxonomy embeddings (return empty tensor)
- Expose via nanobind/Rcpp bindings

**Files to modify**:
- `src/core/cpp_src/predictor.cpp` — Implement extraction logic
- `src/core/python/bindings.cpp` — Expose to Python
- `r/src/resolve_bindings.cpp` — Expose to R
