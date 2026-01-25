# RESOLVE - Claude Code Context

## Architecture

RESOLVE is a **standalone C++ engine** with thin language bindings. The C++ core handles EVERYTHING including data loading, preprocessing, training, and inference. Python/R wrappers are minimal pass-through layers.

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
3. **Thin bindings** - R/Python wrappers are just API translations, no logic
4. **Single source of truth** - All behavior defined in C++, no divergence between languages

## Paper Project

The research paper using RESOLVE is located at:
- **Path**: J:/Phd Local/Gilles_paper_resolve/
- **Title**: Species composition as biotic context: predicting plot area and habitat from assemblages
- **Data**: ASAAS dataset (~1.9M vegetation plots)
- **Targets**: Area (regression) + EUNIS habitat (9-class classification)

## Key Directories

- src/core/ - C++ libtorch implementation (the actual code)
  - cpp_src/ - Implementation files
  - include/resolve/ - Headers
  - cuda/ - CUDA kernels
  - python/ - Python bindings (nanobind)
  - cli/ - CLI application (TODO)
- r/ - R package with Rcpp bindings
- reference/ - UNTRACKED Python/PyTorch reference implementation (for comparison only)

## Tech Stack Preferences

- Python bindings: nanobind (not pybind11)
- R bindings: Rcpp
- Build system: CMake + scikit-build-core (Python), devtools (R)
- CSV parsing: fast-cpp-csv-parser
- CLI parsing: CLI11

## Development Philosophy

- Prefer newest tools over safest - Use modern, actively developed libraries
- Single C++ implementation with language bindings (no duplicate implementations)
- All data processing in C++ - language wrappers are pass-through only

## Implementation Plan: Standalone C++ Engine

### Phase 1: CSV Loading and Role Mapping

**Goal**: ResolveDataset::from_csv(header_path, species_path, roles, targets)

**Files to create**:
- include/resolve/csv_reader.hpp
- include/resolve/role_mapping.hpp
- include/resolve/dataset.hpp
- cpp_src/csv_reader.cpp
- cpp_src/dataset.cpp

**Dependencies**: fast-cpp-csv-parser (via FetchContent)

### Phase 2: Dataset-First Trainer API

**Goal**: Trainer trainer(dataset, config); trainer.fit();

**Changes**:
- Add Trainer(ResolveDataset, TrainConfig) constructor
- Merge ModelConfig fields into TrainConfig (hash_dim, hidden_dims, etc.)
- Add Trainer::predict(dataset, confidence_threshold)

### Phase 3: CLI Application

**Goal**: resolve train --header h.csv --species s.csv --output model.pt

**Commands**:
- train: Load data, train model, save checkpoint
- predict: Load model, run inference, output CSV
- info: Print model schema and training history

**Files to create**:
- cli/main.cpp
- cli/train_cmd.cpp
- cli/predict_cmd.cpp
- cli/info_cmd.cpp

**Dependencies**: CLI11 (via FetchContent)

### Phase 4: Update Bindings

**Python** (nanobind):
- Add ResolveDataset class binding
- Add RoleMapping struct binding
- Update Trainer to accept dataset

**R** (Rcpp):
- Add resolve_load_dataset()
- Update resolve_train() to use new API

### Phase 5: Testing and Validation

- C++ unit tests for CSV loading, dataset creation
- Integration: train with CLI, load in Python/R
- Paper validation: run experiments, compare metrics

### Implementation Order

1. CSV loading (Phase 1) - foundation
2. ResolveDataset (Phase 1) - data container
3. Trainer refactor (Phase 2) - dataset-first API
4. Python bindings (Phase 4) - test with paper
5. CLI (Phase 3) - standalone tool
6. R bindings (Phase 4) - parity
7. Testing (Phase 5) - validation
