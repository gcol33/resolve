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

---

## Performance Optimization Plan

### Current Bottleneck Analysis (Feb 2026)

**Observed Performance:**
- 1.5M plots, batch_size=32768, 17M params
- ~49 seconds per epoch
- GPU utilization: 1-3% (RTX 5080)
- Memory: 4GB / 16GB VRAM

**Root Cause:** Kernel launch overhead from many small operations:
1. **60+ embedding lookups per forward pass**: Separate `nn.Embedding.forward()` for each position
   - 20 species positions × 1 lookup each
   - 3 genus positions × 1 lookup each
   - 3 family positions × 1 lookup each
   - Each is a tiny CUDA kernel with ~5μs launch overhead
2. **Small matrix multiplications**: MLP layers finish in microseconds, GPU is idle waiting for next kernel
3. **Per-batch index_select**: Creates synchronization points

### Optimization Phases

#### Phase A: Fused Embedding Lookups (Expected 2-3x speedup)

**Goal:** Replace 60+ individual embedding lookups with 1-3 batched operations.

**Current code (`encoder.cpp:534-548`):**
```cpp
// SLOW: 20 separate kernel launches
for (int k = 0; k < top_k_species_; ++k) {
    auto sp_emb = species_embeddings_[k](species_ids.select(1, k));
    parts.push_back(sp_emb);
}
```

**Optimized approach:**
```cpp
// FAST: Single embedding lookup with offset indexing
// Flatten species_ids: (batch, top_k) -> (batch * top_k,)
// Add position offsets: id + k * vocab_size
// Single embedding lookup: (batch * top_k, embed_dim)
// Reshape: (batch, top_k * embed_dim)
auto flat_ids = species_ids.flatten() + position_offsets;
auto all_emb = unified_species_embedding_(flat_ids);
auto reshaped = all_emb.view({batch_size, top_k_species_ * embed_dim});
```

**Files to modify:**
- `include/resolve/encoder.hpp`: Add `FusedEmbeddingTable` class
- `cpp_src/encoder.cpp`: Implement fused forward pass
- Keep backward-compatible: old per-position embeddings still loadable

**Checkpoint compatibility:**
- Migration function to convert old per-position tables to unified table
- Version flag in checkpoint format

#### Phase B: CUDA Custom Kernels (Expected 1.5-2x additional speedup)

**Goal:** Write custom CUDA kernels for the embedding + concat + first linear layer.

**Approach:**
1. Fused embedding lookup + concatenation kernel
2. Fused linear + activation kernel (for small hidden dims)
3. Use shared memory for intermediate results

**Files to create:**
- `cuda/fused_embed.cu`: Custom embedding kernel
- `cuda/fused_linear.cu`: Fused linear + activation

**When to do this:** Only if Phase A is insufficient for paper experiments.

#### Phase C: torch.compile / TorchScript (Expected 1.5x speedup)

**Goal:** Let PyTorch optimize the computation graph automatically.

**Approach (libtorch 2.x):**
```cpp
// Enable torch.compile equivalent for C++
auto optimized_model = torch::jit::optimize_for_inference(model_);
```

**Alternative: Export to TorchScript:**
```cpp
auto traced = torch::jit::trace(model_, example_input);
traced.save("optimized_model.pt");
```

**When to do this:** After Phase A, if more speedup needed.

#### Phase D: Async Data Pipeline (Expected 1.2x speedup)

**Goal:** Overlap CPU work with GPU compute.

**Current:** Synchronous batch preparation
**Optimized:** Double-buffered prefetching

```cpp
// Prepare next batch on CPU while GPU processes current batch
std::thread prefetch_thread;
torch::Tensor next_batch;

for (batch_idx = 0; batch_idx < n_batches; batch_idx++) {
    // Start prefetching next batch
    if (batch_idx + 1 < n_batches) {
        prefetch_thread = std::thread([&]{
            next_batch = prepare_batch(batch_idx + 1);
        });
    }

    // Process current batch on GPU
    train_step(current_batch);

    // Wait for prefetch and swap
    if (prefetch_thread.joinable()) prefetch_thread.join();
    current_batch = std::move(next_batch);
}
```

### Implementation Priority

| Phase | Speedup | Effort | Priority |
|-------|---------|--------|----------|
| A: Fused Embeddings | 2-3x | Medium | **HIGH** |
| B: CUDA Kernels | 1.5-2x | High | Low (only if needed) |
| C: torch.compile | 1.5x | Low | Medium |
| D: Async Pipeline | 1.2x | Medium | Low |

### Target Performance

- **Current:** 49 sec/epoch, 1-3% GPU
- **After Phase A:** ~20 sec/epoch, 10-20% GPU
- **After Phase A+C:** ~15 sec/epoch, 20-30% GPU
- **Theoretical max:** ~5 sec/epoch (compute-bound)

### Backward Compatibility

All optimizations must:
1. Load existing checkpoints without modification
2. Produce identical outputs (within floating-point tolerance)
3. Be optional via config flag for debugging
