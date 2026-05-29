# RESOLVE - Claude Code Context

## Never skip work on RESOLVE (CRITICAL)

**Do every piece of RESOLVE work fully and properly. Never take shortcuts, never defer "to keep things simple", never drop features to make calling code easier.** When a feature is needed in C++, port it end-to-end (header, cpp impl, CSV loader integration, schema, encoder forward path, checkpoint save/load, nanobind bindings, R bindings, Catch2 tests, smoke verify, docs update) — not "as a follow-up." The more work the better; the longer it takes the better; S+++ modular code quality is the bar.

**Forbidden moves on RESOLVE work:**
- Dropping a feature from the calling script "for now" so the C++ side doesn't need it (e.g. "C++ has no categoricals so I'll drop the categorical column"). The user explicitly called this out 2026-05-18 as cheating.
- Stubbing a C++ enum without an implementation behind it (`RankPool`, `Transformer` are existing examples of this anti-pattern; resolve them, don't replicate them).
- Marking a port as "multi-day, deferred" without an explicit user sign-off on the deferral.
- Mirroring Python with copy-paste rather than extracting shared C++ helpers.
- Skipping nanobind/Rcpp bindings, tests, or docs because "the core works."

**Required for every C++ port:** header + cpp impl + CSV/loader integration (if data-side) + Schema/Config updates + encoder forward integration (if model-side) + checkpoint save/load (if stateful) + nanobind bindings + Rcpp bindings + Catch2 unit tests + a smoke-run that exercises the new path end-to-end + update both RESOLVE `CLAUDE.md` "Remaining Work" and paper repo `CLAUDE.md` if the gap was tracked there.

**Why:** 2026-05-18 — proposed dropping the `Dataset` categorical column from the C++ parity run because C++ `RoleMapping` has no `categoricals` field. User: "THEN IMPLEMENT IT TO MAKE THE PYTHON EASIER YOU CHEATER, ALSO MAKE SURE RESOLVE STAYS TOP QUALITY MODULAR S+++ code quality / never ever skip any work on resolve, the more work the better, the longer it takes, the better".

**How to apply:** before any "we could just do X simpler" framing on RESOLVE work, stop. The answer is the full implementation. Surface the actual cost ("this is ~1-2 days of careful C++ work across 9 files"), and proceed.

## Status: C++ engine is THE engine. Python in `src/resolve/{encode,model,train,inference,data}/` is a POC.

> **Read this first before touching anything.** The Python implementation under `src/resolve/{encode,model,train,inference,data}/` was an early proof-of-concept and is *not* a maintained parallel backend. The C++ engine in `src/core/cpp_src/` (Python bindings: `resolve_core`, R bindings: `r/`) is the production engine and the codebase the paper, R package, and CLI all depend on going forward.
>
> - **Default to C++ (`resolve_core`) for all new work, paper experiments, benchmarks, and downstream packages.** Do not extend the Python POC; do not add features to it; do not "keep both backends in sync."
> - The legacy Python pieces remain in-tree only as reference for any C++ feature still being ported (currently: `rank_pool` and `transformer` encoding modes — see "Remaining Work" below). Once those are ported, the Python POC will be removed.
> - The `src/resolve/backend/` dispatch shim is therefore *also* legacy. The end state is direct use of `resolve_core` from Python and `resolve` (Rcpp) from R, with no Python-side encoder/model/trainer classes.
> - If a user asks for "the Python API," explain this status and steer them to `resolve_core` unless they have a specific reason to use the POC (e.g., comparing against an in-progress C++ port).

## Architecture

RESOLVE is a **standalone C++ engine** (libtorch) with thin nanobind/Rcpp language bindings. The Python `src/resolve/` package is a legacy POC kept in-tree only for reference during C++ feature ports.

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
4. **Single source of truth** - The C++ engine is the only maintained implementation. Python POC under `src/resolve/` is legacy and will be removed once C++ has feature parity.

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
- C++ is the only maintained engine. Python `src/resolve/` is a POC kept only as porting reference.
- All data processing available in C++ for standalone use (CLI, R, Python bindings)
- New features go in C++ first. Do not add features to the Python POC.

## Completed Infrastructure (all phases done)

- **CSV Loading & Role Mapping**: `ResolveDataset::from_csv()`, fast-cpp-csv-parser via FetchContent
- **Dataset-First Trainer**: `Trainer(ResolveModel, TrainConfig)` with `prepare_data(ResolveDataset)`, cross-validation, calibration, residual analysis
- **Categorical covariates**: `RoleMapping::categoricals` accepts string-typed covariate columns. CSV loader auto-factorizes via `CategoricalVocab` (sorted-unique non-NA → codes 1..K, reserved 0 = UNK/NA). `CategoricalEmbedder` (one `nn::Embedding` table per column) is held by `ResolveModelImpl` and concatenates its output into `continuous` before the encoder runs, matching the Python POC integration point. Vocab is captured on the Trainer at `prepare_data` time and serialized through `Trainer::save`; `Predictor::load` returns it via `Predictor::categorical_vocab()` for inference on raw CSVs. See `include/resolve/categorical.hpp` + `cpp_src/categorical.cpp`, schema fields `categorical_names`/`categorical_vocab_sizes`/`categorical_embed_dim`, and `tests/test_categorical.cpp` for the behavioral contract.
- **Rank-pool / transformer species encoding**: `PlotEncoderRankPool` (single shared species/genus/family embedding tables + weighted-mean pool + cover dropout) and `PlotEncoderTransformer` (additive token embeddings in `d_model` space + optional self-attention + attention/CLS pooling) wired end-to-end through dataset → model → trainer → checkpoint → predictor. Pool tensors (`pool_genus_ids`, `pool_family_ids`, `pool_weights`, `pool_mask`, `pool_has_cover`) are populated at CSV load time via the standalone `RankPoolEncoder` (single source of truth for `PoolWeighting` semantics — `binary`/`abundance`/`log1p`/`norm`/`rank`) and threaded through every forward / get_latent / encode / predict path. `DatasetConfig.pool_weighting` selects the weighting scheme. `ModelConfig.cover_dropout` / `d_model` / `n_heads` / `n_attention_layers` / `transformer_ff_dim` / `transformer_pooling` / `transformer_dropout` round-trip through `save_model_config` / `load_model_config` for Predictor reconstruction. See `include/resolve/encoder.hpp` (encoder classes), `cpp_src/encoder_pool.cpp` (forward impl), `cpp_src/dataset.cpp` (pool tensor population via `RankPoolEncoder::transform`), `tests/test_rank_pool_encoder.cpp` + `tests/test_transformer_encoder.cpp` (behavior contract). End-to-end smoke at `J:\Phd Local\Gilles_paper_resolve\cpp_smoke_rank_pool.py`.
- **CLI**: `resolve train/predict/info` commands via CLI11. `--vram-fraction FLOAT` (default 1.0) on both `train` and `predict` caps the PyTorch CUDA caching allocator. Pass an explicit lower value (e.g. 0.80) when sharing the GPU with a desktop so WDDM doesn't spill VRAM into shared system memory and hang under load.
- **VRAM cap (default 1.0)**: `TrainConfig.vram_fraction` + standalone helper `resolve::set_vram_fraction(fraction, device_index)` in `include/resolve/gpu.hpp` / `cpp_src/gpu.cpp`. Applied automatically in `Trainer::fit()` before any large device allocation and in `Trainer::load` / `Predictor::load` before model upload, so both training and inference respect the cap. Exposed on nanobind (`resolve_core.set_vram_fraction`, `TrainConfig.vram_fraction`, `Predictor.load(vram_fraction=)`) and Rcpp (`resolve_set_vram_fraction()`, `Trainer` config field, `Predictor_load(vram_fraction=)`). Persisted in checkpoint as informational `train_vram_fraction` key. Catch2 tests in `tests/test_gpu.cpp`. Default 1.0 lets dedicated training jobs on a solo GPU use all VRAM; users sharing the GPU with a desktop should set `vram_fraction = 0.80` (or lower) to leave headroom. Compute-cap (Green Contexts) deferred — see `dev_notes/compute_cap_plan.md`.
- **Chunked predict (issue #2)**: `Predictor::predict(const ResolveDataset&, bool return_latent, int64_t batch_size = 4096)` chunks the forward pass along dim 0 and concatenates results on CPU, bounding peak VRAM regardless of `n_plots`. `batch_size = -1` keeps the legacy one-shot path; positive values forward each slice through `model_->forward()` and accumulate `.detach().to(kCPU)` chunks. Zero / non-(-1) negative values raise `std::invalid_argument`. `Predictor::load` defaults `device = torch::kCPU` (inference on a ~5M-param MLP over 300k plots is ~12s CPU vs ~1s GPU; the OOM cost on 16 GiB-class cards outweighs the speedup). Threaded through nanobind (`Predictor.predict_dataset(ds, return_latent=False, batch_size=4096)`, `Predictor.load(device="cpu")`), Rcpp (`RPredictor::predict_dataset(ds, return_latent=FALSE, batch_size=4096L)`, `Predictor_load(device="cpu")`), and the CLI (`resolve predict --predict-batch-size N`, default 4096; `-1` opts out). Header: `include/resolve/predictor.hpp`; impl: `cpp_src/predictor.cpp` (private `slice0`). Catch2 contract tests in `tests/test_predictor.cpp` (one-shot match, large/small `batch_size`, `batch_size=1`, `return_latent` parity, invalid-value rejection). Smoke at `dev_notes/predictor_batch_size_smoke.py` exercises the path on 120k synthetic plots and asserts `torch.allclose(out_chunked, out_oneshot, rtol=1e-5, atol=1e-6)`; log at `dev_notes/predictor_batch_size_smoke.log`.
- **Bindings**: nanobind (`_resolve_core`) with dataset/model/trainer bindings (including `CategoricalVocab`, `roles.categoricals`, schema categorical fields); Rcpp with `resolve_load_dataset()` and matching categorical exposure.
- **Testing**: Catch2 unit tests (dataset, loss, model, categorical, new modules, gpu) + CUDA benchmarks

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

## Architecture Improvements (completed)

- **Unified encoder interface**: `BaseSpeciesEncoder(ABC)` in `src/resolve/encode/base.py` — all species encoders (hash, embedding, bag, rank-pool) implement `fit()`, `transform()`, `state_dict()`, `load_state_dict()`, `is_fitted`
- **Centralized constants**: `src/resolve/constants.py` — `DEFAULT_HIDDEN_DIMS`, `PREFETCH_BATCH_THRESHOLD`, `SCHEDULER_PCT_START`, `ETA_WINDOW`, `NAN_THRESHOLD_PCT`, `MAX_GRAD_NORM`
- **Config dataclasses**: `ModelConfig`, `TrainingConfig`, `DataConfig`, `CheckpointConfig` in `src/resolve/train/_types.py` — grouped alternative to Trainer's 64 individual kwargs (fully backwards compatible)
- **Batch prediction API**: `Predictor.predict_batched(dataset, batch_size)` and `Predictor.predict_generator(dataset, batch_size)` in `src/resolve/inference/predictor.py`
- **Model input validation**: `ResolveModel._get_latent()` validates tensor shapes, batch size consistency, and required inputs per encoding mode
- **Encoder deduplication**: `_get_single_embedding_weights()` helper in `src/resolve/model/encoder.py` shared by PlotEncoderRankPool and PlotEncoderTransformer
- **Full encoder exports**: `src/resolve/model/__init__.py` exports all 5 encoder classes + `MixtureOfExperts`
- **Backend logging**: `logging.getLogger("resolve.backend")` reports backend selection at import time
- **C++ feature parity prep**: `SpeciesEncodingMode::RankPool` and `Transformer` enums, `pool_*` fields in `ResolveBatch`, informative error messages for unimplemented modes
- **C++ adapter tests**: Catch2 tests for TabNet, SAINT, GNN, HeterogeneousGNN adapters
- **C++ const-correctness**: const overload for `ResolveModelImpl::head()`
- **Native fuzzy-string index**: `_resolve_core.fuzzy.FuzzyIndex` — generic Damerau-Levenshtein top-N matcher (trie + DP-row Levenshtein automaton, UTF-8 codepoint level, optional bucket hint, OpenMP `query_batch`). Wired into `WFOBackbone._match_fuzzy` with automatic difflib fallback when `_resolve_core` is unavailable. Header: `src/core/include/resolve/fuzzy.hpp`; sources: `cpp_src/fuzzy_{index,search,automaton}.cpp`.

## Remaining Work

### Benchmark note: Fused embed+concat+linear Triton kernel

The Triton kernel in `csrc/fused_embed_linear.py` is **correct** but ~90x slower than PyTorch's cuBLAS path for typical RESOLVE dimensions (D_in=83, D_out=2048). The intermediate concat tensor (~1.3 MB at B=4096) is too small to justify the fusion overhead. The kernel is disabled by default (`force_triton=False`); the PyTorch fallback path is always used. The Triton kernel could be revisited for architectures with much larger intermediate tensors.

### MoE for embed and sparse encoding modes

**Goal**: Extend Mixture of Experts routing (currently hash-only) to embed and sparse modes.

**What exists**: `PlotEncoderMoE` with gating network and expert routing for hash mode.

**What's needed**:
- Adapter layers for embed mode (input = concatenated embedding IDs -> expert input)
- Adapter layers for sparse mode (input = explicit species vector -> expert input)
- Update model construction in both C++ and Python backends

### Embedding weight extraction API

**Goal**: Extract learned genus/family embedding weights for downstream analysis and export.

**Current state** (`src/core/cpp_src/predictor.cpp`): Two stubs returning empty tensors.

**What's needed**:
- Access embedding tables from the encoder (varies by type)
- Clone and return weight tensors
- Expose via nanobind/Rcpp bindings

(C++ rank_pool/transformer implementation: **DONE** 2026-05-19 — see the
"Rank-pool / transformer species encoding" bullet under "Completed
Infrastructure" above.)
