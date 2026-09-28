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

## Status: the C++ engine is the only implementation.

> **Read this first before touching anything.** The engine in `src/core/cpp_src/` is the production codebase the paper, R package, CLI, and Python bindings all depend on. `resolve_core` (nanobind) and `resolve` (Rcpp, over the `resolve_c` C ABI) are thin bindings over it, carrying API translation only.
>
> - **Default to C++ for all new work, paper experiments, benchmarks, and downstream packages.** A feature lands in the engine, then in every binding.
> - "The Python API" means `resolve_core`. There is no Python-side encoder, model, or trainer class.

## Architecture

RESOLVE is a **standalone C++ engine** (libtorch) with thin nanobind/Rcpp language bindings.

```
+------------------------------------------------------------------+
|                  RESOLVE C++ Engine (libtorch)                   |
|                                                                  |
|  Data Layer:                                                     |
|  - CSV loading (hand-rolled reader)                              |
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
4. **Single source of truth** - The C++ engine is the only implementation; every binding reaches the same code.

## Paper Project

The research paper using RESOLVE spans two repos:
- **Manuscript**: C:/GillesC/Documents/writing/papers/paper_resolve_2026 (`gcol33/paper_resolve_2026`) -- the sources, and where a manuscript edit is committed
- **Experiments**: C:/GillesC/Documents/code/resolve-2026 (`gcol33/resolve-2026`) -- the runs, `PROVENANCE.md`, and a junction at `paper/` onto the manuscript checkout, so `rev build` and the figure scripts read one path
- **Title**: Species composition as ecological memory: decoding environment from 1.9 million European vegetation plots
- **Data**: European Vegetation Archive (~1.9M plots, ~20M species-plot records)
- **Targets**: area, altitude, slope, aspect, survey year, geographic location (regression) + EUNIS habitat (classification)
- **Results tree**: `$RESEARCH_DATA/projects/resolve-2026/results` on E:, mirrored on LiSC

Read `resolve-2026/PROVENANCE.md` before quoting any manuscript number. The J:
drive the paper used to live on is retired.

## Key Directories

- src/core/ - C++ libtorch engine
  - cpp_src/ - Implementation files
  - include/resolve/ - Headers
  - cuda/ - CUDA kernels (hash embedding, benchmarks)
  - python/ - Python bindings (nanobind → `_resolve_core`)
  - cli/ - CLI application (train, predict, info commands)
  - tests/ - Catch2 unit tests and benchmarks
  - tests/fixtures/ - Tiny committed CSVs the CLI end-to-end CI job trains on
- tests/core/ - pytest suite over `resolve_core` (bindings + held-out recovery fits)
- r/ - R package `resolveR` (CRAN name; Bioconductor holds RESOLVE), an Rcpp client over the `resolve_c` C ABI

## Tech Stack Preferences

- Python bindings: nanobind (not pybind11)
- R bindings: Rcpp
- Build system: CMake + scikit-build-core (Python), devtools (R)
- CSV parsing: a hand-rolled reader (`src/core/include/resolve/csv_reader.hpp`), no external dependency
- CLI parsing: a hand-rolled declarative flag table (`src/core/cli/arg_parser.hpp` + `cli_spec.hpp`), no external dependency

## Development Philosophy

- Prefer newest tools over safest — use modern, actively developed libraries
- C++ is the only engine; the bindings hold no logic of their own
- All data processing available in C++ for standalone use (CLI, R, Python bindings)
- A feature lands in C++ first, then in every binding, with tests on both sides

## Completed Infrastructure

The full per-feature record (design, file paths, tests, verification, issue numbers) is in `ENGINE_HISTORY.md`. Read the matching entry before changing an area. **A newly finished feature gets its full write-up there and at most one line here**, so this file stays under the context limit.

Standing facts that bind new work:

- **Data**: hand-rolled `CSVReader` (`csv_reader.hpp`; quoted fields, CRLF, BOM, duplicate-header error), no CSV dependency. `RowSource` is the one seam behind CSV and in-memory (`from_dataframe*`) loaders. Missing covariate/coordinate cells load as NaN; `DatasetConfig::missing_values` (`Indicate` default, `Zero`) is applied by `continuous_block.{hpp,cpp}`, the single owner of the continuous-block layout.
- **Vocabularies**: taxonomy IDs are sorted; checkpoints carry species/genus/family vocabularies on `ResolveSchema`; `ExternalVocabs` is the seam for every vocab-reusing loader; `Predictor::validate_dataset_vocabs` rejects a mismatched dataset instead of remapping it. Unknown-species fraction/count come from `compute_unknown_species_stats`.
- **Species selection**: each encoding takes its per-plot budget from the knob that fixes its width (`top_k` hash, `top_k_species` embed, `species_budget` rank_pool/transformer/sparse, 0 = none); the schema records the effective selection.
- **Config**: one X-macro field registry per config struct (`config_registry.hpp`). A new field = one row, which reaches the checkpoint, C ABI, nanobind, JSON sidecar, `resolve info` and the CLI architecture flags (`cli/config_flags.hpp`); the arity `static_assert` enforces the row. The registry does not prove the engine reads a field: check that separately. Enum spellings live once in `enum_names.hpp`.
- **Model**: all five species encoders end in the shared `EncoderTail` (MLP / TabM / MoE with `moe_placement` tail|post / parallel block, at most one). `freeze_composition` freezes the composition tables. The heterogeneous GNN's species graph is built by `species_graph.{hpp,cpp}` in `Trainer::prepare_data` and saved in the checkpoint.
- **Training**: OOM auto-halves to `batch_size_floor` (`effective_batch_size` recorded); early stopping counts patience once `MultiTaskLoss::objective_settled`; CV resets each fold to the pristine init and restores the split; pretraining is seeded through `PretrainRng` and runs in the one `run_pretrain_loop`; `Trainer::prepare_data` refuses a target-less dataset; `fixed_epochs` runs a fixed number of epochs of the `max_epochs` schedule and keeps the final weights, the one mode that fits with `test_size=0`.
- **Prediction**: chunked `Predictor::predict` (default 4096, CPU device); class probabilities in `ResolvePredictions::probabilities`; checkpoints load on the requested device.
- **CLI**: declarative flag tables (`cli/cli_spec.hpp` + `arg_parser.hpp`), rejects undeclared flags. Never CLI11.
- **Suites**: released weights are a directory + `manifest.json` (`suite.hpp`); `SuitePredictor` verifies checksums, checks members against the input contract, encodes once per vocabulary and reports per plot the combined prediction, agreement / dispersion and species recognition, never a combined support level. The manifest's JSON and checksums come from `json.hpp` / `sha256.hpp`, no dependency.
- **Bindings**: R is a thin Rcpp client over the `resolve_c` C ABI (`resolve_capi.h`; vendored copy in `r/src/resolve/`, CI `vendor-drift` job). R list keys are snake_case, R function arguments camelCase; R front doors reject unnamed or unknown keys. nanobind re-exports every public name (`tests/core/test_bindings_surface.py`).
- **Platform**: Windows crash handler (`process.hpp`), storage I/O retry (`io_retry.hpp`), env access only via `env.hpp`, VRAM cap (`gpu.hpp`), allocator config at import, CPU runs never touch the CUDA runtime (`use_hash_prefetch`).
- **Build**: warnings on (`resolve_warnings.cmake`), `-Werror` on the Linux CI job only. On Linux libstdc++ must stay ahead of libtorch in `DT_NEEDED` (`src/core/CMakeLists.txt`; CI checks it).
- **Version**: repo-root `VERSION`, propagated and checked by `tools/version.py`.
- **Hard cutovers** (retrain before comparing numbers across them): taxonomy vocab order (#5); hash/sparse/MoE/adapter taxonomy tables (#99); GNN taxonomy input (#73); MoE placement for non-hash encoders (2026-08-31); HeterogeneousGNN `n_heads` (#109); TraitNet `interaction_dim` (2026-09-23); TabNet `use_sparsemax=false` (#103); SMAPE before #95 is half scale; unknown-species columns for pre-fix checkpoints scored on novel species.

## Performance Optimization Status

| Phase | Status | Description |
|-------|--------|-------------|
| A: Fused Embeddings | **DONE** | `FusedPositionalEmbedding` in the encoder — single lookup with offset indexing |
| B: CUDA Kernels | **DONE** | Hash kernels (5 variants + auto-select in `cuda/kernels.cu`) |
| C: JIT Inference | **DONE** | BN fusion in `Predictor::optimize_for_inference()` |
| D: Async Pipeline | **DONE** | CUDA-hash prefetch on a side stream in `Trainer::train_epoch`, overlapping the next batch's hash with the current forward |

### Additional done optimizations

- **GPU-resident data**: `Trainer::cache_data_to_gpu()` uploads the split tensors once and indexes batches on device
- **Fused rank-pool**: `embedding_bag(mode=sum, per_sample_weights)` instead of a materialized `(batch, max_species, embed_dim)` gather

---

## Architecture Improvements (completed)

- **C++ adapter tests**: Catch2 tests for TabNet, SAINT, GNN, HeterogeneousGNN adapters
- **C++ const-correctness**: const overload for `ResolveModelImpl::head()`
- **Native fuzzy-string index**: `_resolve_core.fuzzy.FuzzyIndex` — generic Damerau-Levenshtein top-N matcher (trie + DP-row Levenshtein automaton, UTF-8 codepoint level, optional bucket hint, OpenMP `query_batch`). Header: `src/core/include/resolve/fuzzy.hpp`; sources: `cpp_src/fuzzy_{index,search,automaton}.cpp`.

## Remaining Work

None tracked. The mixture-of-experts gap that stood here is closed (see the
`moe_placement` bullet in Completed Infrastructure).
