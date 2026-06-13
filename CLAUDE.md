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
- **Categorical covariates**: `RoleMapping::categoricals` accepts string-typed covariate columns. CSV loader auto-factorizes via `CategoricalVocab` (sorted-unique non-NA → codes 1..K, reserved 0 = UNK/NA). `CategoricalEmbedder` (one `nn::Embedding` table per column) is held by `ResolveModelImpl` and concatenates its output into `continuous` before the encoder runs, matching the Python POC integration point. Vocab is captured on the Trainer at `prepare_data` time and serialized through `Trainer::save`; `Predictor::load` returns it via `Predictor::categorical_vocab()` for inference on raw CSVs. See `include/resolve/categorical.hpp` + `cpp_src/categorical.cpp`, schema fields `categorical_names`/`categorical_vocab_sizes`/`categorical_embed_dim`, and `tests/test_categorical.cpp` for the behavioral contract. The R `RResolveModel` methods `forward` / `get_latent` / `forward_with_aux` / `forward_single` / `encode_with_activations` thread `categorical_ids` (11th arg for the first three, 7th for `forward_single`, 4th for `encode_with_activations`) through to the C++ method, matching the nanobind surface in `bindings_model.cpp`; without it a categorical model invoked through those low-level R methods fed the encoder a narrower-than-constructed `continuous` and shape-errored or predicted from zero-padded categoricals (issue #12). `get_gate_probs` intentionally takes no `categorical_ids` on either binding because the C++ method does not (MoE gating is hash-mode, continuous-only). The fix mirrors the already-correct `RPredictor::predict` / `RTrainer::predict_from_trainer` threading; verified on the matched-toolchain `r-cmd-check` CI (Linux/macOS libtorch), the Windows mingw/MSVC libtorch ABI wall (issue #12 toolchain note) precluding a local Rcpp compile.
- **Rank-pool / transformer species encoding**: `PlotEncoderRankPool` (single shared species/genus/family embedding tables + weighted-mean pool + cover dropout) and `PlotEncoderTransformer` (additive token embeddings in `d_model` space + optional self-attention + attention/CLS pooling) wired end-to-end through dataset → model → trainer → checkpoint → predictor. Pool tensors (`pool_genus_ids`, `pool_family_ids`, `pool_weights`, `pool_mask`, `pool_has_cover`) are populated at CSV load time via the standalone `RankPoolEncoder` (single source of truth for `PoolWeighting` semantics — `binary`/`abundance`/`log1p`/`norm`/`rank`) and threaded through every forward / get_latent / encode / predict path. `DatasetConfig.pool_weighting` selects the weighting scheme. `ModelConfig.cover_dropout` / `d_model` / `n_heads` / `n_attention_layers` / `transformer_ff_dim` / `transformer_pooling` / `transformer_dropout` round-trip through `save_model_config` / `load_model_config` for Predictor reconstruction. See `include/resolve/encoder.hpp` (encoder classes), `cpp_src/encoder_pool.cpp` (forward impl), `cpp_src/dataset.cpp` (pool tensor population via `RankPoolEncoder::transform`), `tests/test_rank_pool_encoder.cpp` + `tests/test_transformer_encoder.cpp` (behavior contract). End-to-end smoke at `J:\Phd Local\Gilles_paper_resolve\cpp_smoke_rank_pool.py`. `PlotEncoderRankPool::forward` pools each table with a fused `torch::nn::functional::embedding_bag` (mode=sum, per-sample weights = normalized pool weights, `padding_idx=0`) instead of materializing the `(batch, max_species, embed_dim)` gather; the explicit-multiply-then-sum transient scaled linearly with `max_species` and was the dominant training-time VRAM spike that spilled into WDDM shared memory and stalled the driver past the watchdog on species-rich targets (issue #6 TDR). The fused path is numerically identical (equivalence guard in `tests/test_rank_pool_encoder.cpp`, "fused pool equals explicit weighted mean") and cuts the pooling peak from ~5.4 GB to ~0.7 GB at batch 16384 / max_species 256 (smoke: `dev_notes/issue6_rank_pool_pool_memory_smoke.py`). The transformer encoder still materializes its `(batch, max_species, d_model)` token tensor — self-attention needs the per-token sequence, so `embedding_bag` does not apply there.
- **CLI**: `resolve train/predict/info` commands via CLI11. `--vram-fraction FLOAT` (default 1.0) on both `train` and `predict` caps the PyTorch CUDA caching allocator. Pass an explicit lower value (e.g. 0.80) when sharing the GPU with a desktop so WDDM doesn't spill VRAM into shared system memory and hang under load.
- **VRAM cap (default 1.0)**: `TrainConfig.vram_fraction` + standalone helper `resolve::set_vram_fraction(fraction, device_index)` in `include/resolve/gpu.hpp` / `cpp_src/gpu.cpp`. Applied automatically in `Trainer::fit()` before any large device allocation and in `Trainer::load` / `Predictor::load` before model upload, so both training and inference respect the cap. Exposed on nanobind (`resolve_core.set_vram_fraction`, `TrainConfig.vram_fraction`, `Predictor.load(vram_fraction=)`) and Rcpp (`resolve_set_vram_fraction()`, `Trainer` config field, `Predictor_load(vram_fraction=)`). Persisted in checkpoint as informational `train_vram_fraction` key. Catch2 tests in `tests/test_gpu.cpp`. Default 1.0 lets dedicated training jobs on a solo GPU use all VRAM; users sharing the GPU with a desktop should set `vram_fraction = 0.80` (or lower) to leave headroom. Compute-cap (Green Contexts) deferred — see `dev_notes/compute_cap_plan.md`.
- **Chunked predict (issue #2)**: `Predictor::predict(const ResolveDataset&, bool return_latent, int64_t batch_size = 4096)` chunks the forward pass along dim 0 and concatenates results on CPU, bounding peak VRAM regardless of `n_plots`. `batch_size = -1` keeps the legacy one-shot path; positive values forward each slice through `model_->forward()` and accumulate `.detach().to(kCPU)` chunks. Zero / non-(-1) negative values raise `std::invalid_argument`. `Predictor::load` defaults `device = torch::kCPU` (inference on a ~5M-param MLP over 300k plots is ~12s CPU vs ~1s GPU; the OOM cost on 16 GiB-class cards outweighs the speedup). Threaded through nanobind (`Predictor.predict_dataset(ds, return_latent=False, batch_size=4096)`, `Predictor.load(device="cpu")`), Rcpp (`RPredictor::predict_dataset(ds, return_latent=FALSE, batch_size=4096L)`, `Predictor_load(device="cpu")`), and the CLI (`resolve predict --predict-batch-size N`, default 4096; `-1` opts out). Header: `include/resolve/predictor.hpp`; impl: `cpp_src/predictor.cpp` (private `slice0`). Catch2 contract tests in `tests/test_predictor.cpp` (one-shot match, large/small `batch_size`, `batch_size=1`, `return_latent` parity, invalid-value rejection). Smoke at `dev_notes/predictor_batch_size_smoke.py` exercises the path on 120k synthetic plots and asserts `torch.allclose(out_chunked, out_oneshot, rtol=1e-5, atol=1e-6)`; log at `dev_notes/predictor_batch_size_smoke.log`.
- **Platform-aware CUDA allocator default + auto-halve `batch_size` on OOM (issue #3)**: `resolve_core.configure_cuda_allocator()` sets `PYTORCH_CUDA_ALLOC_CONF` at module import (BEFORE `import torch`) to `expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256` on Linux/mac and the same without the `expandable_segments` prefix on Windows (cuMemMap-backed allocator not implemented on win32; libtorch warns otherwise). Late-bound C++/Rcpp shims (`_configure_cuda_allocator_native`, `resolve_configure_cuda_allocator`) for callers who imported torch first. `Trainer::fit()` wraps the training loop in a try/catch around `c10::OutOfMemoryError`: releases optimizer/AMP/GPU caches via `Trainer::release_training_state()`, calls `c10::cuda::CUDACachingAllocator::emptyCache()`, halves `config_.batch_size`, restores the initial model snapshot, and restarts from epoch 0. Stops at `TrainConfig::batch_size_floor` (default 1024); below the floor the original OOM is rethrown as `std::runtime_error` with the original requested bs, post-halve value, floor, and underlying allocator message. Decision logic extracted into `resolve::decide_oom_retry()` (pure-logic, unit-tested at `tests/test_oom_retry.cpp`). Effective batch size persisted in checkpoint under `train_batch_size` + `train_effective_batch_size` + `train_batch_size_floor` (and the JSON sidecar). Exposed through nanobind (`TrainConfig.batch_size_floor`, `resolve_core.configure_cuda_allocator`) and Rcpp (`config_list$batch_size_floor`, `resolve_configure_cuda_allocator`), CLI flag `--batch-size-floor N`. Python tests at `tests/test_cuda_allocator_config.py`. Resolves the Windows-WDDM-fragmentation problem documented in issue #3 and obsoletes per-project retry ladders downstream (paper repo's E4/E5 queues, etc.).
- **Checkpoint evaluation surface (issue #4)**: scoring a saved `Trainer` checkpoint from Python/R is now first-class. `Trainer::load_state(path, device="cpu", vram_fraction=1.0)` loads weights + scalers + categorical vocab into an existing trainer in place — the fix for the static `Trainer::load` returning a `std::tuple<ResolveModel, Scalers, CategoricalVocab>` that has no nanobind/Rcpp converter (it threw `TypeError`); no more `Predictor.model.state_dict()` back door. `Trainer::compute_classification_predictions(target_name)` returns a `ClassificationPredictions` struct (`predicted_classes` int64 `[n_test]`, `probabilities` float `[n_test, n_classes]`, `actuals` int64 `[n_test]`, plus `class_names`) — the classification counterpart to the regression-only `compute_residuals`, so per-class F1 / confusion / top-k are reachable from the trainer's own test fold. `Trainer::test_indices()` / `train_indices()` (int64 tensors) and `test_plot_ids()` / `train_plot_ids()` (populated when `prepare_data(ResolveDataset)` was used) expose the held-out fold so a matching test set can be rebuilt downstream. The three test-fold evaluators (`compute_residuals`, `compute_calibration`, `compute_classification_predictions`) now share one `forward_test_fold()` helper (single source of truth; also fixes the CUDA-hash test-fold forward that the calibration/residual paths previously omitted), and the static `load` + `load_state` share `load_weights_into`. See `include/resolve/types.hpp` (`ClassificationPredictions`), `include/resolve/trainer.hpp` + `cpp_src/trainer.cpp`, bindings in `python/src/bindings_{trainer,types}.cpp` and `r/src/{rcpp_trainer.h,rcpp_common.h,resolve_rcpp.cpp}`, Catch2 contract in `tests/test_trainer_eval.cpp`, and the end-to-end smoke at `dev_notes/evaluate_saved_checkpoint_smoke.py`.
- **Deterministic taxonomy vocab (issue #5)**: `TaxonomyVocab::fit` (`include/resolve/types.hpp`) now assigns genus/family IDs in sorted (alphabetical) order — collect unique non-empty names into `std::set`, then number them after the reserved `<UNK>` slot at 0 — instead of by first-appearance order. First-appearance ordering made the genus/family ID maps depend on CSV row order, so a checkpoint trained on one ordering and scored against a differently-ordered rebuild (`from_csv_with_schema` in another process) silently misaligned the genus/family embedding lookups (observed as a ~5pp EUNIS accuracy drop downstream). Now the mapping is a pure function of the name set, matching `SpeciesVocab::from_records` which was already sorted. Regression coverage in `tests/test_taxonomy_vocab.cpp` (vocab order-independence + sorted IDs + save/load round-trip + `from_csv_with_schema` genus/family-id invariance across schema-source row order). Note: this changes the trained vocab order, so pre-fix checkpoints with taxonomy embeddings must be retrained (hard cutover; the issue's optional "persist TaxonomyVocab in the checkpoint" robustness path was not taken — deterministic fitting makes a same-data rebuild match by construction).
- **Checkpoint config/metadata loaders (issue #14)**: `save_train_config` / `save_run_metadata` now have matching readers `load_train_config(archive) -> TrainConfig` and `load_run_metadata(archive) -> RunMetadata` (`cpp_src/checkpoint.cpp`), closing the write-only asymmetry where ~18 `train_*` keys + the metrics tree were persisted but never read back. Path-based convenience: `Trainer::load_train_config(path)` / `Trainer::load_run_metadata(path)` (static, open the archive and call the free functions; the unqualified name would recurse, so the impl qualifies `resolve::load_*`). `load_train_config` recovers the persisted training hyperparameters (batch_size, lr, weight_decay, phase_boundaries, loss_config, lr_scheduler + params, band_thresholds, vram_fraction, batch_size_floor, max_epochs, patience); fields `save_train_config` does not write (device, checkpoint_dir, AMP/cuDNN flags, log callback) keep their `TrainConfig` defaults. Every read uses a FRESH tensor — `InputArchive::read` copies into the passed tensor, so reusing one across reads of different dtype/size trips a setStorage size-mismatch. nanobind: `Trainer.load_train_config(path)` / `Trainer.load_run_metadata(path)` (`def_static`), with `RunMetadata` now re-exported from `resolve_core/__init__.py`. Catch2 round-trip contract in `tests/test_trainer_eval.cpp` (`Checkpoint train-config + run-metadata round-trip`); Python smoke at `dev_notes/checkpoint_config_loader_smoke.py`. Rcpp: `Trainer_load_train_config(path)` / `Trainer_load_run_metadata(path)` free functions returning R lists (`r/src/rcpp_{common,trainer}.h`, `resolve_rcpp.cpp`), wrapped by `resolve.load_train_config()` / `resolve.load_run_metadata()` (`r/R/resolve.R`). The Rcpp side cannot be compiled on this Windows host (Rtools mingw g++ rejects the MSVC CUDA libtorch headers, issue #12); the R wrappers parse clean and the bindings mirror the proven `Predictor_load` / `nested_metrics_to_list` patterns, with full compile verification gated on the `r-cmd-check` CI (matched Linux toolchain).
- **Review-sweep robustness + R/Python parity (issue #16)**: a tracker of verified review findings, all closed. C++ (built + Catch2-covered on this MSVC/CUDA host): (1) `MultiTaskLoss::compute` seeds its accumulator deterministically — device from the first target in `targets_` order, true scalar `{}` float32 — instead of `predictions.begin()->second.options()` shape `{1}` (`cpp_src/loss.cpp`, `tests/test_loss.cpp`). (2) The auto (p99) rank-pool species cap now matches the POC's `int(np.percentile(lengths, 99))` via the extracted single-source `percentile_linear_trunc()` (numpy "linear" interp + truncation) in `species_encoding.{hpp,cpp}`; the old floor-rank index took only the lower order statistic and over-truncated skewed length distributions (sorted `[1,5,100]` -> 5 vs numpy 98). Tests in `tests/test_species_encoders.cpp`. (3) `safe_stof` / `parse_regression_target` route through the new locale-free `parse_float_strict()` in `include/resolve/csv_utils.hpp` (classic-locale `istringstream`: dot-decimal, rejects trailing garbage, so `"12abc"`/`"1,234"` no longer silently truncate to 12/1 and a comma-decimal process locale cannot change parsed coords/abundances/targets); `safe_stoi` also rejects trailing garbage. `parse_strict_int64` was already strict + locale-independent and is unchanged. `std::from_chars<float>` was avoided because libc++ (macOS CI) gained it too recently for a single cross-platform path. (4) `CSVReader::parse_header` strips a leading UTF-8 BOM and throws on duplicate header names (was: last duplicate silently won; a BOM broke first-column role lookup) — `include/resolve/csv_reader.hpp`. Parse + CSV-reader tests in the new `tests/test_csv_utils.cpp`. R bindings (CI-gated on `r-cmd-check`, mirror proven patterns): `RResolveDataset` threads `pool_species_cap` (and `pool_weighting` on `from_species_csv`) through config parsing; species/genus/family ID accessors return `IntegerMatrix` (int64) instead of float32 to match the Python int64 accessors and avoid >2^24 precision loss; `categorical_vocab()` is registered on `ResolveDataset`/`Trainer`/`Predictor` via the shared `categorical_vocab_to_list` helper (`r/src/rcpp_common.h`); and `ResolveDataset::from_csv_with_schema` is exposed through `resolve.dataset.csv(..., schemaSource=)` (the front-door verb gains an argument rather than a sibling function), with `from_csv`/`from_csv_with_schema` sharing a single `parse_csv_inputs` helper instead of duplicating the roles/targets/config block. R signature + `schemaSource` validation verified locally with Windows R; full compile on the matched Linux/macOS CI. The issue's deferred "Notes" findings were also traced and resolved: (a) the nanobind `ResolveModel` methods (`forward`/`__call__`/`get_latent`/`forward_with_aux`/`forward_single`/`encode_with_activations`/`get_gate_probs`/`set_traits`) unpacked `continuous`/genus/family/species/vector via `THPVariable_Unpack(obj.ptr())` with no `is_none()` guard, so passing `None` for an unused input reinterpreted the `None` singleton as a `THPVariable` (UB) — now routed through shared `unpack_optional_tensor` / `unpack_required_tensor` helpers in `bindings_common.hpp` (the duplicate local `unpack_or_empty` in `bindings_trainer.cpp` was folded into them), with `nb::none()` defaults so the taxonomy/species args are genuinely omittable; smoke at `dev_notes/issue16_none_unpack_smoke.py`; (b) `Model.forward`/`get_latent`, `Trainer.predict`, `Predictor.predict` held the GIL through their pure-C++ forward (≈12 s on CPU over ~300k plots, blocking other Python threads) — now wrapped in `nb::gil_scoped_release` around only the compute (inputs held as `at::Tensor`, no Python C-API inside), with `Predictor.predict_dataset` using `nb::call_guard<nb::gil_scoped_release>` like `fit`/`cross_validate`; (c) `compute_diagnostics` running a continuous-only forward for non-hash encoders is a non-issue — `encode_with_activations` deliberately returns empty activations for non-hash encoders (`model.cpp`) and `compute_diagnostics` bails with "Diagnostics not available for this encoder type."
- **R package builds on Windows via a C ABI facade (issue #17)**: the R package no longer `#include`s libtorch or links it directly, so the recurring "Rcpp can't compile on this Windows host — mingw rejects the MSVC CUDA libtorch headers" wall (the issue #12 / #14 / #16 toolchain notes above) is **removed**. A flat `extern "C"` facade (`include/resolve/resolve_capi.h` + `cpp_src/resolve_capi.cpp`) wraps the engine behind opaque handles (`resolve_dataset_t` / `resolve_model_t` / `resolve_trainer_t` / `resolve_predictor_t`) and a single `resolve_value_t` tagged tree (scalars / arrays / row-major double|int matrices / ordered maps / lists) that carries all structured input (roles / targets / config / forward inputs) and output (results, accessor tensors) across the boundary. No `torch::Tensor`, `std::string`, STL, or Rcpp type crosses the C line; only `double*` / `int64_t*` / `const char*` / lengths / opaque pointers / `int` status codes do. All tensor↔buffer marshaling, the config/enum parsers (moved out of `r/src/rcpp_common.h`), and the result-struct→tree converters live on the MSVC side in `resolve_capi.cpp`; C++ exceptions are translated to a thread-local `resolve_last_error()` + a NULL/`-1` return. A new CMake `resolve_c` SHARED target (`BUILD_R_CAPI=ON`, built on all platforms) links `resolve_core` + libtorch and exports the C API. The `r/src/*` wrappers are now a thin client: each `R*` class holds a `shared_ptr<resolve_*_t>` and marshals through the value tree, keeping every Rcpp module class / method / free-function signature byte-identical, so `R/resolve.R`, `RcppExports.*`, and the testthat suite are unchanged (only `R/zzz.R` gains loader-path setup). mingw links the MSVC import library (`resolve_c.lib`) directly; the de-risking spike confirmed all three link modes (direct DLL, dlltool import lib, MSVC `.lib`) work with rtools45 binutils. **The R package's resolve_c is built CPU-only (`USE_CUDA=OFF` against a real CPU libtorch, not the local CUDA nightly whose `TorchConfig.cmake` force-links `torch_cuda`) and OpenMP-free (`RESOLVE_USE_OPENMP=OFF`)**: a CUDA-built, vcomp-linked DLL crashes non-deterministically inside the mingw R process (CUDA init + a third OpenMP runtime stacked on R's libgomp and torch's libiomp5md); the R bindings never call the only OpenMP user (`fuzzy::query_batch`), so dropping it is free. The package links resolve_c via `RESOLVE_C_LIB` (Windows) / `RESOLVE_C_HOME` (Unix `-L … -lresolve_c`) and resolves the runtime DLLs by putting the resolve_c directory on the loader path (CI/install set it; `.onLoad` best-effort prepends `RESOLVE_C_HOME`). **Verified locally on this Windows box** (rtools45 mingw g++ 14.3.0): `R CMD INSTALL` + `R CMD check` are clean (`Status: OK`) and the full testthat suite (129 assertions incl. dataset/classification/trainer end-to-end) passes via R/Rterm — where the package previously could not build at all. The Windows `r-cmd-check` CI job is re-enabled alongside Linux/macOS, all three using the same C-client path. Note: the `Rscript.exe` launcher (as opposed to `R`/`Rterm`/`R CMD check`) exhibits a process-teardown instability with the loaded libtorch DLLs (issue #18); the process-hardening bullet below addresses it. Design + bring-up notes: `src/core/dev_notes/issue17_c_abi_facade_design.md`.
- **Windows process hardening: teardown crash + JIT-debugger hang (issues #18 / #19)**: a single engine module `resolve::process` (`include/resolve/process.hpp` + `cpp_src/process.cpp`, mirroring the `gpu.hpp` standalone-helper pattern) turns a native fault from a hang or a launcher-teardown crash into a fast, deterministic exit. `install_crash_handler(shutdown_exit_code)` (Windows-only; no-op elsewhere) sets `SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)`, `WerSetFlags(WER_FAULT_REPORTING_NO_UI)` (dynamically resolved from kernel32, no wer.lib link), a `SetUnhandledExceptionFilter` (mid-run faults), and a first-in-line `AddVectoredExceptionHandler` (teardown-window faults the unhandled filter never sees). Both terminate via `TerminateProcess`, which never consults the AeDebug `vsjitdebugger` key -- so a headless worker that hits an unhandled native exception fails fast with the fault's NTSTATUS instead of hanging forever on the JIT-debugger handshake (issue #19, observed as a `0xC0000006` worker holding the GPU ~13 h). `signal_work_complete()` flips the policy so a fault during normal teardown-after-success exits with the shutdown code (0) rather than a failure code; the exit-code split is the pure, unit-tested `crash_exit_code()`. `set_thread_pools(intraop, interop)` pins libtorch's host pools (the `at::set_num_threads(1)` mitigation #18 points at: no worker threads to join during process exit, the suspected teardown-race source). Wired through the C ABI (`resolve_capi_set_thread_pools` / `_install_crash_handler` / `_signal_work_complete`), nanobind (auto-armed at `resolve_core` import + `atexit(_signal_work_complete)`), Rcpp + `zzz.R` `.onLoad` (install always; Windows thread-pin unless `RESOLVE_R_NO_THREAD_PIN`; on-exit finalizer marks work complete), and the CLI (`main.cpp`: arm at entry, carry the command exit code, signal before returning). Catch2 in `tests/test_process.cpp`; end-to-end smoke `tests/crash_smoke.cpp` + `dev_notes/crash_smoke_drive.ps1` verifies both branches on this box (fault-midrun -> 0xC0000005 fast/no-hang; fault-teardown -> 0). **#19 is verified end to end** (crash_smoke). **#18's specific Rscript teardown crash could not be reproduced on this dev box** -- the documented `Rscript -e 'library(resolve); resolve:::resolve_mae(...)'` repro exits 0 here with the fix enabled *and* disabled (incl. heavy ops), and this box's `Rscript.exe -f` is independently broken at startup (crashes on an empty script; `R.exe -f`/`Rterm`/`Rscript -e` work) -- so the #18 fix rests on the confirmed root-cause class (pytorch/pytorch#61111 static-dtor teardown AV), the standard thread-pin mitigation, and the verified handler mechanism; final confirmation needs the affected environment. Supported paths (`Rterm`, `R.exe -f`, `R CMD check`) regress clean (`MAE = 0.3333333`, exit 0). Full analysis: `src/core/dev_notes/issue18_19_process_hardening.md`.
- **Bindings**: nanobind (`_resolve_core`) with dataset/model/trainer bindings (including `CategoricalVocab`, `roles.categoricals`, schema categorical fields); Rcpp with `resolve_load_dataset()` and matching categorical exposure. The Rcpp layer is now a thin client over the `resolve_c` C ABI (issue #17), not a direct libtorch consumer.
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
