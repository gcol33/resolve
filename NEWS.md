# RESOLVE Changelog

## v0.6.2 (2026-06-14)

### Bug fixes

- **Bounded retry on transient storage I/O (#20).** The engine now retries its
  explicit, idempotent file I/O on a transient storage fault instead of aborting
  the run, the complement of #19. `resolve::io::with_retry` (header-only) backs a
  new `io::IOError` thrown by the CSV reader on a failed open or a mid-read
  stream error. The dataset loaders (`from_csv` / `from_csv_with_schema` /
  `from_species_csv`) restart the whole load into a fresh dataset on a transient
  read, while a CSV *parse* error propagates immediately and never re-reads a
  multi-GB file; checkpoint save/load (`Trainer::save` / `load` / `load_state`
  and `Predictor::load`) retry the archive read/write. Tunable via
  `RESOLVE_IO_RETRY_ATTEMPTS` (3) and `RESOLVE_IO_RETRY_BACKOFF_MS` (100).
  mmap-backed page-ins and DLL code-page faults remain out of scope (they cannot
  be resumed at app level; that is #19's fail-fast domain).

## v0.6.1 (2026-06-14)

### Bug fixes

- **Windows process-crash hardening (#18, #19).** A native fault in a headless
  training worker no longer hangs forever on the Windows JIT debugger
  (`vsjitdebugger`): the engine installs an unhandled-exception filter plus a
  first-in-line vectored handler that terminate via `TerminateProcess`, so the
  worker fails fast with the fault's exit code and the orchestrator can record
  and skip it instead of waiting on the AeDebug handshake. On Windows the R
  bindings additionally pin libtorch's host thread pools to 1 and arm an on-exit
  finalizer, mitigating the libtorch teardown access violation that could crash
  the `Rscript.exe` launcher; set `RESOLVE_R_NO_THREAD_PIN` to keep libtorch's
  default threading.

### Internal

- New `resolve::process` engine module (`process.{hpp,cpp}`) with the
  `install_crash_handler` / `signal_work_complete` / `set_thread_pools` surface,
  wired through the C ABI facade, nanobind (`resolve_core.install_crash_handler`,
  `set_thread_pools`; armed at import with an `atexit` hook), Rcpp + `zzz.R`
  `.onLoad`, and the CLI. Catch2 `test_process.cpp` plus a crash-handler smoke.
- `tests/test_cuda_allocator_config.py` skips cleanly when the compiled
  `resolve_core` extension is not built.

## v0.5.0 (2026-05-18)

### New Features

- **Native FuzzyIndex backbone for `WFOBackbone`**: When `resolve_core` is
  installed, `WFOBackbone` now builds a C++ `FuzzyIndex` (Damerau-Levenshtein,
  genus-bucketed, case-insensitive) over the WFO names at construction time
  and routes `_match_fuzzy` through it. The stdlib `difflib` path remains the
  silent fallback when the native backend is unavailable. Reported
  `fuzzy_dist` is an integer edit distance on the native path; the legacy
  `1 - SequenceMatcher.ratio()` semantic is preserved on the difflib path.
- **Auto-categorical encoding in `from_fast_csv`**: Classification target columns
  whose values are non-numeric (e.g. EUNIS letters `M..V`) are now loaded as
  strings and automatically encoded to nullable `Int64` codes. Integer-string
  values (e.g. `"0".."8"`) are preserved verbatim; non-numeric values are
  factorized in sorted order. `num_classes` is auto-filled from the resulting
  mapping size, so it can be omitted from the target config.
- **`categorical_covariates` kwarg on `from_fast_csv`**: Pass a `{column: mapping}`
  dict to encode covariates with non-numeric values (e.g. `{"ReSurvey (Y/N)":
  {"Y": 1, "N": 0}}`). Use `None` for the mapping to auto-encode by sorted unique
  value. The encoded mappings are accessible via the new `dataset.categorical_mappings`
  property.

### Internal

- `Trainer.predict()` now batches the forward pass via `_batched_forward`,
  removing the OOM on large held-out sets for rank-pool / hash / embed modes.
- "Training complete" is a first-class checkpoint state: `save_checkpoint`
  takes a `completed: bool` kwarg; resumes that find a completed checkpoint
  fast-return instead of raising `UnboundLocalError` on an empty epoch range.
- `_pretrain.py` rebuilt for the pre-padded tuple layout produced by
  `_build_tensors` (the v3 cache refactor). Adds `MaskedSpeciesCollateWrapper`
  for pre-padded batches and fixes a latent categoricals slot off-by-one.
- Dead code removed: `_RankPoolPreparedData` / `RankPoolBatchDataset` /
  `_rank_pool_collate_fn`, deprecated `track_unknown_count` kwarg, the
  `ext/wfo.py` rapidfuzz fallback (single algorithm: native FuzzyIndex when
  available, else difflib).
- `Trainer._best_state`, `_ema_state`, `_using_gpu_loader` initialized in
  `__init__`; defensive `hasattr`/`getattr` at the seven call sites removed.
- `_cv.py` block_size deprecation now uses `warnings.warn(DeprecationWarning)`.

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
