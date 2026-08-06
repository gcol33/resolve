# RESOLVE Changelog

## v0.7.3 (2026-08-04)

### R package

- **GPU training from R.** `resolve.install_backend()` gains CUDA variants:
  `variant = "cu128"`, `"cu130"`, or `"cuda"` (auto-selects the line from the
  installed NVIDIA driver -- `cu130` for CUDA >= 13, else `cu128`). The CUDA
  builds ship only the small `resolve_c` library on the GitHub release and fetch
  the matching official libtorch from `download.pytorch.org` on first install,
  pinned to the exact version `resolve_c` was built against so the ABI matches;
  GPU training then runs through the ordinary `device = "cuda"` path. A
  backend-variant registry (`{os, arch, variant} -> {asset, libtorch}`) drives
  the downloader, so adding a CUDA line later is one table row.
- **GPU nudge.** On attach, if an NVIDIA GPU is detected but the CPU backend is
  loaded (or none is), the package points to the GPU build.
- The backend is loaded at runtime (`dlopen`/`LoadLibrary`) rather than linked,
  so the package installs and `R CMD check`s with no backend present; libtorch
  threads default to all cores (`RESOLVE_R_TORCH_THREADS=N` to pin/cap).

## v0.7.2 (2026-07-19)

A review sweep of the whole engine, issues #37-#100. Highlights below; each
commit carries the per-issue detail.

### Correctness

- **Checkpoints round-trip the full architecture.** `save_model_config` /
  `load_model_config` serialize every architecture sub-config (FT-Transformer,
  TabNet, SAINT, GNN, TraitNet, ExcelFormer, heterogeneous GNN, parallel
  branches), and weight loading now throws on a missing parameter instead of
  silently leaving it at random init (#37). Classification `class_weights` and
  the rank-pool weighting scheme + species cap are persisted too, so a reloaded
  model keeps its loss and its pooling semantics (#38, #91).
- **Gradients reach the objectives they belong to.** The phase-3 band penalty is
  a differentiable hinge, ExcelFormer's semi-permeable mask is a soft gate with
  an additive log-bias, and BERT-style MLM feeds the 10%-random / 10%-keep ids
  through to the encoder (#42, #43, #44).
- **Self-supervised views hide the answer.** JEPA and SCARF mask the species and
  taxonomy side of each view, so the pretext task cannot be solved by species
  identity alone (#44, #93).
- **Cross-validation starts each fold from the untrained weights** and restores
  the trainer's split afterwards, so CV after `fit()` no longer warm-starts from
  weights that saw the held-out rows (#45, #97).
- **Loaders fail loudly.** A named role column that cannot be resolved throws
  rather than dropping the feature; coordinates parse NA-aware; ranking is dense;
  `num_classes` follows the class list (#40, #46, #47, #94).
- **Determinism knobs are honored.** `cudnn_benchmark = false` survives the
  training loop instead of being re-enabled inside `cache_data_to_gpu` (#92).

### Reported metrics

- **SMAPE** uses the standard `(|p|+|t|)/2` denominator (range 0-2, matches
  sklearn); values previously came out at half scale, and the phased-loss SMAPE
  term shifts by the same constant factor (#95).
- **The VAE ELBO** sums KL over the latent dimension and means over the batch, so
  `kl_weight = 1` is beta = 1 (#96).

### Retraining required

- Taxonomy embedding tables for the hash / sparse / MoE / adapter encoders lose
  the one over-allocated row (`n_genera` already counts `<UNK>`), and the
  coordinate-kNN GNN embeds taxonomy ids instead of concatenating them as
  magnitudes and trains full-batch. Checkpoints from before these changes cannot
  be loaded (#73, #99).

### Tooling

- CI gains a vendored-header drift guard for the R C facade; `resolve info`
  prints the transformer / rank-pool hyperparameters; pretraining configs
  validate their batch size, mask ratio and corruption rate; the header loader
  reads the file in a single streaming pass instead of a count prepass (#100).

## v0.7.1 (2026-06-19)

### Packaging

- **PyPI wheel build repaired across all platforms.** The `resolve-core` wheel
  pipeline (broken since 0.6.x) now builds cleanly on Linux, Windows, and both
  macOS architectures. Linux/Windows install `cmake` and `ninja` explicitly
  because the no-isolation build asks scikit-build-core for `ninja>=1.5`, which
  the manylinux container does not ship; the pip self-upgrade uses `python -m
  pip` so the Windows step no longer aborts. The macOS extension links with
  `-Wl,-undefined,dynamic_lookup` so the Python C-API symbols pulled in via
  `libtorch_python` (absent from nanobind's restricted macOS symbol list)
  resolve from the host interpreter at load time instead of failing the arm64
  link.

## v0.7.0 (2026-06-19)

### New features

- **In-memory (DataFrame) dataset loaders (#22).** Build a dataset directly from
  frames already in RAM, eliminating the write-to-temp-CSV / re-read round-trip
  the CSV loaders force when the header is filtered or subset per fit. Python:
  `ResolveDataset.from_pandas(header, species=, roles, targets, config=,
  schema_source=)` (alias `.from_dataframe`), where `species` may be a
  DataFrame, a CSV path (the large species table is read once from disk while
  the header stays in memory), or `None` (single long frame). R:
  `resolve.dataset.frame(header, species=, ...)`. C++: `from_dataframe` /
  `from_dataframe_header` / `from_species_dataframe` / `from_dataframe_with_schema`.
  A shared `RowSource` seam (implemented by both `CSVReader` and an in-memory
  `ColumnTable`) makes `from_dataframe` byte-identical to `from_csv` on the
  equivalent CSV by construction (an empty cell is a missing value). The
  previously-missing `categorical_ids` R accessor was also registered.

### Bug fixes

- **AMP fp32-normalization guard (#21).** `run_norm_fp32` / `Fp32Norm` force
  every normalization layer to compute in fp32 inside a CUDA autocast region
  while the surrounding Linear/embedding matmuls stay fp16, guarding against
  fp16 BatchNorm-statistic corruption/overflow (a running variance saturating to
  `inf` collapses eval-mode normalization to mean-prediction). On the current
  libtorch build autocast already promotes `batch_norm` to fp32, so the guard is
  defensive there; it removes the dependency on that implicit, version-dependent
  autocast policy. Toggle with `RESOLVE_FP32_NORM=0`; diagnose with
  `RESOLVE_AMP_DEBUG=1`.

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
  and skip it instead of waiting on the AeDebug handshake. The R bindings arm an
  on-exit finalizer alongside the crash handler, mitigating the libtorch teardown
  access violation that could crash the `Rscript.exe` launcher. libtorch's thread
  pools are left at their multi-threaded default so training and prediction use
  all cores; set `RESOLVE_R_TORCH_THREADS=N` (a positive integer) to pin both
  pools to N threads -- to cap CPU use on a shared machine, or as a workaround
  (`N=1`) if a Windows environment still hits the teardown crash.

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
