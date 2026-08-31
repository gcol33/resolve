# RESOLVE Changelog

## Unreleased

### Added

- **A mixture of experts is available to every species encoding, and
  `moe_routing` means one thing.** The knob used to select two different
  architectures depending on `species_encoding`: hash built a dedicated encoder
  whose mixture REPLACED the last MLP stages, embed / sparse / rank_pool /
  transformer got a dim-preserving mixture bolted onto the finished latent, and
  the adapter architectures were refused outright. Nothing in any suite
  constructed a model with routing on, so none of it was observable.

  Placement is now explicit, as **`ModelConfig.moe_placement`**:

  - `tail` (the default) makes the experts the encoder's final stage --
    `hidden_dims` minus its last two widths becomes the backbone and the
    mixture projects that to `hidden_dims.back()`, which stays the latent. This
    is what hash mode always did, and it is now open to all five species
    encodings.
  - `post` runs the mixture over the finished latent, preserving its width.
    This is the placement for an encoder with no MLP tail to give up, so the
    adapter architectures (FT-Transformer, TabNet, SAINT, GNN, ExcelFormer,
    HeterogeneousGNN) gain a mixture where they previously got a refusal.

  Asking a tail-less encoder for `tail` now raises an error naming `post`, and
  asking for TabM and a mixture together raises rather than dropping TabM in
  silence, which is what the dedicated MoE encoder did (it took no `TabMConfig`
  at all).

  The mixture reaches every encoding because the encoder tail is now one shared
  thing (`EncoderTail` in `encoder.hpp`) rather than a per-encoder copy: all
  five encoders build it through `build_encoder_tail` and run it through
  `forward_encoder_tail`, so an MLP, a TabM ensemble and a backbone-plus-mixture
  are three settings of one seam. `PlotEncoderMoE`, which duplicated the hash
  encoder's featurisation to bolt a mixture on the end, is gone, and the three
  copies of the encoder-dispatch if-else chain in `model.cpp` collapse into one
  `encode_all`.

- **`resolve train --moe-routing / --moe-placement / --n-experts /
  --expert-hidden-dims / --moe-top-k / --moe-noise-std /
  --moe-aux-loss-weight`.** The CLI could not reach the mixture at all, so a
  standalone run could not train the architecture the bindings could. `resolve
  info` prints the placement alongside the routing.

- **R: `resolve.train.dataset(moeRouting =, moePlacement =, nExperts =,
  expertHiddenDims =, moeTopK =, moeNoiseStd =, moeAuxLossWeight =)`**, and
  Python `resolve_core.MoEPlacement` with `ModelConfig.moe_placement`.

### Fixed

- **`ResolveModel::get_gate_probs` reported nothing for every non-hash model.**
  It returned an undefined tensor unless the hash-mode MoE encoder was built,
  which was indistinguishable from "MoE is off" even when a mixture was
  actively routing. It now returns the real probabilities for the encoders its
  three-argument signature can drive (hash, TraitNet, the adapter
  architectures) at either placement, and raises for embed / sparse /
  rank_pool / transformer -- whose species inputs that signature does not
  carry -- pointing at `forward_with_aux`, which does.

### Changed

- **Retrain a mixture-of-experts checkpoint on a non-hash encoding.** Under the
  `tail` default an embed / sparse / rank_pool / transformer model with
  `moe_routing` set now puts the experts in the encoder tail (`encoder.backbone`
  + `encoder.moe`) where it previously appended a block to the latent
  (`post_moe`). Set `moe_placement = post` to keep the old architecture.
  **Hash-mode MoE checkpoints are unaffected** -- `tail` is exactly what they
  already were, down to the parameter names -- and **a checkpoint with
  `moe_routing = none`, which is every checkpoint anyone has trained through
  the paper pipeline, is untouched.**

### Tests

- `src/core/tests/test_moe_placement.cpp` (17 cases): a tail mixture builds,
  runs and trains on all five encodings; the tail's parameters are named
  `backbone` + `moe` and a plain run still writes `mlp`; the backbone/mixture
  split follows `hidden_dims`; the load-balancing loss reaches the optimizer
  (weighting it moves the trained gate); post preserves the latent width and
  covers the tail-less encoders; both refusals; gate probabilities; and
  checkpoint round-trips at either placement, including a parameter-by-parameter
  equality check on reload. `tests/core/test_moe.py` (27) covers the Python
  surface, `r/tests/testthat/` (33) the R one, and `tests.yml` gains a CLI
  end-to-end step asserting the flags change the model and the refusal names
  the placement that works.

## v0.8.2 (2026-08-31)

Two fixes that share a root: a guard that was compile-time where it needed to
be run-time, and a seed that covered less than the tests assumed.

### Fixed

- **A CPU run no longer touches the CUDA runtime (#114).** `Trainer::train_epoch`
  acquired its two CUDA streams before checking whether the run was on CUDA at
  all. `RESOLVE_HAS_CUDA` is a COMPILE-time guard, so a CUDA-enabled build ran
  those lines on a `device="cpu"` run too, initializing the CUDA runtime and
  killing the first epoch on a host with no usable driver -- a CPU queue node, a
  CI runner, a laptop. The streams are now held in `std::optional` and acquired
  inside the branch that already gated every USE of them, so the GPU prefetch
  path is unchanged and the CPU path is driver-free. The decision is the pure
  `resolve::use_hash_prefetch()` (`gpu.hpp`), unit-tested without a CUDA device
  the way `decide_oom_retry` is.

- **A seeded run now means what the tests assumed (#115).** The seed passed to
  `prepare_data`, `cross_validate` and `cross_validate_spatial` governs the
  SPLIT. Model weight initialisation draws from the process-global torch RNG,
  as any PyTorch module does, and nothing on the library path seeds it -- so two
  runs with the same seed started from different weights. Measured: six seeded
  `cross_validate_spatial` runs give six different fold losses at one thread as
  at twenty-four, while seeding the global RNG first makes five bit-identical.
  Two test suites were asserting the reproducibility this does not provide and
  getting it only by luck, which is why CI went red at random on unchanged
  code.

  The engine is deliberately unchanged: seeding a global stream from inside
  `fit()` is the side effect issue #107 avoided for pretraining, and weight
  init following the global RNG is the ordinary PyTorch contract. Instead the
  contract is documented (`docs/api/trainer.md` shows the two-seed form) and
  pinned from both sides, and the 17 parameter-recovery cases in
  `test_recovery.cpp` now fix their starting weights, so a correlation
  threshold is no longer evaluated on a fresh random draw each run.

**No library behaviour changed by #115** -- it is a test and documentation fix.
If you rely on reproducible fits, seed `torch.manual_seed()` before
constructing the model; the CLI's `--seed` already does.

## v0.8.1 (2026-08-30)

Three reported defects, all of the same shape: a value the API accepts and
persists, and then does not act on.

### Fixed

- **`SelectionMode` is honoured outside the hash encoding (#113).**
  `apply_selection` was called only inside the hash branch of the loader, so a
  `rank_pool` / `transformer` / `sparse` dataset recorded the selection it was
  given on its schema and encoded every species anyway, and the embed branch
  hardcoded `Top` whatever it was asked for. Each encoding now takes its
  per-plot species budget from the knob that also fixes its width: `top_k` for
  hash, `top_k_species` for embed, and the new **`DatasetConfig.species_budget`**
  for the variable-width encodings. The new knob defaults to `0` -- no budget --
  so every existing configuration encodes exactly what it encoded before;
  setting it makes a top-versus-bottom species ablation reachable on the pooled
  encoders for the first time. The species vocabulary is still fitted over every
  record, so the arms of an ablation share one integer-code namespace and stay
  comparable. The schema now records the selection the run APPLIED, which is
  `All` for a pooled or sparse load with no budget, so a checkpoint can no
  longer report a selection that never happened. Threaded through the schema,
  the checkpoint (`schema_species_budget`, absent on older checkpoints and read
  as `0`), `dataset_config_from_checkpoint`, the C ABI, nanobind, R
  (`config = list(species_budget = ...)`) and the CLI (`--species-budget N`).

- **`Predictor.load(device="cpu")` works on a machine with no CUDA device
  (#112).** `Trainer::load` called `InputArchive::load_from(path)` without the
  requested device, so the unpickler restored every tensor to the device the
  checkpoint was SAVED on and the `model->to(device)` that follows never got the
  chance -- reading a GPU-trained checkpoint on a GPU-less node threw "No CUDA
  GPUs are available" from inside deserialization. The device is now passed to
  the unpickler, in `Trainer::load`, `Trainer::load_state`, and (forced to CPU,
  since they return only scalars) `load_train_config` / `load_run_metadata`.

- **An optional role can be cleared (#111).** `roles.latitude = None` raised
  `TypeError` on the Python bindings: `def_rw` on a `std::optional` member gave
  a getter that read back `None` and a setter that refused it, because a
  nanobind function with no argument annotations takes a fast dispatch path that
  rejects every `None` argument before any caster runs. The five optional role
  columns are now bound with an explicit `str | None` setter. The empty string
  also means unset engine-wide (`RoleMapping::as_column`), so the sentinel
  downstream code already uses keeps working instead of failing with `column not
  found: ""`. A non-empty column name the file does not carry is still the loud
  configuration error it has been since #94.

- **A checkpoint saved before `fit()` recorded `train_batch_size` as 0.**
  `Trainer`'s requested-batch-size tracker started at `0`, which
  `save_train_config` reads as a genuine request, rather than at the `-1` that
  means "no separate request known" and persists the configured size.

### Added

- `resolve info` prints a **Data Encoding** block: the loading-side
  `DatasetConfig` the checkpoint implies, which is the one `resolve predict`
  rebuilds. Driven by the shared field registry, like the Training Configuration
  block beside it.
- `resolve_core.effective_selection(config)` reports the selection a dataset
  built under a config will actually apply.

## v0.8.0 (2026-08-07)

A sweep of issues #102-#110. The engine is the only implementation, the CLI
covers what the bindings cover, and four knobs that were persisted but wired to
nothing now do what they say.

### Breaking

- **The Python POC is gone.** `src/resolve/` (55 files) is deleted; `import
  resolve` no longer resolves. `resolve_core` is the Python surface, and the
  root `pyproject.toml` no longer declares a package. It was kept in-tree for
  one stated reason, the unported `rank_pool` and `transformer` encoders, and
  both have been wired end to end in C++ for some time.

### Retrain before comparing numbers

Existing checkpoints all load. These three change what a loaded model predicts:

- **HeterogeneousGNN attention was single-head.** `HeterogeneousGNNConfig::n_heads`
  reached `TypedMessagePassingLayerImpl` and was dropped on the floor. It is now
  real multi-head attention over disjoint `out_features / n_heads` slices. The
  default is 4, so parameter shapes are unchanged and predictions are not.
- **TabNet checkpoints recording `use_sparsemax = false`** now genuinely run
  1.5-entmax where they previously ran sparsemax regardless.
- **`unknown_fraction` / `unknown_count` carry values when scoring.** Training
  through the plain `from_csv` path still reads 0.0 for every plot, which is the
  correct value there, so training is bit-for-bit unchanged. Scoring through the
  vocabulary-reusing loaders now feeds real values through a weight that only
  ever saw zeros.

### Correctness

- **Checkpoints carry the fitted vocabularies** (#102). A checkpoint stored only
  the *sizes* of the species and taxonomy vocabularies, so scoring new data from
  a checkpoint alone re-fitted the codes and every non-hash encoder looked up
  other species' embedding rows: wrong predictions, no error. `ResolveSchema`
  now carries the ordered species, genus and family vocabularies, and
  `Predictor` rejects a dataset whose vocabularies are not the model's rather
  than silently scoring it. A pre-0.8.0 checkpoint still loads, with a warning.
- **1.5-entmax is the published operator** (#103). `entmax15` dropped the
  `(alpha - 1)` factor of Eq. 13 in Peters, Niculae and Martins (ACL 2019), so
  it ran at a different temperature and collapsed onto sparsemax's support. It
  is now that paper's exact sort-based Algorithm 2, with the closed-form
  Proposition 1 backward.
- **`LossConfigMode::NCA` trained something else.** `PhasedLoss::from_config`
  had no NCA case and fell through to `Combined`, while `NCALossImpl` had zero
  call sites. The preset is live, and its three hyperparameters are now
  `TrainConfig` fields instead of unreachable constants. R's
  `resolve.train.dataset()` also rejected `lossConfig = "nca"` outright.
- **The effective batch size was unreadable after `fit()`** (#105). The OOM
  auto-halve report compared a value `fit()` restores before returning, so it
  was unreachable, and a `save()` after `fit()` recorded the requested batch size
  as the effective one.
- **Uninitialized read in the GNN adapter.** An out-of-range `GNNType` from a
  newer checkpoint or the C ABI read an uninitialized enum.
- **The CLI silently dropped every covariate** (#104). `resolve train` read
  `--header` but nothing populated `RoleMapping::covariates` or `categoricals`,
  so a CLI-trained model was structurally different from the same configuration
  trained through `resolve_core` or R.

### Reproducibility

- **Pretraining is seeded** (#107). `PretrainConfig`, `MLMPretrainConfig` and
  `VAEConfig` gain a `seed`, and every shuffle, mask, corruption and
  reparameterization draw goes through one `PretrainRng` seam. A pretraining run
  no longer advances the global RNG stream, so it cannot shift the dropout draws
  of the finetuning that follows. Module dropout is the exception and is
  documented as such: `torch::nn::Dropout` takes no generator.
- **`resolve train --seed N`** seeds weight initialization, the split and the
  cross-validation folds. Two identical invocations previously produced
  different models.

### CLI

- `--covariate` and `--categorical` (repeatable) on `train` and `predict`,
  `--seed`, cross-validation (`--cv-folds`, `--cv-spatial`, ...), and roughly
  thirty `TrainConfig` / `ModelConfig` / `DatasetConfig` flags the bindings
  already exposed.
- A declarative flag table per subcommand generates the usage text and **rejects
  unknown flags**, naming the near miss. `--maxepochs 10` was previously ignored
  and the default used, with no diagnostic.
- `resolve info` prints every architecture sub-config and the training
  configuration.
- `resolve predict` writes a classification target as the original label plus a
  `<target>_code` column.

### Maintainability

- **One field registry per config struct** (#108). An X-macro list gives each
  field its name and checkpoint key exactly once, and the checkpoint reader and
  writer, the C ABI value tree, the nanobind bindings, the JSON sidecar and
  `resolve info` are all visitors over it. A member added without a registry row
  fails a `static_assert`. Every archive key spelling is unchanged.
- **Compiler warnings are on** (#109). `-Wall -Wextra -Wshadow
  -Wnon-virtual-dtor` on GCC/Clang and `/W4 /permissive-` on MSVC, applied to
  the engine, the C ABI, the CLI and the tests but not to vendored
  dependencies. 619 MSVC and 56 GCC warnings fixed; `-Werror` is armed on the
  Linux CI job.
- Four pretraining loops that each carried their own copy of the epoch scaffold
  now share one, so a fifth pretext task is one loss function.

### Testing and CI

- The Catch2 suite goes from 281 to 375 cases (2999 to 4563 assertions).
- New `tests/core/`, a pytest suite over `resolve_core` including
  parameter-recovery cases that fit to convergence and assert held-out
  correlation and accuracy. The 8-job `python-tests` matrix it replaces was
  installing and exercising the deleted POC, and the production Python surface
  had no automated test at all.
- A CLI end-to-end job trains, inspects and predicts over a committed fixture,
  asserting covariates reach the model, that `--seed` reproduces, and that every
  rejection path exits non-zero. CI previously ran `resolve version` and
  `resolve help`.
- A mechanical check that every public nanobind name is re-exported from
  `resolve_core`. Twelve types and the `fuzzy` submodule were reachable only
  through the private module.

### Removed

- `EncodedSpecies`, a struct with no producer and no caller.
- Documentation claims that the build fetches CLI11 and fast-cpp-csv-parser.
  Neither is fetched anywhere; the argument parser and the CSV reader are both
  hand-rolled.

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
