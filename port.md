# Port: `PlotEncoderRankPool` (+ `PlotEncoderTransformer`) to C++ Engine

**Mission:** end-to-end port of the Python POC's rank-pool species encoder
to the C++ engine in this repo, with **the same S+++ rigor we applied to
the categorical-covariate port on 2026-05-19**. No shortcuts. The bar is
"a v7 hash-headline experiment can be re-run on the rank-pool variant
without any Python-side feature being dropped or worked around."

Read **all** of this document before touching code. The categorical port
revealed seven distinct failure modes that this document explicitly tells
you how to avoid; if you skim, you will re-hit them.

---

## 1. Why this matters

The paper headline is `rank_log1p_big` (29M-param MLP with rank-pool
species encoding). Numbers: lifts area band25 from 53.9% (hash_32) to
69.0% and EUNIS to 87.2%. **None of the headline numbers can be reproduced
on the C++ engine today** because the encoder is unimplemented.

Hash-mode v7 variants are already full-feature-parity on C++ after the
2026-05-19 categorical port. Closing the rank-pool gap is the last C++
blocker before the paper can claim "high-performance C++/CUDA engine" for
the headline result.

Without this port the paper either ships on the legacy Python POC backend
(undercuts the engine story) or ships without the rank-pool headline
(undercuts the result). Neither is acceptable.

---

## 2. Hard rules (CRITICAL)

- **Never skip work on RESOLVE.** No "we'll add this later", no "the
  current code mostly handles it", no "we don't need bindings for this".
  Every port deliverable is: header + cpp impl + integration into
  dataset/model/trainer/predictor/checkpoint + nanobind + Rcpp + Catch2
  tests + smoke run that exercises the path end-to-end + CLAUDE.md
  update. The categorical port has all eleven of those pieces. So must
  this one.
- **Mirror the Python POC behaviorally, not aesthetically.** The C++
  implementation should produce numerically-equivalent latents on
  matched inputs. Sort orders, weight schemes, masking conventions,
  un-known-species padding — match the POC exactly, then validate via
  the Python<->C++ parity smoke at the end.
- **Do not stub.** No `throw std::runtime_error("not implemented yet")`
  paths that the rest of the system relies on. If a feature is
  incomplete, the build should not link. If you find a stub you didn't
  write, replace it; don't extend it.
- **Update `CLAUDE.md` last.** Once the port is functional and smoke is
  green, edit `CLAUDE.md`'s "Remaining Work" section to remove
  `rank_pool` / `transformer` from the deferred list and add them to
  "Completed Infrastructure". Do not edit `CLAUDE.md` before that.

---

## 3. What exists today

### 3.1 C++ stubs (just enums + ResolveBatch fields)

- `include/resolve/types.hpp`
  - `enum class SpeciesEncodingMode { Hash, Embed, Sparse, RankPool, Transformer };`
  - `ResolveBatch::pool_genus_ids / pool_family_ids / pool_weights / pool_mask / pool_has_cover` tensors (with `.to(device)` already wired).
  - `ResolveDataset::pool_*` accessors + private members (already in `dataset.hpp` lines ~90–95, 162–168). The loader does NOT populate them today.
- `include/resolve/encoder.hpp` — declares (but does not implement)
  `PlotEncoderRankPoolImpl` and `PlotEncoderTransformerImpl` shells
  around line 622–693. Forward methods take pool tensors and a few
  hidden-dim args; bodies are stubs.
- `cpp_src/model.cpp` lines 120–162 — the constructor branches for
  `SpeciesEncodingMode::RankPool` and `Transformer` already wire the
  encoder slot via `register_module("encoder", PlotEncoderRankPool(...))`,
  but the encoder's forward path is incomplete.
- `cpp_src/encoder_pool.cpp` exists as the intended home for the pool
  encoders' implementation.

### 3.2 Python POC behavioral spec — READ THIS FIRST

The Python implementation in `src/resolve/` is a complete, working
rank-pool encoder. It is the **behavioral spec**. Read these files in
order before writing any C++:

1. **Data pipeline** — how pool tensors are built from species records:
   - `src/resolve/encode/rank_pool.py` — full implementation of the
     rank-pool encoder (vocab building, weight schemes, mask handling).
   - `src/resolve/data/dataset.py` (`from_fast_csv` path) — how
     pool_genus_ids / pool_family_ids / pool_weights / pool_mask /
     pool_has_cover get assembled per-plot from species observations.
     This is what the C++ CSV loader needs to mirror.
   - `src/resolve/csrc/fast_loader.py` if there's any C++ ingest
     helper for the pool path; otherwise the polars path is the spec.
2. **Model layer** — how pool tensors are consumed by the encoder:
   - `src/resolve/model/encoder.py` — `PlotEncoderRankPool` and
     `PlotEncoderTransformer` classes. These are the reference forward
     implementations. Note in particular:
     - genus/family embedding tables (per-species lookups, not per-slot)
     - the weight scheme switch (`log1p` vs `rank` vs `raw`)
     - cover-dropout behavior (`cover_dropout` zeros out the cover
       channel during training)
     - the `has_cover` per-plot flag and how it gates the cover term
     - mask handling: masked species contribute zero to the weighted
       pool; the pool divides by the *masked* count, not the padded
       count
     - the `weights * embed` -> sum -> normalize path (don't double-
       normalize)
3. **Where rank-pool is invoked from the higher layer**:
   - `src/resolve/model/resolve.py` — the `ResolveModelImpl.forward`
     flow that dispatches to `self.encoder` when
     `species_encoding=="rank_pool"`. Lines 397–404 show how
     pool_genus_ids, pool_family_ids, pool_weights, pool_mask,
     pool_has_cover are threaded.

**You must read each of these files in full**. Grep snippets will mislead
you. The categorical port hit a near-miss because we assumed encoders
needed modification before reading `resolve.py:386–389` and discovering
the concat happened at the model layer, not the encoder layer. Don't
repeat the mistake.

### 3.3 The categorical port as a template (look at this)

The just-completed categorical port (2026-05-19) is the canonical
template for this kind of work. Read all of:

- `include/resolve/categorical.hpp` + `cpp_src/categorical.cpp` —
  freestanding module (vocab + embedder). Same shape: a freestanding
  `pool_encoder.hpp` + `pool_encoder.cpp` module is probably the right
  factoring here too, **unless** mirroring `src/resolve/encode/rank_pool.py`
  argues for splitting the vocab side into `pool_vocab.hpp`.
- `cpp_src/model.cpp` `fuse_categoricals_` helper — the pattern of "fuse
  domain-specific inputs into a base tensor before the encoder runs" is
  reusable. RankPool is *different* though: the rank-pool encoder itself
  consumes the pool tensors. There is no concat-into-continuous step.
  The pattern to copy is "thread the new inputs through every
  forward() / forward_with_aux / forward_single / get_latent /
  encode / encode_with_aux", not the fuse-into-continuous bit.
- `tests/test_categorical.cpp` — the Catch2 structure (TempFile, sectioned
  tests for fit/encode/save-load/back-compat/forward/checkpoint roundtrip).
  Pattern this for `test_rank_pool.cpp`.
- `cpp_smoke_cat.py` (in the paper repo at
  `J:\Phd Local\Gilles_paper_resolve\cpp_smoke_cat.py`) — end-to-end
  Python smoke. Mirror this for rank-pool with a tiny synthetic dataset
  that has enough species per plot to exercise the pool tensors.

---

## 4. Concrete scope

The port must deliver all of these. Each bullet is independently
testable; check them off as you go.

### 4.1 Vocab + data pipeline

- [ ] **Species vocabulary** for rank-pool mode. Hash mode hashes species
      IDs on the fly; rank-pool needs a fitted vocab so each unique
      species gets a stable integer ID, which then becomes the lookup
      index into a learned embedding table. Mirror
      `src/resolve/encode/rank_pool.py`'s vocab build: sort by
      frequency descending, keep top-K (`top_k_species` from
      `DatasetConfig`) or all (`selection==All`), reserve 0 for UNK.
      Implement as a `RankPoolVocab` class with the same save/load
      pattern as `CategoricalVocab` (length-prefixed UInt8 archive
      serialization, back-compat `try_read` on load).
- [ ] **CSV loader pool-tensor population.** Today
      `ResolveDataset::load_species_data` builds raw species records
      and (for hash mode) computes hash embeddings. For rank-pool, the
      loader must also build:
      - `pool_genus_ids_` : (n_plots, max_species) int64. Per-species
        genus ID (lookup via `TaxonomyVocab`), padded with 0.
      - `pool_family_ids_` : same shape, family IDs.
      - `pool_weights_` : (n_plots, max_species) float32. Weights per
        species per plot, computed according to the `pool_weighting`
        scheme:
          - `log1p` : `log1p(abundance)`
          - `rank`  : reverse rank within the plot (most-abundant = 1,
                      next = 2, ...) — see Python POC for exact
                      tie-breaking
          - `raw`   : abundance as-is
      - `pool_mask_` : (n_plots, max_species) bool. True where a real
        species exists, false in the padding region.
      - `pool_has_cover_` : (n_plots,) bool. True iff the plot had real
        cover values (not all-1 default).
      Populate these in `load_species_data` or a dedicated
      `build_pool_tensors_` helper. **Important:** `max_species` should
      be the actual max over the dataset (some plots have 5 species,
      some 200). Do NOT cap it artificially — that loses signal. The
      memory cost is `n_plots * max_species * (16 bytes for ints + 5
      bytes for weights+mask)` which is manageable up to ~1M plots ×
      ~200 species ≈ 4 GB — for larger datasets, consider CSR/COO
      format (defer that decision, but document it in CLAUDE.md if you
      do).
- [ ] **`DatasetConfig` fields** — add a `pool_weighting` enum field if
      not present (`Log1p`, `Rank`, `Raw`). Update bindings + Rcpp.
- [ ] **`ResolveSchema` fields** — `n_species_vocab` already exists; the
      schema needs to carry the vocab itself for save/load. Mirror
      categorical port: vocab lives on `ResolveDataset`, captured by
      `Trainer` at `prepare_data` time, persisted by `Trainer::save`,
      restored by `Trainer::load` (extend the 3-tuple return to a
      4-tuple, OR store on schema — pick the cleaner approach and apply
      consistently).

### 4.2 Encoder implementation

- [ ] **`PlotEncoderRankPoolImpl` forward path** in
      `cpp_src/encoder_pool.cpp`. The forward signature (from
      encoder.hpp) is:
      ```cpp
      torch::Tensor forward(
          torch::Tensor continuous,
          torch::Tensor species_ids,     // (B, max_species) int64
          torch::Tensor pool_genus_ids,  // (B, max_species) int64
          torch::Tensor pool_family_ids, // (B, max_species) int64
          torch::Tensor pool_weights,    // (B, max_species) float32
          torch::Tensor pool_mask,       // (B, max_species) bool
          torch::Tensor pool_has_cover   // (B,) bool
      );
      ```
      Behavior must match `src/resolve/encode/rank_pool.py`:
      1. Look up species embedding for each species in the plot
         (vocab_size, species_embed_dim).
      2. Look up genus + family embeddings (per-species, not per-slot —
         this is different from hash-mode taxonomy which uses per-slot).
      3. Concat species + genus + family per-species embeddings.
      4. Multiply by `pool_weights`, zero out via `pool_mask`.
      5. Sum across the species axis -> per-plot pooled vector.
      6. Apply cover_dropout if training and pool_has_cover[plot] is true.
      7. Concat with `continuous` -> MLP.
- [ ] **`PlotEncoderTransformerImpl` forward path** — same data,
      different reduction (attention pooling instead of weighted sum).
      Implement the simplest viable version (standard transformer block
      + attention pool) that matches the Python POC. Do not stub.
- [ ] **`encoder_pool.cpp` weight extraction methods** —
      `get_species_weights()`, `get_genus_weights()`,
      `get_family_weights()`. These return the learned embedding tables
      (detached, cloned). Used by `Predictor::get_*_embeddings()`.

### 4.3 Model integration

- [ ] Confirm the existing `cpp_src/model.cpp` lines 120–162 wiring of
      rank-pool / transformer encoders compiles and actually invokes
      the forward you write. The existing scaffolding looks right but
      needs to be exercised end-to-end.
- [ ] `latent_dim()` must report the right value (continuous +
      species_embed + genus_embed + family_embed all concatenated). The
      existing logic delegates to `encoder_rank_pool_->latent_dim()`;
      make sure that returns the post-MLP hidden dim, not the pre-MLP
      input dim.
- [ ] Thread `pool_genus_ids` / `pool_family_ids` / `pool_weights` /
      `pool_mask` / `pool_has_cover` through every model API surface
      that already takes them (forward, forward_with_aux,
      forward_single, get_latent, encode, encode_with_aux,
      encode_with_activations). Most of these are already plumbed —
      verify each one actually reaches the encoder for RankPool /
      Transformer.

### 4.4 Trainer integration

- [ ] The trainer already has `train_pool_*_` and `test_pool_*_`
      members + GPU caching + per-batch slicing. Verify those are
      actually populated by `prepare_data(ResolveDataset)` from
      `dataset.pool_genus_ids()` etc. — the C++ side may have stubs.
- [ ] `cross_validate` and `cross_validate_spatial` already
      concat+split pool tensors across folds. Verify that survives the
      port without re-introducing dim mismatches.
- [ ] Smoke run a 2-epoch CPU training to confirm prepare_data → fit
      → save works for rank-pool, the same way `cpp_smoke_cat.py`
      verifies it for categoricals.

### 4.5 Checkpoint integration

- [ ] `RankPoolVocab::save` / `::load` mirroring `CategoricalVocab` and
      `TaxonomyVocab`. Length-prefixed UInt8 tensors so it survives the
      LibTorch archive on Windows + Linux. Back-compat `try_read` for
      pre-port checkpoints.
- [ ] Persist the vocab via `Trainer::save` (extend the existing
      `categorical_vocab_.save(archive, "trainer_categorical_")` line
      in `cpp_src/trainer.cpp`).
- [ ] `Trainer::load` extends its return tuple to include the rank-pool
      vocab (or stores on schema — be consistent with the categorical
      port's decision).
- [ ] Update `Predictor`'s constructors + `Predictor::load` to thread
      the vocab through. Expose `Predictor::species_vocab()` if it
      doesn't exist already.
- [ ] CLI `info_cmd.cpp` destructures `Trainer::load`'s tuple — update
      that destructure to the new arity (the categorical port hit this).

### 4.6 Bindings

- [ ] **nanobind** (`python/src/bindings_*.cpp`):
  - `bindings_types.cpp` — expose `PoolWeighting` enum if added; expose
    any new schema/config fields.
  - `bindings_dataset.cpp` — expose `pool_*` accessors via
    `THPVariable_Wrap` lambdas (the pattern fixed during the
    categorical port — do NOT use `def_prop_ro(&accessor)` for tensor
    returns).
  - `bindings_model.cpp` — every `forward / forward_with_aux /
    forward_single / get_latent / encode_with_activations` already
    takes pool args; confirm the kwargs and defaults are right.
  - `bindings_trainer.cpp` — same for predict and prepare_data_raw.
- [ ] **Rcpp** (`r/src/rcpp_*.hpp`):
  - `rcpp_dataset.hpp` — accept `pool_weighting` in DatasetConfig if
    added; expose `pool_*` accessors as `NumericMatrix` /
    `LogicalMatrix` / `LogicalVector` as appropriate.
  - `rcpp_common.hpp` may need a `tensor_to_r_bmat` helper for the bool
    pool_mask if R doesn't auto-convert.

### 4.7 Tests

- [ ] **`tests/test_rank_pool_encoder.cpp` already exists** — read it.
      It may already have the right shape of tests; extend or replace
      to cover the post-port behavior. New SECTIONs to add at minimum:
  - Vocab build + save/load roundtrip + back-compat on missing keys.
  - Forward shape: `(B, max_species)` pool tensors -> `(B, latent_dim)`.
  - Mask correctness: masked species do not contribute to the pool.
  - Weight scheme correctness: log1p vs rank vs raw produce different
    but predictable pool outputs.
  - End-to-end ResolveDataset (with a small CSV fixture) -> ResolveModel
    (rank-pool encoder) -> forward -> save -> Predictor.load ->
    predict_dataset roundtrip.
- [ ] Same shape for `test_transformer_encoder.cpp` (also exists).
- [ ] Add tests to `tests/CMakeLists.txt` if you create new files.

### 4.8 Smoke + parity

- [ ] **`cpp_smoke_rank_pool.py`** in the paper repo at
      `J:\Phd Local\Gilles_paper_resolve\` — mirror `cpp_smoke_cat.py`
      but with a small synthetic fixture that exercises rank-pool
      encoding. Verify schema fields, vocab roundtrip via predictor,
      end-to-end forward.
- [ ] **`scripts/run_v7_one_cpp.py` update**: remove the
      `if enc_cfg["species_encoding"] != "hash":` early-return that
      currently refuses rank-pool variants. Add `rank_log1p_big` etc.
      to the supported list.
- [ ] **2-epoch CPU smoke on the real dataset** for
      `rank_log1p_big slope seed=3` to validate the runner end-to-end
      on the paper's data, the same way the categorical port was
      smoke-verified.
- [ ] **Full 500-epoch parity** (or whatever cell the user picks) is
      OUT OF SCOPE for this port. Hand back to the user with the smoke
      output once it's green; they decide when to spend the GPU time.

### 4.9 Docs

- [ ] `CLAUDE.md` (this repo): in "Remaining Work" section, remove the
      "C++ rank_pool/transformer implementation" entry. Add a bullet
      under "Completed Infrastructure" describing the port (mirror the
      "Categorical covariates" bullet I added 2026-05-19 for tone).
- [ ] `CLAUDE.md` (paper repo at
      `J:\Phd Local\Gilles_paper_resolve\CLAUDE.md`): update the
      "RESOLVE backend" section's note that `rank_log1p_big` is blocked
      on the C++ rank_pool port — remove that note.
- [ ] Auto-memory at
      `C:\Users\Gilles Colling\.claude\projects\J--Phd-Local-Gilles-paper-resolve\memory\`:
      mark `project_cpp_rankpool_gap.md` as a "closed gap" the same way
      I did for the categorical port. Add a new
      `project_cpp_rankpool_port_done.md` describing what landed.
      Update `MEMORY.md` index.

---

## 5. Build / install / smoke cycle (use this exactly)

The build script is `src/core/build_cuda/launch_build.ps1`. It defaults
to pyenv's torch, but the paper venv uses a different torch and a
mismatch causes segfaults. **Always override `TORCH_DIR` and
`PYTHON_EXE` for the paper venv:**

```cmd
cmd.exe /c "set ""TORCH_DIR=J:\Phd Local\Gilles_paper_resolve\.venv\Lib\site-packages\torch\share\cmake\Torch"" && set ""PYTHON_EXE=J:\Phd Local\Gilles_paper_resolve\.venv\Scripts\python.exe"" && powershell -ExecutionPolicy Bypass -File C:\Users\GillesC\Documents\dev\RESOLVE\src\core\build_cuda\launch_build.ps1"
```

Notes (these were *all* learned the hard way during the cat port):

1. The path **must** use the `C:\Users\GillesC` junction (no spaces);
   PowerShell trips on `C:\Users\Gilles Colling` even when quoted.
2. The build log goes to `C:\tmp\resolve_build_log.txt`. Grep for
   `BUILD SUCCEEDED` / `BUILD FAILED` / `error C` to determine outcome.
3. **The build does NOT install the `.pyd` into the paper venv.** After
   a successful build, copy it manually:
   ```bash
   cp "C:/Users/GillesC/Documents/dev/RESOLVE/src/core/python/src/resolve_core/_resolve_core.cp312-win_amd64.pyd" \
      "J:/Phd Local/Gilles_paper_resolve/.venv/Lib/site-packages/resolve_core/_resolve_core.cp312-win_amd64.pyd"
   ```
   Otherwise your Python smoke loads the *old* pyd.
4. Run the build in the background (`run_in_background: true`) — full
   rebuild is ~5–15 min, incremental ~1–3 min.

Smoke after every meaningful change set:

```bash
cd "J:/Phd Local/Gilles_paper_resolve" && \
  ".venv/Scripts/python.exe" cpp_smoke_rank_pool.py 2>&1 | \
  grep -vE "^00007|<unknown" | tail -30
```

(The `grep -v` strips the unsymbolicated stack frames that libtorch
spews on any exception — they are useless for triage. Read the real
error message above them.)

---

## 6. Known pitfalls from the categorical port

Re-read each of these. They are the exact bugs I hit during 8 build
cycles for the categorical port. Avoiding them shortens your loop by
days.

### 6.1 nanobind cannot auto-convert `at::Tensor` from a property accessor

```cpp
// WRONG — produces "Unable to convert function return value to a Python type"
.def_prop_ro("categorical_ids", &resolve::ResolveDataset::categorical_ids)

// RIGHT — wrap via THPVariable_Wrap inside a lambda
.def_prop_ro("categorical_ids", [](const resolve::ResolveDataset& self) {
    const auto& t = self.categorical_ids();
    return t.defined() ? nb::steal(THPVariable_Wrap(t)) : nb::none();
})
```

This wasn't a categorical-port bug — every tensor accessor in
`bindings_dataset.cpp` had it. I fixed all of them along the way. If you
add new tensor accessors for rank-pool, use the lambda pattern from the
start.

### 6.2 `Trainer::load` tuple arity changes propagate to the CLI

`info_cmd.cpp:22` destructures `Trainer::load`'s return tuple. The
categorical port changed it from 2 to 3 elements. Re-do for rank-pool
if you change the arity again. **The build error is loud
(`tuple_size mismatch`) but only fires on the CLI link step**, so
if you only test the Python smoke you'll miss it.

### 6.3 Adding a positional arg to `prepare_data` breaks all in-repo callers

The 6 `prepare_data` call sites in `tests/benchmark_catch2.cpp` need
explicit `{}` slots for any new positional arg you add to the raw
overload. If you add a new arg at the END but BEFORE `test_size, seed`,
every benchmark call breaks until you patch them.

Less footgun-y option: add new args strictly at the END of the signature
(after `seed`), then existing callers don't need to change. (The
categorical port did NOT do this — it slotted the new arg between
pool_has_cover and test_size, which is why we had to patch the
benchmarks.)

### 6.4 MSVC rejects 2-D nested-initializer-list `torch::tensor` literals

```cpp
// WRONG on MSVC: "cannot convert argument 1 from 'initializer list' to 'TensorDataContainer'"
auto ids = torch::tensor({{0L, 1L}, {1L, 2L}}, torch::kInt64);

// RIGHT: build element-wise
auto ids = torch::zeros({2, 2}, torch::kInt64);
auto a = ids.accessor<int64_t, 2>();
a[0][0] = 0; a[0][1] = 1;
a[1][0] = 1; a[1][1] = 2;
```

### 6.5 ModuleHolder constructor disambiguation

```cpp
// WRONG: ambiguous with the ModuleHolder default-construct-from-{}
CategoricalEmbedder embedder({}, /*embed_dim=*/8);

// RIGHT
CategoricalEmbedder embedder(std::vector<int64_t>{}, /*embed_dim=*/8);
```

### 6.6 `compute_diagnostics` bypasses the model's forward chain

It calls `model_->encode_with_activations(...)` which goes
straight to the encoder. **Any data transformation you put in
`fuse_*_()` or in `forward_with_aux` is bypassed by this path.**

For categoricals I added a `categorical_ids` param to
`encode_with_activations` and call `fuse_categoricals_` inside it. For
rank-pool: this method only supports hash mode today (`if (encoder_hash_)`
in `model.cpp`). Either extend it to support rank-pool too (with all
pool tensor args), OR explicitly return empty when called on a rank-pool
model. **Document the choice in `compute_diagnostics`'s docstring.**

### 6.7 `save_model_config` / `load_model_config` are easy to forget

The categorical port saved `categorical_embed_dim` to the schema but
forgot to save it to the model config too. On reload, the ModelConfig
defaulted to 8 even though the saved value was 4, and the model rebuilt
with the wrong embed_dim ate the saved weights and failed at the first
matmul.

**Lesson:** any new field on `ModelConfig` or `DatasetConfig` that
affects the model architecture must be added to BOTH `save_model_config`
and `load_model_config` (with `try_read` for back-compat). Cover this
with a test that does
`Predictor.load(trainer.save(...))` and asserts the loaded ModelConfig
matches the saved one field-for-field.

---

## 7. Done criteria (checklist for the user to verify)

Before declaring done, all of these must be true:

- [ ] `BUILD SUCCEEDED` in the build log, full rebuild (delete
      `build_cuda/CMakeCache.txt` first).
- [ ] All Catch2 tests pass: `build_cuda/tests/resolve_tests.exe` exits 0.
- [ ] `cpp_smoke_rank_pool.py` prints `RANK POOL SMOKE OK`.
- [ ] `cpp_smoke_cat.py` still prints `CAT SMOKE OK` (no regression).
- [ ] `cpp_smoke.py` still prints `SMOKE OK` (the original hash smoke
      from the bug-fix session, no regression).
- [ ] A 2-epoch CPU smoke of
      `python scripts/run_v7_one_cpp.py --encoding rank_log1p_big --target slope --seed 3 --max-epochs 2 --cpu`
      from the paper repo finishes with a `done.json` containing real
      metrics (not NaN, not zeros).
- [ ] `RESOLVE/CLAUDE.md` "Remaining Work" no longer mentions
      `rank_pool` / `transformer`; "Completed Infrastructure" has a new
      bullet for the port.
- [ ] Paper repo `CLAUDE.md` no longer says `rank_log1p_big` is blocked
      on the C++ rank_pool port.
- [ ] Auto-memory has a new `project_cpp_rankpool_port_done.md`.

---

## 8. Out of scope (do NOT do these)

- The full 500-epoch parity run. That's the user's call after the smoke
  is green; it costs hours of GPU and they want to decide.
- Commits / PRs to gcol33/resolve. Leave the diff uncommitted; the user
  reviews before any push.
- Bindings polish for languages other than Python + R (Rcpp).
- Any refactor of the existing hash/embed/sparse encoders. If they need
  touching, that's a separate task — flag it and stop.
- "Optimization" beyond the obvious (use `index_select` not Python
  loops; use `bmm` for the weighted sum if it's simpler than `(emb *
  weights.unsqueeze(-1)).sum(1)`). No torch.compile, no custom CUDA
  kernels, no fused ops. Match Python POC's perf at minimum; a 2x
  speedup is gravy but not required.

---

## 9. Pointers + quick refs

- **Repo root:** `C:\Users\Gilles Colling\Documents\dev\RESOLVE` (use
  the `C:\Users\GillesC` junction in all build / shell invocations).
- **Paper repo:** `J:\Phd Local\Gilles_paper_resolve`. The paper venv
  has the installed resolve_core that you'll need to overwrite after
  every rebuild.
- **Categorical port memory:**
  `C:\Users\Gilles Colling\.claude\projects\J--Phd-Local-Gilles-paper-resolve\memory\project_cpp_categorical_port_done.md`
  has the full file-level diff of the categorical port — use it as a
  cross-reference for "which files did I touch and why" while you do
  the same for rank-pool.
- **RESOLVE CLAUDE.md "Never skip work" rule** at the very top —
  re-read it. The user enforces it strictly. Do not surface "we could
  defer X" suggestions; if you find a gap, port through it.

Estimate: 1–2 days of careful work end-to-end. Track progress with the
TaskCreate tool, mirror the 11-step task structure I used for the
categorical port (you can read those subtask titles via TaskList for
inspiration — they're still in the harness as `cat-port 1..11`).

Good luck. The standard is S+++. Anything less, the user calls out as
cheating — they've done it before, they'll do it again.
