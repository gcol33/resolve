# RESOLVE — fuzzy_index_plan.md implementation: progress

## What is done and working

### C++ FuzzyIndex (verified, all tests pass)

Implementation of `dev_notes/fuzzy_index_plan.md`:

- `src/core/include/resolve/fuzzy.hpp` — public API (`FuzzyIndex`, `Match`, `BuildOptions`, `QueryOptions`).
- `src/core/cpp_src/fuzzy_internal.h` — internal trie + automaton + `FuzzyIndex::Impl`.
- `src/core/cpp_src/fuzzy_automaton.cpp` — UTF-8 decode, codepoint lowercase, Damerau–Levenshtein DP-row state machine.
- `src/core/cpp_src/fuzzy_index.cpp` — packed-vector trie, per-bucket sub-tries, build pipeline.
- `src/core/cpp_src/fuzzy_search.cpp` — iterative DFS over trie driven by the automaton, top-N heap with adaptive k tightening, OpenMP `query_batch`.
- `src/core/tests/test_fuzzy.cpp` — 15 Catch2 cases, 531 assertions, **all passing** (full test suite: 136 cases / 1090 assertions, no regressions).

Two bugs caught and fixed during testing:
1. `Trie::find_or_create_child` held a reference into `child_lists_` after `emplace_back` (dangling).
2. Empty-string entry at trie root was never visited — DFS started at `root.first_child`. Fixed by checking root acceptance against the initial DP row before descending.

### WFO Python integration

- `src/resolve/ext/wfo.py` — `_match_fuzzy` routes through native `_resolve_core.fuzzy.FuzzyIndex` when present, falls back to `difflib` otherwise. Bucket fn = lowercase genus prefix.

### Build wiring

- `src/core/CMakeLists.txt` — fuzzy sources + header added; OpenMP discovery added.
- `src/core/python/CMakeLists.txt` — fuzzy sources + `bindings_fuzzy.cpp` added.
- `src/core/python/src/bindings.cpp` — `register_fuzzy(m);` call (currently **commented out**, see below).
- `src/core/python/src/bindings_common.hpp` — forward decl for `register_fuzzy`.
- `src/core/python/src/bindings_fuzzy.cpp` — nanobind glue.
- `src/core/tests/CMakeLists.txt` — `test_fuzzy.cpp` added.

### CLAUDE.md

Updated under "Architecture Improvements" with a one-line mention of the new fuzzy submodule.

## CUDA build infrastructure fixes (done as part of this session)

These were all stale-environment problems, **not caused by the fuzzy code**:

1. **MSVC version drift.** `do_build.bat` referenced `14.50.35717`, which no longer existed on disk (only `14.51.36231`). Also: VS2026 preview's `14.51.36231` is too new for CUDA 13.1's `cudafe++` (ACCESS_VIOLATION inside cudafe++ when parsing preprocessed output from MSVC 14.51).
2. **Fix:** installed MSVC 14.44 toolset side-by-side via VS Installer (component `Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`). Updated `do_build.bat` to use `14.44.35207` for everything. Updated `cuda_toolchain.cmake`'s `MSVC_ROOT` reference (cosmetic only — not used at build time, but stale path pointed at the gutted VS2022 BuildTools dir).
3. **Wrong Python env.** `do_build.bat` hardcoded `3.14.4` for torch + python. That env had `torch 2.11.0+cpu` — no CUDA headers, broke libtorch's `c10/cuda/CUDAMacros.h` (`cuda_cmake_macros.h` missing from CPU wheels). The actual CUDA nightly (`2.13.0.dev20260504+cu130`) was in pyenv `3.12.10`.
4. **Fix:** updated `do_build.bat` to point at `3.12.10`. Then ran `pip install --pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu130 torch` in **3.14.4** as well (per user request) — now `3.14.4` has `2.13.0.dev20260518+cu130` available too, but `do_build.bat` stays on `3.12.10`.

## Current state of the build

- Full CUDA build with MSVC 14.44 + cu130 nightly torch: **SUCCEEDED** end-to-end. Produced `src/core/python/src/resolve_core/_resolve_core.cp312-win_amd64.pyd`.
- Catch2 test binary builds and all fuzzy tests pass (verified via a separate CPU-only build earlier; the CPU build dir has since been deleted).

## Resolved: bad_cast was in `register_trainer`, not fuzzy

Bisection (commenting `register_trainer` / `register_pretraining` / `register_fuzzy` independently) localized the failure to `register_trainer`. Root cause: `bindings_trainer.cpp` bound `Trainer::predict`, `Predictor::predict`, `Predictor::get_embeddings`, etc. either via raw method pointers or lambdas taking `torch::Tensor` directly, and used `nb::arg(...) = torch::Tensor()` as defaults. nanobind 2.4.0 has no type caster for `torch::Tensor`, so at module init (when the default values are baked into the function signature) it throws `bad_cast`. Likely surfaced now (not at the commit time of 109bd57) because of the torch 2.13 nightly + nanobind 2.4.0 combination; older torch / older nanobind tolerated this.

**Fix.** Rewrote `bindings_trainer.cpp` to follow the same `nb::object` + `THPVariable_Unpack` pattern that `bindings_model.cpp` already uses (the pattern `bindings_model.cpp` explicitly documents as the workaround for the missing caster). Defaults are `nb::none()`; the lambda body converts `None` to an empty `at::Tensor` via the `unpack_or_empty` helper. Return values that are `torch::Tensor` go through `THPVariable_Wrap`; maps of tensors go through the existing `tensor_map_to_dict` helper.

**Verification.**
- `import _resolve_core` now succeeds. `Trainer`, `Predictor`, `JEPAPretrainer`, `fuzzy.FuzzyIndex` all present.
- `resolve.ext.wfo` imports cleanly and the backend logger reports the C++ backend in use.
- Smoke test on `FuzzyIndex.build` + `.query` + `.query_batch` (incl. Damerau transposition and `case_insensitive`) returns the expected matches.
- Full Catch2 suite: 136 cases / 1090 assertions, all passing (15 cases / 531 assertions just for `[fuzzy]`).

## Files touched (full list)

Modified:
- `CLAUDE.md`
- `src/core/CMakeLists.txt`
- `src/core/build_cuda/do_build.bat` (MSVC version + Python env)
- `src/core/cuda_toolchain.cmake` (cosmetic MSVC_ROOT path)
- `src/core/python/CMakeLists.txt`
- `src/core/python/src/bindings.cpp` (register_fuzzy now re-enabled)
- `src/core/python/src/bindings_common.hpp`
- `src/core/python/src/bindings_trainer.cpp` (rewritten — see bad_cast section above)
- `src/core/tests/CMakeLists.txt`
- `src/resolve/ext/wfo.py`

Added:
- `src/core/include/resolve/fuzzy.hpp`
- `src/core/cpp_src/fuzzy_automaton.cpp`
- `src/core/cpp_src/fuzzy_index.cpp`
- `src/core/cpp_src/fuzzy_internal.h`
- `src/core/cpp_src/fuzzy_search.cpp`
- `src/core/python/src/bindings_fuzzy.cpp`
- `src/core/tests/test_fuzzy.cpp`

## Background tasks left over

None — all builds completed, all tests pass, end-to-end import + smoke test verified.
