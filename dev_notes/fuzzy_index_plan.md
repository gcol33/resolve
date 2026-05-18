# Plan: Native C++ FuzzyIndex for RESOLVE

**Author note (read first):** This plan replaces the dropped `rapidfuzz`-or-`difflib` split in `src/resolve/ext/wfo.py` with a generic native fuzzy-string index in the C++ core. The index is general-purpose (no domain assumptions baked in), exposes an optional bucket hint for callers that have structural knowledge (e.g. WFO knowing the first word is usually a genus), and falls back to the existing Python difflib path when `_resolve_core` is not installed — same dual-backend pattern as the rest of RESOLVE.

---

## 1. Problem statement

### Current state

`src/resolve/ext/wfo.py:488` (`_match_fuzzy`) calls `difflib.get_close_matches` against the full `_name_index` of ~1.4M WFO scientific names. Per query:

- Ratcliff/Obershelp similarity, not Levenshtein (wrong metric for typos)
- Linear scan of every key, single-threaded Python
- No early termination, no index pruning
- ~seconds per query → minutes-to-tens-of-minutes for a 50k-species ingest

A previous code path tried `rapidfuzz` first and fell back to `difflib`. That dual-path was dropped. The previous design conversation considered "hard-require rapidfuzz" but that violates the project rule *"never depend on an external package for something you can implement natively in < 200 lines"* and would re-introduce the dropped antipattern.

### What we want

A native C++ fuzzy index that:

1. Works for arbitrary string vocabularies (no domain assumptions in core)
2. Lets callers opt into structural pruning hints (e.g. WFO's genus-prefix bucket)
3. Returns top-N matches within a max edit distance, with the actual distance
4. Builds in seconds, queries in microseconds, runs on the user's i9 + parallel batch
5. Drops cleanly into the existing C++ core + nanobind binding scaffolding
6. Pure-Python `difflib` path stays as fallback for environments without the C++ build (consistent with RESOLVE's existing dual-backend architecture)

---

## 2. Design decisions (decided — don't re-litigate without good reason)

### Algorithm: Trie + Levenshtein automaton (Schulz–Mihov 2002)

Considered alternatives and why rejected:

| Alternative | Why not |
|---|---|
| Naive pairwise DP / bit-parallel Myers across full vocab | O(N) per query, ~seconds at N=1.4M even bit-parallel |
| BK-tree | ~10k distance evals per query, slower than automaton walk; less cache-friendly |
| q-gram inverted index | Prunes well but distance verification still costly; awkward k-tuning |
| SymSpell | O(1) per query but 2–4 GB memory at N=1.4M, k=2; doesn't scale to k=3+; "vocab must fit in tuned memory budget" baked in |
| Levenshtein automaton + trie | Sublinear per query, ~100–200 MB memory at N=1.4M, scales smoothly in k. Production-proven (Lucene, Elasticsearch, Tantivy) |

**Schulz–Mihov** is the canonical reference. Build a DFA that accepts all strings within edit distance ≤ k of the query, then walk the dictionary trie and the DFA simultaneously: prune any subtree whose DFA state is dead.

For Damerau-Levenshtein (transpositions) we use the extended automaton from Mihov–Schulz 2004. Transpositions matter for species names ("Querucs" ↔ "Quercus").

### Bucket hint: opt-in, never baked into the index logic

Domain hints are passed as a `bucket_fn` at build time and a `bucket_hint` at query time. Without them, the index works as one global trie. With them:

- Build splits entries into N sub-tries keyed by `bucket_fn(entry)`
- Query first searches the hinted bucket's sub-trie
- If that bucket returns 0 matches within `max_edit_distance`, falls back to scanning all sub-tries (handles "genus itself was mistyped")

This is the **same code path** with a pruning step — not a separate fast/slow code path. The fallback is automatic and correct.

### Dual backend: not an antipattern violation

The CLAUDE.md rule against "primary path + fallback path" forbids duplicating algorithm logic. The C++/Python backend split is the project's *architecture*, not duplication: when `_resolve_core` is installed, the C++ path is the only path; when it isn't, the Python path is the only path. WFO follows the same convention as the rest of RESOLVE (see `src/resolve/backend/`).

### No internal dependency on `rapidfuzz`

After this lands, `rapidfuzz` should not appear anywhere in `src/resolve/`. The current `wfo.py` already dropped its rapidfuzz branch; this plan does not reintroduce it.

---

## 3. File layout

All new files. Nothing existing is moved or renamed.

```
src/core/
├── include/resolve/
│   └── fuzzy.hpp                      # Public API: FuzzyIndex, BuildOptions, QueryOptions, Match
├── cpp_src/
│   ├── fuzzy_automaton.cpp            # Schulz-Mihov NFA->DFA construction
│   ├── fuzzy_index.cpp                # Trie storage, bucket map, build pipeline
│   └── fuzzy_search.cpp               # DFS over trie driven by DFA state
├── python/src/
│   └── bindings_fuzzy.cpp             # nanobind: init_fuzzy(m)
└── tests/
    ├── test_fuzzy.cpp                 # Cross-check vs brute-force DP, edge cases, bucket fallback
    └── benchmark_fuzzy.cpp            # Throughput vs difflib at 10k / 100k / 1.4M vocab

src/resolve/ext/
└── wfo.py                             # MODIFY: route _match_fuzzy through FuzzyIndex when C++ available
```

CMake change: `src/core/CMakeLists.txt` already globs `cpp_src/*.cpp`, so the three new files are picked up automatically. Verify the glob is `cpp_src/*.cpp` and not an explicit list before assuming.

Bindings registration: `src/core/python/src/bindings.cpp` calls one new line `init_fuzzy(m);`. Add a `void init_fuzzy(nb::module_&);` forward decl alongside the existing ones.

---

## 4. Public API (final)

```cpp
// include/resolve/fuzzy.hpp
#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace resolve::fuzzy {

struct Match {
    std::string entry;   // Matched dictionary entry (original casing)
    int id;              // Index into the original `entries` vector passed at build time
    int distance;        // Damerau-Levenshtein distance from needle
};

struct BuildOptions {
    int max_edit_distance = 2;                              // Max k the index supports at query time
    std::function<std::string(std::string_view)> bucket_fn; // Optional; returns bucket key per entry. nullptr = one global bucket.
    bool case_insensitive = true;                           // Lowercase during index/query
    bool damerau = true;                                    // Allow adjacent transpositions
};

struct QueryOptions {
    int max_edit_distance = 2;                  // Must be <= BuildOptions::max_edit_distance
    int top_n = 5;
    std::optional<std::string> bucket_hint;     // Try this bucket first; fall back to full scan on miss
};

class FuzzyIndex {
public:
    static FuzzyIndex build(std::span<const std::string> entries, BuildOptions opts = {});

    std::vector<Match> query(std::string_view needle, QueryOptions opts = {}) const;

    // OpenMP-parallel batch. `bucket_fn` (if non-null) is applied per-needle to derive bucket_hint.
    std::vector<std::vector<Match>> query_batch(
        std::span<const std::string> needles,
        QueryOptions opts = {},
        std::function<std::string(std::string_view)> bucket_fn = nullptr
    ) const;

    std::size_t size() const noexcept;     // Number of indexed entries
    std::size_t bucket_count() const noexcept;

private:
    // PIMPL or direct members — your call during implementation
};

} // namespace resolve::fuzzy
```

### nanobind binding (sketch)

```cpp
// python/src/bindings_fuzzy.cpp
void init_fuzzy(nb::module_& m) {
    auto sub = m.def_submodule("fuzzy");

    nb::class_<resolve::fuzzy::Match>(sub, "Match")
        .def_ro("entry", &resolve::fuzzy::Match::entry)
        .def_ro("id", &resolve::fuzzy::Match::id)
        .def_ro("distance", &resolve::fuzzy::Match::distance);

    nb::class_<resolve::fuzzy::FuzzyIndex>(sub, "FuzzyIndex")
        .def_static("build",
            [](std::vector<std::string> entries, int max_dist, nb::object bucket_fn, bool ci, bool damerau) {
                resolve::fuzzy::BuildOptions opts{
                    .max_edit_distance = max_dist,
                    .bucket_fn = bucket_fn.is_none() ? nullptr : [bucket_fn](std::string_view s) {
                        return nb::cast<std::string>(bucket_fn(nb::str(s.data(), s.size())));
                    },
                    .case_insensitive = ci,
                    .damerau = damerau,
                };
                return resolve::fuzzy::FuzzyIndex::build(entries, std::move(opts));
            },
            "entries"_a, "max_edit_distance"_a = 2, "bucket_fn"_a = nb::none(),
            "case_insensitive"_a = true, "damerau"_a = true)
        .def("query", /* QueryOptions kwargs */)
        .def("query_batch", /* ... */)
        .def("__len__", &resolve::fuzzy::FuzzyIndex::size);
}
```

Exact kwarg shape: figure out during implementation. The point is the Python surface mirrors the C++ surface.

---

## 5. WFO integration

```python
# src/resolve/ext/wfo.py — replace _match_fuzzy and update __init__

class WFOBackbone:
    def __init__(self, classification_path):
        # ... existing load code ...
        self._build_indices()
        self._build_fuzzy_index()

    def _build_fuzzy_index(self):
        try:
            from _resolve_core import fuzzy as _fz
            self._fuzzy = _fz.FuzzyIndex.build(
                list(self._name_index.keys()),
                max_edit_distance=3,
                bucket_fn=lambda s: s.split(" ", 1)[0].lower() if " " in s else "",
                case_insensitive=True,
                damerau=True,
            )
            self._fuzzy_native = True
        except ImportError:
            self._fuzzy = None
            self._fuzzy_native = False

    def _match_fuzzy(self, name, cutoff, n, original_input=None):
        input_label = original_input or name
        if self._fuzzy_native:
            max_k = max(1, int(round(len(name) * cutoff)))
            first_word = name.split(" ", 1)[0].lower() if " " in name else None
            matches = self._fuzzy.query(name, max_edit_distance=max_k, top_n=n, bucket_hint=first_word)
            if not matches:
                return None
            best = matches[0]
            best_idx = self._pick_best(self._name_index[best.entry])
            result = self._resolve_accepted(best_idx)
            result["input"] = input_label
            result["match_method"] = "fuzzy"
            result["fuzzy_dist"] = best.distance
            return result
        else:
            # Existing difflib path unchanged
            close = get_close_matches(name, self._name_index.keys(), n=n, cutoff=1.0 - cutoff)
            # ... rest of existing implementation
```

Key decisions in WFO usage:
- **`max_edit_distance` interpretation**: `cutoff` in the current API is a fraction of string length (e.g. 0.1 = 10%). Convert to absolute edit distance for the native path: `max_k = max(1, int(round(len(name) * cutoff)))`. Keep the existing API of `_match_fuzzy` unchanged so callers don't notice.
- **`fuzzy_dist` units change**: native returns integer edit distance; existing code returns `1.0 - similarity_ratio`. Decide whether to renormalize for output consistency or document the unit change. **Recommendation**: keep it as integer edit distance and update the docstring — Ratcliff/Obershelp ratio was never a useful number for the user anyway, and edit distance is what taxonomy people actually reason about.
- **Bucket fallback**: handled inside `FuzzyIndex::query`. WFO doesn't need to do anything special; if the user passes `"Querucs robur"` and the bucket `"querucs"` has no hits, the index automatically scans the global trie and finds `"Quercus robur"` at distance 2.

---

## 6. Implementation order

Do this in order. Each step is independently testable and committable.

### Step 1 — `fuzzy.hpp` + skeleton + brute-force reference
- Write `include/resolve/fuzzy.hpp` exactly as in §4
- Write `fuzzy_index.cpp` with a *brute-force* `query()` that does pairwise DP across all entries (slow but correct)
- Write `bindings_fuzzy.cpp`
- Wire into `bindings.cpp` and confirm `import _resolve_core; _resolve_core.fuzzy.FuzzyIndex.build([...])` works from Python
- Write `test_fuzzy.cpp` with hand-crafted cases (single chars, empty, identical, transposition, multi-byte UTF-8)

This gets the surface and binding working before any algorithm cleverness.

### Step 2 — Trie storage
- In `fuzzy_index.cpp`, build a trie over all entries (per bucket if `bucket_fn` provided)
- Use compact node representation: `struct Node { char c; int first_child; int next_sibling; int entry_id_terminal; };` packed into a single `std::vector<Node>` — cache-friendly, no per-node allocation
- Test: trie traversal returns all entries; entry_id round-trips correctly
- Brute-force `query()` still in use

### Step 3 — Levenshtein automaton
- Implement `fuzzy_automaton.cpp` per Schulz & Mihov 2002
- NFA states are `(position_in_needle, edits_so_far)` pairs; subset-construct to DFA
- For Damerau, extend with the transposition state per Mihov & Schulz 2004
- Test the DFA in isolation: feed strings, check it accepts iff within k

**Pull the paper first.** Schulz & Mihov 2002 "Fast string correction with Levenshtein automata" (IJDAR) is the canonical reference. Read §3–4 before writing code. Apply the project rule: read the paper before patching the algorithm.

### Step 4 — Trie + automaton search
- Write `fuzzy_search.cpp`: DFS over trie, carrying current DFA state. At each trie node, transition the DFA on the node's character. If DFA dead → prune subtree. If trie node is terminal AND DFA in accepting state → emit a match with the distance recoverable from the DFA's `(position, edits)` state.
- Maintain a top-N min-heap of best matches; tighten the effective max distance as the heap fills.
- Test: results match brute-force `query()` exactly on a 10k-entry test vocab across 1000 random queries.

### Step 5 — Replace brute force, benchmark
- Switch `FuzzyIndex::query` to use the trie+automaton path
- Benchmark `tests/benchmark_fuzzy.cpp`: throughput at vocab sizes 1k / 10k / 100k / 1.4M, k=1/2/3
- Target: ≥ 1000 queries/sec single-threaded on the i9 at N=1.4M, k=2. If significantly slower than that, investigate before moving on.

### Step 6 — OpenMP batch
- Implement `query_batch` with `#pragma omp parallel for`
- Test: results identical to serial `query` in a loop
- Benchmark scaling on the i9's cores

### Step 7 — WFO integration
- Modify `src/resolve/ext/wfo.py` as in §5
- Existing tests in `tests/test_wfo.py` (if any — check) must still pass
- Add a test that confirms native and Python paths give same matches on a small vocab
- Confirm the WFO ingest of a 1000-species sample is dramatically faster end-to-end

### Step 8 — Documentation
- Docstring on `FuzzyIndex` in the binding
- One short paragraph in `CLAUDE.md` under the project's existing "Architecture Improvements" list mentioning the new `fuzzy` submodule
- No new top-level README, no marketing copy

---

## 7. Testing strategy

### Correctness (Catch2, `test_fuzzy.cpp`)
- Hand-crafted: empty needle, empty entry, identical strings, all distance-1 single-edit cases (insert/delete/substitute/transpose)
- Multi-byte UTF-8 entries (Latin author names appear in WFO backbone)
- Cross-check against brute-force DP on a 10k-entry random vocab × 1k random queries — assert exact agreement on returned IDs and distances
- Bucket-hint fallback: build with bucket_fn, query with a wrong hint → must still find matches in other buckets
- `case_insensitive=True` round-trips
- `damerau=true/false` distinguishes "abc" ↔ "bac" (distance 1 vs 2)

### Recovery test (not just smoke)
Apply the project rule: tests must validate behavior, not just plumbing. Specifically:
- Generate 100 known-target words; for each, generate 5 noisy queries with 1–3 random edits; assert `query()` returns the original target in top-5 for ≥ 95% of cases at appropriate `max_edit_distance`
- This is a *recall test*, not just shape-correct output

### Performance (Catch2 benchmarks, `benchmark_fuzzy.cpp`)
- Throughput at vocab 10k / 100k / 1.4M
- Speedup vs `difflib.get_close_matches` reported as a ratio in the benchmark output
- Memory footprint of the index (report bytes per entry)

### Integration (Python, `tests/test_wfo.py` if it exists, else new)
- `WFOBackbone.match_one("Querucs robur")` — transposition typo, no exact match — returns `Quercus robur` via fuzzy path
- `match_one("Quercus")` — no species epithet, must match via bucket-hint=empty fallback
- `match_batch` of 1k species with 10% noisy names — assert speedup vs current difflib path is large (>10×) and match results agree on the unambiguous cases

---

## 8. Decision log (for future-you reading this cold)

| Q | A | Why |
|---|---|---|
| Why not rapidfuzz? | No external dep for < 500-line core algorithm | Project rule: no dependency shortcuts for core logic |
| Why not SymSpell? | Memory blowup at 1.4M × k=2 | 2–4 GB index is unwieldy; trie+automaton is 100–200 MB |
| Why bake the genus hint into the *index* instead of the *caller*? | We don't | The bucket_fn is provided by the caller (WFO), not hardcoded in the index. RESOLVE-core has zero domain assumptions. |
| Why keep difflib path? | Dual-backend architecture, not antipattern | Same pattern RESOLVE already uses for C++ vs Python — when `_resolve_core` is absent, the Python path is the only path, not a parallel one |
| Why Damerau (transpositions) not plain Levenshtein? | Species names suffer transposition typos | "Querucs" ↔ "Quercus" is one transposition, distance 1 not 2 under Damerau |
| Why not just hard-require rapidfuzz? | Considered and rejected | Re-introduces the dropped fallback split *and* violates project dependency rule. Previous design conversation arrived at this same conclusion. |
| What's the upper bound on `max_edit_distance`? | Suggest hard-cap at 5 in QueryOptions validation | DFA size grows ~2^k; beyond k=5 the automaton walk loses to brute-force anyway |

---

## 9. Out of scope (don't do these)

- **No similarity metrics other than Damerau-Levenshtein.** No Jaro-Winkler, no Jaccard, no cosine. Single metric, single code path.
- **No on-disk serialization of the index.** Build at runtime. WFO already loads 1.4M strings into RAM; building the trie alongside takes a few extra seconds.
- **No FST compression yet.** Plain trie. Revisit only if memory becomes a real complaint.
- **No GPU implementation.** This is not a hot training-path operation; CPU is the right substrate.
- **No "fuzzy SQL"-style multi-field matching.** Single string in, top-N matches out. If WFO ever wants to match against (genus, epithet, author) tuples, that's a new layer on top.
- **No rapidfuzz anywhere in `src/resolve/`.** Search the codebase and confirm before finishing.

---

## 10. Definition of done

- [ ] `_resolve_core.fuzzy.FuzzyIndex` exists and is importable
- [ ] `test_fuzzy.cpp` passes including the recovery test (≥ 95% top-5 recall on synthetic noisy queries)
- [ ] `benchmark_fuzzy.cpp` shows ≥ 1000 queries/sec single-threaded at N=1.4M, k=2 on the i9
- [ ] OpenMP `query_batch` shows ≥ 8× speedup over single-threaded on the i9
- [ ] `WFOBackbone._match_fuzzy` uses the native path when `_resolve_core` is installed
- [ ] WFO ingest of a 1000-species sample is ≥ 30× faster than current difflib path (measure and report actual number in commit message)
- [ ] No `rapidfuzz` reference anywhere in `src/resolve/`
- [ ] CLAUDE.md mentions the new `fuzzy` submodule under "Architecture Improvements"

---

## 11. Quick reference for the new session

Files to read first when picking this up:
- `src/resolve/ext/wfo.py` — what we're replacing the fuzzy path in
- `src/core/include/resolve/dataset.hpp` — example of an existing public API header to mirror the style
- `src/core/python/src/bindings.cpp` — where `init_fuzzy(m)` gets called
- `src/core/python/src/bindings_dataset.cpp` — example of an existing nanobind binding to mirror the style
- `src/core/tests/test_dataset.cpp` — Catch2 style used in the project

Build:
- `powershell -ExecutionPolicy Bypass -File "src/core/build_cuda/launch_build.ps1"` (or the cuda-build skill at `~/.claude/skills/cuda-build/SKILL.md`)

Papers to read before Step 3:
- Schulz, K.U. & Mihov, S. (2002). "Fast string correction with Levenshtein automata." *IJDAR* 5(1).
- Mihov, S. & Schulz, K.U. (2004). "Fast approximate search in large dictionaries." *Computational Linguistics* 30(4).
