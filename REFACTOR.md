# RESOLVE Refactor Plan: C++ Core with Python/R Frontends

## Goal
Single C++ implementation with Python (pybind11) and R (Rcpp) bindings.
Delete redundant pure Python implementation once C++ is feature-complete.

---

## Phase 1: C++ Core - Species Encoding ✅ COMPLETE

- [x] **1.1** Add `selection` parameter to SpeciesEncoder ("top", "bottom", "top_bottom", "all")
- [x] **1.2** Add `representation` parameter ("abundance", "presence_absence")
- [x] **1.3** Add `min_species_frequency` filtering
- [x] **1.4** Implement `get_taxonomy_ids()` for top/bottom/top_bottom selection
- [x] **1.5** Implement `build_abundance_vector()` for selection="all"
- [x] **1.6** Implement `build_presence_absence_vector()` for binary mode
- [x] **1.7** Add `n_taxonomy_slots()` method (2*top_k for top_bottom)
- [x] **1.8** Add `uses_explicit_vector()` method
- [x] **1.9** Update EncodedSpecies struct to include species_vector field

---

## Phase 2: C++ Core - Model Architecture ✅ COMPLETE

- [x] **2.1** Add PlotEncoderSparse class (for explicit species vectors)
- [x] **2.2** Add PlotEncoderEmbed class (for learned species embeddings)
- [x] **2.3** Update ResolveModel to support `uses_explicit_vector` parameter
- [x] **2.4** Update forward() methods to handle species_vector input
- [x] **2.5** Ensure MLP architecture matches Python (BatchNorm, GELU, Dropout)

---

## Phase 3: C++ Core - Training ✅ COMPLETE

- [x] **3.1** Feature hashing implemented in SpeciesEncoder::hash_species() using std::hash
- [x] **3.2** Implement StandardScaler equivalent for continuous features
- [x] **3.3** Implement train/test split with seed
- [x] **3.4** Implement phased loss with configurable components
- [x] **3.5** Implement AdamW optimizer
- [x] **3.6** Implement gradient clipping
- [x] **3.7** Implement checkpoint save/load
- [x] **3.8** Implement model state restoration for best epoch

---

## Phase 4: Python Bindings (pybind11) ✅ COMPLETE

- [x] **4.1** Expose SpeciesEncoder with all new parameters (selection, representation, etc.)
- [x] **4.2** Expose all enums (SelectionMode, RepresentationMode, NormalizationMode, etc.)
- [x] **4.3** Expose ResolveModel with full API (forward with species_ids, species_vector)
- [x] **4.4** Expose Trainer with fit/save/load and updated prepare_data
- [x] **4.5** Expose Predictor
- [x] **4.6** Add backwards-compatible aliases (SpaccSchema, SpaccModel, etc.)

---

## Phase 5: R Bindings (Rcpp) ✅ COMPLETE

- [x] **5.1** Create Rcpp bindings for SpeciesEncoder
- [x] **5.2** Create Rcpp bindings for ResolveModel
- [x] **5.3** Create Rcpp bindings for Trainer
- [x] **5.4** Create Rcpp bindings for Predictor
- [x] **5.5** Update r/R/resolve.R to use C++ instead of reticulate

---

## Phase 6: Validation & Cleanup

- [ ] **6.1** Cross-validate Python binding outputs vs old pure Python
- [ ] **6.2** Cross-validate R binding outputs
- [ ] **6.3** Run all existing tests against new bindings
- [ ] **6.4** Performance benchmark C++ vs pure Python
- [ ] **6.5** Delete src/resolve/ (pure Python implementation)
- [x] **6.6** Update pyproject.toml for C++ build (scikit-build or cmake)
- [x] **6.7** Update r/DESCRIPTION for C++ compilation

---

## Current Status

**Completed:**
- Phase 1-4: C++ Core (SpeciesEncoder, Model, Training)
- Phase 5: R bindings (Rcpp)
- Python bindings (pybind11)
- R package updated to use C++ directly (no more reticulate)

**Remaining (Phase 6):**
- Validation: Run tests to verify C++ bindings match Python behavior
- Cleanup: Delete `src/resolve/` after validation is complete

**Architecture:**
```
RESOLVE/
├── src/
│   ├── core/                    # C++ implementation (single source of truth)
│   │   ├── include/resolve/     # Headers
│   │   ├── cpp_src/             # Implementation
│   │   └── python/              # pybind11 bindings -> _resolve_core
│   └── resolve/                 # Pure Python (to be deleted after validation)
└── r/
    ├── src/                     # Rcpp bindings
    └── R/                       # R interface (now uses C++ directly)
```

---

## Files Created/Modified

**C++ Core:**
- `src/core/include/resolve/` - Headers
- `src/core/cpp_src/` - Implementation
- `src/core/python/src/bindings.cpp` - Python bindings

**R Package:**
- `r/src/resolve_rcpp.cpp` - Rcpp bindings
- `r/src/RcppExports.cpp` - Generated exports
- `r/src/Makevars`, `r/src/Makevars.win` - Build config
- `r/R/resolve.R` - Updated R interface
- `r/R/RcppExports.R` - Generated R exports
- `r/R/zzz.R` - Package init (Rcpp module loading)
- `r/DESCRIPTION` - Updated for C++ compilation
- `r/NAMESPACE` - Updated exports

---

## Notes

- C++ namespace is `resolve` (renamed from `spacc`)
- API is identical between Python and R frontends
- R package requires libtorch to be installed separately
