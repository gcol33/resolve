# R Package Architecture Plan

## Core Principle
**R package orchestrates, Python computes.**

## Structure

```
resolve/                    # R package (CRAN-compatible)
├── R/                      # Pure R code
│   ├── api.R               # User-facing functions
│   ├── backend.R           # Python bridge via reticulate
│   ├── validation.R        # Argument validation
│   └── utils.R             # Helpers
├── man/                    # Documentation
├── vignettes/              # CPU-only, small examples
├── DESCRIPTION             # reticulate as dependency, NO compiled code
└── tests/
    └── testthat/           # Skip if Python/torch unavailable
```

## How It Works

1. **R package is pure R** - no compiled code, no CUDA, no libtorch
2. **Python accessed via reticulate** at runtime
3. **Defensive checks** - clear error if Python/torch missing
4. **Helper function** `setup_backend()` guides users to install Python + PyTorch

## Installation Story

- DESCRIPTION lists `reticulate` as dependency
- `setup_backend()` helper:
  - Guides users to install Python
  - Installs PyTorch in virtualenv/conda
  - Documents in README and vignette
- CRAN checks never run that code

## Test Requirements (Non-negotiable)

- Skip if Python unavailable
- Skip if torch unavailable
- Never require GPU
- Never download large models
- CRAN checks run on minimal machines

## What NOT to Do

- Do NOT bundle libtorch
- Do NOT compile CUDA on install
- Do NOT require GPU for examples
- Do NOT auto-download large binaries during checks

## CUDA/Performance

- Lives in Python environment
- Power users install themselves
- CRAN only verifies R package behaves when deps exist

## Status

**Deferred** - Python backend must be stable first.

## Next Steps (when ready)

1. Sketch minimal DESCRIPTION for CRAN
2. Design R API so users never notice Python
3. Plan fallback CPU path for CRAN examples
