# Packaging Strategy: No Toolchain Required for Users

## Problem

The C++ backend (`_resolve_core`) requires MSVC cl.exe to compile, which means ~6 GB VS Build Tools install, UAC elevation, and 10–30 min wait. Unacceptable for new users.

## Solution

Ship pre-compiled wheels for `_resolve_core` via CI. Users never need cl.exe.

### Install tiers

| Command | Backend | Toolchain required |
|---|---|---|
| `pip install resolve` | Python/PyTorch | none |
| `pip install resolve[cuda]` | C++ (pre-compiled) | none |
| build from source | C++ (compiled locally) | MSVC + CUDA SDK |

The Python backend is already feature-complete — tier 1 is not a degraded experience, just slower on large datasets.

### How

1. GitHub Actions matrix builds `_resolve_core` wheels for `win-cuda12x`, `linux-cuda12x`, `macos-arm64`.
2. Wheels published to PyPI as a companion package (e.g. `resolve-core-cu121`).
3. `resolve[cuda]` optional dependency pulls the correct wheel; `backend/` auto-activates C++ when `_resolve_core` is importable.
4. Wheel naming follows PyTorch convention (Python version + CUDA ABI tag) — study `download.pytorch.org/whl/` as the reference.

### What needs to change

- `pyproject.toml`: add `[project.optional-dependencies] cuda = ["resolve-core-cu121"]`
- CI: add wheel-build matrix job (libtorch static link recommended to avoid shipping libtorch separately)
- Docs: one-liner install instructions, no mention of VS Build Tools for end users

### Dev workflow (unchanged)

Source builds use the cuda-build skill (`~/.claude/skills/cuda-build/SKILL.md`) with VS 2022 BuildTools + VC++ workload. This is only needed for contributors modifying the C++ engine.
