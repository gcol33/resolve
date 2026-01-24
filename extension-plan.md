# RESOLVE Extension Plan: Species Selection Modes

## Overview

This plan addresses supervisor requirements for flexible species encoding:
1. **Top + Bottom K species selection** - Select most and/or least abundant species
2. **Full species abundance encoding** - Use all species with abundance weights
3. **Presence/absence encoding** - Binary species occurrence

All implemented through a unified `species_selection` parameter.

---

## Unified API: `species_selection` Parameter

The `species_selection` parameter controls which species are included and how:

**`species_selection`**: Which species to include

| Selection Mode | Description | Uses Hash |
|----------------|-------------|-----------|
| `"top"` (default) | Top-K most abundant/frequent | Yes |
| `"bottom"` | Bottom-K least abundant (rarest) | Yes |
| `"top_bottom"` | Top-K + Bottom-K (2K total) | Yes |
| `"all"` | All species (explicit vector) | No |

**`species_representation`**: How to represent species (only for `selection="all"`)

| Representation | Description |
|----------------|-------------|
| `"abundance"` (default) | Weighted by abundance |
| `"presence_absence"` | Binary 0/1 |

### Usage Examples

**Standard hash mode (top-k species):**
```python
trainer = Trainer(
    dataset,
    species_encoding="hash",
    species_selection="top",  # default
    top_k=10,
)
```

**Rare species focus:**
```python
trainer = Trainer(
    dataset,
    species_encoding="hash",
    species_selection="bottom",
    top_k=10,
)
```

**Mixed top + bottom:**
```python
trainer = Trainer(
    dataset,
    species_encoding="hash",
    species_selection="top_bottom",  # 5 top + 5 bottom = 10 total
    top_k=5,
)
```

**Full abundance matrix:**
```python
trainer = Trainer(
    dataset,
    species_encoding="hash",
    species_selection="all",
    species_embed_dim=64,
    min_species_frequency=5,  # Only species in 5+ plots
)
```

**Presence/absence (binary):**
```python
trainer = Trainer(
    dataset,
    species_selection="all",
    species_representation="presence_absence",
    species_embed_dim=64,
    min_species_frequency=5,
)
```

---

## Implementation Details

### Species Encoding Flow

1. **Hash modes** (`top`, `bottom`, `top_bottom`):
   - SpeciesEncoder builds hash embedding from selected species
   - Hash embedding included in continuous features
   - Model uses PlotEncoder

2. **Explicit vector modes** (`all`, `presence_absence`):
   - SpeciesEncoder builds (n_plots, n_species) matrix
   - Matrix passed as separate input to model
   - Model uses PlotEncoderSparse (learned linear projection)

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `species_encoding` | `"hash"` or `"embed"` |
| `species_selection` | `"top"`, `"bottom"`, `"top_bottom"`, `"all"`, `"presence_absence"` |
| `top_k` | Number of taxa for taxonomy embeddings |
| `species_embed_dim` | Embedding dimension for explicit vector modes |
| `min_species_frequency` | Filter species appearing in fewer than N plots |

---

## Files Modified

- `src/resolve/encode/species.py` - Added all selection modes and `_build_species_vector()`
- `src/resolve/model/resolve.py` - Added `uses_explicit_vector` parameter
- `src/resolve/model/encoder.py` - `PlotEncoderSparse` handles explicit vectors
- `src/resolve/train/trainer.py` - Unified species selection handling
- `tests/test_resolve.py` - Tests for all selection modes

---

## Tests

All 39 tests pass:
- `TestSpeciesEncoder::test_selection_modes` - Tests top/bottom/top_bottom
- `TestTrainer::test_fit_top_bottom_selection` - Integration test
- `TestTrainer::test_fit_all_species_mode` - Full abundance vector
- `TestTrainer::test_fit_presence_absence_mode` - Binary species vector
