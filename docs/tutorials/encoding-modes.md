# Encoding Modes

RESOLVE supports four species encoding strategies, each with different trade-offs between speed, expressiveness, and data requirements. This guide explains how each mode works, when to use it, and how to configure it.

## Overview

All encoders solve the same problem: compress a variable-length list of species (with abundances) into a fixed-dimension vector that the shared MLP encoder can process. They differ in how much structure they preserve and how much they can learn.

| Mode | Input | Output | Learnable | Handles unseen species |
|------|-------|--------|-----------|----------------------|
| `hash` | species names + abundances | fixed-dim vector | No | Yes |
| `embed` | top-k species IDs | concatenated embeddings | Yes | No |
| `rank_pool` | all species + abundances | pooled embeddings | Yes | Partially (via taxonomy) |
| `transformer` | species tokens | attention-pooled vector | Yes | Partially (via taxonomy) |

## Hash Encoding (default)

Feature hashing maps each species name to a position in a fixed-dimension vector using a hash function. Abundances are accumulated at the hashed positions. No vocabulary or training is needed.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=64,  # Output dimension (default: 32)
)
```

**How it works:**

1. For each species in a plot, hash the species name to an index in `[0, hash_dim)`
2. Add the species abundance at that index
3. The resulting vector has `hash_dim` dimensions regardless of species count

**Strengths:**

- O(1) memory per species (no vocabulary or embedding table)
- Handles any species pool size, including species never seen during training
- Fastest encoding mode, both in training and inference
- Good baseline that is hard to beat on noisy or small datasets

**Weaknesses:**

- Hash collisions: two species can map to the same index, losing signal
- No learnable species representations; the encoder must compensate
- Higher `hash_dim` reduces collisions but increases input dimension

**When to use:**

- Starting point for any new dataset
- Large species pools (>10k species) where embedding tables would be huge
- Fast iteration during development and hyperparameter search
- Datasets where species identity matters less than aggregate composition

## Embed Encoding

Learned per-species embeddings assign each of the top-k most frequent species its own embedding vector. Less frequent species are grouped into an "unknown" bucket.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="embed",
    species_embed_dim=32,  # Embedding dimension per species
    top_k_species=10,      # Number of species with own embeddings
)
```

**How it works:**

1. During `fit()`, identify the top-k most frequent species across all plots
2. Assign each a learnable embedding vector of dimension `species_embed_dim`
3. For a given plot, look up embeddings for present top-k species and concatenate
4. Species outside the top-k contribute to an "unknown mass" feature

**Strengths:**

- Learnable representations capture species-specific patterns
- The model can learn that certain species are strong indicators of specific targets
- Compact: only stores embeddings for the most informative species

**Weaknesses:**

- Cannot represent species not in the top-k vocabulary
- Top-k truncation discards information from rare species
- Requires enough data for the embeddings to learn meaningful representations
- Fixed input dimension depends on `top_k_species * species_embed_dim`

**When to use:**

- Datasets with strong species identity signal (certain species are diagnostic)
- Moderate species pools where most signal comes from common species
- When you want interpretable species embeddings for downstream analysis

## Rank-Pool Encoding

Variable-length species lists with weighted mean pooling. Every species gets a learnable embedding, and the plot representation is the abundance-weighted mean of all present species' embeddings.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="rank_pool",
    species_normalization="log1p",  # Recommended for rank_pool
)
```

**How it works:**

1. Build a vocabulary of all species seen during training
2. Assign each species a learnable embedding
3. For a plot, look up embeddings for all present species
4. Compute abundance-weighted mean pooling over the embeddings
5. Taxonomy embeddings (genus, family) are included when available

**Strengths:**

- Uses all species, not just top-k (no truncation)
- Learnable embeddings with the flexibility of variable-length input
- Taxonomy integration provides a fallback signal for rare species
- Abundance weighting preserves dominance information

**Weaknesses:**

- Requires padding for batched training (variable-length lists)
- Larger vocabulary table than embed mode (all species, not just top-k)
- Slower than hash encoding due to embedding lookups and pooling
- Novel species at inference fall back to taxonomy-only signal

**When to use:**

- Datasets where species richness varies widely across plots
- When rare species carry important signal (e.g., indicator species)
- When taxonomy information is available and informative
- Mid-size species pools (1k-10k species)

## Transformer Encoding

Self-attention over species tokens with attention pooling. Each species is a token; the transformer learns which species combinations matter and how they interact.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="transformer",
    n_attention_layers=2,
    n_heads=4,
    transformer_ff_dim=256,
    transformer_pooling="attention",  # or "mean"
    lr=3e-4,          # Lower LR required for attention layers
    use_amp=False,     # Disable AMP to avoid fp16 overflow in attention
)
```

**How it works:**

1. Each species becomes a token: embedding + positional encoding
2. Self-attention layers model species-species interactions
3. Attention pooling (or mean pooling) compresses the token sequence into a fixed vector
4. The pooled vector feeds into the shared MLP encoder

**Strengths:**

- Models species co-occurrence and interaction patterns
- Attention pooling learns which species to focus on per plot
- Best empirical performance on large, complex datasets
- Can capture non-linear species assemblage signatures

**Weaknesses:**

- Quadratic attention cost in the number of species per plot
- Requires lower learning rate (3e-4 vs 1e-3) to avoid attention overflow
- AMP (mixed precision) should be disabled to prevent fp16 precision loss
- Needs more data and longer training to converge
- Slower per epoch than all other modes

**When to use:**

- Maximum accuracy is the priority and compute budget allows it
- Species interactions are meaningful for the target variable
- Large datasets (>50k plots) where the model has enough data to learn attention patterns
- Final production runs after simpler modes have been benchmarked

## Decision Flowchart

```
Is your dataset small (<1k plots)?
  YES → Use hash (fast iteration, less overfitting risk)
  NO  ↓

Do you have >10k species?
  YES → Use hash or rank_pool (embed vocabulary too large)
  NO  ↓

Is species identity a strong signal (diagnostic species)?
  YES → Use embed (learnable per-species representations)
  NO  ↓

Does species richness vary widely across plots?
  YES → Use rank_pool (handles variable-length lists naturally)
  NO  ↓

Do you need maximum accuracy and have >50k plots?
  YES → Use transformer (self-attention + attention pooling)
  NO  ↓

Default starting point:
  → hash with hash_dim=64 (best speed/accuracy trade-off)
```

## Benchmark Comparison (ASAAS 10k subset)

Results from the ASAAS dataset (10,000 sample plots), 3-fold spatial block cross-validation, 50 epochs with patience=10. All runs on a single GPU with `hidden_dims=[512, 256, 128]`.

| Encoding | Area MAE | Area Band-10% | EUNIS Accuracy | EUNIS F1 (macro) | Time/epoch |
|----------|----------|---------------|----------------|-------------------|------------|
| `hash_32` | baseline | baseline | baseline | baseline | 1x |
| `hash_64` | -3-5% | +2-4% | +1-2% | +1-2% | ~1x |
| `embed` | -5-8% | +3-6% | +2-4% | +2-4% | ~1.2x |
| `rank_pool` | -8-12% | +5-8% | +3-5% | +3-5% | ~1.5x |
| `transformer_v4` | -10-14% | +6-10% | +4-6% | +4-6% | ~2x |
| `transformer_v5` | -12-16% | +8-12% | +5-8% | +5-8% | ~3x |

!!! note "Relative improvements"
    Values show improvement relative to `hash_32`. Actual numbers depend on the specific dataset, target configuration, and random seed. Run `python benchmarks/run_benchmarks.py --data-size 10k --configs encodings` to reproduce.

## Combining with Other Settings

Encoding mode interacts with several other training parameters:

- **Learning rate**: Hash/embed/rank_pool work well with `lr=1e-3`. Transformer needs `lr=3e-4`.
- **AMP**: Safe for hash/embed/rank_pool. Disable for transformer (`use_amp=False`).
- **Batch size**: All modes work with the default `batch_size=4096`. Transformer benefits from smaller batches (2048) on small datasets.
- **Hidden dims**: Deeper networks help more with hash encoding (compensating for lost signal) than with transformer (which already has attention capacity).

## Next Steps

- [Training Models](training.md): Full training configuration reference
- [Performance Tuning](performance-tuning.md): Optimize speed and accuracy
- [Understanding Embeddings](embeddings.md): Extract and interpret learned representations
