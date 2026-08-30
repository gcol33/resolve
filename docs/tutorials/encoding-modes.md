# Encoding Modes

Every encoder solves the same problem: turn a variable-length set of species,
with optional abundances, into something a neural network can read. They differ
in how much structure survives and how much of it is learned.

The mode is chosen once and used twice, because the loader and the model have to
agree on the tensors that pass between them:

```python
mode = rc.SpeciesEncodingMode.RankPool

data_config  = rc.DatasetConfig()
model_config = rc.ModelConfig()
data_config.species_encoding  = mode
model_config.species_encoding = mode
```

| Mode | Model input | Learned | Unseen species |
|------|-------------|---------|----------------|
| `Hash` | hashed vector | no | hashed like any name |
| `Embed` | concatenated top-k embeddings | yes | unknown token |
| `Sparse` | abundance vector over the vocabulary | first layer | no column |
| `RankPool` | weight-pooled embeddings | yes | unknown token |
| `Transformer` | attention-pooled tokens | yes | unknown token |

## Hash

Each species name is hashed to a bucket in `[0, hash_dim)` and to a sign, and
its weight is accumulated there. The signed variant makes collisions cancel in
expectation rather than accumulate.

```python
data_config.species_encoding  = rc.SpeciesEncodingMode.Hash
data_config.hash_dim          = 64
data_config.selection         = rc.SelectionMode.All      # hash every species
data_config.normalization     = rc.NormalizationMode.Log1p

model_config.species_encoding = rc.SpeciesEncodingMode.Hash
model_config.hash_dim         = 64
```

Before hashing, the loader applies `selection` with `top_k` and then
`normalization`:

| `selection` | Species kept per plot |
|-------------|-----------------------|
| `Top` (default) | the `top_k` most abundant |
| `Bottom` | the `top_k` least abundant |
| `TopBottom` | both ends, deduplicated |
| `All` | every species in the plot |

`top_k` defaults to `3`, so the default hash embedding summarizes a plot's three
most abundant species. Set `selection = SelectionMode.All` to hash the whole
list. `top_k` also sets the number of fixed genus and family slots, so raising
it widens the taxonomy input as well.

Each encoding takes its per-plot budget from the knob that also fixes its width:
`top_k` for hash, `top_k_species` for embed, and `species_budget` for the
variable-width `RankPool`, `Transformer` and `Sparse` encodings. `species_budget`
defaults to `0`, which is no budget: those encodings encode every species a plot
records, and the schema reports `All` to say so. Set it to run a species
ablation on the pooled encoders:

```python
data_config.species_encoding = rc.SpeciesEncodingMode.RankPool
data_config.selection        = rc.SelectionMode.Bottom
data_config.species_budget   = 20     # the 20 least abundant species per plot
```

Embed writes a fixed number of ranked slots, so `All` has no encoding there and
is rejected rather than quietly treated as `Top`; under `TopBottom` it takes
half its slots from each end.

`normalization` rescales the weights: `Raw` uses abundance as recorded, `Norm`
divides by the plot total, `Log1p` compresses the range.

**Strengths.** No embedding table, so memory is flat in vocabulary size. A
species the model never saw still lands in a bucket, so the representation
degrades smoothly on new data. Fastest mode to train and to score, and the only
one with a dedicated CUDA kernel and with layer diagnostics.

**Costs.** Two species can share a bucket, and the encoder has no way to tell
them apart. Nothing about a species is learned; whatever the model knows about
composition it infers from the hashed mixture.

**Reach for it** as the first thing you run on a new dataset, on very large
vocabularies where an embedding table would dominate memory, and during
hyperparameter search where iteration speed matters.

## Embed

The `top_k_species` most abundant species in a plot are looked up in a
frequency-ranked vocabulary, and their embeddings are concatenated. Genus and
family go into `n_taxonomy_slots` fixed slots, filled with the most abundant
distinct genera and families.

```python
data_config.species_encoding   = rc.SpeciesEncodingMode.Embed
data_config.top_k_species      = 10

model_config.species_encoding  = rc.SpeciesEncodingMode.Embed
model_config.top_k_species     = 10
model_config.species_embed_dim = 32
```

The input width is fixed at `top_k_species * species_embed_dim`, and a plot with
fewer species pads with the reserved unknown row at index 0.

**Strengths.** A species gets its own vector, so the model can learn that one
species is diagnostic of a target. The table only covers the species that
actually appear, and the concatenation keeps the slots positional, so the
encoder can treat "most abundant" differently from "fourth most abundant".

**Costs.** Everything past rank `top_k_species` is discarded, which is most of a
species-rich plot. A species the model never saw resolves to the unknown row.

**Reach for it** when a handful of dominant species carries the signal, and when
you want per-species vectors to inspect afterwards.

## Sparse

The plot becomes one row of an explicit abundance vector, `n_species_vocab`
wide, with each known species at its own position.

```python
data_config.species_encoding  = rc.SpeciesEncodingMode.Sparse
data_config.representation    = rc.RepresentationMode.Abundance  # or PresenceAbsence

model_config.species_encoding = rc.SpeciesEncodingMode.Sparse
```

`RepresentationMode.PresenceAbsence` writes `1.0` for every present species
instead of its abundance.

**Strengths.** Nothing is discarded and nothing collides; the first layer sees
the exact community matrix, which is the representation classical vegetation
analysis works on. The first weight matrix is a per-species vector, so it stays
interpretable.

**Costs.** The input width is the whole vocabulary, so the first layer grows
linearly with it and most entries in a row are zero. A species absent from the
training vocabulary has no column at all.

**Reach for it** on small to moderate vocabularies, and when you want the model
input to line up column for column with a community matrix.

## RankPool

Every species in a plot contributes. The loader emits padded per-species tensors
`(n_plots, max_species)` of species, genus, and family IDs plus a weight and a
mask, and the encoder pools each table with a weighted `embedding_bag`.

```python
data_config.species_encoding  = rc.SpeciesEncodingMode.RankPool
data_config.pool_weighting    = rc.PoolWeighting.Log1p
data_config.pool_species_cap  = -1        # auto p99

model_config.species_encoding = rc.SpeciesEncodingMode.RankPool
model_config.species_embed_dim = 32
model_config.cover_dropout    = 0.1
```

`pool_weighting` sets the per-species weight before pooling:

| Value | Weight |
|-------|--------|
| `Binary` | `1` for every present species |
| `Abundance` | the recorded abundance |
| `Log1p` (default) | `log(1 + abundance)` |
| `Norm` | abundance divided by the plot total |
| `Rank` | `1 / rank`, dense-ranked by descending abundance |

`pool_species_cap` bounds the padded width: `0` (default) pads to the longest
plot in the dataset, `-1` truncates at the 99th percentile of species counts and
prints what it dropped, and a positive value truncates at that many species.

`cover_dropout` replaces a training plot's weights with the plain presence mask
and clears its `has_cover` flag, at that probability per plot per epoch. The
encoder receives `has_cover` as a feature, so a model trained with cover dropout
learns to work from presence alone and still score plots that arrive without
abundances.

**Strengths.** No truncation and no collisions: every species contributes at its
own weight. Genus and family are pooled the same way, so rare species arrive
with taxonomic company. The fused pooling keeps memory flat in `max_species`
rather than materializing a per-species embedding tensor.

**Costs.** Pooling is order-free, so it holds which species are present and how
strongly, and nothing about how they interact. Padding to `max_species` means one
species-rich plot widens every row unless you cap it.

**Reach for it** when species richness varies widely across plots, when rare
species matter, and when taxonomy is available.

## Transformer

The same pool tensors as rank pooling, read differently. Species, genus, and
family embeddings are summed into one token per species in `d_model` space,
optionally passed through self-attention layers, and then pooled.

```python
data_config.species_encoding      = rc.SpeciesEncodingMode.Transformer
data_config.pool_weighting        = rc.PoolWeighting.Log1p

model_config.species_encoding     = rc.SpeciesEncodingMode.Transformer
model_config.d_model              = 128
model_config.n_heads              = 4
model_config.n_attention_layers   = 2
model_config.transformer_ff_dim   = 256
model_config.transformer_pooling  = "attention"   # or "cls"
model_config.transformer_dropout  = 0.1
```

`transformer_pooling = "attention"` learns a weight per token and takes the
weighted sum. `"cls"` prepends a learned token and reads its output, so it needs
at least one attention layer; `n_attention_layers = 0` with `"cls"` is rejected.

With `n_attention_layers = 0` the encoder is additive token embeddings plus
attention pooling, which is rank pooling with a learned weighting.

**Strengths.** Attention lets a species' contribution depend on what else is in
the plot, which is where co-occurrence structure can enter the representation.
Attention pooling learns which species to read from, per plot.

**Costs.** Attention is quadratic in species per plot, so a cap matters more
here. Attention layers are the part of the model most sensitive to learning rate
and to fp16 range, so lower `lr` and disabling AMP are the usual starting point.
It is the slowest mode per epoch.

**Reach for it** when species interactions plausibly carry signal, on datasets
large enough to fit attention weights, and after the simpler modes have given
you a number to beat.

## Choosing

| Situation | Start with |
|-----------|-----------|
| First run on a new dataset | `Hash`, `hash_dim=64`, `selection=All` |
| Vocabulary in the tens of thousands | `Hash` or `RankPool` |
| Small vocabulary, community matrix semantics | `Sparse` |
| A few dominant species carry the signal | `Embed` |
| Richness varies widely, rare species matter | `RankPool` |
| Interactions matter and the data is large | `Transformer` |

Nothing here substitutes for measuring. `benchmarks/run_benchmarks.py` runs the
modes against each other on your own data, under one cross-validation split:

```bash
python benchmarks/run_benchmarks.py --data-size 10k --configs encodings
python benchmarks/run_benchmarks.py --synthetic --configs hash_64,rank_pool
```

It writes a JSON file of per-fold metrics and prints a comparison table.

## Settings that travel with the mode

| Setting | Hash / Embed / Sparse | RankPool | Transformer |
|---------|----------------------|----------|-------------|
| `lr` | `1e-3` | `1e-3` | lower, `3e-4` is a common starting point |
| `use_amp` | safe | safe | try it off first if the loss goes to NaN |
| `pool_weighting` | ignored | used | used |
| `pool_species_cap` | ignored | used | used, and matters more |
| `selection` | used, with `top_k` (hash) or `top_k_species` (embed) | used, with `species_budget` | used, with `species_budget` |
| `species_budget` | ignored | used | used |
| Layer diagnostics | hash only | unavailable | unavailable |

## Next steps

- [Training Models](training.md): the rest of the configuration
- [Performance Tuning](performance-tuning.md): speed, memory, and accuracy levers
- [Understanding Embeddings](embeddings.md): reading what the encoders learned
