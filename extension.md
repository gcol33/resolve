# RESOLVE Extension: Learnable Species Embeddings

## Problem

Current RESOLVE uses feature hashing for species encoding:
- **Result**: 64% band_25 accuracy
- **Baseline**: 80% band_25 with learnable embeddings
- **Gap**: 16 percentage points

Feature hashing maps species/taxonomy to fixed hash buckets with no learned relationships. The original implementation uses `nn.Embedding` layers that learn which genera/families are similar through backpropagation.

## Solution

Add three species encoding modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `hash` | Feature hashing (current) | Quick experiments, unseen species |
| `embed` | Learnable embeddings | Maximum accuracy |
| `pretrained` | Pre-trained embeddings | Transfer learning |

## API Design

### Mode 1: Hash (default, unchanged)
```python
trainer = Trainer(dataset, species_encoding="hash", hash_dim=64)
```

### Mode 2: Learnable Embeddings
```python
trainer = Trainer(
    dataset,
    species_encoding="embed",
    embed_dim=32,
    taxonomy_embed_dim=8,  # for genus/family
    top_k_species=10,
    top_k_taxonomy=3,
)
```

### Mode 3: Pre-trained Embeddings
```python
# Step 1: Train embeddings separately
from resolve import train_species_embeddings

embeddings = train_species_embeddings(
    species_data,           # DataFrame with plot_id, species_id, abundance
    method="skipgram",      # or "cbow", "cooccurrence"
    embed_dim=32,
    window_size=None,       # None = all species in plot co-occur
    min_count=5,            # ignore rare species
    epochs=10,
)
embeddings.save("species_emb.pt")

# Step 2: Use in training
trainer = Trainer(
    dataset,
    species_encoding="pretrained",
    embeddings="species_emb.pt",
    freeze_embeddings=False,  # allow fine-tuning
)
```

## Implementation Plan

### Phase 1: Vocabulary Building (in ResolveDataset)

**File**: `src/resolve/data/dataset.py`

1. Add vocab building during dataset creation:
```python
class ResolveDataset:
    def __init__(self, ..., build_vocab=True):
        if build_vocab:
            self._species_vocab = self._build_vocab(species, "species_id")
            self._genus_vocab = self._build_vocab(species, "genus")
            self._family_vocab = self._build_vocab(species, "family")
```

2. Add top-k extraction per plot:
```python
def _extract_top_k(self, plot_id, k=10, by="abundance"):
    """Extract top-k species/genus/family by abundance for a plot."""
    plot_species = self._species[self._species[plot_id_col] == plot_id]
    top_k = plot_species.nlargest(k, abundance_col)
    return top_k[species_col].map(self._species_vocab).fillna(0).values
```

3. Store in schema:
```python
@dataclass
class DatasetSchema:
    # ... existing fields ...
    species_vocab: dict[str, int] | None = None
    genus_vocab: dict[str, int] | None = None
    family_vocab: dict[str, int] | None = None
```

### Phase 2: Embedding Encoder (new file)

**File**: `src/resolve/encode/embedding.py`

```python
class EmbeddingEncoder(nn.Module):
    """Learnable embeddings for species and taxonomy."""

    def __init__(
        self,
        n_species: int,
        n_genera: int,
        n_families: int,
        species_embed_dim: int = 32,
        taxonomy_embed_dim: int = 8,
        top_k_species: int = 10,
        top_k_taxonomy: int = 3,
    ):
        super().__init__()
        self.top_k_species = top_k_species
        self.top_k_taxonomy = top_k_taxonomy

        # Species embeddings (one per top-k slot)
        self.species_embeddings = nn.ModuleList([
            nn.Embedding(n_species + 1, species_embed_dim, padding_idx=0)
            for _ in range(top_k_species)
        ])

        # Taxonomy embeddings (one per top-k slot)
        self.genus_embeddings = nn.ModuleList([
            nn.Embedding(n_genera + 1, taxonomy_embed_dim, padding_idx=0)
            for _ in range(top_k_taxonomy)
        ])
        self.family_embeddings = nn.ModuleList([
            nn.Embedding(n_families + 1, taxonomy_embed_dim, padding_idx=0)
            for _ in range(top_k_taxonomy)
        ])

    @property
    def output_dim(self) -> int:
        return (
            self.top_k_species * self.species_embeddings[0].embedding_dim +
            self.top_k_taxonomy * self.genus_embeddings[0].embedding_dim +
            self.top_k_taxonomy * self.family_embeddings[0].embedding_dim
        )

    def forward(self, species_ids, genus_ids, family_ids):
        """
        Args:
            species_ids: (batch, top_k_species) - top-k species indices
            genus_ids: (batch, top_k_taxonomy) - top-k genus indices
            family_ids: (batch, top_k_taxonomy) - top-k family indices
        """
        # Embed each top-k slot separately
        sp_embs = [emb(species_ids[:, i]) for i, emb in enumerate(self.species_embeddings)]
        g_embs = [emb(genus_ids[:, i]) for i, emb in enumerate(self.genus_embeddings)]
        f_embs = [emb(family_ids[:, i]) for i, emb in enumerate(self.family_embeddings)]

        return torch.cat(sp_embs + g_embs + f_embs, dim=1)
```

### Phase 3: Model Integration

**File**: `src/resolve/model/resolve_net.py`

Modify `ResolveNet` to support both encoders:

```python
class ResolveNet(nn.Module):
    def __init__(
        self,
        schema: DatasetSchema,
        species_encoding: str = "hash",  # "hash", "embed", "pretrained"
        hash_dim: int = 64,
        embed_dim: int = 32,
        taxonomy_embed_dim: int = 8,
        top_k_species: int = 10,
        top_k_taxonomy: int = 3,
        pretrained_embeddings: str | None = None,
        ...
    ):
        if species_encoding == "hash":
            self.encoder = SpeciesEncoder(hash_dim=hash_dim, ...)
            encoder_dim = hash_dim
        elif species_encoding == "embed":
            self.encoder = EmbeddingEncoder(
                n_species=len(schema.species_vocab),
                n_genera=len(schema.genus_vocab),
                n_families=len(schema.family_vocab),
                species_embed_dim=embed_dim,
                taxonomy_embed_dim=taxonomy_embed_dim,
                top_k_species=top_k_species,
                top_k_taxonomy=top_k_taxonomy,
            )
            encoder_dim = self.encoder.output_dim
        elif species_encoding == "pretrained":
            self.encoder = PretrainedEmbeddingEncoder(pretrained_embeddings)
            encoder_dim = self.encoder.output_dim
```

### Phase 4: Data Preparation Changes

**File**: `src/resolve/train/trainer.py`

Modify `_prepare_data` to extract top-k indices when using embed mode:

```python
def _prepare_data(self):
    if self.species_encoding == "embed":
        # Extract top-k species/genus/family indices per plot
        species_ids = self._extract_top_k_species(self.dataset)
        genus_ids = self._extract_top_k_genus(self.dataset)
        family_ids = self._extract_top_k_family(self.dataset)
        # Store as tensors alongside other features
```

### Phase 5: Pre-training Function

**File**: `src/resolve/pretrain/species2vec.py`

```python
def train_species_embeddings(
    species_data: pd.DataFrame,
    method: str = "skipgram",
    embed_dim: int = 32,
    window_size: int | None = None,
    min_count: int = 5,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 0.025,
) -> SpeciesEmbeddings:
    """
    Train species embeddings from co-occurrence data.

    Treats each plot as a "sentence" and species as "words".
    Species that appear together in plots will have similar embeddings.

    Args:
        species_data: DataFrame with columns [plot_id, species_id, abundance]
        method: "skipgram" or "cbow"
        embed_dim: Embedding dimension
        window_size: Co-occurrence window (None = all species in plot)
        min_count: Minimum species frequency to include
        epochs: Training epochs

    Returns:
        SpeciesEmbeddings object with save/load methods
    """
    # Build vocabulary
    species_counts = species_data["species_id"].value_counts()
    vocab = {sp: i+1 for i, sp in enumerate(species_counts[species_counts >= min_count].index)}

    # Create co-occurrence pairs
    pairs = _create_cooccurrence_pairs(species_data, vocab, window_size)

    # Train with negative sampling
    model = Word2VecModel(len(vocab) + 1, embed_dim, method)
    _train_word2vec(model, pairs, epochs, batch_size, lr)

    return SpeciesEmbeddings(vocab, model.embeddings.weight.detach())
```

## File Changes Summary

| File | Changes |
|------|---------|
| `src/resolve/data/dataset.py` | Add vocab building, top-k extraction |
| `src/resolve/data/schema.py` | Add vocab fields to DatasetSchema |
| `src/resolve/encode/__init__.py` | New package |
| `src/resolve/encode/hashing.py` | Move existing SpeciesEncoder |
| `src/resolve/encode/embedding.py` | New EmbeddingEncoder |
| `src/resolve/encode/pretrained.py` | New PretrainedEmbeddingEncoder |
| `src/resolve/model/resolve_net.py` | Support multiple encoder types |
| `src/resolve/train/trainer.py` | Add species_encoding param, data prep |
| `src/resolve/pretrain/__init__.py` | New package |
| `src/resolve/pretrain/species2vec.py` | train_species_embeddings() |
| `src/resolve/__init__.py` | Export train_species_embeddings |

## Testing Plan

1. **Unit tests**: Vocab building, top-k extraction, embedding encoder
2. **Integration test**: Train with embed mode on small dataset
3. **Benchmark**: Compare hash vs embed vs pretrained on ASAAS
4. **Target**: Match or exceed 80% band_25 on ASAAS

## Migration

- Default remains `species_encoding="hash"` for backward compatibility
- Existing code works unchanged
- New parameter surfaced in Trainer API

## Timeline

1. Phase 1 (Vocab): ~2 hours
2. Phase 2 (Encoder): ~2 hours
3. Phase 3 (Integration): ~2 hours
4. Phase 4 (Data prep): ~1 hour
5. Phase 5 (Pretrain): ~3 hours
6. Testing: ~2 hours

**Total**: ~12 hours implementation
