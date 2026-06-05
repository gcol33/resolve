# RESOLVE Roadmap

**R**epresentation **E**ncoding for **S**tructured **O**bservation **L**earning with **V**ector **E**mbeddings

---

## v2.0 - Advanced Architectures (NEW)

### Overview

Expanding RESOLVE beyond MLP-based encoders to include attention mechanisms, graph networks, and foundation model integration.

### Architecture 1: FT-Transformer

**Priority: HIGH** | **Status: IMPLEMENTED**

Feature Tokenizer + Transformer. Each input feature is transformed into an embedding, then self-attention is applied across all feature tokens.

- Outperforms GBDTs on most tabular tasks
- Works well with mixed categorical/numerical data
- Built-in feature interaction learning

```python
trainer = Trainer(
    dataset,
    architecture="ft_transformer",
    d_model=192,
    n_heads=8,
    n_layers=3,
)
```

### Architecture 2: TabNet

**Priority: HIGH** | **Status: IMPLEMENTED**

Sequential attention mechanism that selects features at each decision step. Provides built-in interpretability through feature importance masks.

- Sparse feature selection (like decision trees)
- Built-in feature importance
- Interpretable predictions

```python
trainer = Trainer(
    dataset,
    architecture="tabnet",
    n_steps=3,
    n_d=64,
    n_a=64,
    relaxation_factor=1.5,
)
```

### Architecture 3: Graph Neural Network (GNN)

**Priority: MEDIUM** | **Status: IMPLEMENTED**

Model relationships between plots or species using graph structure. Captures spatial proximity, taxonomic relationships, or species co-occurrence.

- Spatial graphs: plots as nodes, k-NN edges
- Species graphs: co-occurrence or phylogenetic edges
- Message passing for neighborhood aggregation

```python
trainer = Trainer(
    dataset,
    architecture="gnn",
    gnn_type="gat",  # gcn, gat, graphsage
    n_layers=3,
    k_neighbors=10,
)
```

### Architecture 4: SAINT

**Priority: MEDIUM** | **Status: IMPLEMENTED**

Self-Attention + Inter-sample Attention. Attention across both features (columns) AND samples (rows) within a batch.

- Row attention captures sample similarities
- Column attention captures feature interactions
- Optional contrastive pre-training

```python
trainer = Trainer(
    dataset,
    architecture="saint",
    d_model=128,
    n_heads=8,
    use_row_attention=True,
)
```

### Architecture 5: Trait-based Multi-species Network

**Priority: MEDIUM** | **Status: IMPLEMENTED**

Incorporates species functional traits to share information across species. Traits mediate the environment-species relationship.

- Species traits inform predictions
- Transfer learning across species
- Ecological interpretability

```python
trainer = Trainer(
    dataset,
    architecture="trait_net",
    trait_data=species_traits_df,
    interaction_mode="bilinear",
)
```

### Architecture 6: TabPFN Integration

**Priority: LOW** | **Status: PLANNED**

Integration with TabPFN v2 foundation model for zero-shot or few-shot predictions.

- No training required for small datasets
- Use embeddings in RESOLVE heads
- Baseline comparison

```python
trainer = Trainer(
    dataset,
    architecture="tabpfn",
    use_embeddings=True,
)
```

### Implementation Checklist

- [x] Shared attention infrastructure (MultiHeadAttention, TransformerBlock)
- [x] FT-Transformer encoder
- [x] TabNet encoder (with sparsemax)
- [x] SAINT encoder (row + column attention)
- [x] Trait-based network
- [x] GNN encoder (GCN, GAT, GraphSAGE)
- [ ] TabPFN Python wrapper
- [x] Python bindings for all configs
- [ ] Benchmarks vs MLP baseline
- [ ] Integration with Trainer (architecture selection)

---

## v1.1 (Planned)

### Multiple observation tables

Support multiple many-to-one relationships in a single model:

```r
resolve(
  ltv ~ age + hash(product_id, from = purchases) + embed(category, from = browsing),
  data = customers,
  obs = list(purchases = purchase_df, browsing = browse_df),
  by = "customer_id"
)
```

Use cases:
- Customers with purchases AND browsing history
- Patients with diagnoses AND prescriptions AND procedures
- Documents with words AND citations AND authors
- Plots with species AND environmental measurements over time

Implementation notes:
- `from = <table_name>` parameter in `hash()`/`embed()`/`onehot()`
- `obs` accepts named list of data frames
- Each table can have different join key (extend `by` to named vector)

---

## v1.0 (Current)

- Generalized PlotEncoder (hash/embed/onehot/numeric/raw)
- R formula interface with `data` + `obs` pattern
- Separate `top_k` and `bottom_k` selection
- Categorical validation (error if bare variable is categorical)
- PyTorch plug-and-play compatibility (parameters, state_dict, optimizers)
- Configurable architecture (activations, normalizations, residuals)
