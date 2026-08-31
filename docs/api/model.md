# Model

Everything in this page lives in `resolve_core`.

```python
import resolve_core as rc
```

---

## ResolveModel

The network: a species encoder, a shared plot encoder, and one head per target.

```python
model = rc.ResolveModel(dataset.schema, rc.ModelConfig())
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schema` | `ResolveSchema` | required | Sizes every input width and output head |
| `config` | `ModelConfig` | `ModelConfig()` | Architecture |

The schema fixes the input widths and the heads, so the model follows the data
rather than being described twice.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `schema` | `ResolveSchema` | The schema it was built from |
| `config` | `ModelConfig` | The config it was built with |
| `latent_dim` | `int` | Width of the shared encoder's output |
| `species_encoding` | `SpeciesEncodingMode` | The species encoder in use |
| `uses_explicit_vector` | `bool` | Whether the model reads `species_vector` |
| `uses_moe` | `bool` | Whether expert routing is active |
| `n_experts` | `int` | Expert count when routing is active |

### Forward passes

```python
outputs = model.forward(continuous, genus_ids=..., family_ids=...)
outputs = model(continuous, genus_ids=..., family_ids=...)      # __call__
```

```
forward(continuous, genus_ids=None, family_ids=None,
        species_ids=None, species_vector=None,
        pool_genus_ids=None, pool_family_ids=None, pool_weights=None,
        pool_mask=None, pool_has_cover=None, categorical_ids=None)
```

Returns `dict[str, Tensor]`, one entry per target. `continuous` is required;
which of the rest are needed follows the species encoding. `continuous` is the
concatenation the trainer builds: coordinates, covariates, the unknown-mass
columns, and the hash embedding in hash mode, in that order.

| Method | Returns |
|--------|---------|
| `forward(...)` / `__call__(...)` | `dict[str, Tensor]` |
| `get_latent(...)` | `Tensor` of shape `(batch, latent_dim)`; same arguments as `forward` |
| `forward_with_aux(...)` | `ModelForwardResult` with `outputs` and `moe_aux_loss` |
| `forward_single(target, continuous, genus_ids=None, family_ids=None, species_ids=None, species_vector=None, categorical_ids=None)` | `Tensor` for one head; rejects pool encoders |
| `encode_with_activations(continuous, genus_ids=None, family_ids=None, categorical_ids=None)` | `(latent, [activations])`; activations are non-empty for the hash encoder |
| `get_gate_probs(continuous, genus_ids=None, family_ids=None)` | `Tensor` of expert gate probabilities, empty without MoE. Covers the encoders whose species signal is already inside `continuous` (hash) or absent (TraitNet, adapter architectures); for an embed / sparse / rank_pool / transformer model it raises, because the signature carries no species input. Read those gates off `forward_with_aux` instead. |

The forward passes release the GIL around the compute, so other Python threads
run during a long pass.

### Embedding tables

| Method | Returns |
|--------|---------|
| `get_species_weights()` | `(n_species_vocab, species_embed_dim)`, or `None` for the hash and sparse encoders |
| `get_genus_weights()` | `(n_genera, genus_emb_dim)`, or `None` without taxonomy |
| `get_family_weights()` | `(n_families, family_emb_dim)`, or `None` without taxonomy |

### PyTorch-shaped surface

| Method | Description |
|--------|-------------|
| `parameters()` | `list[Tensor]`, ready for a torch optimizer |
| `named_parameters()` | `dict[str, Tensor]` |
| `state_dict()` | `dict[str, Tensor]` of cloned parameters and buffers |
| `load_state_dict(state_dict, strict=True)` | `(missing_keys, unexpected_keys)` |
| `n_parameters()` | `int` total element count |
| `train(mode=True)` / `eval()` | Switch training mode |
| `to(device)` | `"cpu"` or `"cuda"` |
| `zero_grad()` | Zero all gradients |
| `requires_grad_(requires_grad=True)` | Freeze or unfreeze every parameter |
| `set_traits(traits)` | Attach the trait matrix; valid only for the `TraitNet` architecture |

---

## ModelConfig

```python
config = rc.ModelConfig()
```

### Species encoding

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `species_encoding` | `SpeciesEncodingMode` | `Hash` | Must match `DatasetConfig.species_encoding` |
| `hash_dim` | `int` | `32` | Must match `DatasetConfig.hash_dim` |
| `uses_explicit_vector` | `bool` | `False` | Read the sparse species vector |
| `species_embed_dim` | `int` | `32` | Species embedding width |
| `top_k` | `int` | `3` | Fixed taxonomy slots |
| `top_k_species` | `int` | `10` | Species slots in embed mode |
| `n_taxonomy_slots` | `int` | `3` | Genus and family slot count; `2 * top_k` under `TopBottom` |
| `genus_emb_dim`, `family_emb_dim` | `int` | `8` | Taxonomy embedding widths |
| `categorical_embed_dim` | `int` | `8` | Embedding width per categorical column |

### Shared encoder

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dims` | `list[int]` | `[2048, 1024, 512, 256, 128, 64]` | Layer widths |
| `dropout` | `float` | `0.3` | Dropout between layers |
| `activation` | `ActivationType` | `GELU` | See enums below |
| `normalization` | `NormLayerType` | `BatchNorm` | See enums below |
| `norm_groups` | `int` | `32` | Groups for `GroupNorm` |
| `use_residual` | `bool` | `False` | Residual connections between blocks |
| `leaky_relu_slope` | `float` | `0.01` | For `LeakyReLU` |
| `elu_alpha` | `float` | `1.0` | For `ELU` |

### Heads

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `head_hidden_dims` | `list[int]` | `[]` | Empty gives a single linear head per target |
| `head_activation` | `ActivationType` | `GELU` | Activation inside multi-layer heads |
| `head_dropout` | `float` | `0.0` | Dropout inside heads |

### Pool and transformer encoders

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `cover_dropout` | `float` | `0.0` | Probability of replacing a plot's pool weights with the presence mask during training |
| `d_model` | `int` | `128` | Token width |
| `n_heads` | `int` | `4` | Attention heads |
| `n_attention_layers` | `int` | `0` | Self-attention layers; `0` gives pooling alone |
| `transformer_ff_dim` | `int` | `256` | Feed-forward width inside an attention layer |
| `transformer_pooling` | `str` | `"attention"` | `"attention"` or `"cls"`; `"cls"` needs at least one attention layer |
| `transformer_dropout` | `float` | `0.1` | Dropout inside attention layers |

### Mixture of experts

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `moe_routing` | `MoERoutingType` | `None_` | `None_`, `Soft`, `TopK` |
| `moe_placement` | `MoEPlacement` | `Tail` | `Tail` or `Post`; where the mixture sits |
| `n_experts` | `int` | `4` | Expert count, at least 2 |
| `expert_hidden_dims` | `list[int]` | `[256, 128]` | Expert MLP widths |
| `moe_top_k` | `int` | `2` | Experts per sample under `TopK` |
| `moe_noise_std` | `float` | `0.1` | Routing noise during training |
| `moe_aux_loss_weight` | `float` | `0.01` | Weight of the load-balancing loss |

Routing is available in every species encoding.

`moe_placement` decides what the mixture replaces:

| Placement | What it does | Available to |
|-----------|--------------|--------------|
| `Tail` (default) | The experts are the encoder's final stage. `hidden_dims` minus its last two widths becomes the backbone; the mixture projects that to `hidden_dims[-1]`, which is the latent. Capacity moves into the experts rather than being stacked on top. | Every species encoding: `Hash`, `Embed`, `Sparse`, `RankPool`, `Transformer` |
| `Post` | The encoder produces its latent as usual and the mixture maps that latent to one of the same width. | Any encoder, including the adapter architectures and `TraitNet` |

`Tail` and TabM both claim the encoder's MLP tail, so asking for both raises.
Asking an adapter architecture or `TraitNet` for `Tail` raises too, naming
`Post` as the placement that works.

### Encoder architecture

`encoder_architecture` replaces the shared MLP. Each value has its own
sub-config object on `ModelConfig`.

```python
config.encoder_architecture = rc.EncoderArchitecture.TabNet
config.tabnet.n_steps = 5
config.tabnet.use_sparsemax = False       # 1.5-entmax instead
```

| Architecture | Sub-config | Fields |
|--------------|------------|--------|
| `MLP` | `hidden_dims` and friends | Default |
| `FTTransformer` | `ft_transformer` | `d_model`, `n_heads`, `n_layers`, `attention_dropout`, `ffn_dropout`, `ffn_multiplier`, `pre_norm` |
| `TabNet` | `tabnet` | `n_steps`, `n_d`, `n_a`, `relaxation_factor`, `sparsity_coefficient`, `virtual_batch_size`, `use_sparsemax` |
| `SAINT` | `saint` | `d_model`, `n_heads`, `n_layers`, `attention_dropout`, `use_row_attention`, `use_contrastive_pretrain`, `mixup_alpha` |
| `ExcelFormer` | `excelformer` | `d_model`, `n_heads`, `n_layers`, `attention_dropout`, `ffn_multiplier`, `importance_threshold`, `pre_norm` |
| `TraitNet` | `trait_net` | `env_dim`, `trait_dim`, `interaction_dim`, `interaction`, `shared_trait_encoder` |
| `GNN` | `gnn` | `gnn_type`, `n_layers`, `hidden_dim`, `n_heads`, `k_neighbors`, `graph_mode`, `edge_dropout`, `use_edge_features` |
| `HeterogeneousGNN` | `heterogeneous_gnn` | `hidden_dim`, `output_dim`, `n_layers`, `n_edge_types`, `n_heads`, `dropout`, `k_cooccurrence`, `cooccurrence_threshold`, `use_taxonomic_edges`, `use_cooccurrence_edges` |

`TabNet.use_sparsemax = False` selects exact 1.5-entmax
(Peters, Niculae & Martins, arXiv:1905.05702, Algorithm 2), which keeps strictly
more features per step than sparsemax.

`GNN` with `graph_mode = Spatial` needs coordinates and trains full-batch, so
its k-nearest-neighbour graph spans every plot rather than an arbitrary batch.

`TraitNet` needs a trait matrix supplied through `model.set_traits(traits)`.

### Parallel branches and TabM

| Attribute | Fields |
|-----------|--------|
| `parallel_layers` | `enabled`, `branches` (a list of `ParallelBranchConfig`), `aggregation`, `attention_heads`, `use_residual` |
| `tabm` | `enabled`, `n_ensembles`, `aggregation` |

`ParallelBranchConfig` carries `hidden_dims`, `activation`, `normalization`,
`dropout`, and `branch_weight`.

---

## ModelForwardResult

Returned by `forward_with_aux`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `outputs` | `dict[str, Tensor]` | One entry per target |
| `moe_aux_loss` | `Tensor` or `None` | Load-balancing loss when routing is active |

---

## Enums

| Enum | Values |
|------|--------|
| `ActivationType` | `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Tanh`, `Mish`, `ELU`, `SELU`, `Softplus`, `PReLU` |
| `NormLayerType` | `BatchNorm`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `None_` |
| `EncoderArchitecture` | `MLP`, `FTTransformer`, `TabNet`, `SAINT`, `TraitNet`, `GNN`, `ExcelFormer`, `HeterogeneousGNN` |
| `MoERoutingType` | `None_`, `Soft`, `TopK` |
| `MoEPlacement` | `Tail`, `Post` |
| `GNNType` | `GCN`, `GAT`, `GraphSAGE` |
| `GraphConstructionMode` | `Spatial`, `Taxonomic`, `CoOccurrence` |
| `TraitInteractionMode` | `Bilinear`, `MLP`, `Attention` |
| `ParallelAggregation` | `Concat`, `Sum`, `Mean`, `Attention`, `Gated` |

---

## Architecture

```
Coordinates ──────┐
Covariates ───────┤
Categorical ids ──┼──→ continuous block ──┐
Unknown mass ─────┘                       │
                                          ├──→ shared encoder ──→ latent
Species set ──→ species encoder ──────────┤
                (hash / embed / sparse /  │
                 rank pool / transformer) │
Genus, family ──→ taxonomy embeddings ────┘
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ↓                         ↓                         ↓
                    head(area)               head(elevation)            head(habitat)
                    regression                 regression              classification
```

---

## See also

- [Encoding Modes](../tutorials/encoding-modes.md)
- [Trainer](trainer.md)
- [Dataset](dataset.md)
