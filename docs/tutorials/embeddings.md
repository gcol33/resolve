# Understanding Embeddings

A trained RESOLVE model holds three kinds of learned representation: one vector
per plot, one per species, and one per genus and family. This guide covers how
to get them out and what they mean.

## Plot embeddings

The shared encoder turns a plot's composition, coordinates, and covariates into
a single latent vector, and the task heads read their predictions off it.

```python
import resolve_core as rc

predictor = rc.Predictor.load("model.pt", device="cpu")
out = predictor.predict_dataset(dataset, return_latent=True)

latent = out.latent                  # (n_plots, latent_dim)
plot_ids = list(out.plot_ids)
```

`latent` is a CPU float tensor with one row per plot, in the dataset's plot
order. `predictor.model.latent_dim` is its width, and it is the width of the
last entry in `ModelConfig.hidden_dims`.

Because every head reads the same vector, two plots that sit close together in
latent space are plots the model expects to behave alike on every target at
once.

## Species embeddings

The encoders that give each species its own row expose the table:

```python
species = predictor.get_species_embeddings()   # (n_species_vocab, species_embed_dim)
names   = predictor.species_vocab              # index -> species name, [0] is "<UNK>"
```

`Embed`, `RankPool`, and `Transformer` build such a table. `Hash` derives its
representation from the name itself and has none, and `Sparse` carries species
identity in the first layer's weight matrix rather than an embedding, so both
return `None`.

Row 0 is the reserved unknown token, which is where every species outside the
training vocabulary is encoded.

## Taxonomy embeddings

```python
genus  = predictor.get_genus_embeddings()      # (n_genera, genus_emb_dim)
family = predictor.get_family_embeddings()     # (n_families, family_emb_dim)

genus_names  = predictor.genus_vocab
family_names = predictor.family_vocab
```

They are present whenever the dataset carried a genus or family column and
`DatasetConfig.use_taxonomy` was left on. Both tables also reserve index 0 for
the unknown token.

The same vocabularies travel on the schema, so a dataset gives them without a
checkpoint:

```python
dataset.schema.genus_vocab
dataset.schema.family_vocab
dataset.species_vocab
```

## Projecting plot embeddings

```python
import numpy as np
import umap
import matplotlib.pyplot as plt

latent = predictor.predict_dataset(dataset, return_latent=True).latent.numpy()

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
xy = reducer.fit_transform(latent)

plt.figure(figsize=(10, 8))
plt.scatter(xy[:, 0], xy[:, 1], c=dataset.targets["area"].numpy(),
            cmap="viridis", alpha=0.6)
plt.colorbar(label="Area")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("plot_embeddings.png", dpi=150)
```

`dataset.targets` is a dict of target-name to tensor, in whatever space the
loader stored it: a `Log1p` target is stored transformed, so invert it with
`np.expm1` before using it as a colour scale in original units.

t-SNE takes the same input:

```python
from sklearn.manifold import TSNE

xy = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(latent)
```

## Comparing species and genera

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

genus = predictor.get_genus_embeddings().numpy()
names = list(predictor.genus_vocab)

similarity = cosine_similarity(genus)

target = "Quercus"
i = names.index(target)
order = np.argsort(similarity[i])[::-1]

print(f"Closest to {target}:")
for j in order[1:6]:
    print(f"  {names[j]}: {similarity[i, j]:.3f}")
```

Two genera end up close when the model found them interchangeable for
predicting its targets, on this dataset. That can line up with ecological or
phylogenetic similarity, and it is not a measurement of either.

Hierarchical clustering over the same table:

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = linkage(genus, method="ward")

plt.figure(figsize=(12, 8))
dendrogram(Z, labels=names, leaf_rotation=90)
plt.tight_layout()
plt.savefig("genus_dendrogram.png", dpi=150)
```

## Linear pooling and what stays readable

RESOLVE aggregates species effects linearly before any nonlinearity mixes them.

- `Hash` accumulates `sign * weight` for each species at its hashed bucket, so a
  plot's hashed vector is a signed sum of per-species contributions.
- `RankPool` takes a weight-normalized mean of per-species embeddings.
- `Sparse` hands the abundance vector straight to a linear layer.

In all three the species contribution to the encoder's input is additive, so a
plot's input decomposes into per-species terms. That decomposition holds up to
the first nonlinearity, past which the encoder mixes them.

The transformer encoder breaks this on purpose: attention makes a species'
contribution depend on the rest of the plot, which is the point of using it.

## Practical uses

### Similar plots

```python
from sklearn.neighbors import NearestNeighbors

latent = predictor.predict_dataset(dataset, return_latent=True).latent.numpy()
plot_ids = np.asarray(dataset.plot_ids)

nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(latent)
distances, indices = nn.kneighbors(latent[:1])

print(f"Closest to {plot_ids[0]}:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"  {plot_ids[idx]}: {dist:.3f}")
```

### Unusual plots

```python
from sklearn.ensemble import IsolationForest

labels = IsolationForest(contamination=0.05, random_state=42).fit_predict(latent)
print("Flagged:", plot_ids[labels == -1])
```

A plot lands far from the rest when its composition is unlike what the model
saw. Read it next to that plot's residual before calling it an outlier.

### Gradients

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit(latent)
for i, var in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {var:.1%}")
```

## Embeddings without a predictor

`ResolveModel.get_latent` runs the encoder on tensors directly, which is what a
custom training loop or an ablation needs:

```python
latent = model.get_latent(
    continuous,
    genus_ids=dataset.genus_ids,
    family_ids=dataset.family_ids,
    species_ids=dataset.species_ids,
)
```

`continuous` is the concatenation the trainer builds: coordinates, covariates,
the unknown-mass columns, and the hash embedding in hash mode, in that order.
`tests/core/conftest.py` has a `trainer_continuous` helper that assembles it.

The untrained weight tables are reachable the same way, from the model rather
than a predictor:

```python
model.get_species_weights()
model.get_genus_weights()
model.get_family_weights()
```

## Next steps

- [Making Predictions](prediction.md): scoring new data and writing results
- [Encoding Modes](encoding-modes.md): which encoders build which tables
- [Training Models](training.md): configuring the encoder these vectors come from
