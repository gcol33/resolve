# Understanding Embeddings

RESOLVE learns several types of embeddings that capture ecological relationships. This guide explains how to extract and interpret them.

## Types of Embeddings

### 1. Plot Embeddings (Latent Representations)

The shared encoder produces a latent vector for each plot that captures its ecological "fingerprint":

```python
predictions = predictor.predict(dataset, return_latent=True)
latent = predictions.latent  # Shape: (n_plots, latent_dim)
```

These embeddings encode:
- Species composition patterns
- Spatial context (if coordinates provided)
- Environmental conditions (if covariates provided)

### 2. Genus Embeddings

Learned representations for each genus:

```python
genus_emb = predictor.get_genus_embeddings()
# Shape: (n_genera, genus_emb_dim)
```

Similar genera (ecologically or phylogenetically related) should have similar embeddings.

### 3. Family Embeddings

Learned representations for each family:

```python
family_emb = predictor.get_family_embeddings()
# Shape: (n_families, family_emb_dim)
```

## Visualizing Plot Embeddings

### UMAP Projection

```python
import umap
import matplotlib.pyplot as plt

# Get embeddings
predictions = predictor.predict(dataset, return_latent=True)
latent = predictions.latent

# Reduce to 2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(latent)

# Plot colored by target
plt.figure(figsize=(10, 8))
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=dataset.get_targets()["area"],
    cmap="viridis",
    alpha=0.6,
)
plt.colorbar(label="Area")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Plot Embeddings Colored by Area")
plt.savefig("plot_embeddings.png", dpi=150)
```

### t-SNE Projection

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding_2d = tsne.fit_transform(latent)
```

## Analyzing Taxonomy Embeddings

### Genus Similarity

Find genera with similar ecological roles:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

genus_emb = predictor.get_genus_embeddings()
genus_names = dataset.schema.genus_vocab  # Get genus names

# Compute similarity matrix
similarities = cosine_similarity(genus_emb)

# Find most similar genera for a target genus
target_genus = "Quercus"
target_idx = genus_names.index(target_genus)
sim_scores = similarities[target_idx]

# Sort by similarity
sorted_idx = np.argsort(sim_scores)[::-1]
print(f"Genera most similar to {target_genus}:")
for i in sorted_idx[1:6]:  # Top 5 (excluding self)
    print(f"  {genus_names[i]}: {sim_scores[i]:.3f}")
```

### Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

genus_emb = predictor.get_genus_embeddings()
genus_names = dataset.schema.genus_vocab

# Compute linkage
Z = linkage(genus_emb, method="ward")

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=genus_names, leaf_rotation=90)
plt.title("Genus Embedding Dendrogram")
plt.tight_layout()
plt.savefig("genus_dendrogram.png", dpi=150)
```

## Linear Compositional Pooling

RESOLVE aggregates species effects linearly before nonlinear mixing. This means:

1. Each species contributes additively to the hash embedding
2. Contributions are weighted by abundance (after normalization)
3. The encoder then applies nonlinear transformations

This design preserves interpretability: you can decompose a plot's embedding into species contributions.

### Decomposing Plot Embeddings

```python
# For a single plot, its hash embedding is:
# h = sum(abundance[i] * hash_vector[species[i]]) for all species in plot
#
# The species encoder's contribution to the latent space
# follows this linear structure before the PlotEncoder's nonlinear layers.
```

## Practical Applications

### 1. Plot Similarity Search

Find plots with similar ecological characteristics:

```python
from sklearn.neighbors import NearestNeighbors

# Fit nearest neighbors model
nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(latent)

# Find similar plots for a query
query_idx = 0
distances, indices = nn.kneighbors([latent[query_idx]])

print(f"Plots similar to {dataset.plot_ids[query_idx]}:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"  {dataset.plot_ids[idx]}: distance={dist:.3f}")
```

### 2. Outlier Detection

Identify plots with unusual species compositions:

```python
from sklearn.ensemble import IsolationForest

detector = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = detector.fit_predict(latent)

outlier_plots = dataset.plot_ids[outlier_labels == -1]
print(f"Potential outliers: {outlier_plots}")
```

### 3. Ecological Gradients

Project embeddings onto interpretable axes:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_emb = pca.fit_transform(latent)

print("Variance explained by first 3 PCs:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.1%}")
```

## Next Steps

- [Training Models](training.md): Customize model architecture
- [Making Predictions](prediction.md): Use models for inference
