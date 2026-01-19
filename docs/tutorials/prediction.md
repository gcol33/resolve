# Making Predictions

This guide covers how to use trained RESOLVE models for inference.

## Loading a Trained Model

```python
import resolve

predictor = resolve.Predictor.load("model.pt", device="auto")
```

## Basic Prediction

```python
# Load new data (same format as training data)
new_dataset = resolve.ResolveDataset.from_csv(
    header="new_plots.csv",
    species="new_species.csv",
    roles=roles,
    targets=targets,  # Can be empty dict if no targets
)

# Predict
predictions = predictor.predict(new_dataset)

# Access results
for target in predictions.predictions:
    print(f"{target}: {predictions[target][:5]}")  # First 5 predictions
```

## Output Formats

### DataFrame Export

```python
df = predictions.to_dataframe()
print(df.head())
```

Output:
```
   plot_id   area  habitat
0    P001  125.3        2
1    P002  340.1        0
2    P003   45.8        2
```

### CSV Export

```python
predictions.to_csv("predictions.csv")
```

## Prediction Options

### Output Space

For regression targets with transforms:

```python
# Get predictions in original scale (default)
predictions = predictor.predict(dataset, output_space="raw")

# Get predictions in transformed space (e.g., log1p)
predictions = predictor.predict(dataset, output_space="transformed")
```

### Include Latent Representations

```python
predictions = predictor.predict(dataset, return_latent=True)

# Access latent vectors
latent = predictions.latent
print(f"Latent shape: {latent.shape}")  # (n_plots, latent_dim)
```

## Extracting Embeddings

### Plot Embeddings

Get learned representations for all plots:

```python
embeddings = predictor.get_embeddings(dataset)
print(f"Shape: {embeddings.shape}")  # (n_plots, latent_dim)
```

Use for:
- Visualization (UMAP, t-SNE)
- Clustering
- Similarity analysis

### Taxonomy Embeddings

Extract learned genus and family representations:

```python
genus_emb = predictor.get_genus_embeddings()
family_emb = predictor.get_family_embeddings()

print(f"Genus embeddings: {genus_emb.shape}")   # (n_genera, emb_dim)
print(f"Family embeddings: {family_emb.shape}") # (n_families, emb_dim)
```

## Handling New Species

RESOLVE tracks species not seen during training:

```python
# During encoding, unknown species contribute to:
# - unknown_fraction: Proportion of abundance from unknown species
# - unknown_count: Number of unknown species (if enabled)
```

The model uses these features to adjust predictions for plots with novel species.

## Batch Processing

For large datasets, predictions are made in batches automatically:

```python
# Predictor handles batching internally
predictions = predictor.predict(large_dataset)
```

## Example: Complete Workflow

```python
import resolve
import pandas as pd

# Load trained model
predictor = resolve.Predictor.load("trained_model.pt")

# Prepare new data
new_header = pd.read_csv("new_plots.csv")
new_species = pd.read_csv("new_species.csv")

roles = {
    "plot_id": "PlotID",
    "species_id": "Species",
    "species_plot_id": "PlotID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover",
    "taxonomy_genus": "Genus",
    "taxonomy_family": "Family",
}

new_dataset = resolve.ResolveDataset(
    header=new_header,
    species=new_species,
    roles=roles,
    targets={},  # No targets for prediction-only
)

# Predict with latent representations
predictions = predictor.predict(new_dataset, return_latent=True)

# Export results
predictions.to_csv("results.csv")

# Save latent representations
import numpy as np
np.save("latent_vectors.npy", predictions.latent)
```

## Next Steps

- [Understanding Embeddings](embeddings.md): Interpret learned representations
- [Training Models](training.md): Train custom models
