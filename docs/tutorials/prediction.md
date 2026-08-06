# Making Predictions

This guide covers how to use trained RESOLVE models for inference.

## Loading a Trained Model

```python
import resolve_core as rc

predictor = rc.Predictor.load("model.pt")
```

`device` defaults to `"cpu"`. Inference on a typical RESOLVE model
(~5M-param MLP, 300k plots) finishes in ~12 s on CPU vs ~1 s on a 16 GiB
GPU; the throughput win on GPU does not amortise the operational cost
of GPU OOMs at predict time, so the default leans towards "always
works". Pass `device="cuda"` explicitly when the GPU is otherwise idle
and the test set is small enough that you know it fits.

## Basic Prediction

```python
# Load new data through the training vocabularies, so species, taxonomy, and
# categorical ids index the same embedding rows the model was trained on.
new_dataset = rc.ResolveDataset.from_csv_with_schema(
    "new_plots.csv", "new_species.csv", roles, targets, training_dataset,
)

predictions = predictor.predict_dataset(new_dataset)

for target, values in predictions.predictions.items():
    print(f"{target}: {values[:5]}")
```

Regression targets come back on the original scale; a `Log1p` target is
inverted before it is returned. Classification targets come back as `int64`
class codes indexed into `schema.targets[i].class_names`.

## Output Formats

`ResolvePredictions` carries `plot_ids` plus a dict of tensors, which writes out
with whatever frame library you already use:

```python
import pandas as pd

df = pd.DataFrame({"plot_id": list(predictions.plot_ids)})
for target, values in predictions.predictions.items():
    df[target] = values.numpy()
df.to_csv("predictions.csv", index=False)
```

The CLI writes the same layout directly:

```bash
resolve predict --model model.pt --header new_plots.csv \
                --species new_species.csv --output predictions.csv
```

```
plot_id,area,habitat,habitat_code
P001,125.3,Forest,2
P002,340.1,Grassland,0
P003,45.8,Forest,2
```

A classification target is written as two columns: `<target>` carries the
original class label the model was trained on, and `<target>_code` the integer
code it predicted. A checkpoint that has no class vocabulary (an
already-integer-coded column) repeats the code in both.

The CLI builds its dataset from the checkpoint's own species, taxonomy, and
categorical vocabularies, so the codes it feeds the model mean what they meant
at training time. Building a dataset for inference yourself needs the same:

```python
vocabs = predictor.external_vocabs
dataset = rc.ResolveDataset.from_csv_with_vocabs(
    "new_plots.csv", "new_species.csv", roles, targets,
    vocabs, predictor.dataset_config,
)
```

`predict_dataset` rejects a dataset built any other way, because a
freshly-fitted vocabulary assigns different integer codes to the same species
and the model would read the wrong embedding rows.

## Prediction Options

### Include Latent Representations

```python
predictions = predictor.predict_dataset(dataset, return_latent=True)

latent = predictions.latent
print(f"Latent shape: {latent.shape}")  # (n_plots, latent_dim)
```

## Extracting Embeddings

### Plot Embeddings

Get learned representations for all plots:

```python
latent = predictor.predict_dataset(dataset, return_latent=True).latent
print(f"Shape: {latent.shape}")  # (n_plots, latent_dim)
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

A dataset built against a checkpoint's vocabulary measures how much of each plot
the model has never seen:

```python
ds = rc.ResolveDataset.from_csv_with_schema(
    header, species, roles, targets, predictor.schema, cfg)

ds.unknown_fraction    # (n_plots,) share of abundance from species outside the vocabulary
ds.unknown_count       # (n_plots,) records naming a species outside the vocabulary
```

Both are computed over each plot's full record list, before top-k selection or a
pool cap, and both are concatenated into the encoder's continuous block when
`track_unknown_fraction` / `track_unknown_count` are set. A dataset built with
plain `from_csv` fits its own vocabulary from the file it reads, so every plot
reads 0.0 there.

## Batch Processing

`Predictor.predict_dataset` chunks the forward pass along dim 0 and
concatenates results on CPU. Default `batch_size = 4096` keeps peak
VRAM bounded on 16 GiB-class GPUs.

```python
# Default: chunked forward at batch_size = 4096.
predictions = predictor.predict_dataset(large_dataset)

# Custom chunk size — useful when a wider hidden layer demands a
# smaller chunk to fit on the device.
predictions = predictor.predict_dataset(large_dataset, batch_size=1024)

# Opt out of chunking entirely (legacy one-shot path). On a 16 GiB GPU
# at the v7 recipe (hash_dim=32, hidden=[2048,1024,512,256,128,64],
# 5.28M params) this OOMs above ~150k plots — only pass -1 when the
# whole test set is known to fit.
predictions = predictor.predict_dataset(large_dataset, batch_size=-1)
```

Outputs from `batch_size > 0` are bit-equivalent to the one-shot path:
each chunk goes through the same `model.forward()` call, and only the
input slicing differs. Returned tensors live on CPU regardless of
`Predictor.device` so callers can free GPU memory immediately after.

## Example: Complete Workflow

```python
import numpy as np
import pandas as pd
import resolve_core as rc

predictor = rc.Predictor.load("trained_model.pt")

roles = rc.RoleMapping()
roles.plot_id    = "PlotID"
roles.species_id = "Species"
roles.latitude   = "Latitude"
roles.longitude  = "Longitude"
roles.abundance  = "Cover"
roles.genus      = "Genus"
roles.family     = "Family"

targets = [
    rc.TargetSpec.regression("Area", rc.TransformType.Log1p),
    rc.TargetSpec.classification("Habitat", 5),
]

# In-memory frames avoid a temp-CSV round trip when the header is filtered
# per run; `from_pandas` is byte-identical to `from_csv` on the same data.
new_dataset = rc.ResolveDataset.from_pandas(
    pd.read_csv("new_plots.csv"),
    pd.read_csv("new_species.csv"),
    roles,
    targets,
    schema_source=training_dataset,
)

predictions = predictor.predict_dataset(new_dataset, return_latent=True)

results = pd.DataFrame({"plot_id": list(predictions.plot_ids)})
for target, values in predictions.predictions.items():
    results[target] = values.numpy()
results.to_csv("results.csv", index=False)

np.save("latent_vectors.npy", predictions.latent.numpy())
```

## Next Steps

- [Understanding Embeddings](embeddings.md): Interpret learned representations
- [Training Models](training.md): Train custom models
