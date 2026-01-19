# Quick Start

This guide walks through a complete RESOLVE workflow: loading data, training a model, and making predictions.

## Data Requirements

RESOLVE expects two data sources:

1. **Header file**: One row per plot with plot-level attributes
2. **Species file**: One row per species occurrence (plot × species)

## Example Workflow

```python
import resolve

# 1. Define semantic roles (map your column names)
roles = {
    "plot_id": "PlotObservationID",
    "species_id": "Species",
    "species_plot_id": "PlotObservationID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover",           # optional
    "taxonomy_genus": "Genus",      # optional
    "taxonomy_family": "Family",    # optional
    "covariates": ["Temperature"],  # optional
}

# 2. Define targets
targets = {
    "area": {
        "column": "Area",
        "task": "regression",
        "transform": "log1p",
    },
    "habitat": {
        "column": "Habitat",
        "task": "classification",
        "num_classes": 5,
    },
}

# 3. Load data
dataset = resolve.ResolveDataset.from_csv(
    header="plots.csv",
    species="species.csv",
    roles=roles,
    targets=targets,
)

# 4. Check schema
print(f"Plots: {dataset.schema.n_plots}")
print(f"Species: {dataset.schema.n_species}")
print(f"Genera: {dataset.schema.n_genera}")
print(f"Families: {dataset.schema.n_families}")

# 5. Build model
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[256, 128, 64],
)

# 6. Train
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    max_epochs=200,
    patience=30,
    batch_size=256,
)
result = trainer.fit()

print(f"Best epoch: {result.best_epoch}")
for target, metrics in result.final_metrics.items():
    print(f"{target}: {metrics}")

# 7. Save model
trainer.save("model.pt")

# 8. Predict on new data
predictor = resolve.Predictor.load("model.pt")
predictions = predictor.predict(new_dataset)
predictions.to_csv("predictions.csv")
```

## Role Mapping

The `roles` dictionary maps your column names to RESOLVE's semantic roles:

| Role | Required | Description |
|------|----------|-------------|
| `plot_id` | Yes | Unique plot identifier in header |
| `species_id` | Yes | Species identifier in species file |
| `species_plot_id` | Yes | Plot identifier in species file (for joining) |
| `coords_lat` | No | Latitude coordinate |
| `coords_lon` | No | Longitude coordinate |
| `abundance` | No | Species abundance/cover value |
| `taxonomy_genus` | No | Genus name for taxonomy embeddings |
| `taxonomy_family` | No | Family name for taxonomy embeddings |
| `covariates` | No | List of additional predictor columns |

## Target Configuration

Each target specifies:

- `column`: Column name in header file
- `task`: `"regression"` or `"classification"`
- `transform`: Optional transform (`"log1p"` for regression)
- `num_classes`: Required for classification tasks
- `weight`: Optional loss weighting (default: 1.0)

## Next Steps

- [Data Preparation](data-preparation.md): Detailed data formatting guide
- [Training Models](training.md): Advanced training options
- [Understanding Embeddings](embeddings.md): Interpreting learned representations
