# Training Models

This guide covers RESOLVE's training configuration and optimization strategies.

## Basic Training

```python
model = resolve.ResolveModel(schema=dataset.schema, targets=targets)
trainer = resolve.Trainer(model, dataset)
result = trainer.fit()
```

## Trainer Configuration

### Core Parameters

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    max_epochs=200,      # Maximum training epochs
    patience=30,         # Early stopping patience
    batch_size=256,      # Samples per batch
    lr=1e-3,             # Learning rate
    device="auto",       # "auto", "cpu", or "cuda"
)
```

### Phased Training

RESOLVE uses phased training for regression targets:

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    phase_boundaries=(50, 150),  # Phase transitions at epochs 50 and 150
)
```

**Phases:**

1. **Phase 1 (epochs 0-50)**: MAE loss - robust initial learning
2. **Phase 2 (epochs 50-150)**: SMAPE loss - scale-invariant refinement
3. **Phase 3 (epochs 150+)**: Band accuracy - calibrated predictions

### Validation Split

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    val_fraction=0.2,  # 20% validation set
)
```

## Model Architecture

### Hidden Dimensions

Configure the shared encoder's hidden layers:

```python
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[256, 128, 64],  # 3 hidden layers
)
```

### Hash Dimension

Control the species feature hashing dimension:

```python
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hash_dim=32,  # Default: 32
)
```

## Species Encoding Options

### Abundance Normalization

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    species_normalization="relative_plot",  # Default
)
```

Options:
- `"raw"`: Use abundance values directly
- `"relative_plot"`: Normalize to sum to 1 per plot
- `"log_scaled"`: Apply log1p transform

### Unknown Species Tracking

Enable detailed tracking of novel species:

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    track_unknown_count=True,  # Include count of unknown species
)
```

## Training Results

The `fit()` method returns a result object:

```python
result = trainer.fit()

print(f"Best epoch: {result.best_epoch}")
print(f"Stopped at epoch: {result.stopped_epoch}")

for target, metrics in result.final_metrics.items():
    print(f"\n{target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
```

### Metrics

**Regression targets:**
- `mae`: Mean Absolute Error
- `smape`: Symmetric Mean Absolute Percentage Error
- `band_5pct`: Fraction within 5% of true value
- `band_10pct`: Fraction within 10% of true value
- `band_20pct`: Fraction within 20% of true value

**Classification targets:**
- `accuracy`: Overall accuracy
- `f1_macro`: Macro-averaged F1 score

## Saving and Loading

```python
# Save trained model
trainer.save("model.pt")

# Load for prediction
predictor = resolve.Predictor.load("model.pt")
```

## GPU Training

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    device="cuda",  # Use GPU
)
```

Or auto-detect:

```python
trainer = resolve.Trainer(
    model=model,
    dataset=dataset,
    device="auto",  # Uses GPU if available
)
```

## Next Steps

- [Making Predictions](prediction.md): Use trained models for inference
- [Understanding Embeddings](embeddings.md): Extract and interpret latent representations
