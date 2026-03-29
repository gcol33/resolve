# Performance Tuning

This guide covers the key levers for optimizing RESOLVE training speed and model accuracy.

## Learning Rate

The learning rate is the single most impactful hyperparameter. The right value depends on the encoding mode and model architecture.

### Defaults by Encoding Mode

| Encoding | Recommended LR | Notes |
|----------|---------------|-------|
| `hash` | `1e-3` | Default. Works reliably across dataset sizes. |
| `embed` | `1e-3` | Same as hash. Embedding gradients are well-behaved. |
| `rank_pool` | `1e-3` | Same as hash. Pooling stabilizes gradients. |
| `transformer` | `3e-4` | Lower LR required. Attention layers overflow at `1e-3`. |

### OneCycle Schedule

RESOLVE uses a OneCycle learning rate schedule by default. The LR ramps up from `lr/25` to `lr` over the first 30% of training, then anneals to near zero. This usually outperforms constant LR and requires no manual scheduling.

```python
trainer = resolve.Trainer(
    dataset,
    lr=1e-3,  # Peak learning rate for OneCycle
)
```

### Signs of Wrong LR

- **LR too high**: Loss spikes or NaN after a few epochs. Reduce by 3-10x.
- **LR too low**: Loss decreases very slowly, never reaching a good minimum. Increase by 3-10x.
- **LR slightly too high**: Loss oscillates without clear downward trend. Reduce by 2-3x.

## Batch Size

Batch size controls the trade-off between gradient noise and training speed.

### Guidelines

| Dataset size | Recommended batch size | Reasoning |
|-------------|----------------------|-----------|
| <10k plots | 512-2048 | Smaller batches add regularization, prevent overfitting |
| 10k-100k plots | 4096 (default) | Good balance of speed and gradient quality |
| >100k plots | 8192-16384 | Larger batches utilize GPU parallelism fully |

```python
trainer = resolve.Trainer(
    dataset,
    batch_size=4096,
)
```

### Batch Size and Learning Rate

When increasing batch size, consider scaling the learning rate proportionally. A common heuristic: if you double the batch size, multiply the LR by 1.5x (not 2x, since OneCycle already adapts).

### Memory Limits

If you hit GPU out-of-memory errors, reduce batch size first. RESOLVE's memory footprint scales linearly with batch size for all encoding modes except transformer, which scales quadratically due to the attention matrix.

## Hidden Dimensions

The `hidden_dims` parameter controls the shared MLP encoder's depth and width.

### Presets

```python
# Small: fast training, less overfitting risk
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[256, 128, 64],
)

# Default: deep network, good for most datasets
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[2048, 1024, 512, 256, 128, 64],
)

# Wide: more capacity per layer, fewer layers
model = resolve.ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[2048, 1024, 512],
)
```

### Choosing Dimensions

- **Small datasets (<5k plots)**: Use `[256, 128, 64]`. Deeper networks overfit without enough data.
- **Medium datasets (5k-50k plots)**: Use `[512, 256, 128]` or the default.
- **Large datasets (>50k plots)**: The default `[2048, 1024, 512, 256, 128, 64]` is a good starting point. Going wider (e.g., `[4096, 2048, 1024, 512]`) can help if the model underfits.
- **With transformer encoding**: The attention layers already provide capacity, so a shallower MLP (`[512, 256, 128]`) often suffices.

## Mixed Precision (AMP)

Automatic Mixed Precision uses fp16 for most operations while keeping critical accumulations in fp32. This roughly halves memory usage and doubles throughput on modern GPUs.

```python
trainer = resolve.Trainer(
    dataset,
    use_amp=True,   # Default: enabled on CUDA
)
```

### When to Disable AMP

Disable AMP for transformer encoding. Self-attention computes softmax over dot products that can overflow in fp16, producing NaN losses:

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="transformer",
    use_amp=False,  # Required for stable transformer training
)
```

For hash, embed, and rank_pool modes, AMP is safe and recommended.

## GPU Optimization

### cuDNN Benchmark Mode

Enabled by default when training on CUDA. Auto-tunes convolution and matmul algorithms for the specific input sizes, providing a one-time warmup cost in exchange for faster subsequent operations.

```python
# Enabled automatically — no configuration needed
# Equivalent to: torch.backends.cudnn.benchmark = True
```

### torch.compile

PyTorch 2.0+ can JIT-compile the model graph for 10-20% speedup. This is experimental and adds compilation overhead at the start of training.

```python
trainer = resolve.Trainer(
    dataset,
    compile_model=True,  # Default: False
)
```

!!! note "Compilation overhead"
    The first epoch will be slower due to compilation. The speedup appears from the second epoch onward. Worth it for long training runs (>50 epochs) but counterproductive for quick experiments.

### Data Prefetching

Double-buffered GPU prefetching overlaps data transfer with computation. Auto-enabled when `batch_size >= 16384` on CUDA.

```python
trainer = resolve.Trainer(
    dataset,
    prefetch_data=True,   # Force enable
    # prefetch_data=False  # Force disable
)
```

For smaller batch sizes, the transfer overhead is negligible and prefetching adds no benefit.

### GPU-Resident Data

For datasets that fit in GPU memory, RESOLVE can load the entire dataset onto the GPU once, eliminating all CPU-to-GPU transfers during training. This is handled automatically via `GPUTensorLoader` when CUDA is available and the dataset is small enough.

## Early Stopping

Early stopping monitors validation loss and halts training when the model stops improving.

```python
trainer = resolve.Trainer(
    dataset,
    patience=50,      # Default: stop after 50 epochs without improvement
    max_epochs=200,   # Hard upper limit
)
```

### Tuning Patience

| Use case | Patience | Reasoning |
|----------|----------|-----------|
| Quick experiments | 10-20 | Fast feedback, accept slightly suboptimal convergence |
| Standard training | 30-50 | Good balance of convergence and compute |
| Final production runs | 100 | Ensure the model has fully converged |
| Transformer encoding | 50-100 | Attention layers converge slower than MLP |

The best epoch's model weights are always restored after early stopping triggers, so higher patience never produces a worse model — it only costs more compute.

## Species Encoding Tips

### Hash Dimension

`hash_dim=64` consistently outperforms `hash_dim=32` on datasets with >1k species. The reduction in hash collisions more than compensates for the increased input dimension. For very large species pools (>50k), `hash_dim=128` can help further.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=64,  # Default is 32; 64 is usually better
)
```

### Rank-Pool Normalization

For rank_pool encoding, `log1p` normalization works best in practice. It compresses the abundance distribution without losing the ordering, which helps the weighted pooling attend to dominant species without ignoring rare ones.

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="rank_pool",
    species_normalization="log1p",
)
```

### Transformer Architecture

A good starting point for transformer encoding:

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="transformer",
    n_attention_layers=2,   # 2 self-attention layers
    n_heads=4,              # 4 attention heads
    transformer_ff_dim=256, # Feed-forward dimension
    transformer_pooling="attention",  # Learned attention pooling
    lr=3e-4,
    use_amp=False,
)
```

- More attention layers (3-4) can help on very large datasets but increase compute quadratically.
- `transformer_pooling="attention"` usually outperforms `"mean"` because it learns which species to weight.
- Keep `transformer_ff_dim` proportional to the embedding dimension (2-4x is typical).

## Recommended Configurations

### Quick Experiment (minutes)

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=64,
    hidden_dims=[256, 128, 64],
    max_epochs=50,
    patience=10,
    batch_size=4096,
)
```

### Standard Training (tens of minutes)

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=64,
    max_epochs=200,
    patience=30,
    batch_size=4096,
)
```

### Maximum Accuracy (hours)

```python
trainer = resolve.Trainer(
    dataset,
    species_encoding="transformer",
    n_attention_layers=2,
    n_heads=4,
    transformer_ff_dim=256,
    transformer_pooling="attention",
    lr=3e-4,
    use_amp=False,
    max_epochs=300,
    patience=100,
    batch_size=2048,
)
```

## Next Steps

- [Encoding Modes](encoding-modes.md): Understand each encoding strategy in depth
- [Training Models](training.md): Full training configuration reference
- [Making Predictions](prediction.md): Use trained models for inference
