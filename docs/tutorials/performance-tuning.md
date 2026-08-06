# Performance Tuning

This guide covers the levers for RESOLVE's training speed, memory footprint, and
accuracy, and what each one actually does in the engine.

## Learning rate

`TrainConfig.lr` feeds AdamW, together with `weight_decay` (default `1e-4`).

```python
cfg = rc.TrainConfig()
cfg.lr           = 1e-3
cfg.weight_decay = 1e-4
```

`1e-3` is the default and a reasonable starting point for the hash, embed,
sparse, and rank-pool encoders. Attention layers are more sensitive: with
`n_attention_layers >= 1`, a lower rate such as `3e-4` is the usual place to
start.

Reading the loss curve:

- Loss spikes or turns to NaN within a few epochs: reduce by 3-10x.
- Loss falls very slowly and flattens high: raise by 3-10x.
- Loss oscillates without a downward trend: reduce by 2-3x.

### Schedules

No schedule runs by default (`lr_scheduler = LRSchedulerType.None_`); the rate
stays at `lr` for the whole run. Two schedules are available:

```python
cfg.lr_scheduler = rc.LRSchedulerType.StepLR
cfg.lr_step_size = 100      # decay every 100 epochs
cfg.lr_gamma     = 0.1      # by this factor

cfg.lr_scheduler = rc.LRSchedulerType.CosineAnnealing
cfg.lr_min       = 1e-6     # reached on the last epoch
```

Cosine annealing spans `max_epochs`, so it lands on `lr_min` at the last epoch
rather than a step short of it.

## Batch size

`batch_size` trades gradient noise against throughput.

```python
cfg.batch_size = 4096       # default
```

Starting points, to be measured rather than trusted:

| Dataset size | Try |
|--------------|-----|
| under 10k plots | 512 to 2048 |
| 10k to 100k plots | 4096 |
| over 100k plots | 8192 to 16384 |

Smaller batches add gradient noise, which acts as regularization on a small
dataset; larger batches keep a GPU busy. When you raise the batch size, the
learning rate usually has to rise with it.

Memory grows linearly with batch size for every encoder. The transformer
encoder additionally holds an attention matrix quadratic in species per plot, so
`pool_species_cap` matters there as much as `batch_size`.

## Model size

`hidden_dims` sets the shared encoder's depth and width. The default is
`[2048, 1024, 512, 256, 128, 64]`.

```python
model_config = rc.ModelConfig()
model_config.hidden_dims = [256, 128, 64]        # small
model_config.hidden_dims = [512, 256, 128]       # mid
model_config.hidden_dims = [2048, 1024, 512]     # wide and shallow
```

- Small datasets (under 5k plots): a deep default overfits before it converges;
  `[256, 128, 64]` is a better place to start.
- Mid-size datasets: `[512, 256, 128]` or the default.
- Large datasets: the default, widened if the training loss plateaus above the
  test loss.
- With the transformer encoder: the attention layers already carry capacity, so
  a shallower MLP above them is usually enough.

`dropout` (default `0.3`) applies between encoder layers, and `head_dropout`
(default `0.0`) inside the per-target heads. `use_residual = True` adds residual
connections, which helps deeper stacks train.

## Mixed precision

AMP runs most operations in fp16 and keeps the accumulations that need range in
fp32. It is off by default:

```python
cfg.use_amp             = True
cfg.amp_init_scale      = 65536.0
cfg.amp_growth_factor   = 2.0
cfg.amp_backoff_factor  = 0.5
cfg.amp_growth_interval = 2000
```

Normalization layers are forced to fp32 inside the autocast region, so
BatchNorm statistics cannot saturate at the fp16 maximum and collapse
eval-mode output. Set `RESOLVE_FP32_NORM=0` to A/B that guard.

If a loss goes to NaN a few steps into an AMP run with attention layers, turn
AMP off before touching anything else.

## GPU settings

### cuDNN and TF32

```python
cfg.cudnn_benchmark = True    # default; auto-tunes algorithms for fixed shapes
cfg.allow_tf32      = True    # default; TF32 matmuls on Ampere and later
```

Both are set once at the start of `fit`, so `cudnn_benchmark = False` for a
deterministic run stays false for the whole run.

### GPU-resident data

On CUDA, the trainer uploads the training split once and indexes batches on the
device, so no batch crosses the bus during the epoch loop. This happens on its
own; there is no knob.

### Hash prefetch

In hash mode with `use_cuda_hash = True`, each batch's hash embedding is
computed by a CUDA kernel. The trainer runs the next batch's hash on a side
stream while the current batch's forward pass runs, so the two overlap. This
also happens on its own.

### Limiting VRAM usage

RESOLVE leaves the PyTorch CUDA caching allocator uncapped by default
(`TrainConfig.vram_fraction = 1.0`) so dedicated training jobs on a solo GPU
use the full device. Pass an explicit lower value when sharing the GPU with a
desktop: on Windows the WDDM driver spills allocations beyond physical VRAM
into shared system memory, which freezes the desktop under load. Leaving
headroom keeps the compositor, browser, and other GPU clients responsive
while training runs.

```python
from resolve_core import TrainConfig

cfg = TrainConfig()
cfg.vram_fraction = 1.0   # default, dedicated training job on a solo GPU
cfg.vram_fraction = 0.80  # sharing the GPU with a desktop or GUI
```

The same cap is applied automatically when `Predictor.load` runs on a CUDA
device, with the same default. The CLI exposes `--vram-fraction FLOAT` for both
`resolve train` and `resolve predict`.

To apply the cap independently of either, before constructing any RESOLVE
object:

```python
import resolve_core
resolve_core.set_vram_fraction(0.80)
```

!!! note "Compute saturation"
    The cap addresses VRAM-exhaustion hangs only. If the desktop becomes
    sluggish rather than fully freezing while RESOLVE trains, that is GPU
    *compute* saturation, not memory. See
    `dev_notes/compute_cap_plan.md` for the design path on a compute cap
    (CUDA Green Contexts), which is not currently implemented.

### Allocator config: `configure_cuda_allocator`

`resolve_core.configure_cuda_allocator()` runs once at module import time
and sets a platform-aware `PYTORCH_CUDA_ALLOC_CONF` default. The PyTorch
CUDA caching allocator reads that variable exactly once, on first
initialization, so the call has to happen *before* the first `import torch`
in the process. `resolve_core` performs this ordering itself; call the helper
explicitly only when your code imports torch before resolve_core, or when you
want to log the active config:

```python
import resolve_core
config = resolve_core.configure_cuda_allocator()  # idempotent; returns the active value
config = resolve_core.configure_cuda_allocator(force=True)  # overwrite an existing value
print("PYTORCH_CUDA_ALLOC_CONF:", config)
```

The chosen defaults differ by platform:

| Platform   | Prefix                          | Tail                                                       |
|------------|---------------------------------|------------------------------------------------------------|
| Linux/mac  | `expandable_segments:True,`     | `garbage_collection_threshold:0.8,max_split_size_mb:256`   |
| Windows    | *(omitted)*                     | `garbage_collection_threshold:0.8,max_split_size_mb:256`   |

The `expandable_segments` allocator uses cuMemMap-backed virtual memory and
lets the allocator release fragmented reserved blocks. PyTorch's OOM message
recommends turning it on, but **on Windows it is not implemented**: libtorch
prints `Warning: expandable_segments not supported on this platform` and
quietly ignores the request. RESOLVE skips the prefix on `win32` so the
warning does not show up in user logs.

The `garbage_collection_threshold:0.8` knob asks the allocator to GC reserved
blocks when `reserved_bytes / cap > 0.8`, and `max_split_size_mb:256` keeps
the allocator from carving very large free blocks into mismatched fragments.
Together they reduce reserved-but-unallocated fragmentation on Windows, and
they cannot match the headroom Linux gets from `expandable_segments`, so for
Windows jobs near the VRAM cap the right fallback is to halve the batch size.

### Auto-halve `batch_size` on OOM: `batch_size_floor`

`Trainer::fit` catches `c10::OutOfMemoryError` from the CUDA caching
allocator, releases the optimizer, AMP scaler, and GPU caches, halves
`config.batch_size`, asks the allocator to empty its cache, and restarts
training from epoch 0 against the original model weights. The retry stops
at `config.batch_size_floor` (default 1024); if halving would breach the
floor the original OOM is rethrown as a `std::runtime_error` carrying the
original requested batch size, the post-halve value, the floor, and the
underlying allocator message.

```python
from resolve_core import TrainConfig

cfg = TrainConfig()
cfg.batch_size = 16384       # what you would like to train with
cfg.batch_size_floor = 1024  # smallest the OOM retry will drop to
```

After `Trainer.fit()` returns, `trainer.config.batch_size` is the *effective*
batch size: equal to `cfg.batch_size` on a clean run, smaller if the retry
fired. The same value is persisted in the checkpoint under both
`train_batch_size` and `train_effective_batch_size`, alongside
`train_batch_size_floor`, so downstream tooling can flag fallback runs.

On Linux the allocator config delays the OOM by reducing fragmentation; on
Windows the batch-size halving recovers when fragmentation can no longer be
papered over.

CLI: `resolve train --batch-size 16384 --batch-size-floor 1024`.

## Early stopping

Early stopping watches the test-fold loss and halts when it stops improving.

```python
cfg.patience   = 50     # default
cfg.max_epochs = 500    # default; hard upper limit
```

| Use case | Patience |
|----------|----------|
| Quick experiments | 10 to 20 |
| Standard training | 30 to 50 |
| Final runs | 100 |
| Attention layers | 50 to 100, since they converge slower |

The best epoch's weights are restored when early stopping fires, so a larger
patience never gives a worse model. It costs compute.

## Encoding settings

### Hash dimension

Raising `hash_dim` gives species more buckets to land in, so fewer of them
collide. It also widens the encoder's input. `32` is the default; `64` and `128`
are the usual next steps on large vocabularies. Measure rather than assume:

```bash
python benchmarks/run_benchmarks.py --configs hash_32,hash_64
```

Remember that `selection` and `top_k` gate what reaches the hash at all; the
default keeps only the three most abundant species per plot.

### Pool weighting and capping

`pool_weighting = PoolWeighting.Log1p` (the default for rank-pool and
transformer modes) compresses the abundance range while preserving order, so
pooling attends to dominant species without ignoring the rest.

`pool_species_cap` bounds the padded per-plot width, which is the dominant term
in pool-mode memory:

```python
data_config.pool_species_cap = 0     # default; pad to the longest plot
data_config.pool_species_cap = -1    # auto: 99th percentile, reports the drop
data_config.pool_species_cap = 256   # manual
```

A cap truncates the species list, so it changes what the model sees. Leave it at
`0` for a result you intend to publish unless the uncapped run does not fit, and
say what you capped at if it does not.

### Transformer settings

```python
model_config.species_encoding    = rc.SpeciesEncodingMode.Transformer
model_config.d_model             = 128
model_config.n_heads             = 4
model_config.n_attention_layers  = 2
model_config.transformer_ff_dim  = 256
model_config.transformer_pooling = "attention"
train_config.lr      = 3e-4
train_config.use_amp = False
```

`transformer_pooling = "cls"` needs at least one attention layer, since the CLS
token only sees the species through attention. `transformer_ff_dim` is normally
2 to 4 times `d_model`.

## Recommended starting points

### Quick experiment

```python
data_config  = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.Hash
data_config.hash_dim         = 64
data_config.selection        = rc.SelectionMode.All

model_config = rc.ModelConfig()
model_config.species_encoding = rc.SpeciesEncodingMode.Hash
model_config.hash_dim         = 64
model_config.hidden_dims      = [256, 128, 64]

train_config = rc.TrainConfig()
train_config.max_epochs = 50
train_config.patience   = 10
train_config.batch_size = 4096
```

### Standard training

```python
model_config.hidden_dims = [512, 256, 128]
train_config.max_epochs  = 200
train_config.patience    = 30
train_config.batch_size  = 4096
```

### Longer run with pooling

```python
data_config.species_encoding  = rc.SpeciesEncodingMode.RankPool
data_config.pool_weighting    = rc.PoolWeighting.Log1p
model_config.species_encoding = rc.SpeciesEncodingMode.RankPool
model_config.cover_dropout    = 0.1
train_config.max_epochs = 300
train_config.patience   = 100
train_config.use_amp    = True
```

## Next steps

- [Encoding Modes](encoding-modes.md): what each encoder does with the species set
- [Training Models](training.md): the full configuration reference
- [Making Predictions](prediction.md): inference-time memory and batching
