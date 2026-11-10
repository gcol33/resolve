# Large data and GPUs

A survey of a few hundred plots trains in seconds on a laptop with the
default settings. An archive of a million plots takes the settings this
article covers: the learning rate and its schedule, the batch size, the
size of the model, the encoding settings that bound memory, and the GPU
settings, from mixed precision to what happens when the GPU runs out of
memory.

Each section says what a setting does in the RESOLVE engine. The
examples run on the small survey that ships with the R package, so they
show the calls and what they return; the speed and memory effects only
appear on large data. The GPU settings need the CUDA build of the
RESOLVE engine and are shown without being run.

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The RESOLVE engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

[Installing the RESOLVE
engine](https://gillescolling.com/resolve/articles/installation.md)
covers the GPU build.

## The survey

The survey has 150 plots in the Alps, each with its elevation and
habitat, and the species recorded on each plot with its genus, family
and cover. The *roles* name the columns of the two tables, and each
*target* is a column to impute.

``` r

plots <- read.csv(system.file("extdata", "plots.csv", package = "resolveR"))
species <- read.csv(system.file("extdata", "species.csv", package = "resolveR"))

roles <- list(plot_id = "plot_id", species_id = "species",
              abundance = "cover", genus = "genus", family = "family")
targets <- list(
  elevation = list(column = "elevation", task = "regression"),
  habitat = list(column = "habitat", task = "classification")
)
dataset <- resolve.dataset.frame(plots, species, roles = roles,
                                 targets = targets)
```

``` python
import pandas as pd
import resolve_core as rc

data = "https://raw.githubusercontent.com/gcol33/resolve/master/r/inst/extdata/"
plots = pd.read_csv(data + "plots.csv")
species = pd.read_csv(data + "species.csv")

roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"

targets = [rc.TargetSpec.regression("elevation"),
           rc.TargetSpec.classification("habitat", 4)]
dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets)
```

[Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) introduces
the training settings and how to pass them.

## Learning rate

The *learning rate* scales each step the optimizer, AdamW, takes. Its
default is 0.001, with a weight decay of 0.0001; both are training
settings.

``` r

fit <- resolve.train.dataset(dataset, lr = 1e-3, maxEpochs = 100L,
                             trainConfig = list(weight_decay = 1e-4),
                             verbose = FALSE)
fit$trainer$get_config()[c("lr", "weight_decay")]
#> $lr
#> [1] 0.001
#> 
#> $weight_decay
#> [1] 1e-04
```

``` python
cfg = rc.TrainConfig()
cfg.lr = 1e-3
cfg.weight_decay = 1e-4
cfg.max_epochs = 100
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
trainer = rc.Trainer(model, cfg)
trainer.prepare_data(dataset, test_size=0.2, seed=42)
result = trainer.fit()
round(trainer.config.lr, 6), round(trainer.config.weight_decay, 6)
#> (0.001, 0.0001)
```

The default is a reasonable start for the hash, embed, sparse and
rank-pool encodings. Attention layers react more strongly to the rate:
with the transformer encoding and at least one attention layer, a lower
rate such as 0.0003 is a common start.

The loss curve tells you which way to move it:

- **The loss jumps or turns to NaN within a few epochs**: lower the rate
  by a factor of 3 to 10.
- **The loss falls slowly and levels off high**: raise it by a factor of
  3 to 10.
- **The loss swings up and down without a downward trend**: lower it by
  a factor of 2 to 3.

### Schedules

A *schedule* changes the rate during the run. By default none runs and
the rate stays at `lr`. Four schedules are available:

- **`step`** multiplies the rate by `lr_gamma` (default 0.1) every
  `lr_step_size` epochs (default 100).
- **`cosine`** lowers the rate along a cosine from `lr` to `lr_min`
  (default 0.000001), reached at the last epoch of `max_epochs`.
- **`linear`** lowers the rate at every optimizer step to `lr_min` at
  the end of `max_epochs`.
- **`plateau`** follows the held-out fold and lowers the rate when the
  held-out value stops improving.

``` r

step <- resolve.train.dataset(dataset, maxEpochs = 100L, fixedEpochs = 100L,
                              verbose = FALSE,
                              lrScheduler = "step", lrStepSize = 30L,
                              lrGamma = 0.1)
signif(step$result$lr[c(1, 30, 31, 61, 91)], 3)
#> [1] 1e-03 1e-03 1e-04 1e-05 1e-06

cosine <- resolve.train.dataset(dataset, maxEpochs = 100L, fixedEpochs = 100L,
                                verbose = FALSE,
                                lrScheduler = "cosine", lrMin = 1e-6)
signif(cosine$result$lr[c(1, 25, 50, 75, 100)], 3)
#> [1] 0.001000 0.000862 0.000508 0.000150 0.000001
```

``` python
cfg = rc.TrainConfig()
cfg.max_epochs = 100
cfg.fixed_epochs = 100
cfg.lr_scheduler = rc.LRSchedulerType.StepLR
cfg.lr_step_size = 30
cfg.lr_gamma = 0.1
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
step = rc.Trainer(model, cfg)
step.prepare_data(dataset, test_size=0.2, seed=42)
step_rates = step.fit().lr_history
[float(f"{step_rates[epoch - 1]:.3g}") for epoch in [1, 30, 31, 61, 91]]
#> [0.001, 0.001, 0.0001, 1e-05, 1e-06]

cfg = rc.TrainConfig()
cfg.max_epochs = 100
cfg.fixed_epochs = 100
cfg.lr_scheduler = rc.LRSchedulerType.CosineAnnealing
cfg.lr_min = 1e-6
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
cosine = rc.Trainer(model, cfg)
cosine.prepare_data(dataset, test_size=0.2, seed=42)
cosine_rates = cosine.fit().lr_history
[float(f"{cosine_rates[epoch - 1]:.3g}") for epoch in [1, 25, 50, 75, 100]]
#> [0.001, 0.000862, 0.000508, 0.00015, 1e-06]
```

Both runs train a fixed 100 epochs, so the rate history, the rate each
epoch trained at, covers the whole schedule. The step schedule drops the
rate after every 30 epochs; the cosine schedule ends at `lr_min` on the
last epoch. Both are laid out over `max_epochs`, so a run that early
stopping ends sooner never reaches their end. The plateau schedule and a
linear warmup at the start of the run are covered in [Training and
reading the
fit](https://gillescolling.com/resolve/articles/training.md).

## Batch size

The *batch size* is the number of plots per optimizer step. A smaller
batch takes more steps per epoch with noisier gradients, which acts as a
regularizer on a small dataset; a larger batch keeps a GPU busy. The
default is 4096, larger than the whole survey, so each epoch here is one
step.

``` r

small_batch <- resolve.train.dataset(dataset, batchSize = 32L,
                                     maxEpochs = 100L, verbose = FALSE)
small_batch$result$effective_batch_size
#> [1] 32
```

``` python
cfg = rc.TrainConfig()
cfg.batch_size = 32
cfg.max_epochs = 100
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
small_batch = rc.Trainer(model, cfg)
small_batch.prepare_data(dataset, test_size=0.2, seed=42)
small_batch.fit().effective_batch_size
#> 32
```

Starting points to try, and to measure on your own data:

| Plots             | Batch size    |
|-------------------|---------------|
| under 10,000      | 512 to 2048   |
| 10,000 to 100,000 | 4096          |
| over 100,000      | 8192 to 16384 |

When you raise the batch size, the learning rate usually has to rise
with it. Memory grows linearly with the batch size under every encoding.
The transformer encoding also holds an attention matrix whose size grows
with the square of the number of species per plot, so there the cap on
species per plot matters as much as the batch size ([Pool weighting and
capping](#pool-weighting-and-capping)).

## Model size

`hidden_dims` sets the depth and the width of the shared encoder, by
default six layers of 2048, 1024, 512, 256, 128 and 64 units. The last
width is the width of the *latent representation*, the vector each plot
is encoded into.

``` r

small <- resolve.train.dataset(dataset, hiddenDims = c(256L, 128L, 64L),
                               maxEpochs = 100L, verbose = FALSE,
                               modelConfig = list(dropout = 0.3,
                                                  head_dropout = 0,
                                                  use_residual = TRUE))
sapply(small$trainer$compute_diagnostics()$layers, `[[`, "n_neurons")
#> [1] 256 128  64
```

``` python
model_config = rc.ModelConfig()
model_config.hidden_dims = [256, 128, 64]
model_config.dropout = 0.3
model_config.head_dropout = 0.0
model_config.use_residual = True
cfg = rc.TrainConfig()
cfg.max_epochs = 100
model_config = rc.model_config_for_dataset(dataset, model_config)
small = rc.Trainer(rc.ResolveModel(dataset.schema, model_config), cfg)
small.prepare_data(dataset, test_size=0.2, seed=42)
small_result = small.fit()
[layer.n_neurons for layer in small.compute_diagnostics().layers]
#> [256, 128, 64]
```

- **Small datasets (under 5,000 plots)**: the deep default can overfit
  before it converges; three layers of 256, 128 and 64 units are a
  better start.
- **Mid-size datasets**: 512, 256 and 128 units, or the default.
- **Large datasets**: the default, made wider if the training loss
  levels off above the held-out loss.
- **With the transformer encoding**: the attention layers already carry
  capacity, so a shallower stack above them is usually enough.

`dropout` (default 0.3) applies between the encoder’s layers and
`head_dropout` (default 0) inside the heads, one per target.
`use_residual` adds residual connections between the encoder’s layers,
which helps a deeper stack train.

## Early stopping

Early stopping watches the loss on the held-out fold and ends the run
when it has not improved for `patience` epochs (default 50), within the
limit of `max_epochs` (default 500). The weights of the best epoch are
restored, so a larger patience never returns a worse model, it only
costs epochs. Starting points:

| Use                                          | Patience  |
|----------------------------------------------|-----------|
| Quick experiments                            | 10 to 20  |
| Standard training                            | 30 to 50  |
| Final runs                                   | 100       |
| Attention layers, which converge more slowly | 50 to 100 |

An epoch counts against the patience only once the loss has reached the
phase it will be in at the last epoch; [Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) explains
the phases.

## Encoding settings

The encoding settings are dataset settings, passed when the data is
loaded ([Encoding the species
set](https://gillescolling.com/resolve/articles/encoding.md)). Three of
them decide how much memory the species take.

### Hash width

The hash encoding maps each species to one of `hash_dim` buckets
(default 32). More buckets mean fewer species share one, and a wider
input layer. 64 and 128 are the usual next steps on a large species
list. The selection decides which species reach the hash at all: the
default keeps the three most abundant species of each plot
(`selection = "top"`, `top_k = 3`); `"all"` keeps every species.

``` r

hashed <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "hash", hash_dim = 64L, selection = "all")
)
dim(hashed$hash_embedding())
#> [1] 150  64
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.Hash
data_config.hash_dim = 64
data_config.selection = rc.SelectionMode.All
hashed = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                       config=data_config)
tuple(hashed.hash_embedding.shape)
#> (150, 64)
```

The model takes the hash width from the dataset (in Python through
`rc.model_config_for_dataset()`), so it needs no setting of its own.
Measure the effect of a wider hash on your own data.

### Pool weighting and capping

The rank-pool and transformer encodings read every species of a plot,
padded to a common width. `pool_weighting` sets the weight each species
carries in the pooling; the default, `log1p`, compresses the range of
the abundances while keeping their order, so the dominant species weigh
most without the rest being ignored. The other weightings are `binary`,
`abundance`, `norm` and `rank`.

`pool_species_cap` bounds the padded width, the largest term in the
memory these encodings use:

- **0** (default): no cap; the plot with the most species sets the
  width.
- **A negative value**: a percentile cap, read as 100 minus the value:
  -1 caps at the 99th percentile of species per plot, -5 at the 95th.
- **A positive value**: a fixed cap.

A plot longer than the cap keeps its most abundant species, so the kept
set does not depend on the order of its records. Here a cap of 4 cuts
the survey’s plots, which hold up to 8 species, to their 4 most
abundant.

``` r

uncapped <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "rank_pool")
)
capped <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "rank_pool", pool_weighting = "log1p",
                pool_species_cap = 4L)
)
c(uncapped = ncol(uncapped$pool_weights()), capped = ncol(capped$pool_weights()))
#> uncapped   capped 
#>        8        4
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.RankPool
uncapped = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                         config=data_config)
data_config.pool_weighting = rc.PoolWeighting.Log1p
data_config.pool_species_cap = 4
capped = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                       config=data_config)
{"uncapped": uncapped.pool_weights.shape[1], "capped": capped.pool_weights.shape[1]}
#> {'uncapped': 8, 'capped': 4}
```

A cap drops species, so it changes what the model sees. Leave it at 0
for a result you publish unless the uncapped run does not fit in memory,
and report the cap if it does not.

### Transformer settings

The transformer encoding has its own model settings: the width of each
species token (`d_model`), the attention heads (`n_heads`), the
attention layers (`n_attention_layers`), the width of the feed-forward
layer inside each block (`transformer_ff_dim`, normally 2 to 4 times
`d_model`) and how the tokens are pooled into one vector per plot
(`transformer_pooling`). Pooling with `cls` reads a separate token that
sees the species only through attention, so it needs at least one
attention layer.

``` r

tokens <- resolve.dataset.frame(plots, species, roles = roles,
                                targets = targets,
                                config = list(species_encoding = "transformer"))
attn <- resolve.train.dataset(tokens, dModel = 64L, nHeads = 4L,
                              nAttentionLayers = 2L, transformerFfDim = 128L,
                              transformerPooling = "attention", lr = 3e-4,
                              maxEpochs = 50L, verbose = FALSE)
attn$result$best_epoch >= 0
#> [1] TRUE
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.Transformer
tokens = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                       config=data_config)
model_config = rc.ModelConfig()
model_config.d_model = 64
model_config.n_heads = 4
model_config.n_attention_layers = 2
model_config.transformer_ff_dim = 128
model_config.transformer_pooling = rc.TransformerPooling.Attention
cfg = rc.TrainConfig()
cfg.lr = 3e-4
cfg.max_epochs = 50
model_config = rc.model_config_for_dataset(tokens, model_config)
attn = rc.Trainer(rc.ResolveModel(tokens.schema, model_config), cfg)
attn.prepare_data(tokens, test_size=0.2, seed=42)
attn.fit().best_epoch >= 0
#> True
```

## GPU settings

Training on a GPU needs the CUDA build of the RESOLVE engine:
`resolve.install_backend(variant = "cuda")` in R, and the CUDA wheel in
Python ([Installing the RESOLVE
engine](https://gillescolling.com/resolve/articles/installation.md)).
The code in this section is shown without being run, since this site is
built on a machine without a GPU.

### Device and mixed precision

The device is a training setting: `"cuda"` for the current GPU,
`"cuda:N"` for GPU number N. A missing CUDA device is an error.

*Mixed precision* (AMP, automatic mixed precision) runs most operations
in 16-bit floating point and keeps the sums that need the range in 32
bits. It is off by default. Its loss scaling starts at `amp_init_scale`
(default 65536), grows by `amp_growth_factor` (2) after
`amp_growth_interval` (2000) steps without overflow and shrinks by
`amp_backoff_factor` (0.5) after one.

``` r

fit <- resolve.train.dataset(
  dataset, device = "cuda",
  trainConfig = list(use_amp = TRUE, amp_init_scale = 65536,
                     amp_growth_factor = 2, amp_backoff_factor = 0.5,
                     amp_growth_interval = 2000L)
)
```

``` python
cfg = rc.TrainConfig()
cfg.device = "cuda"
cfg.use_amp = True
cfg.amp_init_scale = 65536.0
cfg.amp_growth_factor = 2.0
cfg.amp_backoff_factor = 0.5
cfg.amp_growth_interval = 2000
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
trainer = rc.Trainer(model, cfg)
trainer.prepare_data(dataset, test_size=0.2, seed=42)
result = trainer.fit()
```

The normalization layers run in 32 bits inside the mixed-precision
region, so batch-normalization statistics cannot overflow the 16-bit
range. If the loss turns to NaN a few steps into a mixed-precision run
with attention layers, turn mixed precision off before changing anything
else.

### cuDNN and TF32

Two settings trade exact reproducibility for speed on the GPU.
`cudnn_benchmark` (default true) lets cuDNN, NVIDIA’s library of network
primitives, time several algorithms on the first batches and keep the
fastest for inputs of a fixed shape. `allow_tf32` (default true) runs
32-bit matrix products in the TF32 format on Ampere and later GPUs. Both
are applied once when the run starts and hold for the whole run.

``` r

fit <- resolve.train.dataset(
  dataset, device = "cuda",
  trainConfig = list(cudnn_benchmark = FALSE, allow_tf32 = FALSE)
)
```

``` python
cfg = rc.TrainConfig()
cfg.device = "cuda"
cfg.cudnn_benchmark = False
cfg.allow_tf32 = False
```

### Data on the GPU

On a GPU, the trainer copies the training and held-out folds to the
device once and takes each batch there, so no batch crosses from main
memory during the epochs. This has no setting.

With the hash encoding, the dataset setting `use_cuda_hash` computes
each batch’s hash on the GPU instead of when the data is loaded. The
trainer then computes the next batch’s hash on a separate CUDA stream
while the current batch runs through the model, so the two overlap. This
also has no setting of its own.

``` r

hashed <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "hash", use_cuda_hash = TRUE)
)
fit <- resolve.train.dataset(hashed, device = "cuda")
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.Hash
data_config.use_cuda_hash = True
hashed = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                       config=data_config)
cfg = rc.TrainConfig()
cfg.device = "cuda"
model = rc.ResolveModel(hashed.schema, rc.model_config_for_dataset(hashed))
trainer = rc.Trainer(model, cfg)
trainer.prepare_data(hashed, test_size=0.2, seed=42)
result = trainer.fit()
```

### Limiting GPU memory

PyTorch keeps the GPU memory it has used in a cache for reuse.
`vram_fraction` caps that cache at a share of the GPU’s memory. The
default, 1, sets no cap, so a training job alone on a GPU uses all of
it. Set a lower value when the GPU also drives a desktop: on Windows,
the graphics driver moves allocations beyond the GPU’s memory into main
memory, which can freeze the desktop under load. The cap only limits
memory; a desktop that turns slow without freezing is short of GPU
compute, which it does not address.

The same cap applies when a saved model is loaded onto a GPU, and it can
be set on its own, before any model is built.

``` r

fit <- resolve.train.dataset(dataset, device = "cuda",
                             trainConfig = list(vram_fraction = 0.8))
predictor <- resolve.load("model.pt", device = "cuda", vram_fraction = 0.8)

resolve.set_vram_fraction(0.8)
```

``` python
cfg = rc.TrainConfig()
cfg.device = "cuda"
cfg.vram_fraction = 0.8
predictor = rc.Predictor.load("model.pt", device="cuda", vram_fraction=0.8)

rc.set_vram_fraction(0.8)
```

On a machine without a GPU, setting the cap does nothing. The command
line takes `--vram-fraction` for `resolve train` and `resolve predict`
([The command
line](https://gillescolling.com/resolve/articles/command-line.md)).

### Allocator configuration

PyTorch reads the settings of its GPU memory cache from the environment
variable `PYTORCH_CUDA_ALLOC_CONF` when it first allocates on a GPU.
Importing resolve_core sets it to the RESOLVE engine’s default, unless
it is already set; in R,
[`resolve_configure_cuda_allocator()`](https://gillescolling.com/resolve/reference/resolve_configure_cuda_allocator.md)
sets it. The default depends on the platform:

| Platform | Setting |
|----|----|
| Linux, macOS | `expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256` |
| Windows | `garbage_collection_threshold:0.8,max_split_size_mb:256` |

`expandable_segments` lets the cache grow and release its memory in
place, which keeps it from fragmenting; PyTorch does not support it on
Windows, so the RESOLVE engine leaves it out there.
`garbage_collection_threshold:0.8` makes the cache free unused blocks
when it holds more than 80 % of its cap, and `max_split_size_mb:256`
keeps it from splitting large free blocks into small pieces. On Windows
these two reduce fragmentation less than `expandable_segments` does on
Linux, so a job close to the memory cap there may need a smaller batch.

The function returns the setting in effect; `force` overwrites a value
set elsewhere. It acts only when it runs before the first GPU allocation
of the process, so in R call it before the first training run on a GPU.

``` r

resolve_configure_cuda_allocator()
resolve_configure_cuda_allocator(force = TRUE)
```

``` python
rc.default_cuda_alloc_conf()
rc.configure_cuda_allocator()
rc.configure_cuda_allocator(force=True)
```

### Running out of GPU memory

When the GPU runs out of memory during training, the trainer frees the
optimizer state, the mixed-precision state and the cached memory, halves
the batch size and starts the run again from the first epoch with the
starting weights. It halves at most down to `batch_size_floor` (default
1024); when halving would go below it, the original out-of-memory error
is raised with the batch size asked for, the size reached and the floor.
The *effective batch size* in the result is the size the run trained at;
the checkpoint records it as the run’s batch size, with the floor.

``` r

fit <- resolve.train.dataset(dataset, device = "cuda", batchSize = 16384L,
                             trainConfig = list(batch_size_floor = 1024L))
fit$result$effective_batch_size
```

``` python
cfg = rc.TrainConfig()
cfg.device = "cuda"
cfg.batch_size = 16384
cfg.batch_size_floor = 1024
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
trainer = rc.Trainer(model, cfg)
trainer.prepare_data(dataset, test_size=0.2, seed=42)
trainer.fit().effective_batch_size
```

The self-supervised pretraining runs the same retry, down to its own
`batch_size_floor`, and caps the cache at its own `vram_fraction`
([Latent and species
embeddings](https://gillescolling.com/resolve/articles/embeddings.md)).
On Linux the allocator setting delays running out of memory; on Windows
the halving recovers when fragmentation leaves no other way. On the
command line:
`resolve train --batch-size 16384 --batch-size-floor 1024`.

## Starting points

The three recipes below are places to start from, to be adjusted against
the loss curve and the held-out metrics.

A quick experiment: the hash encoding over every species, a small
encoder and a short run.

``` r

quick_data <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "hash", hash_dim = 64L, selection = "all")
)
quick <- resolve.train.dataset(quick_data, hiddenDims = c(256L, 128L, 64L),
                               maxEpochs = 50L, patience = 10L,
                               batchSize = 4096L, verbose = FALSE)
round(quick$result$final_metrics$habitat$accuracy, 3)
#> [1] 0.9
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.Hash
data_config.hash_dim = 64
data_config.selection = rc.SelectionMode.All
quick_data = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                           config=data_config)
model_config = rc.ModelConfig()
model_config.hidden_dims = [256, 128, 64]
cfg = rc.TrainConfig()
cfg.max_epochs = 50
cfg.patience = 10
cfg.batch_size = 4096
model_config = rc.model_config_for_dataset(quick_data, model_config)
quick = rc.Trainer(rc.ResolveModel(quick_data.schema, model_config), cfg)
quick.prepare_data(quick_data, test_size=0.2, seed=42)
round(quick.fit().final_metrics["habitat"]["accuracy"], 3)
#> 0.8
```

Standard training: a mid-size encoder and a longer run.

``` r

standard <- resolve.train.dataset(quick_data, hiddenDims = c(512L, 256L, 128L),
                                  maxEpochs = 200L, patience = 30L,
                                  batchSize = 4096L, verbose = FALSE)
round(standard$result$final_metrics$habitat$accuracy, 3)
#> [1] 0.8
```

``` python
model_config = rc.ModelConfig()
model_config.hidden_dims = [512, 256, 128]
cfg = rc.TrainConfig()
cfg.max_epochs = 200
cfg.patience = 30
cfg.batch_size = 4096
model_config = rc.model_config_for_dataset(quick_data, model_config)
standard = rc.Trainer(rc.ResolveModel(quick_data.schema, model_config), cfg)
standard.prepare_data(quick_data, test_size=0.2, seed=42)
round(standard.fit().final_metrics["habitat"]["accuracy"], 3)
#> 0.833
```

A longer run with the rank-pool encoding on a GPU: log-weighted pooling,
cover dropout (`cover_dropout`, which during training replaces the cover
values of a random share of the plots in each batch with presence
alone), a patient early stop and mixed precision.

``` r

pooled_data <- resolve.dataset.frame(
  plots, species, roles = roles, targets = targets,
  config = list(species_encoding = "rank_pool", pool_weighting = "log1p")
)
pooled <- resolve.train.dataset(pooled_data, coverDropout = 0.1,
                                maxEpochs = 300L, patience = 100L,
                                device = "cuda",
                                trainConfig = list(use_amp = TRUE))
```

``` python
data_config = rc.DatasetConfig()
data_config.species_encoding = rc.SpeciesEncodingMode.RankPool
data_config.pool_weighting = rc.PoolWeighting.Log1p
pooled_data = rc.ResolveDataset.from_pandas(plots, species, roles, targets,
                                            config=data_config)
model_config = rc.ModelConfig()
model_config.cover_dropout = 0.1
cfg = rc.TrainConfig()
cfg.max_epochs = 300
cfg.patience = 100
cfg.device = "cuda"
cfg.use_amp = True
model_config = rc.model_config_for_dataset(pooled_data, model_config)
pooled = rc.Trainer(rc.ResolveModel(pooled_data.schema, model_config), cfg)
pooled.prepare_data(pooled_data, test_size=0.2, seed=42)
result = pooled.fit()
```

## Next steps

- [Encoding the species
  set](https://gillescolling.com/resolve/articles/encoding.md): what
  each encoding does with the species of a plot.
- [Training and reading the
  fit](https://gillescolling.com/resolve/articles/training.md): every
  training setting and how to read the result.
- [Imputing new
  plots](https://gillescolling.com/resolve/articles/prediction.md):
  imputation in chunks and on the GPU.
