# Predictor

Everything in this page lives in `resolve_core`.

```python
import resolve_core as rc
```

---

## Predictor

Inference over a trained checkpoint.

```python
predictor = rc.Predictor.load("model.pt", device="cpu", vram_fraction=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Checkpoint written by `Trainer.save` |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` |
| `vram_fraction` | `float` | `1.0` | Cap on the CUDA caching allocator, applied before the model uploads |

`device` defaults to CPU. Inference on a ~5M-parameter MLP over 300k plots takes
roughly 12 s on CPU against roughly 1 s on a 16 GiB GPU, and the GPU path can
run out of memory, so the default leans towards the option that always works.
Pass `device="cuda"` when the GPU is idle and the test set is known to fit.

`device` is also what the checkpoint is deserialized onto, not just where the
weights end up, so a GPU-trained checkpoint loads on a machine with no CUDA
device at all.

A predictor can also be assembled from parts:

```python
predictor = rc.Predictor(model, scalers, device="cpu")
```

Loading retries on a transient filesystem error, with exponential backoff,
tunable through `RESOLVE_IO_RETRY_ATTEMPTS` and `RESOLVE_IO_RETRY_BACKOFF_MS`.

### Predicting on a dataset

```python
predictions = predictor.predict_dataset(dataset, return_latent=False, batch_size=4096)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `ResolveDataset` | required | Must be built against this checkpoint's vocabularies |
| `return_latent` | `bool` | `False` | Also return the shared encoder's output |
| `batch_size` | `int` | `4096` | Chunk size along dim 0; `-1` runs one forward over the whole dataset |

Returns a `ResolvePredictions`. Chunked output is equal to the one-shot path:
each chunk goes through the same forward call, and only the input slicing
differs. Results are moved to CPU as they are produced, so peak device memory
stays bounded regardless of plot count. `0` and negative values other than `-1`
are rejected.

The dataset has to be built in this checkpoint's integer-code namespace, or
every non-hash encoder reads the wrong embedding rows:

```python
vocabs = predictor.external_vocabs
config = predictor.dataset_config

dataset = rc.ResolveDataset.from_csv_with_vocabs(
    "new_plots.csv", "new_species.csv", roles, targets, vocabs, config,
)
predictions = predictor.predict_dataset(dataset)
```

`predict_dataset` rejects a dataset built any other way, comparing the full
ordered vocabularies when the checkpoint carries them and the sizes alone when
it does not. Hash encoding is exempt for species, because its representation is
derived from the name, and its genus and family slots are still checked.

### Predicting on tensors

```
predict(coordinates, covariates, hash_embedding, species_ids, species_vector,
        genus_ids, family_ids, unknown_fraction, unknown_count,
        pool_genus_ids=None, pool_family_ids=None, pool_weights=None,
        pool_mask=None, pool_has_cover=None, categorical_ids=None,
        return_latent=False)
```

Returns a `ResolvePredictions`. The first nine arguments are positional; pass
`None` for anything the encoding does not use.

### Embeddings

| Method | Returns |
|--------|---------|
| `get_species_embeddings()` | `(n_species_vocab, species_embed_dim)`, or `None` for hash and sparse encoders |
| `get_genus_embeddings()` | `(n_genera, genus_emb_dim)`, or `None` without taxonomy |
| `get_family_embeddings()` | `(n_families, family_emb_dim)`, or `None` without taxonomy |
| `get_embeddings(coordinates, covariates, hash_embedding, genus_ids, family_ids)` | Latent vectors for those inputs |

Tables come back detached on CPU.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `ResolveModel` | The loaded model |
| `scalers` | `Scalers` | Feature and per-target scaling from training |
| `device` | `str` | `"cpu"` or `"cuda"` |
| `schema` | `ResolveSchema` | The schema the model was built from |
| `external_vocabs` | `ExternalVocabs` | Every training vocabulary, including the categorical maps |
| `dataset_config` | `DatasetConfig` | The loading-side config this checkpoint implies |
| `categorical_vocab` | `CategoricalVocab` | String-to-code maps per categorical column |
| `species_vocab` | `list[str]` | Ordered species vocabulary; `[0]` is `"<UNK>"` |
| `genus_vocab` | `list[str]` | Ordered genus vocabulary |
| `family_vocab` | `list[str]` | Ordered family vocabulary |

### Optimizing for inference

```python
predictor.optimize_for_inference()
```

Folds batch-normalization parameters into the preceding linear layers. Call it
after loading and before a large prediction run.

---

## ResolvePredictions

| Attribute | Type | Description |
|-----------|------|-------------|
| `predictions` | `dict[str, Tensor]` | One entry per target |
| `targets` | `dict[str, Tensor]` | Ground truth, when the dataset carried it |
| `plot_ids` | `list[str]` | Plot identifiers, in row order |
| `latent` | `Tensor` or `None` | Shape `(n_plots, latent_dim)` when `return_latent=True` |

Regression targets come back on the original scale; a `Log1p` target is inverted
before it is returned. Classification targets come back as int64 class codes,
indexing into `schema.targets[i].class_names`.

Writing them out:

```python
import pandas as pd

frame = pd.DataFrame({"plot_id": list(predictions.plot_ids)})
for target, values in predictions.predictions.items():
    frame[target] = values.numpy()
frame.to_csv("predictions.csv", index=False)
```

---

## Scalers

Feature and target scaling fitted on the training fold.

| Attribute | Type | Description |
|-----------|------|-------------|
| `continuous_mean` | `Tensor` or `None` | Per continuous input column |
| `continuous_scale` | `Tensor` or `None` | Per continuous input column |
| `target_scalers` | `dict[str, (float, float)]` | `(mean, scale)` per regression target |

---

## Process and memory helpers

| Function | Description |
|----------|-------------|
| `rc.set_vram_fraction(fraction, device_index=0)` | Cap the CUDA caching allocator directly |
| `rc.configure_cuda_allocator(force=False)` | Set a platform-aware `PYTORCH_CUDA_ALLOC_CONF`; runs at import, before torch loads |
| `rc.set_thread_pools(intraop, interop)` | Pin libtorch's host thread pools |
| `rc.install_crash_handler(shutdown_exit_code)` | Windows only; turn a native fault into a fast, deterministic exit rather than a JIT-debugger hang |
| `rc.retry_io(fn, what=...)` | The retry wrapper applied to `Trainer.save`, `Trainer.load`, and `Predictor.load` |

The crash handler is installed and an `atexit` hook registered when
`resolve_core` is imported, so a headless worker that faults fails fast instead
of hanging.

---

## Command line

```bash
resolve predict --model model.pt \
                --header new_plots.csv --species new_species.csv \
                --plot-id plot_id --species-id species --abundance cover \
                --output predictions.csv \
                --predict-batch-size 4096
```

The CLI rebuilds its `DatasetConfig` and its vocabularies from the checkpoint,
so its codes mean what they meant at training time. See
[Making Predictions](../tutorials/prediction.md) for the output layout.

---

## See also

- [Making Predictions](../tutorials/prediction.md)
- [Understanding Embeddings](../tutorials/embeddings.md)
- [Trainer](trainer.md)
