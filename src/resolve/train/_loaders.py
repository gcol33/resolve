"""Data loading utilities for RESOLVE training.

Provides GPU-optimized data loading and rank-pool batch handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


class CUDAPrefetcher:
    """
    Prefetches batches to GPU asynchronously.

    Overlaps data transfer with computation by loading the next batch
    while the current batch is being processed. Uses CUDA streams for
    true async transfer.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self._next_batch = tuple(
                    t.to(self.device, non_blocking=True) for t in self._next_batch
                )

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        batch = self._next_batch
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


class GPUTensorLoader:
    """
    GPU-resident tensor loader for maximum training throughput.

    Stores all training tensors on GPU and performs fast GPU-based indexing
    for batch sampling. This eliminates the CPU->GPU transfer bottleneck
    that dominates training time with large datasets.

    With shuffle=True, generates random permutation indices on GPU each epoch.
    Batch extraction uses fast GPU tensor indexing (~0.1ms vs ~400ms for CPU).
    """

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        device: torch.device = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move all tensors to GPU
        self.tensors = tuple(t.to(device) for t in tensors)
        self.n_samples = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device

        # Calculate number of batches
        if drop_last:
            self.n_batches = self.n_samples // batch_size
        else:
            self.n_batches = (self.n_samples + batch_size - 1) // batch_size

        # Pre-allocate index tensor for shuffling (stays on GPU)
        self._indices = torch.arange(self.n_samples, device=device)

    def __iter__(self):
        # Shuffle indices on GPU (very fast)
        if self.shuffle:
            self._indices = torch.randperm(self.n_samples, device=self.device)
        else:
            self._indices = torch.arange(self.n_samples, device=self.device)

        self._batch_idx = 0
        return self

    def __next__(self):
        if self._batch_idx >= self.n_batches:
            raise StopIteration

        start = self._batch_idx * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]

        # Fast GPU indexing - all tensors already on GPU
        batch = tuple(t[indices] for t in self.tensors)

        self._batch_idx += 1
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def dataset(self):
        """Return a wrapper for compatibility with DataLoader interface (for len(loader.dataset))."""
        class _DatasetWrapper:
            def __init__(wrapper_self, n_samples):
                wrapper_self._n_samples = n_samples
            def __len__(wrapper_self):
                return wrapper_self._n_samples
        return _DatasetWrapper(self.n_samples)


@dataclass
class _RankPoolPreparedData:
    """Holds ragged rank-pool arrays + pre-scaled continuous/target tensors.

    Used instead of a flat tensor tuple for rank_pool mode, enabling per-batch
    padding in the DataLoader collate function.
    """

    continuous: torch.Tensor            # (N, d) pre-scaled float32
    target_tensors: list[torch.Tensor]  # one per target
    species_ids: list[np.ndarray]       # ragged per-plot (int32)
    genus_ids: list[np.ndarray] | None  # ragged, None if no taxonomy
    family_ids: list[np.ndarray] | None
    weights: list[np.ndarray]           # ragged per-plot (float32)
    has_cover: np.ndarray               # (N,) float32, 1.0 if cover info present
    has_taxonomy: bool
    n_samples: int
    categorical_ids: torch.Tensor | None = None  # (N, n_cat) int64, None if no categoricals


class RankPoolBatchDataset(Dataset):
    """PyTorch Dataset wrapping ragged rank-pool data for per-batch padding."""

    def __init__(self, data: _RankPoolPreparedData):
        self._data = data

    def __len__(self):
        return self._data.n_samples

    def __getitem__(self, idx):
        result = {
            "continuous": self._data.continuous[idx],
            "species_ids": self._data.species_ids[idx],
            "weights": self._data.weights[idx],
            "has_cover": self._data.has_cover[idx],
            "targets": [t[idx] for t in self._data.target_tensors],
        }
        if self._data.has_taxonomy:
            result["genus_ids"] = self._data.genus_ids[idx]
            result["family_ids"] = self._data.family_ids[idx]
        if self._data.categorical_ids is not None:
            result["categorical_ids"] = self._data.categorical_ids[idx]
        return result


def _rank_pool_collate_fn(samples: list[dict]) -> tuple:
    """Collate ragged rank-pool samples into a padded batch.

    Pads species_ids/genus_ids/family_ids/weights to the batch-level max
    species count (not the global max). Uses numpy for bulk padding, then
    converts to torch tensors once (avoids per-sample torch.from_numpy).

    Returns a tuple matching the existing batch layout:
      has_taxonomy=True:  (continuous, species_ids, genus_ids, family_ids, weights, mask, has_cover, *targets)
      has_taxonomy=False: (continuous, species_ids, weights, mask, has_cover, *targets)
    """
    n = len(samples)
    has_taxonomy = "genus_ids" in samples[0]
    has_categoricals = "categorical_ids" in samples[0]

    # Stack continuous (already a tensor from __getitem__)
    continuous = torch.stack([s["continuous"] for s in samples])

    # Find batch-level max species count
    sp_arrays = [s["species_ids"] for s in samples]
    lengths = [len(a) for a in sp_arrays]
    max_sp = max(max(lengths), 1)

    # Build padded numpy arrays (single allocation, numpy slice assignment)
    sp_np = np.zeros((n, max_sp), dtype=np.int64)
    w_np = np.zeros((n, max_sp), dtype=np.float32)
    mask_np = np.zeros((n, max_sp), dtype=np.bool_)

    if has_taxonomy:
        g_np = np.zeros((n, max_sp), dtype=np.int64)
        f_np = np.zeros((n, max_sp), dtype=np.int64)

    for i in range(n):
        k = lengths[i]
        if k > 0:
            sp_np[i, :k] = sp_arrays[i]
            w_np[i, :k] = samples[i]["weights"]
            mask_np[i, :k] = True
            if has_taxonomy:
                g_np[i, :k] = samples[i]["genus_ids"]
                f_np[i, :k] = samples[i]["family_ids"]

    # Single torch.from_numpy per array (zero-copy)
    sp_ids = torch.from_numpy(sp_np)
    w = torch.from_numpy(w_np)
    mask = torch.from_numpy(mask_np)

    # has_cover scalar per sample
    has_cover = torch.tensor([s["has_cover"] for s in samples], dtype=torch.float32)

    # Stack targets
    n_targets = len(samples[0]["targets"])
    targets = [torch.stack([s["targets"][t] for s in samples]) for t in range(n_targets)]

    # Stack categorical_ids if present (already tensors from __getitem__)
    cat_ids = None
    if has_categoricals:
        cat_ids = torch.stack([s["categorical_ids"] for s in samples])

    # Build batch tuple
    if has_taxonomy:
        g_ids = torch.from_numpy(g_np)
        f_ids = torch.from_numpy(f_np)
        batch = (continuous, sp_ids, g_ids, f_ids, w, mask, has_cover)
    else:
        batch = (continuous, sp_ids, w, mask, has_cover)

    if cat_ids is not None:
        batch = batch + (cat_ids,)

    return batch + tuple(targets)
