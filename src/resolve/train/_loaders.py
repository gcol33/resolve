"""Data loading utilities for RESOLVE training.

Provides GPU-optimized data loading for pre-padded tensor batches.
"""

from __future__ import annotations

import torch


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


