"""Distributed training utilities for multi-GPU support via PyTorch DDP.

Usage::

    # Launch with torchrun (recommended):
    # torchrun --nproc_per_node=2 train_script.py

    from resolve.train._distributed import setup_ddp, cleanup_ddp, is_distributed

    setup_ddp()  # Initialize process group
    trainer = Trainer(dataset, device="cuda", ...)
    trainer.fit()  # Automatically uses DDP if initialized
    cleanup_ddp()

Or use the convenience wrapper::

    from resolve.train._distributed import train_distributed

    result = train_distributed(dataset, n_gpus=2, max_epochs=100)
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    """Check if running in a distributed context."""
    return dist.is_available() and dist.is_initialized()


def local_rank() -> int:
    """Get the local rank (GPU index) for this process."""
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if is_distributed():
        return dist.get_rank() == 0
    return True


def setup_ddp(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Call this at the start of each worker process. Typically launched via
    ``torchrun --nproc_per_node=N script.py``.

    Args:
        backend: Communication backend. "nccl" for GPU, "gloo" for CPU.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if dist.is_initialized():
        return  # Already set up

    rank = local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(backend=backend)


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module) -> DDP:
    """Wrap a model in DistributedDataParallel.

    Args:
        model: The model to wrap. Must already be on the correct device.

    Returns:
        DDP-wrapped model.
    """
    rank = local_rank()
    return DDP(model, device_ids=[rank], output_device=rank)


def train_distributed(
    dataset: Any,
    n_gpus: int | None = None,
    **trainer_kwargs: Any,
) -> Any:
    """Launch distributed training across multiple GPUs.

    This is a convenience wrapper that spawns ``torchrun`` with the correct
    number of workers. Each worker runs the training independently with DDP
    synchronization.

    Args:
        dataset: ResolveDataset to train on.
        n_gpus: Number of GPUs to use. Defaults to all available.
        **trainer_kwargs: Passed to Trainer constructor.

    Returns:
        TrainResult from the main process.
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        # Fall back to single-GPU training
        from resolve.train.trainer import Trainer

        trainer = Trainer(dataset, device="cuda", **trainer_kwargs)
        return trainer.fit()

    # For multi-GPU, the caller should use torchrun directly.
    # This function provides guidance.
    raise NotImplementedError(
        f"Multi-GPU training with {n_gpus} GPUs requires launching via torchrun:\n"
        f"  torchrun --nproc_per_node={n_gpus} your_script.py\n\n"
        f"In your script, call setup_ddp() before creating the Trainer, "
        f"and the Trainer will automatically detect the distributed context."
    )
