"""
CUDA kernels for RESOLVE.

Provides GPU-accelerated hash embedding computation.
Uses Triton kernels (PyTorch 2.4+) or pure PyTorch scatter_add as fallback.
"""

from .hash_ops import (
    hash_aggregate,
    hash_batch_csr,
    is_triton_available,
    hash_aggregate_pure_torch,
    hash_batch_csr_pure_torch,
)

__all__ = [
    "hash_aggregate",
    "hash_batch_csr",
    "is_triton_available",
    "hash_aggregate_pure_torch",
    "hash_batch_csr_pure_torch",
]


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available (Triton or pure PyTorch on GPU)."""
    import torch
    return torch.cuda.is_available()
