"""
CUDA kernels for RESOLVE.

Provides GPU-accelerated hash embedding computation and fused linear+CE loss.
Uses Triton kernels (PyTorch 2.4+) or pure PyTorch fallback.
"""

from .hash_ops import (
    hash_aggregate,
    hash_batch_csr,
    is_triton_available,
    hash_aggregate_pure_torch,
    hash_batch_csr_pure_torch,
)
from .fused_linear_ce import fused_linear_cross_entropy

__all__ = [
    "hash_aggregate",
    "hash_batch_csr",
    "is_triton_available",
    "hash_aggregate_pure_torch",
    "hash_batch_csr_pure_torch",
    "fused_linear_cross_entropy",
]


def is_cuda_available() -> bool:
    """Check if CUDA acceleration is available (Triton or pure PyTorch on GPU)."""
    import torch
    return torch.cuda.is_available()
