"""
CUDA hash embedding kernels using torch.library + Triton.

Modern PyTorch 2.4+ approach - no C++ compilation needed.
"""

import torch
import torch.library

# Check if triton is available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _murmurhash3_x64_128_fmix64(h: int) -> int:
    """MurmurHash3 64-bit finalizer (Python version for reference)."""
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h


if TRITON_AVAILABLE:
    @triton.jit
    def _murmur_hash(key):
        """MurmurHash3 finalizer in Triton."""
        h = key.to(tl.uint64)
        h ^= h >> 33
        h *= 0xFF51AFD7ED558CCD
        h ^= h >> 33
        h *= 0xC4CEB9FE1A85EC53
        h ^= h >> 33
        return h.to(tl.int64)

    @triton.jit
    def hash_aggregate_kernel(
        plot_indices_ptr,
        species_ids_ptr,
        weights_ptr,
        output_ptr,
        n_elements,
        hash_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Hash and aggregate species into embeddings."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        plot_idx = tl.load(plot_indices_ptr + offsets, mask=mask, other=0)
        species_id = tl.load(species_ids_ptr + offsets, mask=mask, other=0)
        weight = tl.load(weights_ptr + offsets, mask=mask, other=0.0)

        # Hash species ID
        h = _murmur_hash(species_id)
        # Get positive modulo
        # Mask to unsigned range before modulo to avoid negative result if h == INT64_MIN
        hash_idx = (tl.abs(h).to(tl.uint64) % hash_dim).to(tl.int64)
        # Sign from hash
        sign = tl.where(h >= 0, 1.0, -1.0)
        contribution = sign * weight

        # Atomic add to output
        output_offset = plot_idx * hash_dim + hash_idx
        tl.atomic_add(output_ptr + output_offset, contribution, mask=mask)


def hash_aggregate_triton(
    plot_indices: torch.Tensor,
    species_ids: torch.Tensor,
    weights: torch.Tensor,
    n_plots: int,
    hash_dim: int,
) -> torch.Tensor:
    """
    Compute hash embedding using Triton kernel.

    Args:
        plot_indices: (n,) int64 - plot index for each species record
        species_ids: (n,) int64 - hashed species ID for each record
        weights: (n,) float32 - abundance weight for each record
        n_plots: Total number of plots
        hash_dim: Dimension of hash embedding

    Returns:
        (n_plots, hash_dim) float32 tensor of hash embeddings
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available. Install with: pip install triton")

    n = plot_indices.shape[0]
    output = torch.zeros(n_plots, hash_dim, dtype=torch.float32, device=plot_indices.device)

    BLOCK_SIZE = 256
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    hash_aggregate_kernel[grid](
        plot_indices,
        species_ids,
        weights,
        output,
        n,
        hash_dim,
        BLOCK_SIZE,
    )

    return output


def _unsigned_rshift_33(x: torch.Tensor) -> torch.Tensor:
    """Simulate unsigned right shift by 33 on int64 tensor."""
    # For unsigned right shift by 33 bits on 64-bit:
    # - Positive numbers: just shift right
    # - Negative numbers: clear sign bit, shift, then set high bits
    # Since 33 > 32, the result fits in 31 bits (positive)
    # Mask: after shifting 64-bit unsigned by 33, we get 31 bits
    return (x >> 33) & 0x7FFFFFFF


def _murmur_hash_torch(species_ids: torch.Tensor) -> torch.Tensor:
    """MurmurHash3 finalizer matching the Triton implementation."""
    h = species_ids.to(torch.int64)

    # MurmurHash3 fmix64 - use signed arithmetic with masking
    # Note: Python int overflow wraps in PyTorch, which is what we want
    h = h ^ _unsigned_rshift_33(h)
    h = h * 0xFF51AFD7ED558CCD
    h = h ^ _unsigned_rshift_33(h)
    h = h * 0xC4CEB9FE1A85EC53
    h = h ^ _unsigned_rshift_33(h)

    return h


def hash_aggregate_pure_torch(
    plot_indices: torch.Tensor,
    species_ids: torch.Tensor,
    weights: torch.Tensor,
    n_plots: int,
    hash_dim: int,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of hash embedding (fallback).

    Uses scatter_add for GPU acceleration without custom kernels.
    """
    device = plot_indices.device
    dtype = weights.dtype

    # MurmurHash3-style mixing (vectorized)
    h = _murmur_hash_torch(species_ids)

    # Get bucket (use abs for positive modulo) and sign
    # Use bitwise AND to get unsigned abs, avoiding overflow when h == INT64_MIN
    hash_idx = (h.abs().to(torch.int64) & 0x7FFFFFFFFFFFFFFF) % hash_dim
    sign = torch.where(h >= 0, torch.ones_like(weights), -torch.ones_like(weights))

    # Compute linear indices for scatter
    linear_idx = plot_indices * hash_dim + hash_idx

    # Weighted contribution
    contribution = sign * weights

    # Scatter add
    output = torch.zeros(n_plots * hash_dim, dtype=dtype, device=device)
    output.scatter_add_(0, linear_idx, contribution)

    return output.view(n_plots, hash_dim)


def hash_batch_csr_pure_torch(
    batch_indices: torch.Tensor,
    plot_offsets: torch.Tensor,
    species_ids: torch.Tensor,
    weights: torch.Tensor,
    hash_dim: int,
) -> torch.Tensor:
    """
    Batch hash embedding from CSR format using pure PyTorch.

    Args:
        batch_indices: (batch_size,) int64 - which plots are in this batch
        plot_offsets: (n_plots+1,) int64 - CSR offsets for species ranges
        species_ids: (n_records,) int64 - all species IDs
        weights: (n_records,) float32 - all weights
        hash_dim: Dimension of hash embedding

    Returns:
        (batch_size, hash_dim) float32 tensor of hash embeddings
    """
    batch_size = batch_indices.shape[0]
    device = batch_indices.device
    dtype = weights.dtype

    output = torch.zeros(batch_size, hash_dim, dtype=dtype, device=device)

    # Get start/end offsets for each batch element
    starts = plot_offsets[batch_indices]
    ends = plot_offsets[batch_indices + 1]

    for batch_idx in range(batch_size):
        start = starts[batch_idx].item()
        end = ends[batch_idx].item()

        if end > start:
            sp_ids = species_ids[start:end]
            wts = weights[start:end]

            # Hash using same function as aggregate
            h = _murmur_hash_torch(sp_ids)
            # Use bitwise AND to get unsigned abs, avoiding overflow when h == INT64_MIN
            hash_idx = (h.abs().to(torch.int64) & 0x7FFFFFFFFFFFFFFF) % hash_dim
            sign = torch.where(h >= 0, torch.ones_like(wts), -torch.ones_like(wts))

            # Scatter into this batch row
            output[batch_idx].scatter_add_(0, hash_idx, sign * wts)

    return output


# Main API - auto-selects best available implementation
def hash_aggregate(
    plot_indices: torch.Tensor,
    species_ids: torch.Tensor,
    weights: torch.Tensor,
    n_plots: int,
    hash_dim: int,
) -> torch.Tensor:
    """
    Compute hash embedding by aggregating species contributions.

    Automatically selects best available implementation:
    1. Triton kernel (fastest, if available)
    2. Pure PyTorch with scatter_add (GPU accelerated fallback)

    Args:
        plot_indices: (n,) int64 - plot index for each species record
        species_ids: (n,) int64 - hashed species ID for each record
        weights: (n,) float32 - abundance weight for each record
        n_plots: Total number of plots
        hash_dim: Dimension of hash embedding

    Returns:
        (n_plots, hash_dim) float32 tensor of hash embeddings
    """
    if TRITON_AVAILABLE and plot_indices.is_cuda:
        return hash_aggregate_triton(plot_indices, species_ids, weights, n_plots, hash_dim)
    else:
        return hash_aggregate_pure_torch(plot_indices, species_ids, weights, n_plots, hash_dim)


def hash_batch_csr(
    batch_indices: torch.Tensor,
    plot_offsets: torch.Tensor,
    species_ids: torch.Tensor,
    weights: torch.Tensor,
    hash_dim: int,
) -> torch.Tensor:
    """
    Compute hash embedding for a batch using CSR format.

    Args:
        batch_indices: (batch_size,) int64 - global plot indices in batch
        plot_offsets: (n_plots+1,) int64 - CSR offsets for species ranges
        species_ids: (n_records,) int64 - all species IDs
        weights: (n_records,) float32 - all weights
        hash_dim: Dimension of hash embedding

    Returns:
        (batch_size, hash_dim) float32 tensor of hash embeddings
    """
    # For now, use pure PyTorch - can add Triton kernel later
    return hash_batch_csr_pure_torch(batch_indices, plot_offsets, species_ids, weights, hash_dim)


def is_triton_available() -> bool:
    """Check if Triton kernels are available."""
    return TRITON_AVAILABLE
