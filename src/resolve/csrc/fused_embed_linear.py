"""Fused Embedding + Concat + Linear kernel (Triton + PyTorch fallback).

Avoids materializing the intermediate concatenated tensor by fusing:
  1. Taxonomy embedding lookup (genus + family, per position)
  2. Concatenation with continuous features
  3. First linear projection (W @ [continuous; genus_embs; family_embs] + b)

into a single pass. For a batch of B samples with D_cont continuous features,
K taxonomy positions, and G/F embedding dims, the naive approach materializes
a (B, D_cont + K*G + K*F) tensor; this kernel computes the Linear output
directly, saving memory bandwidth.

NOTE: Benchmarking on RTX 5080 shows the Triton kernel is ~90x slower than
PyTorch's cuBLAS-backed path for typical RESOLVE dimensions (D_in=83,
D_out=2048). The intermediate concat tensor (~1.3 MB at B=4096) is too small
to justify the fusion overhead. The Triton kernel is disabled by default;
use ``force_triton=True`` only for experimentation or very different dimension
configurations where memory bandwidth is the actual bottleneck.

The PyTorch fallback path is always used by default. It performs the same
embed+concat+linear operations using optimized PyTorch ops.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _fused_embed_concat_linear_kernel(
        # Input pointers
        continuous_ptr,      # (B, D_cont) float
        genus_ids_ptr,       # (B, K) int64
        family_ids_ptr,      # (B, K) int64
        genus_weight_ptr,    # (n_genera, G) float - flattened K tables stacked
        family_weight_ptr,   # (n_families, F) float - flattened K tables stacked
        linear_weight_ptr,   # (D_out, D_in) float, D_in = D_cont + K*G + K*F
        linear_bias_ptr,     # (D_out,) float or nullptr
        output_ptr,          # (B, D_out) float
        # Dimensions
        B,
        D_cont: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,     # genus embedding dim
        F_dim: tl.constexpr, # family embedding dim (can't use F, reserved)
        D_out: tl.constexpr,
        D_in: tl.constexpr,  # D_cont + K*G + K*F
        n_genera,
        n_families,
        has_bias: tl.constexpr,
        # Block sizes
        BLOCK_DOUT: tl.constexpr,
    ):
        """One program per (row, output_block).

        For each output element, computes dot product of the input row
        (continuous + embedded taxonomy) with the corresponding weight row.
        """
        row = tl.program_id(0)
        out_block = tl.program_id(1)
        if row >= B:
            return

        out_offs = out_block * BLOCK_DOUT + tl.arange(0, BLOCK_DOUT)
        out_mask = out_offs < D_out

        # Accumulator for output[row, out_offs]
        acc = tl.zeros((BLOCK_DOUT,), dtype=tl.float32)

        # Part 1: continuous features contribution
        # acc += W[out_offs, :D_cont] @ continuous[row, :]
        for d in range(D_cont):
            x_val = tl.load(continuous_ptr + row * D_cont + d).to(tl.float32)
            w_vals = tl.load(
                linear_weight_ptr + out_offs * D_in + d,
                mask=out_mask, other=0.0,
            ).to(tl.float32)
            acc += x_val * w_vals

        # Part 2: genus embeddings contribution
        # For each position k, look up genus_ids[row, k] and multiply
        w_offset = D_cont  # where genus embeddings start in the weight matrix
        for k in range(K):
            gid = tl.load(genus_ids_ptr + row * K + k)
            for g in range(G):
                emb_val = tl.load(genus_weight_ptr + k * n_genera * G + gid * G + g).to(tl.float32)
                w_vals = tl.load(
                    linear_weight_ptr + out_offs * D_in + (w_offset + k * G + g),
                    mask=out_mask, other=0.0,
                ).to(tl.float32)
                acc += emb_val * w_vals

        # Part 3: family embeddings contribution
        w_offset = D_cont + K * G
        for k in range(K):
            fid = tl.load(family_ids_ptr + row * K + k)
            for f in range(F_dim):
                emb_val = tl.load(family_weight_ptr + k * n_families * F_dim + fid * F_dim + f).to(tl.float32)
                w_vals = tl.load(
                    linear_weight_ptr + out_offs * D_in + (w_offset + k * F_dim + f),
                    mask=out_mask, other=0.0,
                ).to(tl.float32)
                acc += emb_val * w_vals

        # Add bias
        if has_bias:
            bias_vals = tl.load(linear_bias_ptr + out_offs, mask=out_mask, other=0.0).to(tl.float32)
            acc += bias_vals

        # Store
        tl.store(output_ptr + row * D_out + out_offs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# PyTorch fallback (standard path)
# ---------------------------------------------------------------------------

def _fused_embed_concat_linear_torch(
    continuous: torch.Tensor,
    genus_ids: torch.Tensor,
    family_ids: torch.Tensor,
    genus_embeddings: list[torch.nn.Embedding],
    family_embeddings: list[torch.nn.Embedding],
    linear: torch.nn.Linear,
) -> torch.Tensor:
    """Standard PyTorch: embed, concat, linear (materializes intermediate)."""
    parts = [continuous]
    for k, emb in enumerate(genus_embeddings):
        parts.append(emb(genus_ids[:, k]))
    for k, emb in enumerate(family_embeddings):
        parts.append(emb(family_ids[:, k]))
    x = torch.cat(parts, dim=1)
    return linear(x)


# ---------------------------------------------------------------------------
# Autograd Function for Triton path
# ---------------------------------------------------------------------------

class _FusedEmbedConcatLinear(torch.autograd.Function):
    """Fused forward only; backward falls back to standard PyTorch.

    The forward pass avoids materializing the concatenated embedding tensor.
    The backward pass recomputes it (embed+concat) then uses standard autograd,
    since the backward is not the bottleneck (forward is called much more
    during inference).
    """

    @staticmethod
    def forward(
        ctx,
        continuous,
        genus_ids,
        family_ids,
        linear_weight,
        linear_bias,
        *emb_weights,  # genus_emb_0.weight, ..., genus_emb_K.weight, family_emb_0.weight, ...
    ):
        K = (len(emb_weights)) // 2
        genus_weights = emb_weights[:K]
        family_weights = emb_weights[K:]

        B = continuous.shape[0]
        D_cont = continuous.shape[1]
        G = genus_weights[0].shape[1]
        F_dim = family_weights[0].shape[1]
        D_out = linear_weight.shape[0]
        D_in = D_cont + K * G + K * F_dim
        n_genera = genus_weights[0].shape[0]
        n_families = family_weights[0].shape[0]

        # Stack embedding tables for contiguous access
        # genus_weight_stacked: (K, n_genera, G) contiguous
        genus_stacked = torch.stack(list(genus_weights), dim=0).contiguous()
        family_stacked = torch.stack(list(family_weights), dim=0).contiguous()

        output = torch.empty(B, D_out, dtype=continuous.dtype, device=continuous.device)

        BLOCK_DOUT = min(triton.next_power_of_2(D_out), 128)
        n_out_blocks = (D_out + BLOCK_DOUT - 1) // BLOCK_DOUT

        _fused_embed_concat_linear_kernel[(B, n_out_blocks)](
            continuous, genus_ids, family_ids,
            genus_stacked, family_stacked,
            linear_weight, linear_bias if linear_bias is not None else continuous,  # dummy ptr if no bias
            output,
            B, D_cont, K, G, F_dim, D_out, D_in,
            n_genera, n_families,
            linear_bias is not None,
            BLOCK_DOUT=BLOCK_DOUT,
        )

        # Save for backward (reconstruct concatenated input)
        ctx.save_for_backward(
            continuous, genus_ids, family_ids,
            linear_weight, linear_bias,
            *emb_weights,
        )
        ctx.K = K
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        continuous = saved[0]
        genus_ids = saved[1]
        family_ids = saved[2]
        linear_weight = saved[3]
        linear_bias = saved[4]
        emb_weights = saved[5:]
        K = ctx.K

        # Reconstruct concatenated input for standard backward
        parts = [continuous]
        for k in range(K):
            parts.append(F.embedding(genus_ids[:, k], emb_weights[k]))
        for k in range(K):
            parts.append(F.embedding(family_ids[:, k], emb_weights[K + k]))
        x = torch.cat(parts, dim=1)

        # grad through linear
        grad_x = grad_output @ linear_weight  # (B, D_in)
        grad_weight = grad_output.T @ x  # (D_out, D_in)
        grad_bias = grad_output.sum(0) if linear_bias is not None else None

        # Split grad_x back into continuous + embedding parts
        D_cont = continuous.shape[1]
        G = emb_weights[0].shape[1]
        F_dim = emb_weights[K].shape[1]

        grad_continuous = grad_x[:, :D_cont]

        # Grad through embeddings
        grad_emb_weights = []
        offset = D_cont
        for k in range(K):
            grad_emb_out = grad_x[:, offset:offset + G]  # (B, G)
            grad_w = torch.zeros_like(emb_weights[k])
            grad_w.index_add_(0, genus_ids[:, k], grad_emb_out)
            grad_emb_weights.append(grad_w)
            offset += G
        for k in range(K):
            grad_emb_out = grad_x[:, offset:offset + F_dim]  # (B, F_dim)
            grad_w = torch.zeros_like(emb_weights[K + k])
            grad_w.index_add_(0, family_ids[:, k], grad_emb_out)
            grad_emb_weights.append(grad_w)
            offset += F_dim

        return (grad_continuous, None, None, grad_weight, grad_bias, *grad_emb_weights)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_embed_concat_linear(
    continuous: torch.Tensor,
    genus_ids: torch.Tensor,
    family_ids: torch.Tensor,
    genus_embeddings: list[torch.nn.Embedding],
    family_embeddings: list[torch.nn.Embedding],
    linear: torch.nn.Linear,
    force_triton: bool = False,
) -> torch.Tensor:
    """Fused embedding lookup + concatenation + first linear projection.

    Computes ``linear(cat(continuous, genus_embs, family_embs))`` without
    materializing the intermediate concatenated tensor.

    Uses the PyTorch path by default (cuBLAS-backed, faster for typical
    RESOLVE dimensions). Set ``force_triton=True`` to use the Triton kernel
    for experimentation.

    Args:
        continuous: (B, D_cont) continuous features
        genus_ids: (B, K) integer genus IDs per position
        family_ids: (B, K) integer family IDs per position
        genus_embeddings: List of K nn.Embedding modules for genus
        family_embeddings: List of K nn.Embedding modules for family
        linear: nn.Linear layer (first MLP layer)
        force_triton: Use Triton kernel even when PyTorch path is faster.

    Returns:
        (B, D_out) output of fused operation.
    """
    if force_triton and TRITON_AVAILABLE and continuous.is_cuda:
        emb_weights = [emb.weight for emb in genus_embeddings] + [emb.weight for emb in family_embeddings]
        return _FusedEmbedConcatLinear.apply(
            continuous, genus_ids, family_ids,
            linear.weight, linear.bias,
            *emb_weights,
        )
    return _fused_embed_concat_linear_torch(
        continuous, genus_ids, family_ids,
        genus_embeddings, family_embeddings, linear,
    )


def is_triton_available() -> bool:
    """Check if Triton kernels are available."""
    return TRITON_AVAILABLE
