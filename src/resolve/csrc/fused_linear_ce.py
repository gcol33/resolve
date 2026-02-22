"""Fused Linear + CrossEntropy kernel (Triton + PyTorch fallback).

Avoids materializing the (N, V) logits tensor by computing the loss and
gradients in a single pass over the weight matrix using online softmax.

Pattern follows hash_ops.py: Triton kernel when available, pure PyTorch
fallback otherwise, with auto-selection in the public API.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Check if triton is available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _fused_linear_ce_fwd_kernel(
        # Pointers
        x_ptr,          # (N, D) input embeddings
        w_ptr,          # (V, D) weight matrix
        targets_ptr,    # (N,) target indices
        loss_ptr,       # (N,) per-sample loss (output)
        lse_ptr,        # (N,) log-sum-exp (saved for backward)
        # Dimensions
        N,
        D: tl.constexpr,
        V,
        # Parameters
        ignore_index,
        label_smoothing: tl.constexpr,
        # Block sizes
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Forward: one program per sample. Online softmax over vocab."""
        row = tl.program_id(0)
        if row >= N:
            return

        target = tl.load(targets_ptr + row)

        # Handle ignore_index: zero loss
        if target == ignore_index:
            tl.store(loss_ptr + row, 0.0)
            tl.store(lse_ptr + row, 0.0)
            return

        # Online softmax: single pass over vocab blocks
        running_max = float("-inf")
        running_sum = 0.0
        z_tgt = 0.0
        sum_z = 0.0  # for label smoothing: need mean_z = sum_z / V

        for v_start in range(0, V, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offsets < V

            # Compute z[j] = dot(x[row], W[j]) for j in this vocab block
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D
                x_chunk = tl.load(x_ptr + row * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                w_block = tl.load(
                    w_ptr + v_offsets[:, None] * D + d_offs[None, :],
                    mask=v_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.sum(w_block * x_chunk[None, :], axis=1)

            z = acc  # (BLOCK_V,) logits for this block

            # Capture z[target] if target falls in this block
            target_in_block = (target >= v_start) & (target < v_start + BLOCK_V)
            if target_in_block:
                local_idx = target - v_start
                z_tgt = tl.sum(tl.where(tl.arange(0, BLOCK_V) == local_idx, z, 0.0))

            # Accumulate sum of logits for label smoothing
            if label_smoothing > 0.0:
                sum_z += tl.sum(tl.where(v_mask, z, 0.0))

            # Online softmax update
            block_max = tl.max(tl.where(v_mask, z, float("-inf")))
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.where(v_mask, tl.exp(z - new_max), 0.0)
            )
            running_max = new_max

        lse = running_max + tl.log(running_sum)
        nll = -z_tgt + lse

        # Label-smoothed CE: (1-s)*NLL + s*(lse - mean_z)
        if label_smoothing > 0.0:
            mean_z = sum_z / V
            loss = (1.0 - label_smoothing) * nll + label_smoothing * (lse - mean_z)
        else:
            loss = nll

        tl.store(loss_ptr + row, loss)
        tl.store(lse_ptr + row, lse)

    @triton.jit
    def _fused_linear_ce_bwd_kernel(
        # Pointers
        x_ptr,            # (N, D)
        w_ptr,            # (V, D)
        targets_ptr,      # (N,)
        lse_ptr,          # (N,)
        grad_output_ptr,  # (N,) per-sample upstream gradient
        grad_x_ptr,       # (N, D) output, accumulated via atomicAdd
        grad_w_ptr,       # (V, D) output, each chunk owned by one program
        # Dimensions
        N,
        D: tl.constexpr,
        V,
        # Parameters
        ignore_index,
        label_smoothing: tl.constexpr,
        # Block sizes
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward: one program per vocab chunk.

        Each program owns grad_W[v_start:v_start+BLOCK_V, :] (no atomics).
        grad_x accumulated via atomicAdd across programs.
        """
        chunk_id = tl.program_id(0)
        v_start = chunk_id * BLOCK_V
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offsets < V

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load W chunk: (BLOCK_V, BLOCK_D) -- kept in registers for the entire loop
        w_block = tl.load(
            w_ptr + v_offsets[:, None] * D + d_offs[None, :],
            mask=v_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate grad_W for this chunk
        grad_w_acc = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)

        for row in range(N):
            target = tl.load(targets_ptr + row)

            # Skip ignored rows (no `continue` in Triton; use nested if)
            is_valid = (target != ignore_index)
            if is_valid:
                go = tl.load(grad_output_ptr + row).to(tl.float32)
                lse_val = tl.load(lse_ptr + row).to(tl.float32)

                # Load x[row]: (BLOCK_D,)
                x_row = tl.load(
                    x_ptr + row * D + d_offs, mask=d_mask, other=0.0,
                ).to(tl.float32)

                # Recompute logits z = W_block @ x_row -> (BLOCK_V,)
                z = tl.sum(w_block * x_row[None, :], axis=1)

                # Softmax probabilities
                p = tl.exp(z - lse_val)

                # Gradient of CE w.r.t. logits
                is_target = (v_offsets == target)
                if label_smoothing > 0.0:
                    grad_z = go * ((1.0 - label_smoothing) * (p - tl.where(is_target, 1.0, 0.0))
                                   + label_smoothing * (p - 1.0 / V))
                else:
                    grad_z = go * (p - tl.where(is_target, 1.0, 0.0))
                grad_z = tl.where(v_mask, grad_z, 0.0)

                # grad_W += grad_z[:, None] * x_row[None, :]
                grad_w_acc += grad_z[:, None] * x_row[None, :]

                # grad_x[row] += sum_j(grad_z[j] * W[j, :])
                grad_x_contrib = tl.sum(grad_z[:, None] * w_block, axis=0)  # (BLOCK_D,)
                tl.atomic_add(
                    grad_x_ptr + row * D + d_offs,
                    grad_x_contrib,
                    mask=d_mask,
                )

        # Store grad_W for this chunk
        tl.store(
            grad_w_ptr + v_offsets[:, None] * D + d_offs[None, :],
            grad_w_acc,
            mask=v_mask[:, None] & d_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _FusedLinearCETriton(torch.autograd.Function):
    """Fused linear + cross-entropy via Triton (forward + backward)."""

    @staticmethod
    def forward(ctx, x, weight, targets, ignore_index, label_smoothing):
        N, D = x.shape
        V = weight.shape[0]

        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_V = 128  # forward: each program handles one row

        loss = torch.empty(N, dtype=torch.float32, device=x.device)
        lse = torch.empty(N, dtype=torch.float32, device=x.device)

        _fused_linear_ce_fwd_kernel[(N,)](
            x, weight, targets,
            loss, lse,
            N, D, V,
            ignore_index,
            label_smoothing,
            BLOCK_V=BLOCK_V,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x, weight, targets, lse)
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing

        # Mean loss over valid (non-ignored) samples
        valid = (targets != ignore_index)
        n_valid = valid.sum().clamp(min=1)
        return loss.sum() / n_valid

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, targets, lse = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing

        N, D = x.shape
        V = weight.shape[0]
        BLOCK_D = triton.next_power_of_2(D)
        # Backward BLOCK_V: smaller to keep register pressure manageable
        # (BLOCK_V, BLOCK_D) floats held in registers for grad_W + W)
        BLOCK_V = 64

        # Per-sample gradient scaling (forward returned mean over valid)
        valid = (targets != ignore_index)
        n_valid = valid.sum().clamp(min=1).float()
        grad_per_sample = torch.full(
            (N,), (grad_output / n_valid).item(),
            dtype=torch.float32, device=x.device,
        )
        grad_per_sample[~valid] = 0.0

        grad_x = torch.zeros(N, D, dtype=torch.float32, device=x.device)
        grad_w = torch.zeros(V, D, dtype=torch.float32, device=x.device)

        n_chunks = (V + BLOCK_V - 1) // BLOCK_V
        _fused_linear_ce_bwd_kernel[(n_chunks,)](
            x, weight, targets, lse, grad_per_sample,
            grad_x, grad_w,
            N, D, V,
            ignore_index,
            label_smoothing,
            BLOCK_V=BLOCK_V,
            BLOCK_D=BLOCK_D,
        )

        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None, None, None


# ---------------------------------------------------------------------------
# Pure PyTorch fallback
# ---------------------------------------------------------------------------

def _fused_linear_ce_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Pure PyTorch: materialize logits and use F.cross_entropy.

    Functionally identical but allocates the (N, V) logits tensor.
    Used as fallback when Triton is unavailable or input is on CPU.
    """
    logits = F.linear(x, weight)  # (N, V)
    return F.cross_entropy(
        logits, targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Fused linear projection + cross-entropy loss.

    Computes ``F.cross_entropy(x @ weight.T, targets)`` without materializing
    the ``(N, V)`` logits tensor, using online softmax in a single pass.

    Automatically selects Triton kernel (GPU) or pure PyTorch fallback (CPU).

    Args:
        x: (N, D) input embeddings (float16 or float32)
        weight: (V, D) weight matrix (same dtype as x)
        targets: (N,) target class indices (int64)
        ignore_index: target value to ignore in loss computation
        label_smoothing: label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar mean loss (with autograd support for x and weight).
    """
    if TRITON_AVAILABLE and x.is_cuda:
        return _FusedLinearCETriton.apply(
            x, weight, targets, ignore_index, label_smoothing,
        )
    return _fused_linear_ce_torch(x, weight, targets, ignore_index, label_smoothing)


def is_triton_available() -> bool:
    """Check if Triton kernels are available."""
    return TRITON_AVAILABLE
