"""Mixture of Experts module for RESOLVE.

Expert weights are stored as stacked 3D tensors and computed via batched
matmul (single cuBLAS GEMM call per layer) instead of sequential
per-expert forward passes. For E experts this reduces GPU kernel launches
from E to 1 per layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MixtureOfExperts(nn.Module):
    """Mixture of Experts with batched expert computation.

    All expert MLPs share the same architecture (same hidden dims). Their
    weights are stacked into ``(n_experts, out_features, in_features)``
    tensors so that a single ``torch.bmm`` computes all experts in parallel.

    For top-k routing with small E (4-8), all experts are computed via one
    batched GEMM and results are gathered, which is faster than E*K small
    kernel launches with dynamic shapes and boolean masking.

    Args:
        input_dim: Input feature dimension.
        expert_hidden_dims: Hidden dimensions for each expert MLP.
        output_dim: Output dimension from each expert.
        n_experts: Number of expert networks.
        routing: ``"soft"`` (all experts contribute) or ``"top_k"``.
        top_k: Number of experts per sample (only for top_k routing).
        noise_std: Noise added to gating logits during training.
        dropout: Dropout rate inside expert MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dims: list[int],
        output_dim: int,
        n_experts: int = 4,
        routing: str = "soft",
        top_k: int = 2,
        noise_std: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if routing not in ("soft", "top_k"):
            raise ValueError(f"routing must be 'soft' or 'top_k', got '{routing}'")

        self.n_experts = n_experts
        self.routing = routing
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std
        self._output_dim = output_dim
        self._dropout = dropout

        # Gating network
        self.gate = nn.Linear(input_dim, n_experts)

        # Stacked expert weights: (E, O, I) per layer.
        # One batched matmul per layer instead of E sequential nn.Linear calls.
        dims = [input_dim] + list(expert_hidden_dims) + [output_dim]
        self._n_layers = len(dims) - 1
        self.layer_weights = nn.ParameterList()
        self.layer_biases = nn.ParameterList()

        for i in range(self._n_layers):
            W = torch.empty(n_experts, dims[i + 1], dims[i])
            b = torch.zeros(n_experts, dims[i + 1])
            # Kaiming uniform init per expert slice (matches nn.Linear default)
            for e in range(n_experts):
                nn.init.kaiming_uniform_(W[e], a=math.sqrt(5))
                bound = 1 / math.sqrt(dims[i])
                nn.init.uniform_(b[e], -bound, bound)
            self.layer_weights.append(nn.Parameter(W))
            self.layer_biases.append(nn.Parameter(b))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _batched_experts(self, x: torch.Tensor) -> torch.Tensor:
        """Run all experts on all inputs via batched matmul.

        First layer uses broadcast matmul to avoid copying x E times.
        Subsequent layers use ``torch.bmm`` over the already-batched
        activations.

        Args:
            x: (B, D_in)

        Returns:
            (E, B, D_out) expert outputs for all samples.
        """
        # First layer: (1, B, I) @ (E, I, H0) → (E, B, H0) via broadcast
        h = torch.matmul(x.unsqueeze(0), self.layer_weights[0].transpose(1, 2))
        h = h + self.layer_biases[0].unsqueeze(1)
        if self._n_layers > 1:
            h = F.gelu(h)
            if self._dropout > 0 and self.training:
                h = F.dropout(h, p=self._dropout, training=True)

        # Remaining layers: (E, B, H_prev) @ (E, H_prev, H_next) → (E, B, H_next)
        for i in range(1, self._n_layers):
            h = torch.bmm(h, self.layer_weights[i].transpose(1, 2))
            h = h + self.layer_biases[i].unsqueeze(1)
            if i < self._n_layers - 1:
                h = F.gelu(h)
                if self._dropout > 0 and self.training:
                    h = F.dropout(h, p=self._dropout, training=True)

        return h  # (E, B, D_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (output, auxiliary_loss).

        Args:
            x: (batch, input_dim)

        Returns:
            output: (batch, output_dim)
            aux_loss: Scalar load-balancing loss.
        """
        logits = self.gate(x)  # (B, E)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        gate_probs = F.softmax(logits, dim=-1)  # (B, E)

        # All expert outputs via batched matmul: (E, B, D_out)
        expert_out = self._batched_experts(x)

        if self.routing == "soft":
            # Weighted sum across all experts
            # gate_probs.T: (E, B), unsqueeze → (E, B, 1)
            output = (gate_probs.T.unsqueeze(-1) * expert_out).sum(dim=0)

            # Load-balancing: CV² of expert importance
            importance = gate_probs.sum(dim=0)  # (E,)
            mean_imp = importance.mean()
            aux_loss = importance.var() / (mean_imp * mean_imp + 1e-8)
        else:
            # Top-k: select and re-normalize
            top_k_probs, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # Gather selected expert outputs
            # (E, B, D) → (B, E, D), then gather along expert dim
            expert_out_bt = expert_out.permute(1, 0, 2)  # (B, E, D)
            idx_expanded = top_k_idx.unsqueeze(-1).expand(-1, -1, self._output_dim)
            selected = torch.gather(expert_out_bt, 1, idx_expanded)  # (B, K, D)
            output = (top_k_probs.unsqueeze(-1) * selected).sum(dim=1)  # (B, D)

            # Load-balancing: Switch Transformer loss = E * sum(f_i * P_i)
            flat_idx = top_k_idx.reshape(-1)
            counts = torch.zeros(self.n_experts, device=x.device)
            counts.scatter_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float),
            )
            f = counts / counts.sum()
            P = gate_probs.mean(dim=0)
            aux_loss = self.n_experts * (f * P).sum()

        return output, aux_loss

    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without auxiliary loss (for inference)."""
        output, _ = self.forward(x)
        return output
