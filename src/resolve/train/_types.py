"""Result types and utilities for RESOLVE training."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class ProfileResult:
    """Results from profiling a training run."""

    total_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    data_time_ms: float
    n_batches: int
    avg_batch_time_ms: float
    samples_per_second: float
    gpu_memory_peak_mb: float = 0.0
    detailed_trace_path: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            "=== Training Profile ===",
            f"Total time:      {self.total_time_ms:.1f} ms",
            f"  Forward:       {self.forward_time_ms:.1f} ms ({100*self.forward_time_ms/self.total_time_ms:.1f}%)",
            f"  Backward:      {self.backward_time_ms:.1f} ms ({100*self.backward_time_ms/self.total_time_ms:.1f}%)",
            f"  Optimizer:     {self.optimizer_time_ms:.1f} ms ({100*self.optimizer_time_ms/self.total_time_ms:.1f}%)",
            f"  Data loading:  {self.data_time_ms:.1f} ms ({100*self.data_time_ms/self.total_time_ms:.1f}%)",
            f"Batches:         {self.n_batches}",
            f"Avg batch time:  {self.avg_batch_time_ms:.2f} ms",
            f"Throughput:      {self.samples_per_second:.0f} samples/sec",
        ]
        if self.gpu_memory_peak_mb > 0:
            lines.append(f"GPU memory peak: {self.gpu_memory_peak_mb:.0f} MB")
        if self.detailed_trace_path:
            lines.append(f"Trace saved to:  {self.detailed_trace_path}")
        return "\n".join(lines)


class Timer:
    """Simple timer for profiling code sections."""

    def __init__(self):
        self.times = {}
        self._starts = {}

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._starts[name]) * 1000  # ms
        if name not in self.times:
            self.times[name] = 0.0
        self.times[name] += elapsed
        return elapsed

    @contextmanager
    def section(self, name: str):
        """Context manager for timing a code section."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get(self, name: str) -> float:
        return self.times.get(name, 0.0)

    def reset(self) -> None:
        self.times.clear()
        self._starts.clear()


@dataclass
class TrainResult:
    """Results from training."""

    best_epoch: int
    final_metrics: dict[str, dict[str, float]]
    history: dict[str, list[float]] = field(default_factory=dict)
    resumed_from_epoch: Optional[int] = None
    train_time: float = 0.0  # Total training time in seconds


@dataclass
class CVResult:
    """Results from cross-validation."""

    fold_results: list[TrainResult]
    fold_metrics: list[dict[str, dict[str, float]]]
    mean_metrics: dict[str, dict[str, float]]
    std_metrics: dict[str, dict[str, float]]
    n_folds: int

    def __str__(self) -> str:
        lines = [f"=== {self.n_folds}-Fold CV Results ==="]
        for target, metrics in self.mean_metrics.items():
            lines.append(f"  {target}:")
            for metric, value in metrics.items():
                std = self.std_metrics[target].get(metric, 0.0)
                lines.append(f"    {metric}: {value:.4f} +/- {std:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grouped configuration dataclasses
# ---------------------------------------------------------------------------

_SENTINEL = object()


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Pass to Trainer as ``model_config=ModelConfig(...)`` to group
    architecture params. Individual kwargs still work and take priority.
    """

    species_encoding: str = "hash"
    hash_dim: int = 32
    species_embed_dim: int = 32
    top_k: int = 5
    top_k_species: int = 10
    hidden_dims: list[int] | None = None
    genus_emb_dim: int = 8
    family_emb_dim: int = 8
    dropout: float = 0.3
    head_hidden_dims: list[int] | None = None
    # Transformer-specific
    n_attention_layers: int = 0
    n_heads: int = 4
    transformer_ff_dim: int = 256
    transformer_pooling: str = "attention"
    transformer_dropout: float = 0.1
    # Advanced architecture (requires C++ backend for non-MLP)
    encoder_architecture: str = "mlp"
    # MoE configuration
    moe_routing: str = "none"  # "none", "soft", "topk"
    n_experts: int = 4
    expert_hidden_dims: list[int] | None = None
    moe_top_k: int = 2
    moe_noise_std: float = 0.1
    moe_aux_loss_weight: float = 0.01
    # Architecture sub-configs (dicts passed to C++ backend)
    ft_transformer_config: dict | None = None
    tabnet_config: dict | None = None
    saint_config: dict | None = None
    gnn_config: dict | None = None
    excelformer_config: dict | None = None


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Pass to Trainer as ``training_config=TrainingConfig(...)`` to group
    training params. Individual kwargs still work and take priority.
    """

    batch_size: int = 32768
    num_workers: int = 0
    max_epochs: int = 500
    patience: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "onecycle"
    lr_factor: float = 0.1
    lr_patience: int = 5
    loss_config: str = "mae"
    device: str = "auto"
    use_amp: bool = True
    compile_model: bool = False
    prefetch_data: bool | None = None
    gpu_data: bool | None = None
    label_smoothing: float = 0.0
    class_weights: torch.Tensor | None = None
    ema_decay: float = 0.0
    verbose: int = 1


@dataclass
class DataConfig:
    """Data preprocessing configuration.

    Pass to Trainer as ``data_config=DataConfig(...)`` to group
    data-related params. Individual kwargs still work and take priority.
    """

    species_aggregation: str = "abundance"
    species_selection: str = "top"
    species_representation: str = "abundance"
    min_species_frequency: int = 1
    cover_dropout: float = 0.0
    categorical_embed_dim: int = 8
    # Pretraining
    pretrain_epochs: int = 0
    pretrain_mask_prob: float = 0.15
    pretrain_lr: float = 1e-4
    pretrain_all_data: bool = False


@dataclass
class CheckpointConfig:
    """Checkpointing and caching configuration.

    Pass to Trainer as ``checkpoint_config=CheckpointConfig(...)`` to group
    checkpoint params. Individual kwargs still work and take priority.
    """

    checkpoint_dir: str | Path | None = None
    checkpoint_every: int = 50
    resume: bool = True
    reset_patience: bool = False
    cache_dir: str | Path | None = None
    max_cache_files: int = 5
