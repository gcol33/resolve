"""Result types and utilities for RESOLVE training."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
