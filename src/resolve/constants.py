"""Centralized constants for the RESOLVE package."""

from __future__ import annotations

# Model architecture defaults
DEFAULT_HIDDEN_DIMS: list[int] = [2048, 1024, 512, 256, 128, 64]

# Training constants
PREFETCH_BATCH_THRESHOLD: int = 16384  # Auto-enable prefetch above this batch size
SCHEDULER_PCT_START: float = 0.1  # OneCycleLR warmup fraction
ETA_WINDOW: int = 10  # Rolling average window for ETA estimation
NAN_THRESHOLD_PCT: int = 50  # NaN batch % above which training is aborted
MAX_GRAD_NORM: float = 1.0  # Gradient clipping max norm
