"""
Resolve Core - C++ backend for species-composition based prediction.

This module provides low-level Python bindings to the C++ core library.
For high-level API, use the `resolve` package instead.

Low-level usage:
    from resolve_core import ResolveModel, Trainer, Predictor, ...

Import-order constraint
-----------------------
``configure_cuda_allocator()`` sets ``PYTORCH_CUDA_ALLOC_CONF`` and MUST run
BEFORE ``torch`` is imported by this process. The PyTorch CUDA caching
allocator reads ``PYTORCH_CUDA_ALLOC_CONF`` exactly once during the first
allocator initialization (which torch triggers on module import). To make this
work for downstream callers, the helper runs unconditionally at the very top
of this module before ``import torch``. If you ``import torch`` in your own
script before ``import resolve_core``, the env var is already too late; in
that case set ``PYTORCH_CUDA_ALLOC_CONF`` yourself before importing torch.
"""

# --- BEGIN: pre-torch allocator configuration ---------------------------------
# Order-sensitive: this block MUST stay above any `import torch` (direct or
# transitive via `from . import _resolve_core`).
import os as _os
import sys as _sys


def configure_cuda_allocator(force: bool = False) -> str:
    """Set a platform-aware ``PYTORCH_CUDA_ALLOC_CONF`` default.

    Sets ``os.environ["PYTORCH_CUDA_ALLOC_CONF"]`` if it is unset, or always
    when ``force=True``. The returned string is the final config in effect
    after this call (the existing value when neither branch overrides it).

    Linux/macOS get ``expandable_segments:True,...`` prefixed; the
    cuMemMap-backed expandable-segments allocator is not implemented on
    Windows (libtorch warns ``expandable_segments not supported on this
    platform``), so on ``win32`` that prefix is intentionally omitted.

    The baseline ``garbage_collection_threshold:0.8,max_split_size_mb:256``
    pair helps recover the most fragmented reserved memory on Windows and
    is harmless on Linux.

    Effective ONLY if called before ``torch`` is imported. After the
    allocator has lazy-initialized, the env var is read at most once and
    later changes are ignored. ``resolve_core`` calls this function once
    at import time, before ``import torch``, so the default applies for
    normal usage. ``force=True`` is useful when callers want to make the
    intent explicit (it overrides any existing value), but cannot rescue
    the case where torch was already imported.
    """
    base = "garbage_collection_threshold:0.8,max_split_size_mb:256"
    if _sys.platform != "win32":
        base = "expandable_segments:True," + base

    if force or "PYTORCH_CUDA_ALLOC_CONF" not in _os.environ:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = base
        return base

    return _os.environ["PYTORCH_CUDA_ALLOC_CONF"]


# Apply the default before torch is imported (below). Side-effecting on import
# is intentional: this is the only ordering that lets the allocator pick the
# config up before its first allocation.
_RESOLVE_CUDA_ALLOC_CONF = configure_cuda_allocator()
# --- END: pre-torch allocator configuration -----------------------------------

# Why: _resolve_core.pyd uses THPVariable_Wrap from torch_python.dll to wrap
# C++ at::Tensor results as Python torch.Tensor objects. THPVariable_Wrap
# requires torch's Python tensor class to be registered first, which only
# happens when the torch package's __init__ has run. Importing torch here
# guarantees that ordering. Without it, any binding that returns a tensor
# segfaults inside THPVariable_Wrap.
import torch  # noqa: F401  (must come AFTER configure_cuda_allocator)

try:
    from ._resolve_core import (
        # Enums
        TaskType,
        TransformType,
        SpeciesEncodingMode,
        LossConfigMode,
        SelectionMode,
        RepresentationMode,
        NormalizationMode,
        AggregationMode,
        MoERoutingType,
        LRSchedulerType,
        ActivationType,
        NormLayerType,
        # Pool-style species encoder weighting (rank_pool / transformer modes)
        PoolWeighting,
        # Architecture enums (v2.0)
        EncoderArchitecture,
        GNNType,
        GraphConstructionMode,
        TraitInteractionMode,
        ParallelAggregation,
        # Config structs
        TargetConfig,
        ResolveSchema,
        ModelConfig,
        TrainConfig,
        TrainResult,
        RunMetadata,
        BaselineMetrics,
        LayerDiagnostics,
        NetworkDiagnostics,
        ResolvePredictions,
        Scalers,
        DatasetConfig,
        # Architecture configs (v2.0)
        FTTransformerConfig,
        TabNetConfig,
        SAINTConfig,
        GNNConfig,
        TraitNetConfig,
        ExcelFormerConfig,
        HeterogeneousGNNConfig,
        ParallelBranchConfig,
        ParallelLayersConfig,
        # Calibration and residual analysis
        CalibrationBin,
        CalibrationResult,
        ResidualAnalysis,
        CrossValidationResult,
        # Species encoding
        TaxonomyVocab,
        SpeciesRecord,
        EncodedSpecies,
        # Model
        ResolveModel,
        # Training
        Trainer,
        # Inference
        Predictor,
        # Metrics
        Metrics,
        # Role mapping
        RoleMapping,
        TargetSpec,
        # Dataset
        ResolveDataset,
        # Pretraining (v3.0)
        MaskStrategy,
        PretrainConfig,
        PretrainResult,
        JEPAPretrainer,
        SCARFPretrainer,
        # VAE (v3.0)
        VAEConfig,
        VAEPretrainResult,
        VAEPretrainer,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import resolve_core C++ extension: {e}\n"
        "Make sure the package was built with CMake and libtorch is available."
    ) from e

# Optional symbols added in newer builds; tolerate older installed .pyd.
try:
    from ._resolve_core import set_vram_fraction
except ImportError:
    def set_vram_fraction(*_args, **_kwargs):
        """Stub for older resolve_core builds that lack set_vram_fraction.

        Rebuild resolve_core to enable GPU VRAM control via this function.
        """
        return None

from ._io_retry import retry_io

# Wrap checkpoint I/O entry points with retry_io so transient OSError on the
# host filesystem (WinError 121 semaphore timeout, EINVAL on a hung handle,
# etc.) doesn't kill a multi-hour training run. The C++ side passes filesystem
# faults through as Python OSError; retry_io catches them, sleeps with
# exponential backoff, and retries.
_Trainer_save_orig = Trainer.save
def _trainer_save(self, path, metadata=None):
    return retry_io(
        lambda: _Trainer_save_orig(self, path, metadata),
        what=f"Trainer.save({path!r})",
    )
Trainer.save = _trainer_save

_Trainer_load_orig = Trainer.load
def _trainer_load(path, device="cpu", vram_fraction=1.0):
    return retry_io(
        lambda: _Trainer_load_orig(path, device, vram_fraction),
        what=f"Trainer.load({path!r})",
    )
Trainer.load = staticmethod(_trainer_load)

_Predictor_load_orig = Predictor.load
def _predictor_load(path, device="cpu", vram_fraction=1.0):
    return retry_io(
        lambda: _Predictor_load_orig(path, device, vram_fraction),
        what=f"Predictor.load({path!r})",
    )
Predictor.load = staticmethod(_predictor_load)

__version__ = "0.6.0"

__all__ = [
    # Enums
    "TaskType",
    "TransformType",
    "SpeciesEncodingMode",
    "LossConfigMode",
    "SelectionMode",
    "RepresentationMode",
    "NormalizationMode",
    "AggregationMode",
    "MoERoutingType",
    "LRSchedulerType",
    "ActivationType",
    "NormLayerType",
    "PoolWeighting",
    # Architecture enums (v2.0)
    "EncoderArchitecture",
    "GNNType",
    "GraphConstructionMode",
    "TraitInteractionMode",
    "ParallelAggregation",
    # Config structs
    "TargetConfig",
    "ResolveSchema",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "RunMetadata",
    "BaselineMetrics",
    "LayerDiagnostics",
    "NetworkDiagnostics",
    "ResolvePredictions",
    "Scalers",
    "DatasetConfig",
    # Architecture configs (v2.0)
    "FTTransformerConfig",
    "TabNetConfig",
    "SAINTConfig",
    "GNNConfig",
    "TraitNetConfig",
    "ExcelFormerConfig",
    "HeterogeneousGNNConfig",
    "ParallelBranchConfig",
    "ParallelLayersConfig",
    # Calibration and residual analysis
    "CalibrationBin",
    "CalibrationResult",
    "ResidualAnalysis",
    "CrossValidationResult",
    # Species encoding
    "TaxonomyVocab",
    "SpeciesRecord",
    "EncodedSpecies",
    # Model
    "ResolveModel",
    # Training
    "Trainer",
    # Inference
    "Predictor",
    # Metrics
    "Metrics",
    # Role mapping
    "RoleMapping",
    "TargetSpec",
    # Dataset
    "ResolveDataset",
    # Pretraining (v3.0)
    "MaskStrategy",
    "PretrainConfig",
    "PretrainResult",
    "JEPAPretrainer",
    "SCARFPretrainer",
    # VAE (v3.0)
    "VAEConfig",
    "VAEPretrainResult",
    "VAEPretrainer",
    # GPU memory management
    "set_vram_fraction",
    "configure_cuda_allocator",
    # I/O resilience helper (also auto-applied to Trainer.save/load and Predictor.load)
    "retry_io",
]
