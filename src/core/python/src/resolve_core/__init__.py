"""
Resolve Core - C++ backend for species-composition based prediction.

This module provides low-level Python bindings to the C++ core library.
For high-level API, use the `resolve` package instead.

Low-level usage:
    from resolve_core import ResolveModel, Trainer, Predictor, ...
"""

# Why: _resolve_core.pyd uses THPVariable_Wrap from torch_python.dll to wrap
# C++ at::Tensor results as Python torch.Tensor objects. THPVariable_Wrap
# requires torch's Python tensor class to be registered first, which only
# happens when the torch package's __init__ has run. Importing torch here
# guarantees that ordering. Without it, any binding that returns a tensor
# segfaults inside THPVariable_Wrap.
import torch  # noqa: F401

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

__version__ = "0.1.0"

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
    # I/O resilience helper (also auto-applied to Trainer.save/load and Predictor.load)
    "retry_io",
]
