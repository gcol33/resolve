"""
Resolve Core - C++ backend for species-composition based prediction.

This module provides low-level Python bindings to the C++ core library.
For high-level API, use the `resolve` package instead.

Low-level usage:
    from resolve_core import ResolveModel, Trainer, Predictor, ...
"""

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
        # Config structs
        TargetConfig,
        ResolveSchema,
        ModelConfig,
        TrainConfig,
        TrainResult,
        ResolvePredictions,
        Scalers,
        DatasetConfig,
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
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import resolve_core C++ extension: {e}\n"
        "Make sure the package was built with CMake and libtorch is available."
    ) from e

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
    # Config structs
    "TargetConfig",
    "ResolveSchema",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "ResolvePredictions",
    "Scalers",
    "DatasetConfig",
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
]
