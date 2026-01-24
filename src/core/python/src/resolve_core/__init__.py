"""
Resolve Core - C++ backend for species-composition based prediction.

This module provides Python bindings to the C++ core library.
"""

try:
    from ._resolve_core import (
        # Enums
        TaskType,
        TransformType,
        SpeciesEncodingMode,
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
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import resolve_core C++ extension: {e}\n"
        "Make sure the package was built with CMake and libtorch is available."
    ) from e

__version__ = "0.1.0"

__all__ = [
    "TaskType",
    "TransformType",
    "SpeciesEncodingMode",
    "SelectionMode",
    "RepresentationMode",
    "NormalizationMode",
    "AggregationMode",
    "TargetConfig",
    "ResolveSchema",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "ResolvePredictions",
    "Scalers",
    "TaxonomyVocab",
    "SpeciesRecord",
    "EncodedSpecies",
    "ResolveModel",
    "Trainer",
    "Predictor",
    "Metrics",
]
