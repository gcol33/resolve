"""
Spacc Core - C++ backend for species-composition based prediction.

This module provides Python bindings to the C++ core library.
"""

try:
    from ._resolve_core import (
        # Enums
        TaskType,
        TransformType,
        # Config structs
        TargetConfig,
        SpaccSchema,
        ModelConfig,
        TrainConfig,
        TrainResult,
        SpaccPredictions,
        Scalers,
        # Species encoding
        TaxonomyVocab,
        EncodedSpecies,
        SpeciesEncoder,
        # Model
        SpaccModel,
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
    "TargetConfig",
    "SpaccSchema",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "SpaccPredictions",
    "Scalers",
    "TaxonomyVocab",
    "EncodedSpecies",
    "SpeciesEncoder",
    "SpaccModel",
    "Trainer",
    "Predictor",
    "Metrics",
]
