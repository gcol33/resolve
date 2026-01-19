"""
Spacc: Species Presence and Abundance for Coordinate-linked Characteristics.

An opinionated torch-based package for predicting plot-level attributes
from species composition, environment, and space.
"""

from spacc.data.dataset import SpaccDataset
from spacc.model.spacc import SpaccModel
from spacc.train.trainer import Trainer
from spacc.inference.predictor import Predictor

__version__ = "0.1.0"
__all__ = ["SpaccDataset", "SpaccModel", "Trainer", "Predictor"]
