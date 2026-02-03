"""
RESOLVE: Representation Encoding of Species Outcomes via Linear Vector Embeddings.

A framework for predicting plot-level ecological attributes from species composition
using linear compositional pooling and learned representations.
"""

from resolve.data.dataset import ResolveDataset, ResolveSchema
from resolve.data.roles import RoleMapping, TargetConfig
from resolve.model.resolve import ResolveModel
from resolve.train.trainer import Trainer
from resolve.train.loss import PhaseConfig
from resolve.inference.predictor import Predictor, ResolvePredictions
from resolve import backend

__version__ = "0.1.0"
__all__ = [
    "ResolveDataset",
    "ResolveSchema",
    "ResolveModel",
    "Trainer",
    "Predictor",
    "ResolvePredictions",
    "RoleMapping",
    "TargetConfig",
    "PhaseConfig",
    "backend",
]
