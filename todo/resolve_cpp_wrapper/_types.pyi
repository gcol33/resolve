"""Type stubs for RESOLVE."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
from torch import Tensor

# Type aliases
PathLike = Union[str, Path]
DeviceType = Union[str, torch.device]
TargetDict = Dict[str, Dict[str, Any]]
RolesDict = Dict[str, str]

class Schema:
    """Dataset schema containing metadata."""
    n_plots: int
    n_species: int
    n_genera: int
    n_families: int
    has_coordinates: bool
    has_taxonomy: bool
    has_covariates: bool
    targets: List[Any]

class TrainResult:
    """Training result with history and metrics."""
    best_epoch: int
    best_loss: float
    history: Dict[str, List[float]]
    final_metrics: Dict[str, Dict[str, float]]

class ResolvePredictions:
    """Prediction results."""
    predictions: Dict[str, Tensor]
    latent: Optional[Tensor]
    unknown_fraction: Tensor

class Scalers:
    """Data scalers for normalization."""
    ...

class RoleMapping:
    """Column role mapping configuration."""
    plot_id: str
    species_id: str
    species_plot_id: str
    coords_lat: Optional[str]
    coords_lon: Optional[str]
    abundance: Optional[str]
    taxonomy_genus: Optional[str]
    taxonomy_family: Optional[str]
    covariates: Optional[List[str]]

class TargetConfig:
    """Target variable configuration."""
    column: str
    task: str  # "regression" or "classification"
    transform: Optional[str]
    num_classes: Optional[int]
    weight: float
