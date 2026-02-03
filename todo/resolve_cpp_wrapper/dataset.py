"""
High-level ResolveDataset class that wraps the C++ core.

Provides a pandas-friendly API matching the paper's expected interface.
"""
from __future__ import annotations

import tempfile
import os
from pathlib import Path
from typing import Optional, Union, Any
from dataclasses import dataclass, field

import pandas as pd
import torch

from resolve_core import (
    ResolveDataset as _CoreDataset,
    RoleMapping as _CoreRoleMapping,
    TargetSpec as _CoreTargetSpec,
    DatasetConfig as _CoreDatasetConfig,
    SpeciesEncodingMode,
    TransformType,
    TaskType,
)

__all__ = [
    "ResolveDataset",
    "RoleMapping",
    "TargetConfig",
]

# Valid encoding modes
_VALID_ENCODINGS = {"hash", "embed", "sparse"}
_VALID_SELECTIONS = {"top", "bottom", "top_bottom", "all"}
_VALID_TASKS = {"regression", "classification"}
_VALID_TRANSFORMS = {"none", "log1p"}


@dataclass
class RoleMapping:
    """Maps column names to their semantic roles."""
    plot_id: str = "plot_id"
    species_id: str = "species_id"
    species_plot_id: str = "plot_id"  # Plot ID column in species table
    abundance: Optional[str] = None
    coords_lat: Optional[str] = None
    coords_lon: Optional[str] = None
    taxonomy_genus: Optional[str] = None
    taxonomy_family: Optional[str] = None
    covariates: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "RoleMapping":
        """Create RoleMapping from a dictionary."""
        return cls(
            plot_id=data.get("plot_id", "plot_id"),
            species_id=data.get("species_id", "species_id"),
            species_plot_id=data.get("species_plot_id", data.get("plot_id", "plot_id")),
            abundance=data.get("abundance"),
            coords_lat=data.get("coords_lat"),
            coords_lon=data.get("coords_lon"),
            taxonomy_genus=data.get("taxonomy_genus"),
            taxonomy_family=data.get("taxonomy_family"),
            covariates=data.get("covariates", []),
        )


@dataclass
class TargetConfig:
    """Configuration for a prediction target.

    Attributes:
        column: Column name in header CSV containing target values.
        task: Task type - "regression" or "classification".
        transform: Transform to apply - "none" or "log1p".
        num_classes: Number of classes for classification (required if task="classification").
        weight: Loss weight for multi-task training.
    """
    column: str
    task: str = "regression"
    transform: str = "none"
    num_classes: int = 0
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got '{self.task}'")
        if self.transform not in _VALID_TRANSFORMS:
            raise ValueError(f"transform must be one of {_VALID_TRANSFORMS}, got '{self.transform}'")
        if self.task == "classification" and self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2 for classification, got {self.num_classes}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetConfig:
        """Create TargetConfig from a dictionary.

        Args:
            data: Dictionary with target configuration. Must contain "column" key.

        Returns:
            TargetConfig instance.

        Raises:
            KeyError: If "column" key is missing.
            ValueError: If configuration is invalid.
        """
        if "column" not in data:
            raise KeyError("TargetConfig dict must contain 'column' key")
        return cls(
            column=data["column"],
            task=data.get("task", "regression"),
            transform=data.get("transform", "none"),
            num_classes=data.get("num_classes", 0),
            weight=data.get("weight", 1.0),
        )


def _convert_targets(targets: dict) -> dict[str, TargetConfig]:
    """Convert dict targets to TargetConfig instances."""
    result = {}
    for name, cfg in targets.items():
        if isinstance(cfg, dict):
            result[name] = TargetConfig.from_dict(cfg)
        else:
            result[name] = cfg
    return result


class ResolveDataset:
    """
    High-level dataset class for species composition prediction.

    Can be created from:
    - CSV file paths (from_csv)
    - Pandas DataFrames (constructor)
    - Dict-based roles (from_csv with dict)
    """

    def __init__(
        self,
        header: pd.DataFrame,
        species: pd.DataFrame,
        roles: Union[RoleMapping, dict],
        targets: dict[str, Union[TargetConfig, dict]],
        species_encoding: str = "hash",
        hash_dim: int = 32,
        top_k: int = 3,
        top_k_species: int = 10,
        selection: str = "top",
        track_unknown_fraction: bool = True,
    ) -> None:
        """
        Create dataset from pandas DataFrames.

        Args:
            header: DataFrame with plot-level data (one row per plot)
            species: DataFrame with species occurrences (multiple rows per plot)
            roles: RoleMapping or dict mapping semantic roles to column names
            targets: Dict of target_name -> TargetConfig or dict
            species_encoding: "hash", "embed", or "sparse"
            hash_dim: Dimension for species hash embedding
            top_k: Number of top genera/families for taxonomy
            top_k_species: Number of top species for embed mode
            selection: "top", "bottom", "top_bottom", or "all"
            track_unknown_fraction: Track fraction of unknown species

        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If header/species are not DataFrames.
        """
        # Validate inputs
        if not isinstance(header, pd.DataFrame):
            raise TypeError(f"header must be a pandas DataFrame, got {type(header).__name__}")
        if not isinstance(species, pd.DataFrame):
            raise TypeError(f"species must be a pandas DataFrame, got {type(species).__name__}")
        if species_encoding not in _VALID_ENCODINGS:
            raise ValueError(f"species_encoding must be one of {_VALID_ENCODINGS}, got '{species_encoding}'")
        if selection not in _VALID_SELECTIONS:
            raise ValueError(f"selection must be one of {_VALID_SELECTIONS}, got '{selection}'")
        if hash_dim < 1:
            raise ValueError(f"hash_dim must be positive, got {hash_dim}")
        if top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k_species < 1:
            raise ValueError(f"top_k_species must be positive, got {top_k_species}")
        if not targets:
            raise ValueError("targets must not be empty")

        # Convert dict roles to RoleMapping if needed
        if isinstance(roles, dict):
            roles = RoleMapping.from_dict(roles)

        # Convert dict targets to TargetConfig
        target_configs = _convert_targets(targets)

        self._roles = roles
        self._target_configs = target_configs
        self._encoding = species_encoding
        self._hash_dim = hash_dim
        self._top_k = top_k
        self._top_k_species = top_k_species

        # Save to temp files and load via C++ core
        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = os.path.join(tmpdir, "header.csv")
            species_path = os.path.join(tmpdir, "species.csv")

            header.to_csv(header_path, index=False)
            species.to_csv(species_path, index=False)

            self._core = self._load_from_csv(
                header_path, species_path, roles, target_configs,
                species_encoding, hash_dim, top_k, top_k_species,
                selection, track_unknown_fraction, use_cuda_hash=False
            )

    @classmethod
    def from_csv(
        cls,
        header: Union[str, Path],
        species: Union[str, Path],
        roles: Union[RoleMapping, dict],
        targets: dict[str, Union[TargetConfig, dict]],
        species_encoding: str = "hash",
        hash_dim: int = 32,
        top_k: int = 3,
        top_k_species: int = 10,
        selection: str = "top",
        track_unknown_fraction: bool = True,
        use_cuda_hash: bool = False,
    ) -> "ResolveDataset":
        """
        Load dataset from CSV files.

        Args:
            header: Path to header CSV (plot-level data)
            species: Path to species CSV (species occurrences)
            roles: RoleMapping or dict mapping semantic roles to column names
            targets: Dict of target_name -> TargetConfig or dict

        Returns:
            ResolveDataset instance
        """
        # Convert dict roles to RoleMapping if needed
        if isinstance(roles, dict):
            roles = RoleMapping.from_dict(roles)

        # Convert dict targets to TargetConfig
        target_configs = _convert_targets(targets)

        instance = cls.__new__(cls)
        instance._roles = roles
        instance._target_configs = target_configs
        instance._encoding = species_encoding
        instance._hash_dim = hash_dim
        instance._top_k = top_k
        instance._top_k_species = top_k_species

        instance._core = cls._load_from_csv(
            str(header), str(species), roles, target_configs,
            species_encoding, hash_dim, top_k, top_k_species,
            selection, track_unknown_fraction, use_cuda_hash
        )

        return instance

    @staticmethod
    def _load_from_csv(
        header_path: str,
        species_path: str,
        roles: RoleMapping,
        targets: dict[str, TargetConfig],
        species_encoding: str,
        hash_dim: int,
        top_k: int,
        top_k_species: int,
        selection: str,
        track_unknown_fraction: bool,
        use_cuda_hash: bool,
    ) -> _CoreDataset:
        """Load via C++ core."""
        # Build core RoleMapping
        core_roles = _CoreRoleMapping()
        core_roles.plot_id = roles.plot_id
        core_roles.species_id = roles.species_id
        if roles.abundance:
            core_roles.abundance = roles.abundance
        if roles.coords_lon:
            core_roles.longitude = roles.coords_lon
        if roles.coords_lat:
            core_roles.latitude = roles.coords_lat
        if roles.taxonomy_genus:
            core_roles.genus = roles.taxonomy_genus
        if roles.taxonomy_family:
            core_roles.family = roles.taxonomy_family
        core_roles.covariates = roles.covariates or []

        # Build target specs
        target_specs = []
        for name, cfg in targets.items():
            spec = _CoreTargetSpec()
            spec.column_name = cfg.column
            spec.target_name = name
            spec.task = TaskType.Classification if cfg.task == "classification" else TaskType.Regression
            spec.transform = TransformType.Log1p if cfg.transform == "log1p" else TransformType.None_
            spec.num_classes = cfg.num_classes
            spec.weight = cfg.weight
            target_specs.append(spec)

        # Build config
        config = _CoreDatasetConfig()
        if species_encoding == "embed":
            config.species_encoding = SpeciesEncodingMode.Embed
        elif species_encoding == "sparse":
            config.species_encoding = SpeciesEncodingMode.Sparse
        else:
            config.species_encoding = SpeciesEncodingMode.Hash
        config.hash_dim = hash_dim
        config.top_k = top_k
        config.top_k_species = top_k_species
        config.track_unknown_fraction = track_unknown_fraction
        config.use_cuda_hash = use_cuda_hash

        return _CoreDataset.from_csv(header_path, species_path, core_roles, target_specs, config)

    @property
    def schema(self):
        """Get dataset schema."""
        return self._core.schema

    @property
    def n_plots(self) -> int:
        """Number of plots in dataset."""
        return self._core.n_plots

    @property
    def plot_ids(self) -> list[str]:
        """List of plot IDs."""
        return self._core.plot_ids

    @property
    def coordinates(self):
        """Coordinates tensor."""
        return self._core.coordinates

    @property
    def hash_embedding(self):
        """Hash embedding tensor."""
        return self._core.hash_embedding

    @property
    def genus_ids(self):
        """Genus ID tensor."""
        return self._core.genus_ids

    @property
    def family_ids(self):
        """Family ID tensor."""
        return self._core.family_ids

    @property
    def species_ids(self):
        """Species ID tensor (embed mode)."""
        return self._core.species_ids

    @property
    def species_vector(self):
        """Species vector tensor (sparse mode)."""
        return self._core.species_vector

    @property
    def targets(self):
        """Target tensors dict."""
        return self._core.targets

    @property
    def unknown_fraction(self):
        """Unknown fraction tensor."""
        return self._core.unknown_fraction

    @property
    def covariates(self):
        """Covariates tensor."""
        return self._core.covariates

    @property
    def _core_dataset(self):
        """Access underlying C++ dataset."""
        return self._core
