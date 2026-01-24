"""ResolveDataset: validated container for plot data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from resolve.data.roles import RoleMapping, TargetConfig


# Valid normalization modes for species abundance
VALID_NORMALIZATIONS = ("raw", "norm", "log1p")


@dataclass
class ResolveSchema:
    """Schema information derived from a ResolveDataset."""

    n_plots: int
    n_species: int
    n_continuous: int
    has_coordinates: bool
    has_abundance: bool
    has_taxonomy: bool
    n_genera: int
    n_families: int
    targets: dict[str, TargetConfig]
    covariate_names: list[str]
    # Species encoding configuration
    species_normalization: str = "norm"
    track_unknown_fraction: bool = True
    track_unknown_count: bool = False
    # Vocabulary sizes for embed mode (populated when using EmbeddingEncoder)
    n_species_vocab: int = 0  # Number of species in vocab (0 = use hash mode)
    n_genera_vocab: int = 0
    n_families_vocab: int = 0


class ResolveDataset:
    """
    Validated container for ecological plot data.

    Holds header (plot-level) and species (occurrence) data with
    semantic role mappings. Validates structure and provides
    train/test splitting.
    """

    def __init__(
        self,
        header: pd.DataFrame,
        species: pd.DataFrame,
        roles: RoleMapping,
        targets: dict[str, TargetConfig],
        species_normalization: str = "norm",
        track_unknown_fraction: bool = True,
        track_unknown_count: bool = False,
    ):
        # Validate normalization mode
        if species_normalization not in VALID_NORMALIZATIONS:
            raise ValueError(
                f"species_normalization must be one of {VALID_NORMALIZATIONS}, "
                f"got {species_normalization!r}"
            )

        self._header = header.copy()
        self._species = species.copy()
        self._roles = roles
        self._targets = targets
        self._species_normalization = species_normalization
        self._track_unknown_fraction = track_unknown_fraction
        self._track_unknown_count = track_unknown_count
        self._validate()

    def _validate(self) -> None:
        """Validate data structure and roles."""
        self._roles.validate()

        # Check required columns in header
        header_cols = set(self._header.columns)
        required_header = {self._roles.plot_id}
        if self._roles.has_coordinates:
            required_header.add(self._roles.coords_lat)
            required_header.add(self._roles.coords_lon)
        missing_header = required_header - header_cols
        if missing_header:
            raise ValueError(f"Missing columns in header: {missing_header}")

        # Check target columns
        for name, cfg in self._targets.items():
            if cfg.column not in header_cols:
                raise ValueError(f"Target '{name}' column '{cfg.column}' not in header")

        # Check covariate columns
        for cov in self._roles.covariates:
            if cov not in header_cols:
                raise ValueError(f"Covariate '{cov}' not in header")

        # Check required columns in species
        species_cols = set(self._species.columns)
        required_species = {self._roles.species_id, self._roles.species_plot_id}
        if self._roles.abundance:
            required_species.add(self._roles.abundance)
        if self._roles.taxonomy_genus:
            required_species.add(self._roles.taxonomy_genus)
        if self._roles.taxonomy_family:
            required_species.add(self._roles.taxonomy_family)

        missing_species = required_species - species_cols
        if missing_species:
            raise ValueError(f"Missing columns in species: {missing_species}")

        # Check foreign key relationship
        header_ids = set(self._header[self._roles.plot_id])
        species_ids = set(self._species[self._roles.species_plot_id])
        orphan_species = species_ids - header_ids
        if orphan_species:
            n_orphan = len(orphan_species)
            raise ValueError(f"{n_orphan} species rows reference plots not in header")

    @classmethod
    def from_csv(
        cls,
        header: str | Path,
        species: str | Path,
        roles: dict[str, str | list[str]],
        targets: dict[str, dict],
        species_normalization: str = "norm",
        track_unknown_fraction: bool = True,
        track_unknown_count: bool = False,
    ) -> ResolveDataset:
        """
        Load dataset from CSV files.

        Args:
            header: Path to plot-level CSV (one row per plot)
            species: Path to species CSV (one row per species-plot occurrence)
            roles: Mapping of semantic roles to column names
            targets: Target configurations {name: {column, task, transform?, num_classes?}}
            species_normalization: Abundance normalization mode
                - "raw": use abundance values directly
                - "norm": normalize to sum to 1 per sample (default)
                - "log1p": apply log(1 + x) transform
            track_unknown_fraction: Track fraction of abundance from unknown species (default True)
            track_unknown_count: Track count of unknown species (default False)
        """
        # Validate file paths exist before loading
        header_path = Path(header)
        species_path = Path(species)

        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")
        if not header_path.is_file():
            raise ValueError(f"Header path is not a file: {header_path}")

        if not species_path.exists():
            raise FileNotFoundError(f"Species file not found: {species_path}")
        if not species_path.is_file():
            raise ValueError(f"Species path is not a file: {species_path}")

        header_df = pd.read_csv(header_path, low_memory=False)
        species_df = pd.read_csv(species_path, low_memory=False)

        role_mapping = RoleMapping.from_dict(roles)
        target_configs = {
            name: TargetConfig.from_dict(name, cfg) for name, cfg in targets.items()
        }

        return cls(
            header_df,
            species_df,
            role_mapping,
            target_configs,
            species_normalization=species_normalization,
            track_unknown_fraction=track_unknown_fraction,
            track_unknown_count=track_unknown_count,
        )

    @property
    def header(self) -> pd.DataFrame:
        """Plot-level data."""
        return self._header

    @property
    def species(self) -> pd.DataFrame:
        """Species occurrence data."""
        return self._species

    @property
    def roles(self) -> RoleMapping:
        """Semantic role mapping."""
        return self._roles

    @property
    def targets(self) -> dict[str, TargetConfig]:
        """Target configurations."""
        return self._targets

    @property
    def species_normalization(self) -> str:
        """Species abundance normalization mode."""
        return self._species_normalization

    @property
    def track_unknown_fraction(self) -> bool:
        """Whether to track fraction of unknown species."""
        return self._track_unknown_fraction

    @property
    def track_unknown_count(self) -> bool:
        """Whether to track count of unknown species."""
        return self._track_unknown_count

    @property
    def plot_ids(self) -> np.ndarray:
        """Array of plot IDs."""
        return self._header[self._roles.plot_id].values

    @property
    def n_plots(self) -> int:
        """Number of plots."""
        return len(self._header)

    @property
    def schema(self) -> ResolveSchema:
        """Derive schema from dataset."""
        n_genera = 0
        n_families = 0
        if self._roles.has_taxonomy:
            n_genera = self._species[self._roles.taxonomy_genus].nunique()
            n_families = self._species[self._roles.taxonomy_family].nunique()

        # n_continuous: coordinates (if present) + covariates
        n_coords = 2 if self._roles.has_coordinates else 0
        n_continuous = n_coords + len(self._roles.covariates)

        return ResolveSchema(
            n_plots=self.n_plots,
            n_species=self._species[self._roles.species_id].nunique(),
            n_continuous=n_continuous,
            has_coordinates=self._roles.has_coordinates,
            has_abundance=self._roles.has_abundance,
            has_taxonomy=self._roles.has_taxonomy,
            n_genera=n_genera,
            n_families=n_families,
            targets=self._targets,
            covariate_names=self._roles.covariates,
            species_normalization=self._species_normalization,
            track_unknown_fraction=self._track_unknown_fraction,
            track_unknown_count=self._track_unknown_count,
        )

    def get_coordinates(self) -> Optional[np.ndarray]:
        """Get (lat, lon) array for all plots, or None if no coordinates.

        Missing coordinates are filled with 0 (becomes mean after standardization).
        """
        if not self._roles.has_coordinates:
            return None
        arr = self._header[[self._roles.coords_lat, self._roles.coords_lon]].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0)

    def get_covariates(self) -> Optional[np.ndarray]:
        """Get covariate array if any covariates defined."""
        if not self._roles.covariates:
            return None
        arr = self._header[self._roles.covariates].values.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0)

    def get_target(self, name: str) -> np.ndarray:
        """Get target array by name."""
        if name not in self._targets:
            raise KeyError(f"Unknown target: {name}")
        cfg = self._targets[name]
        values = self._header[cfg.column].values

        if cfg.task == "regression":
            values = values.astype(np.float32)
            if cfg.transform == "log1p":
                values = np.log1p(values)
        else:
            # Classification: encode as integers
            values = pd.Categorical(values).codes.astype(np.int64)

        return values

    def get_target_mask(self, name: str) -> np.ndarray:
        """Get boolean mask for non-null target values."""
        cfg = self._targets[name]
        return ~self._header[cfg.column].isna().values

    def split(
        self,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> tuple[ResolveDataset, ResolveDataset]:
        """
        Split into train and test datasets.

        Splits by plot ID, keeping species rows with their plots.
        """
        plot_ids = self._header[self._roles.plot_id].values
        train_ids, test_ids = train_test_split(
            plot_ids, test_size=test_size, random_state=seed
        )

        train_ids_set = set(train_ids)
        test_ids_set = set(test_ids)

        train_header = self._header[self._header[self._roles.plot_id].isin(train_ids_set)]
        test_header = self._header[self._header[self._roles.plot_id].isin(test_ids_set)]

        train_species = self._species[
            self._species[self._roles.species_plot_id].isin(train_ids_set)
        ]
        test_species = self._species[
            self._species[self._roles.species_plot_id].isin(test_ids_set)
        ]

        train_ds = ResolveDataset(
            train_header, train_species, self._roles, self._targets,
            species_normalization=self._species_normalization,
            track_unknown_fraction=self._track_unknown_fraction,
            track_unknown_count=self._track_unknown_count,
        )
        test_ds = ResolveDataset(
            test_header, test_species, self._roles, self._targets,
            species_normalization=self._species_normalization,
            track_unknown_fraction=self._track_unknown_fraction,
            track_unknown_count=self._track_unknown_count,
        )

        return train_ds, test_ds

    def filter_by_target(self, name: str) -> ResolveDataset:
        """Return dataset filtered to rows with non-null target values."""
        mask = self.get_target_mask(name)
        filtered_header = self._header[mask].copy()
        plot_ids = set(filtered_header[self._roles.plot_id])
        filtered_species = self._species[
            self._species[self._roles.species_plot_id].isin(plot_ids)
        ].copy()
        return ResolveDataset(
            filtered_header, filtered_species, self._roles, self._targets,
            species_normalization=self._species_normalization,
            track_unknown_fraction=self._track_unknown_fraction,
            track_unknown_count=self._track_unknown_count,
        )
