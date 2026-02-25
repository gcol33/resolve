"""ResolveDataset: validated container for plot data."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split

from resolve.data.roles import RoleMapping, TargetConfig


# Valid normalization modes for species abundance
VALID_NORMALIZATIONS = ("raw", "norm", "log1p")


def _read_csv_with_progress(
    path: Path,
    desc: str = "Loading",
    verbose: bool = True,
) -> pl.DataFrame:
    """Read CSV with progress indicator using polars (multithreaded)."""
    file_size_mb = path.stat().st_size / (1024 * 1024)

    if not verbose:
        return pl.read_csv(path, infer_schema_length=10000, null_values=["NA", ""])

    start = time.time()
    print(f"  {desc}: {path.name} ({file_size_mb:.1f} MB)...", end=" ", flush=True)

    df = pl.read_csv(path, infer_schema_length=10000, null_values=["NA", ""])
    elapsed = time.time() - start
    print(f"done ({len(df):,} rows, {elapsed:.1f}s) [polars]")
    return df


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
    # Categorical feature configuration
    categorical_names: list[str] = None  # type: ignore[assignment]
    categorical_vocab_sizes: dict[str, int] = None  # type: ignore[assignment]
    categorical_embed_dim: int = 8

    def __post_init__(self):
        if self.categorical_names is None:
            self.categorical_names = []
        if self.categorical_vocab_sizes is None:
            self.categorical_vocab_sizes = {}

    @property
    def has_categoricals(self) -> bool:
        return len(self.categorical_names) > 0


class ResolveDataset:
    """
    Validated container for ecological plot data.

    Holds header (plot-level) and species (occurrence) data with
    semantic role mappings. Validates structure and provides
    train/test splitting.
    """

    def __init__(
        self,
        header: pl.DataFrame,
        species: pl.DataFrame,
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

        self._header = header
        self._species = species
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

        # Check categorical columns
        for cat in self._roles.categoricals:
            if cat not in header_cols:
                raise ValueError(f"Categorical '{cat}' not in header")

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
        header_ids = set(self._header[self._roles.plot_id].to_list())
        species_ids = set(self._species[self._roles.species_plot_id].to_list())
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
        verbose: bool = True,
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
            verbose: Show progress during loading (default True)
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

        if verbose:
            print("Loading data...")
        header_df = _read_csv_with_progress(header_path, desc="Header", verbose=verbose)
        species_df = _read_csv_with_progress(species_path, desc="Species", verbose=verbose)

        role_mapping = RoleMapping.from_dict(roles)
        target_configs = {
            name: TargetConfig.from_dict(name, cfg) for name, cfg in targets.items()
        }

        return cls(
            header_df,  # already pl.DataFrame from _read_csv_with_progress
            species_df,
            role_mapping,
            target_configs,
            species_normalization=species_normalization,
            track_unknown_fraction=track_unknown_fraction,
            track_unknown_count=track_unknown_count,
        )

    @classmethod
    def from_fast_csv(
        cls,
        header: str | Path,
        species: str | Path,
        roles: dict[str, str | list[str]],
        targets: dict[str, dict],
        species_normalization: str = "norm",
        track_unknown_fraction: bool = True,
        track_unknown_count: bool = False,
        verbose: bool = True,
    ) -> ResolveDataset:
        """
        Load dataset using C++ fast loader for both header and species data.

        Uses memory-mapped file reading for maximum speed (~10x faster than pandas).

        The C++ loader produces pre-hashed species IDs and COO/CSR format data,
        which is stored directly and used by FastSpeciesEncoder.

        Args:
            header: Path to plot-level CSV (one row per plot)
            species: Path to species CSV (one row per species-plot occurrence)
            roles: Mapping of semantic roles to column names
            targets: Target configurations {name: {column, task, transform?, num_classes?}}
            species_normalization: Abundance normalization mode
            track_unknown_fraction: Track fraction of abundance from unknown species
            track_unknown_count: Track count of unknown species
            verbose: Show progress during loading
        """
        from resolve.csrc.fast_loader import load_grouped_csv, load_header_csv_full

        # Validate file paths
        header_path = Path(header)
        species_path = Path(species)

        if not header_path.exists():
            raise FileNotFoundError(f"Header file not found: {header_path}")
        if not species_path.exists():
            raise FileNotFoundError(f"Species file not found: {species_path}")

        if verbose:
            print("Loading data (C++ fast loader)...")

        # Parse roles to get column names
        role_mapping = RoleMapping.from_dict(roles)

        # Collect all columns needed from header
        # String columns: plot_id + categoricals
        header_string_cols = [role_mapping.plot_id]
        header_string_cols.extend(role_mapping.categoricals)

        # Numeric columns: targets, coords, covariates
        header_numeric_cols = []
        for name, cfg in targets.items():
            header_numeric_cols.append(cfg["column"])
        if role_mapping.has_coordinates:
            header_numeric_cols.extend([role_mapping.coords_lat, role_mapping.coords_lon])
        header_numeric_cols.extend(role_mapping.covariates)

        # Remove duplicates while preserving order
        header_numeric_cols = list(dict.fromkeys(header_numeric_cols))

        # Load header with C++ fast loader
        header_data = load_header_csv_full(
            str(header_path),
            numeric_cols=header_numeric_cols,
            string_cols=header_string_cols,
            verbose=verbose,
        )

        # Build DataFrame from C++ results
        header_dict = {}
        for col in header_string_cols:
            if col in header_data:
                header_dict[col] = header_data[col]
        for col in header_numeric_cols:
            if col in header_data:
                header_dict[col] = header_data[col].numpy()

        header_df = pl.DataFrame(header_dict)

        # Filter out rows where target values are NaN
        original_count = len(header_df)
        filter_exprs = []
        target_cols_filtered = []
        for name, cfg in targets.items():
            col = cfg["column"]
            if col in header_df.columns:
                n_null = header_df[col].null_count()
                if n_null > 0:
                    target_cols_filtered.append(col)
                filter_exprs.append(pl.col(col).is_not_null())

        # Track which rows were filtered for later species filtering
        valid_indices = None
        if filter_exprs:
            combined_mask = filter_exprs[0]
            for expr in filter_exprs[1:]:
                combined_mask = combined_mask & expr
            valid_mask_series = header_df.select(combined_mask.alias("_valid"))["_valid"]
            if valid_mask_series.all():
                pass  # No filtering needed
            else:
                valid_indices = valid_mask_series.to_numpy()
                header_df = header_df.filter(combined_mask)
                filtered_count = original_count - len(header_df)
                pct_remaining = 100 * len(header_df) / original_count
                if verbose:
                    cols_str = ", ".join(target_cols_filtered) if target_cols_filtered else "targets"
                    print(f"  Filtered: {filtered_count:,} rows with NA in {cols_str} ({len(header_df):,} remaining, {pct_remaining:.1f}%)")

        # Collect columns needed from species file
        # Numeric columns: abundance
        species_numeric_cols = []
        if role_mapping.abundance:
            species_numeric_cols.append(role_mapping.abundance)

        # String columns: species_id + taxonomy (genus, family)
        # These will be hashed to int64 for efficient encoding
        species_string_cols = [role_mapping.species_id]
        if role_mapping.has_taxonomy:
            if role_mapping.taxonomy_genus:
                species_string_cols.append(role_mapping.taxonomy_genus)
            if role_mapping.taxonomy_family:
                species_string_cols.append(role_mapping.taxonomy_family)

        # Load species with C++ generic loader
        species_data = load_grouped_csv(
            str(species_path),
            group_id_col=role_mapping.species_plot_id,
            numeric_cols=species_numeric_cols,
            string_cols=species_string_cols,
            hash_string_cols=True,  # Hash species/taxonomy to int64
            verbose=verbose,
        )

        # Build species tensors dict from generic loader output
        # Note: internally we use "plot_" naming for backwards compatibility with Trainer
        species_tensors = {
            "plot_indices": species_data["group_indices"],
            "plot_offsets": species_data["group_offsets"],
            "species_ids": species_data[role_mapping.species_id],
        }
        if role_mapping.abundance and role_mapping.abundance in species_data:
            species_tensors["weights"] = species_data[role_mapping.abundance]
        else:
            # Default weights if no abundance column
            import torch
            species_tensors["weights"] = torch.ones(len(species_data["group_indices"]), dtype=torch.float32)

        # Add taxonomy tensors if present
        if role_mapping.has_taxonomy:
            if role_mapping.taxonomy_genus and role_mapping.taxonomy_genus in species_data:
                species_tensors["genus_ids"] = species_data[role_mapping.taxonomy_genus]
            if role_mapping.taxonomy_family and role_mapping.taxonomy_family in species_data:
                species_tensors["family_ids"] = species_data[role_mapping.taxonomy_family]

        n_records = species_data["_n_records"]
        n_groups = species_data["_n_groups"]

        if verbose:
            print(f"  Species: {n_records:,} records, {n_groups:,} unique groups [C++ tensors]")

        # If header was filtered, filter species data to match
        # The C++ loader creates group indices based on order of plot IDs in species file
        # We need to filter out species records for plots that were removed from header
        # and remap indices to match the filtered header
        if valid_indices is not None:
            import torch

            plot_indices = species_tensors["plot_indices"]

            # Create mapping from old index to new index (or -1 if invalid)
            old_to_new = torch.full((original_count,), -1, dtype=torch.int64)
            new_idx = 0
            for old_idx, is_valid in enumerate(valid_indices):
                if is_valid:
                    old_to_new[old_idx] = new_idx
                    new_idx += 1

            # Filter species records: keep only those whose plot index is valid
            valid_record_mask = old_to_new[plot_indices] >= 0
            n_filtered_records = (~valid_record_mask).sum().item()

            if n_filtered_records > 0:
                # Filter all species tensors
                for key in species_tensors:
                    if key != "plot_offsets":
                        species_tensors[key] = species_tensors[key][valid_record_mask]

                # Remap plot indices to new header row indices
                species_tensors["plot_indices"] = old_to_new[species_tensors["plot_indices"]]

                # Rebuild plot_offsets from filtered data
                new_n_plots = len(header_df)
                plot_indices_np = species_tensors["plot_indices"].numpy()
                new_offsets = np.zeros(new_n_plots + 1, dtype=np.int64)
                np.add.at(new_offsets[1:], plot_indices_np, 1)
                new_offsets = np.cumsum(new_offsets)
                species_tensors["plot_offsets"] = torch.from_numpy(new_offsets)

                if verbose:
                    print(f"  Filtered {n_filtered_records:,} species records for invalid plots")

        # Create a minimal species DataFrame with required columns
        species_dict = {
            role_mapping.species_plot_id: species_tensors["plot_indices"].numpy(),
            role_mapping.species_id: species_tensors["species_ids"].numpy(),
        }
        if role_mapping.abundance:
            species_dict[role_mapping.abundance] = species_tensors["weights"].numpy()
        if role_mapping.has_taxonomy:
            if "genus_ids" in species_tensors:
                species_dict[role_mapping.taxonomy_genus] = species_tensors["genus_ids"].numpy()
            if "family_ids" in species_tensors:
                species_dict[role_mapping.taxonomy_family] = species_tensors["family_ids"].numpy()
        species_df = pl.DataFrame(species_dict)

        # Build target configs
        target_configs = {
            name: TargetConfig.from_dict(name, cfg) for name, cfg in targets.items()
        }

        # Create dataset instance
        instance = cls.__new__(cls)
        instance._header = header_df
        instance._species = species_df
        instance._roles = role_mapping
        instance._targets = target_configs
        instance._species_normalization = species_normalization
        instance._track_unknown_fraction = track_unknown_fraction
        instance._track_unknown_count = track_unknown_count

        # Store the fast-loaded tensors for direct use by FastSpeciesEncoder
        instance._fast_species_tensors = species_tensors

        # Skip full validation since we have tensor-based species data
        # Just validate header requirements
        instance._validate_header_only()

        return instance

    def _validate_header_only(self) -> None:
        """Validate header structure only (for fast-loaded datasets)."""
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

        # Check categorical columns
        for cat in self._roles.categoricals:
            if cat not in header_cols:
                raise ValueError(f"Categorical '{cat}' not in header")

    @property
    def has_fast_species_tensors(self) -> bool:
        """Check if dataset has pre-loaded species tensors from C++ loader."""
        return hasattr(self, '_fast_species_tensors') and self._fast_species_tensors is not None

    @property
    def fast_species_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get pre-loaded species tensors if available."""
        return getattr(self, '_fast_species_tensors', None)

    @property
    def header(self) -> pl.DataFrame:
        """Plot-level data."""
        return self._header

    @property
    def species(self) -> pl.DataFrame:
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
        return self._header[self._roles.plot_id].to_numpy()

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
            n_genera = self._species[self._roles.taxonomy_genus].n_unique()
            n_families = self._species[self._roles.taxonomy_family].n_unique()

        # n_continuous: coordinates (if present) + covariates
        n_coords = 2 if self._roles.has_coordinates else 0
        n_continuous = n_coords + len(self._roles.covariates)

        # Categorical feature vocab sizes
        categorical_names = list(self._roles.categoricals)
        categorical_vocab_sizes = {}
        for cat in categorical_names:
            categorical_vocab_sizes[cat] = self._header[cat].drop_nulls().n_unique() + 1  # +1 for unknown

        return ResolveSchema(
            n_plots=self.n_plots,
            n_species=self._species[self._roles.species_id].n_unique(),
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
            categorical_names=categorical_names,
            categorical_vocab_sizes=categorical_vocab_sizes,
        )

    def get_coordinates(self) -> Optional[np.ndarray]:
        """Get (lat, lon) array for all plots, or None if no coordinates.

        Missing coordinates are filled with 0 (becomes mean after standardization).
        """
        if not self._roles.has_coordinates:
            return None
        arr = (
            self._header
            .select(self._roles.coords_lat, self._roles.coords_lon)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )
        return arr

    def get_covariates(self) -> Optional[np.ndarray]:
        """Get covariate array if any covariates defined."""
        if not self._roles.covariates:
            return None
        arr = (
            self._header
            .select(self._roles.covariates)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float32)
        )
        return arr

    def get_categoricals(self) -> dict[str, pl.Series] | None:
        """Get categorical feature columns as polars Series, or None if none defined."""
        if not self._roles.categoricals:
            return None
        return {
            col: self._header[col]
            for col in self._roles.categoricals
        }

    def get_target(self, name: str) -> np.ndarray:
        """Get target array by name."""
        if name not in self._targets:
            raise KeyError(f"Unknown target: {name}")
        cfg = self._targets[name]
        col = self._header[cfg.column]

        if cfg.task == "regression":
            values = col.cast(pl.Float32, strict=False).to_numpy()
            if cfg.transform == "log1p":
                values = np.log1p(values)
        else:
            # Classification: cast to int directly (column should already be integer-encoded)
            values = col.cast(pl.Int64).to_numpy()

        return values

    def get_target_mask(self, name: str) -> np.ndarray:
        """Get boolean mask for non-null target values."""
        cfg = self._targets[name]
        return self._header[cfg.column].is_not_null().to_numpy()

    def split(
        self,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> tuple[ResolveDataset, ResolveDataset]:
        """
        Split into train and test datasets.

        Splits by plot ID, keeping species rows with their plots.
        """
        plot_ids = self._header[self._roles.plot_id].to_numpy()
        train_ids, test_ids = train_test_split(
            plot_ids, test_size=test_size, random_state=seed
        )

        train_ids_list = train_ids.tolist()
        test_ids_list = test_ids.tolist()

        pid = self._roles.plot_id
        spid = self._roles.species_plot_id

        train_header = self._header.filter(pl.col(pid).is_in(train_ids_list))
        test_header = self._header.filter(pl.col(pid).is_in(test_ids_list))

        train_species = self._species.filter(pl.col(spid).is_in(train_ids_list))
        test_species = self._species.filter(pl.col(spid).is_in(test_ids_list))

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
        cfg = self._targets[name]
        filtered_header = self._header.filter(pl.col(cfg.column).is_not_null())
        plot_ids_list = filtered_header[self._roles.plot_id].to_list()
        filtered_species = self._species.filter(
            pl.col(self._roles.species_plot_id).is_in(plot_ids_list)
        )
        return ResolveDataset(
            filtered_header, filtered_species, self._roles, self._targets,
            species_normalization=self._species_normalization,
            track_unknown_fraction=self._track_unknown_fraction,
            track_unknown_count=self._track_unknown_count,
        )
