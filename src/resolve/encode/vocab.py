"""Vocabulary building for learned embeddings (species and taxonomy)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class SpeciesVocab:
    """
    Vocabulary mapping for species IDs.

    Index 0 is reserved for unknown/padding.
    Provides mapping from species ID strings to integer indices for nn.Embedding.
    """

    species_to_id: dict[str, int]

    @property
    def n_species(self) -> int:
        """Number of species including unknown."""
        return len(self.species_to_id) + 1

    def encode(self, species_id: Optional[str]) -> int:
        """Encode species ID to integer. Returns 0 for unknown."""
        if species_id is None:
            return 0
        return self.species_to_id.get(str(species_id), 0)

    def encode_batch(self, species_ids: pl.Series) -> np.ndarray:
        """Encode a series of species IDs to integers (vectorized)."""
        mapping = self.species_to_id
        return species_ids.map_elements(
            lambda x: mapping.get(str(x), 0) if x is not None else 0,
            return_dtype=pl.Int64,
        ).to_numpy()

    @classmethod
    def from_species_data(
        cls,
        species_df: pl.DataFrame,
        species_col: str,
        min_count: int = 1,
    ) -> SpeciesVocab:
        """
        Build vocabulary from species data.

        Args:
            species_df: Species occurrence dataframe (polars)
            species_col: Column name for species ID
            min_count: Minimum occurrences to include in vocab (default 1 = all)
        """
        # Count occurrences
        counts = (
            species_df
            .select(pl.col(species_col))
            .drop_nulls()
            .group_by(species_col)
            .len()
        )
        if min_count > 1:
            counts = counts.filter(pl.col("len") >= min_count)

        # Sort alphabetically for deterministic ordering
        species = sorted(str(s) for s in counts[species_col].to_list())
        species_to_id = {s: i + 1 for i, s in enumerate(species)}

        return cls(species_to_id)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump({"species_to_id": self.species_to_id}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SpeciesVocab:
        """Load vocabulary from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(data["species_to_id"])


@dataclass
class TaxonomyVocab:
    """
    Vocabulary mapping for genus and family names.

    Index 0 is reserved for unknown/padding.
    """

    genus_to_id: dict[str, int]
    family_to_id: dict[str, int]

    @property
    def n_genera(self) -> int:
        """Number of genera including unknown."""
        return len(self.genus_to_id) + 1

    @property
    def n_families(self) -> int:
        """Number of families including unknown."""
        return len(self.family_to_id) + 1

    def encode_genus(self, genus: Optional[str]) -> int:
        """Encode genus name to integer ID. Returns 0 for unknown."""
        if genus is None:
            return 0
        return self.genus_to_id.get(genus, 0)

    def encode_family(self, family: Optional[str]) -> int:
        """Encode family name to integer ID. Returns 0 for unknown."""
        if family is None:
            return 0
        return self.family_to_id.get(family, 0)

    @classmethod
    def from_species_data(
        cls,
        species_df: pl.DataFrame,
        genus_col: str,
        family_col: str,
    ) -> TaxonomyVocab:
        """
        Build vocabulary from species data.

        Args:
            species_df: Species occurrence dataframe (polars)
            genus_col: Column name for genus
            family_col: Column name for family
        """
        genera = sorted(species_df[genus_col].drop_nulls().unique().to_list())
        families = sorted(species_df[family_col].drop_nulls().unique().to_list())

        genus_to_id = {g: i + 1 for i, g in enumerate(genera)}
        family_to_id = {f: i + 1 for i, f in enumerate(families)}

        return cls(genus_to_id, family_to_id)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON file."""
        path = Path(path)
        data = {
            "genus_to_id": self.genus_to_id,
            "family_to_id": self.family_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TaxonomyVocab:
        """Load vocabulary from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(data["genus_to_id"], data["family_to_id"])


@dataclass
class CategoricalVocab:
    """
    Vocabulary mapping for a single categorical feature (e.g. ecoregion, country).

    Index 0 is reserved for unknown/unseen categories.
    """

    name: str
    category_to_id: dict[str, int]

    @property
    def n_categories(self) -> int:
        """Number of categories including unknown (index 0)."""
        return len(self.category_to_id) + 1

    def encode(self, value: Optional[str]) -> int:
        """Encode a category string to integer ID. Returns 0 for unknown/None."""
        if value is None:
            return 0
        return self.category_to_id.get(str(value), 0)

    def encode_array(self, series: pl.Series) -> np.ndarray:
        """Encode a polars Series of category strings to integer IDs (vectorized).

        Uses polars native replace() instead of per-element Python lambda.
        For 1.2M rows this is ~100x faster than map_elements.
        """
        str_series = series.cast(pl.Utf8).fill_null("")
        return (
            str_series
            .replace_strict(self.category_to_id, default=0, return_dtype=pl.Int64)
            .to_numpy()
        )

    @classmethod
    def from_series(cls, name: str, series: pl.Series) -> CategoricalVocab:
        """Build vocabulary from a polars Series of category values.

        Assigns 1-based IDs in sorted order for deterministic ordering.
        """
        categories = sorted(str(v) for v in series.drop_nulls().unique().to_list())
        category_to_id = {cat: i + 1 for i, cat in enumerate(categories)}
        return cls(name=name, category_to_id=category_to_id)
