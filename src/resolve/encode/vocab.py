"""Vocabulary building for learned embeddings (species and taxonomy)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


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
        if species_id is None or pd.isna(species_id):
            return 0
        return self.species_to_id.get(str(species_id), 0)

    def encode_batch(self, species_ids: pd.Series) -> np.ndarray:
        """Encode a series of species IDs to integers (vectorized)."""
        return species_ids.map(lambda x: self.species_to_id.get(str(x), 0) if pd.notna(x) else 0).values

    @classmethod
    def from_species_data(
        cls,
        species_df: pd.DataFrame,
        species_col: str,
        min_count: int = 1,
    ) -> SpeciesVocab:
        """
        Build vocabulary from species data.

        Args:
            species_df: Species occurrence dataframe
            species_col: Column name for species ID
            min_count: Minimum occurrences to include in vocab (default 1 = all)
        """
        # Count occurrences
        counts = species_df[species_col].dropna().value_counts()
        if min_count > 1:
            counts = counts[counts >= min_count]

        # Sort alphabetically for deterministic ordering
        species = sorted(str(s) for s in counts.index)
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
        if genus is None or pd.isna(genus):
            return 0
        return self.genus_to_id.get(genus, 0)

    def encode_family(self, family: Optional[str]) -> int:
        """Encode family name to integer ID. Returns 0 for unknown."""
        if family is None or pd.isna(family):
            return 0
        return self.family_to_id.get(family, 0)

    @classmethod
    def from_species_data(
        cls,
        species_df: pd.DataFrame,
        genus_col: str,
        family_col: str,
    ) -> TaxonomyVocab:
        """
        Build vocabulary from species data.

        Args:
            species_df: Species occurrence dataframe
            genus_col: Column name for genus
            family_col: Column name for family
        """
        genera = sorted(species_df[genus_col].dropna().unique())
        families = sorted(species_df[family_col].dropna().unique())

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
