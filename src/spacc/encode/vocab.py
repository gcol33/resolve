"""Taxonomy vocabulary building for learned embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


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
