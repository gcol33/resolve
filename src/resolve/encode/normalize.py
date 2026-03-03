"""Taxonomy normalization: map species names to canonical forms via GBIF/WFO backbones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import polars as pl


class TaxonomyNormalizer:
    """Normalizes species names using a taxonomy backbone (GBIF, WFO, or custom).

    Maps original species names to canonical accepted names. Synonyms collapse
    to a single canonical form. Species not in the backbone are kept as-is.

    Usage:
        # From a GBIF cache (produced by normalize_species_gbif.py)
        normalizer = TaxonomyNormalizer.from_gbif_cache("gbif_species_map.json")

        # Normalize a single name
        canonical = normalizer.normalize("Quercus robur L.")

        # Normalize a polars Series (vectorized)
        species_col = normalizer.normalize_series(df["species"])

        # Save/load for checkpoint persistence
        normalizer.save("normalizer.json")
        normalizer = TaxonomyNormalizer.load("normalizer.json")
    """

    def __init__(self, mapping: dict[str, str], backbone: str = "custom"):
        """
        Args:
            mapping: Dict of {original_name: canonical_name}.
                     Identity mappings (name → name) are valid for unresolved species.
            backbone: Source identifier ("gbif", "wfo", or "custom").
        """
        if not isinstance(mapping, dict):
            raise TypeError(f"mapping must be a dict, got {type(mapping).__name__}")
        self._mapping = mapping
        self.backbone = backbone

    def normalize(self, name: str) -> str:
        """Return canonical name, or original if not in mapping."""
        if name is None:
            return name
        return self._mapping.get(name, name)

    def normalize_series(self, series: pl.Series) -> pl.Series:
        """Vectorized normalization of a polars Series via join-based replacement."""
        if not self._mapping:
            return series
        # Build mapping DataFrame and join
        keys = list(self._mapping.keys())
        vals = list(self._mapping.values())
        mapping_df = pl.DataFrame({"_key": keys, "_val": vals})
        # Left join: original values → mapped values, coalesce with original
        name = series.name
        df = pl.DataFrame({name: series})
        result = (
            df.join(mapping_df, left_on=name, right_on="_key", how="left")
            .select(pl.coalesce(pl.col("_val"), pl.col(name)).alias(name))
        )
        return result[name]

    @classmethod
    def from_gbif_cache(cls, path: str | Path) -> TaxonomyNormalizer:
        """Load from GBIF normalization cache (full format with canonical field).

        Supports two formats:
        1. Full cache: {name: {input, canonical, match_type, ...}}
        2. Simple map: {original: canonical}
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Detect format
        first_val = next(iter(data.values())) if data else None
        if isinstance(first_val, dict):
            # Full cache format
            mapping = {}
            for name, info in data.items():
                canonical = info.get("canonical")
                mapping[name] = canonical if canonical else name
        else:
            # Simple {original: canonical} format
            mapping = {k: v if v else k for k, v in data.items()}

        return cls(mapping, backbone="gbif")

    @classmethod
    def _from_simple_json_map(cls, path: str | Path, backbone: str) -> TaxonomyNormalizer:
        """Load from {original: canonical} JSON mapping with given backbone label."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mapping = {k: v if v else k for k, v in data.items()}
        return cls(mapping, backbone=backbone)

    @classmethod
    def from_wfo_cache(cls, path: str | Path) -> TaxonomyNormalizer:
        """Load from WFO normalization cache.

        Expected format: {original_name: canonical_name} JSON.
        """
        return cls._from_simple_json_map(path, backbone="wfo")

    @classmethod
    def from_wfo_backbone(
        cls,
        species_names: list[str],
        backbone_path: str | Path,
        fuzzy: float = 0.0,
        verbose: bool = True,
    ) -> TaxonomyNormalizer:
        """Build normalizer by matching species against a local WFO backbone.

        Requires a downloaded WFO classification.txt file (see WFOBackbone.download).

        Args:
            species_names: Unique species names to match.
            backbone_path: Path to WFO classification.txt.
            fuzzy: Fuzzy matching threshold (0 = exact only, 0.1 = 10% tolerance).
            verbose: Print progress.

        Returns:
            TaxonomyNormalizer with {original_name: accepted_name} mapping.
        """
        from resolve.ext.wfo import WFOBackbone

        backbone = WFOBackbone(backbone_path)
        return backbone.to_normalizer(species_names, fuzzy=fuzzy, verbose=verbose)

    @classmethod
    def from_json(cls, path: str | Path) -> TaxonomyNormalizer:
        """Load from simple {original: canonical} JSON mapping."""
        return cls._from_simple_json_map(path, backbone="custom")

    def save(self, path: str | Path) -> None:
        """Save mapping + backbone metadata to JSON."""
        path = Path(path)
        payload = {
            "backbone": self.backbone,
            "mapping": self._mapping,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> TaxonomyNormalizer:
        """Load normalizer from saved JSON (with backbone metadata)."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(data["mapping"], backbone=data.get("backbone", "custom"))

    @property
    def n_original(self) -> int:
        """Number of original species names in the mapping."""
        return len(self._mapping)

    @property
    def n_canonical(self) -> int:
        """Number of unique canonical names after normalization."""
        return len(set(self._mapping.values()))

    @property
    def n_collapsed(self) -> int:
        """Number of synonyms that collapsed (original - canonical)."""
        return self.n_original - self.n_canonical

    def __repr__(self) -> str:
        return (
            f"TaxonomyNormalizer(backbone={self.backbone!r}, "
            f"n_original={self.n_original:,}, n_canonical={self.n_canonical:,}, "
            f"n_collapsed={self.n_collapsed:,})"
        )


def normalize_species_df(
    normalizer: Optional[TaxonomyNormalizer],
    species_df: pl.DataFrame,
    roles,
) -> pl.DataFrame:
    """Apply taxonomy normalization to species names if normalizer is set.

    Shared helper used by BagOfSpeciesEncoder, EmbeddingEncoder,
    RankPoolEncoder, and SpeciesEncoder.
    """
    if normalizer is None:
        return species_df
    normalized = normalizer.normalize_series(species_df[roles.species_id])
    return species_df.with_columns(normalized.alias(roles.species_id))
