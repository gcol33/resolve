"""Hybrid species encoding: hashing + learned taxonomic embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from resolve.data.dataset import ResolveDataset
from resolve.encode.vocab import TaxonomyVocab


@dataclass
class EncodedSpecies:
    """Output of species encoding for a dataset."""

    hash_embedding: np.ndarray  # (n_plots, hash_dim)
    genus_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    family_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    plot_ids: np.ndarray  # (n_plots,) to match back to dataset
    unknown_fraction: np.ndarray  # (n_plots,) fraction of abundance from unknown species
    unknown_count: Optional[np.ndarray] = None  # (n_plots,) count of unknown species


class SpeciesEncoder:
    """
    Encodes species composition per plot.

    Hybrid approach:
        1. Feature hashing for full species list -> fixed-dim embedding
        2. Top-k genera/families -> integer IDs for learned embeddings

    Design contract:
        Species contributions are aggregated linearly (weighted sum).
        No species-species interactions occur before the PlotEncoder.
        This enforces the inductive bias that species effects are additive
        at the plot level; interactions are learned only at the plot scale.

    Parameters:
        hash_dim: Dimension of the hashed species embedding
        top_k: Number of top genera/families to track
        aggregation: How to select top-k taxonomy ("abundance" or "count")
        normalization: How to weight species contributions:
            - "raw": Use abundance values directly
            - "relative": Normalize to sum to 1 per sample (default)
            - "log": Apply log(1 + abundance) transformation

    If taxonomy is absent, only produces hash embedding.
    """

    VALID_NORMALIZATIONS = ("raw", "relative", "log")

    def __init__(
        self,
        hash_dim: int = 32,
        top_k: int = 3,
        aggregation: str = "abundance",
        normalization: str = "relative",
        track_unknown_count: bool = False,
    ):
        if aggregation not in ("abundance", "count"):
            raise ValueError(f"aggregation must be 'abundance' or 'count', got {aggregation!r}")
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self.VALID_NORMALIZATIONS}, got {normalization!r}"
            )

        self.hash_dim = hash_dim
        self.top_k = top_k
        self.aggregation = aggregation
        self.normalization = normalization
        self.track_unknown_count = track_unknown_count
        self._vocab: Optional[TaxonomyVocab] = None
        self._species_vocab: set[str] = set()  # Known species IDs from training
        # FeatureHasher with dict input allows explicit weighting
        self._hasher = FeatureHasher(
            n_features=hash_dim,
            input_type="dict",
            dtype=np.float32,
            alternate_sign=True,
        )
        self._fitted = False

    @property
    def vocab(self) -> Optional[TaxonomyVocab]:
        """Taxonomy vocabulary (None if no taxonomy columns)."""
        return self._vocab

    @property
    def n_genera(self) -> int:
        """Number of genera in vocabulary (including unknown)."""
        if self._vocab is None:
            return 0
        return self._vocab.n_genera

    @property
    def n_families(self) -> int:
        """Number of families in vocabulary (including unknown)."""
        if self._vocab is None:
            return 0
        return self._vocab.n_families

    @property
    def n_known_species(self) -> int:
        """Number of known species from training."""
        return len(self._species_vocab)

    def fit(self, dataset: ResolveDataset) -> SpeciesEncoder:
        """
        Build vocabulary from training data.

        Tracks:
            - Species IDs seen during training (for unknown mass calculation)
            - Taxonomy vocabulary if taxonomy columns are present
        """
        roles = dataset.roles

        # Track all species IDs seen during training
        species_col = dataset.species[roles.species_id]
        self._species_vocab = set(species_col.dropna().unique())

        if roles.has_taxonomy:
            self._vocab = TaxonomyVocab.from_species_data(
                dataset.species,
                roles.taxonomy_genus,
                roles.taxonomy_family,
            )

        self._fitted = True
        return self

    def transform(self, dataset: ResolveDataset) -> EncodedSpecies:
        """
        Encode species composition for all plots.

        Returns:
            EncodedSpecies with:
                - hash_embedding: weighted species hash
                - genus_ids/family_ids: top-k taxonomy (if available)
                - unknown_fraction: fraction of abundance from unknown species
                - unknown_count: count of unknown species (if track_unknown_count=True)
        """
        if not self._fitted:
            raise RuntimeError("SpeciesEncoder must be fit before transform")

        roles = dataset.roles
        species_df = dataset.species
        plot_ids = dataset.plot_ids

        # Build hash embedding from species list per plot
        hash_emb = self._build_hash_embedding(species_df, roles, plot_ids)

        # Build taxonomic IDs if available
        genus_ids = None
        family_ids = None
        if roles.has_taxonomy and self._vocab is not None:
            genus_ids, family_ids = self._build_taxonomy_ids(
                species_df, roles, plot_ids
            )

        # Compute unknown mass (fraction and optionally count)
        unknown_fraction, unknown_count = self._compute_unknown_mass(
            species_df, roles, plot_ids
        )

        return EncodedSpecies(
            hash_embedding=hash_emb,
            genus_ids=genus_ids,
            family_ids=family_ids,
            plot_ids=plot_ids,
            unknown_fraction=unknown_fraction,
            unknown_count=unknown_count if self.track_unknown_count else None,
        )

    def _compute_unknown_mass(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute unknown species mass per plot.

        Returns:
            unknown_fraction: (n_plots,) fraction of abundance from unknown species
            unknown_count: (n_plots,) count of unknown species per plot
        """
        # Determine abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
        else:
            species_df = species_df.copy()
            species_df["_abundance"] = 1
            abundance_col = "_abundance"

        n_plots = len(plot_ids)
        unknown_fraction = np.zeros(n_plots, dtype=np.float32)
        unknown_count = np.zeros(n_plots, dtype=np.int32)

        for i, pid in enumerate(plot_ids):
            plot_species = species_df[species_df[roles.species_plot_id] == pid]
            if len(plot_species) == 0:
                continue

            total_abundance = 0.0
            unknown_abundance = 0.0
            n_unknown = 0

            for _, row in plot_species.iterrows():
                species_id = row[roles.species_id]
                abundance = row[abundance_col] if pd.notna(row[abundance_col]) else 0

                if pd.isna(species_id):
                    continue

                total_abundance += abundance

                # Check if species is unknown (not seen during training)
                if species_id not in self._species_vocab:
                    unknown_abundance += abundance
                    n_unknown += 1

            # Compute fraction (avoid division by zero)
            if total_abundance > 0:
                unknown_fraction[i] = unknown_abundance / total_abundance
            unknown_count[i] = n_unknown

        return unknown_fraction, unknown_count

    def _normalize_weights(
        self,
        abundances: np.ndarray,
        plot_totals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply normalization to abundance weights.

        Args:
            abundances: Raw abundance values
            plot_totals: Optional pre-computed plot totals for relative normalization

        Returns:
            Normalized weights according to self.normalization setting
        """
        if self.normalization == "raw":
            return abundances.astype(np.float32)
        elif self.normalization == "log":
            return np.log1p(abundances).astype(np.float32)
        elif self.normalization == "relative":
            if plot_totals is None:
                # Single-value case, return as-is (will be normalized at plot level)
                return abundances.astype(np.float32)
            # Avoid division by zero
            totals = np.where(plot_totals > 0, plot_totals, 1.0)
            return (abundances / totals).astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def _build_hash_embedding(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Build hash embedding from species composition.

        Uses weighted feature hashing where weights are determined by
        the normalization setting. This implements linear aggregation:
        z_species = Σ w_i * h(species_i)
        """
        # Determine abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
        else:
            # Use count of 1 for each species
            species_df = species_df.copy()
            species_df["_abundance"] = 1
            abundance_col = "_abundance"

        # Compute plot totals for relative normalization
        plot_totals = None
        if self.normalization == "relative":
            plot_totals = species_df.groupby(roles.species_plot_id)[abundance_col].sum()

        # Build weighted token dicts per plot
        weighted_tokens_list = []
        for pid in plot_ids:
            plot_species = species_df[species_df[roles.species_plot_id] == pid]
            if len(plot_species) == 0:
                weighted_tokens_list.append({})
                continue

            token_weights = {}
            plot_total = plot_totals[pid] if plot_totals is not None else None

            for _, row in plot_species.iterrows():
                species_id = row[roles.species_id]
                if pd.isna(species_id):
                    continue
                raw_abundance = row[abundance_col]
                weight = self._normalize_weights(
                    np.array([raw_abundance]),
                    np.array([plot_total]) if plot_total is not None else None,
                )[0]
                token = f"sp={species_id}"
                token_weights[token] = token_weights.get(token, 0) + weight

            weighted_tokens_list.append(token_weights)

        # Hash to fixed dimension (linear aggregation via weighted sum)
        hash_emb = self._hasher.transform(weighted_tokens_list).toarray()
        return hash_emb.astype(np.float32)

    def _build_taxonomy_ids(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build top-k genus and family IDs using configured aggregation and normalization.

        Normalization is applied consistently with hash embedding to ensure
        the same species weighting logic across both pathways.
        """
        species_df = species_df.copy()

        # Determine base abundance column
        if self.aggregation == "abundance" and roles.has_abundance:
            abundance_col = roles.abundance
        else:
            species_df["_count"] = 1
            abundance_col = "_count"

        # Apply normalization to weights (consistent with hash embedding)
        if self.normalization == "log":
            species_df["_weight"] = np.log1p(species_df[abundance_col].values)
        elif self.normalization == "relative":
            plot_totals = species_df.groupby(roles.species_plot_id)[abundance_col].transform("sum")
            plot_totals = np.where(plot_totals > 0, plot_totals, 1.0)
            species_df["_weight"] = species_df[abundance_col] / plot_totals
        else:  # raw
            species_df["_weight"] = species_df[abundance_col]

        weight_col = "_weight"

        # Top-k genera by normalized weight
        genus_agg = (
            species_df.groupby([roles.species_plot_id, roles.taxonomy_genus])[weight_col]
            .sum()
            .reset_index(name="_total")
        )
        genus_agg = genus_agg.sort_values(
            [roles.species_plot_id, "_total"], ascending=[True, False]
        )
        genus_agg["_rank"] = genus_agg.groupby(roles.species_plot_id).cumcount()
        genus_top = genus_agg[genus_agg["_rank"] < self.top_k]

        # Top-k families by normalized weight
        family_agg = (
            species_df.groupby([roles.species_plot_id, roles.taxonomy_family])[weight_col]
            .sum()
            .reset_index(name="_total")
        )
        family_agg = family_agg.sort_values(
            [roles.species_plot_id, "_total"], ascending=[True, False]
        )
        family_agg["_rank"] = family_agg.groupby(roles.species_plot_id).cumcount()
        family_top = family_agg[family_agg["_rank"] < self.top_k]

        # Build ID arrays aligned to plot_ids
        n_plots = len(plot_ids)
        genus_ids = np.zeros((n_plots, self.top_k), dtype=np.int64)
        family_ids = np.zeros((n_plots, self.top_k), dtype=np.int64)

        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}

        for _, row in genus_top.iterrows():
            pid = row[roles.species_plot_id]
            if pid in plot_id_to_idx:
                idx = plot_id_to_idx[pid]
                rank = int(row["_rank"])
                genus_ids[idx, rank] = self._vocab.encode_genus(row[roles.taxonomy_genus])

        for _, row in family_top.iterrows():
            pid = row[roles.species_plot_id]
            if pid in plot_id_to_idx:
                idx = plot_id_to_idx[pid]
                rank = int(row["_rank"])
                family_ids[idx, rank] = self._vocab.encode_family(row[roles.taxonomy_family])

        return genus_ids, family_ids

    def save(self, path: str) -> None:
        """Save encoder state (vocabulary)."""
        if self._vocab is not None:
            self._vocab.save(path)

    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        self._vocab = TaxonomyVocab.load(path)
        self._fitted = True
