"""Hybrid species encoding: hashing + learned taxonomic embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from spacc.data.dataset import SpaccDataset
from spacc.encode.vocab import TaxonomyVocab


@dataclass
class EncodedSpecies:
    """Output of species encoding for a dataset."""

    hash_embedding: np.ndarray  # (n_plots, hash_dim)
    genus_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    family_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    plot_ids: np.ndarray  # (n_plots,) to match back to dataset


class SpeciesEncoder:
    """
    Encodes species composition per plot.

    Hybrid approach:
        1. Feature hashing for full species list -> fixed-dim embedding
        2. Top-k genera/families by abundance -> integer IDs for learned embeddings

    If abundance is absent, uses presence counts.
    If taxonomy is absent, only produces hash embedding.
    """

    def __init__(
        self,
        hash_dim: int = 32,
        top_k: int = 3,
    ):
        self.hash_dim = hash_dim
        self.top_k = top_k
        self._vocab: Optional[TaxonomyVocab] = None
        self._hasher = FeatureHasher(
            n_features=hash_dim,
            input_type="string",
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

    def fit(self, dataset: SpaccDataset) -> SpeciesEncoder:
        """
        Build vocabulary from training data.

        Only builds taxonomy vocabulary if taxonomy columns are present.
        The hash embedding doesn't need fitting.
        """
        roles = dataset.roles

        if roles.has_taxonomy:
            self._vocab = TaxonomyVocab.from_species_data(
                dataset.species,
                roles.taxonomy_genus,
                roles.taxonomy_family,
            )

        self._fitted = True
        return self

    def transform(self, dataset: SpaccDataset) -> EncodedSpecies:
        """
        Encode species composition for all plots.

        Returns:
            EncodedSpecies with hash_embedding and optionally genus_ids/family_ids
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

        return EncodedSpecies(
            hash_embedding=hash_emb,
            genus_ids=genus_ids,
            family_ids=family_ids,
            plot_ids=plot_ids,
        )

    def _build_hash_embedding(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """Build hash embedding from species lists."""
        # Group species by plot
        grouped = species_df.groupby(roles.species_plot_id)[roles.species_id].agg(list)

        # Align to plot_ids order
        tokens_list = []
        for pid in plot_ids:
            if pid in grouped.index:
                species = grouped[pid]
                tokens = [f"sp={s}" for s in species if pd.notna(s)]
            else:
                tokens = []
            tokens_list.append(tokens)

        # Hash to fixed dimension
        hash_emb = self._hasher.transform(tokens_list).toarray()
        return hash_emb.astype(np.float32)

    def _build_taxonomy_ids(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build top-k genus and family IDs by abundance."""
        # Determine abundance column or use count
        if roles.has_abundance:
            abundance_col = roles.abundance
        else:
            # Add dummy count column
            species_df = species_df.copy()
            species_df["_count"] = 1
            abundance_col = "_count"

        # Top-k genera by total abundance
        genus_agg = (
            species_df.groupby([roles.species_plot_id, roles.taxonomy_genus])[abundance_col]
            .sum()
            .reset_index(name="_total")
        )
        genus_agg = genus_agg.sort_values(
            [roles.species_plot_id, "_total"], ascending=[True, False]
        )
        genus_agg["_rank"] = genus_agg.groupby(roles.species_plot_id).cumcount()
        genus_top = genus_agg[genus_agg["_rank"] < self.top_k]

        # Top-k families by total abundance
        family_agg = (
            species_df.groupby([roles.species_plot_id, roles.taxonomy_family])[abundance_col]
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
