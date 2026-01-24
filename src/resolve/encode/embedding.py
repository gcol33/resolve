"""Learnable embedding encoder for species and taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from resolve.data.dataset import ResolveDataset
from resolve.encode.vocab import SpeciesVocab, TaxonomyVocab


@dataclass
class EmbeddedSpecies:
    """Output of embedding-based species encoding."""

    species_ids: np.ndarray  # (n_plots, top_k_species) - integer IDs
    genus_ids: np.ndarray  # (n_plots, top_k_taxonomy) - integer IDs
    family_ids: np.ndarray  # (n_plots, top_k_taxonomy) - integer IDs
    plot_ids: np.ndarray  # (n_plots,)
    unknown_fraction: np.ndarray  # (n_plots,) fraction from unknown species


class EmbeddingEncoder:
    """
    Encodes species composition using learnable embeddings.

    Unlike hash-based encoding, this learns separate embeddings for each
    species/genus/family that appear in training data. Unknown species
    map to a shared unknown embedding (index 0).

    This approach can capture species-specific patterns but requires
    a fixed vocabulary and won't generalize to unseen species.
    """

    VALID_SELECTIONS = ("top", "bottom", "top_bottom")

    def __init__(
        self,
        top_k_species: int = 10,
        top_k_taxonomy: int = 3,
        aggregation: str = "abundance",
        selection: str = "top",
    ):
        """
        Args:
            top_k_species: Number of top species per plot to track
            top_k_taxonomy: Number of top genera/families per plot to track
            aggregation: How to rank species - "abundance" or "count"
            selection: Which species to select:
                - "top": Most abundant/frequent (default)
                - "bottom": Least abundant/frequent (rarest)
                - "top_bottom": Half top + half bottom
        """
        if aggregation not in ("abundance", "count"):
            raise ValueError(f"aggregation must be 'abundance' or 'count', got {aggregation!r}")
        if selection not in self.VALID_SELECTIONS:
            raise ValueError(f"selection must be one of {self.VALID_SELECTIONS}, got {selection!r}")

        self.top_k_species = top_k_species
        self.top_k_taxonomy = top_k_taxonomy
        self.aggregation = aggregation
        self.selection = selection

        self._species_vocab: Optional[SpeciesVocab] = None
        self._taxonomy_vocab: Optional[TaxonomyVocab] = None
        self._fitted = False

    @property
    def species_vocab(self) -> Optional[SpeciesVocab]:
        return self._species_vocab

    @property
    def taxonomy_vocab(self) -> Optional[TaxonomyVocab]:
        return self._taxonomy_vocab

    @property
    def n_species(self) -> int:
        """Number of species in vocabulary (including unknown)."""
        return self._species_vocab.n_species if self._species_vocab else 0

    @property
    def n_genera(self) -> int:
        """Number of genera in vocabulary (including unknown)."""
        return self._taxonomy_vocab.n_genera if self._taxonomy_vocab else 0

    @property
    def n_families(self) -> int:
        """Number of families in vocabulary (including unknown)."""
        return self._taxonomy_vocab.n_families if self._taxonomy_vocab else 0

    def fit(self, dataset: ResolveDataset) -> EmbeddingEncoder:
        """
        Build vocabularies from training data.

        Creates mappings for species, genus, and family to integer IDs.
        """
        roles = dataset.roles

        # Build species vocabulary
        self._species_vocab = SpeciesVocab.from_species_data(
            dataset.species,
            roles.species_id,
        )

        # Build taxonomy vocabulary if available
        if roles.has_taxonomy:
            self._taxonomy_vocab = TaxonomyVocab.from_species_data(
                dataset.species,
                roles.taxonomy_genus,
                roles.taxonomy_family,
            )

        self._fitted = True
        return self

    def transform(self, dataset: ResolveDataset) -> EmbeddedSpecies:
        """
        Encode species composition as top-k integer IDs.

        Returns:
            EmbeddedSpecies with species_ids, genus_ids, family_ids arrays
        """
        if not self._fitted:
            raise RuntimeError("EmbeddingEncoder must be fit before transform")

        roles = dataset.roles
        species_df = dataset.species
        plot_ids = dataset.plot_ids

        # Extract top-k species IDs
        species_ids = self._extract_top_k(
            species_df, roles, plot_ids,
            group_col=roles.species_id,
            vocab=self._species_vocab,
            k=self.top_k_species,
        )

        # Extract top-k genus IDs
        genus_ids = None
        family_ids = None
        if roles.has_taxonomy and self._taxonomy_vocab:
            genus_ids = self._extract_top_k(
                species_df, roles, plot_ids,
                group_col=roles.taxonomy_genus,
                vocab=self._taxonomy_vocab,
                k=self.top_k_taxonomy,
                encode_fn="encode_genus",
            )
            family_ids = self._extract_top_k(
                species_df, roles, plot_ids,
                group_col=roles.taxonomy_family,
                vocab=self._taxonomy_vocab,
                k=self.top_k_taxonomy,
                encode_fn="encode_family",
            )

        # Compute unknown fraction
        unknown_fraction = self._compute_unknown_fraction(species_df, roles, plot_ids)

        return EmbeddedSpecies(
            species_ids=species_ids,
            genus_ids=genus_ids if genus_ids is not None else np.zeros((len(plot_ids), self.top_k_taxonomy), dtype=np.int64),
            family_ids=family_ids if family_ids is not None else np.zeros((len(plot_ids), self.top_k_taxonomy), dtype=np.int64),
            plot_ids=plot_ids,
            unknown_fraction=unknown_fraction,
        )

    def _select_by_mode(
        self,
        agg_df: pd.DataFrame,
        plot_id_col: str,
        k: int,
    ) -> pd.DataFrame:
        """
        Select k items per plot based on selection mode.

        Args:
            agg_df: DataFrame with plot_id_col and "_total" weight column
            plot_id_col: Name of the plot ID column
            k: Number of items to select per plot

        Returns:
            DataFrame with selected items and "_rank" column (0 to k-1)
        """
        if self.selection == "top":
            # Top-k: highest weights first
            agg_df = agg_df.sort_values([plot_id_col, "_total"], ascending=[True, False])
            agg_df["_rank"] = agg_df.groupby(plot_id_col).cumcount()
            return agg_df[agg_df["_rank"] < k]

        elif self.selection == "bottom":
            # Bottom-k: lowest weights first (rarest species)
            agg_df = agg_df.sort_values([plot_id_col, "_total"], ascending=[True, True])
            agg_df["_rank"] = agg_df.groupby(plot_id_col).cumcount()
            return agg_df[agg_df["_rank"] < k]

        else:  # top_bottom
            # Get K top items + K bottom items (total 2K)
            # Get top items
            agg_top = agg_df.sort_values([plot_id_col, "_total"], ascending=[True, False])
            agg_top["_rank"] = agg_top.groupby(plot_id_col).cumcount()
            top_selected = agg_top[agg_top["_rank"] < k].copy()

            # Get bottom items (excluding already selected)
            top_keys = set(zip(top_selected[plot_id_col], top_selected.iloc[:, 1]))
            agg_bottom = agg_df.copy()
            agg_bottom["_in_top"] = list(zip(agg_bottom[plot_id_col], agg_bottom.iloc[:, 1]))
            agg_bottom["_in_top"] = agg_bottom["_in_top"].isin(top_keys)
            agg_bottom = agg_bottom[~agg_bottom["_in_top"]].drop(columns=["_in_top"])

            # Sort ascending (rarest first)
            agg_bottom = agg_bottom.sort_values([plot_id_col, "_total"], ascending=[True, True])
            agg_bottom["_rank"] = agg_bottom.groupby(plot_id_col).cumcount()
            bottom_selected = agg_bottom[agg_bottom["_rank"] < k].copy()
            # Offset rank to place after top items
            bottom_selected["_rank"] = bottom_selected["_rank"] + k

            # Combine
            return pd.concat([top_selected, bottom_selected], ignore_index=True)

    def _extract_top_k(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
        group_col: str,
        vocab,
        k: int,
        encode_fn: str = "encode",
    ) -> np.ndarray:
        """Extract k items by selection mode and encode to integer IDs."""
        df = species_df.copy()

        # Determine weight column
        if self.aggregation == "abundance" and roles.has_abundance:
            weight_col = roles.abundance
        else:
            df["_weight"] = 1
            weight_col = "_weight"

        # Aggregate by plot and group_col
        agg = (
            df.groupby([roles.species_plot_id, group_col])[weight_col]
            .sum()
            .reset_index(name="_total")
        )
        selected = self._select_by_mode(agg, roles.species_plot_id, k)

        # Build ID arrays
        # For top_bottom mode, we get 2K items (K top + K bottom)
        n_plots = len(plot_ids)
        n_items = k * 2 if self.selection == "top_bottom" else k
        ids = np.zeros((n_plots, n_items), dtype=np.int64)
        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}

        # Get encoding function
        encoder = getattr(vocab, encode_fn)

        for _, row in selected.iterrows():
            pid = row[roles.species_plot_id]
            if pid in plot_id_to_idx:
                idx = plot_id_to_idx[pid]
                rank = int(row["_rank"])
                ids[idx, rank] = encoder(row[group_col])

        return ids

    def _compute_unknown_fraction(
        self,
        species_df: pd.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute fraction of abundance from unknown species."""
        df = species_df.copy()

        if roles.has_abundance:
            weight_col = roles.abundance
        else:
            df["_weight"] = 1
            weight_col = "_weight"

        df = df.dropna(subset=[roles.species_id])
        df[weight_col] = df[weight_col].fillna(0)

        # Mark unknown species
        known_species = set(self._species_vocab.species_to_id.keys())
        df["_is_unknown"] = ~df[roles.species_id].astype(str).isin(known_species)
        df["_unknown_abundance"] = df[weight_col] * df["_is_unknown"].astype(float)

        # Aggregate per plot
        stats = df.groupby(roles.species_plot_id).agg({
            weight_col: "sum",
            "_unknown_abundance": "sum",
        }).rename(columns={weight_col: "total", "_unknown_abundance": "unknown"})

        stats = stats.reindex(plot_ids, fill_value=0)
        total = stats["total"].values
        unknown = stats["unknown"].values

        return np.divide(unknown, total, out=np.zeros_like(unknown), where=total > 0).astype(np.float32)


class SpeciesEmbeddingModule(nn.Module):
    """
    PyTorch module for learnable species embeddings.

    Embeds top-k species, genera, and families using separate embedding
    tables per position slot. This allows the model to learn position-aware
    representations (e.g., most dominant species vs second most dominant).
    """

    def __init__(
        self,
        n_species: int,
        n_genera: int,
        n_families: int,
        species_embed_dim: int = 32,
        taxonomy_embed_dim: int = 8,
        top_k_species: int = 10,
        top_k_taxonomy: int = 3,
    ):
        super().__init__()
        self.top_k_species = top_k_species
        self.top_k_taxonomy = top_k_taxonomy

        # Species embeddings (one table per top-k slot)
        self.species_embeddings = nn.ModuleList([
            nn.Embedding(n_species, species_embed_dim, padding_idx=0)
            for _ in range(top_k_species)
        ])

        # Taxonomy embeddings (one table per slot)
        self.genus_embeddings = nn.ModuleList([
            nn.Embedding(n_genera, taxonomy_embed_dim, padding_idx=0)
            for _ in range(top_k_taxonomy)
        ])
        self.family_embeddings = nn.ModuleList([
            nn.Embedding(n_families, taxonomy_embed_dim, padding_idx=0)
            for _ in range(top_k_taxonomy)
        ])

        self._species_embed_dim = species_embed_dim
        self._taxonomy_embed_dim = taxonomy_embed_dim

    @property
    def output_dim(self) -> int:
        """Total output dimension from all embeddings."""
        return (
            self.top_k_species * self._species_embed_dim +
            self.top_k_taxonomy * self._taxonomy_embed_dim +
            self.top_k_taxonomy * self._taxonomy_embed_dim
        )

    def forward(
        self,
        species_ids: torch.Tensor,
        genus_ids: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            species_ids: (batch, top_k_species) integer IDs
            genus_ids: (batch, top_k_taxonomy) integer IDs
            family_ids: (batch, top_k_taxonomy) integer IDs

        Returns:
            Concatenated embeddings (batch, output_dim)
        """
        # Embed each slot
        sp_embs = [
            emb(species_ids[:, i])
            for i, emb in enumerate(self.species_embeddings)
        ]
        g_embs = [
            emb(genus_ids[:, i])
            for i, emb in enumerate(self.genus_embeddings)
        ]
        f_embs = [
            emb(family_ids[:, i])
            for i, emb in enumerate(self.family_embeddings)
        ]

        return torch.cat(sp_embs + g_embs + f_embs, dim=1)
