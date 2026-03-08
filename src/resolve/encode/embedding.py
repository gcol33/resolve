"""Learnable embedding encoder for species and taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
from torch import nn

from resolve.data.dataset import ResolveDataset
from resolve.encode.base import BaseSpeciesEncoder, compute_unknown_stats, select_top_k_by_mode
from resolve.encode.mixins import TaxonomyEncoderMixin
from resolve.encode.normalize import TaxonomyNormalizer, normalize_species_df
from resolve.encode.vocab import SpeciesVocab, TaxonomyVocab


@dataclass
class EmbeddedSpecies:
    """Output of embedding-based species encoding."""

    species_ids: np.ndarray  # (n_plots, top_k_species) - integer IDs
    genus_ids: np.ndarray  # (n_plots, top_k_taxonomy) - integer IDs
    family_ids: np.ndarray  # (n_plots, top_k_taxonomy) - integer IDs
    plot_ids: np.ndarray  # (n_plots,)
    unknown_fraction: np.ndarray  # (n_plots,) fraction from unknown species


class EmbeddingEncoder(BaseSpeciesEncoder, TaxonomyEncoderMixin):
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
        normalizer: Optional[TaxonomyNormalizer] = None,
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
            normalizer: Optional taxonomy normalizer for species name normalization
        """
        if aggregation not in ("abundance", "count"):
            raise ValueError(f"aggregation must be 'abundance' or 'count', got {aggregation!r}")
        if selection not in self.VALID_SELECTIONS:
            raise ValueError(f"selection must be one of {self.VALID_SELECTIONS}, got {selection!r}")

        self.top_k_species = top_k_species
        self.top_k_taxonomy = top_k_taxonomy
        self.aggregation = aggregation
        self.selection = selection
        self.normalizer = normalizer

        self._species_vocab: Optional[SpeciesVocab] = None
        self._taxonomy_vocab: Optional[TaxonomyVocab] = None
        self._fitted = False

    @property
    def species_vocab(self) -> Optional[SpeciesVocab]:
        return self._species_vocab

    @property
    def taxonomy_vocab(self) -> Optional[TaxonomyVocab]:
        return self._taxonomy_vocab

    def fit(self, dataset: ResolveDataset) -> EmbeddingEncoder:
        """
        Build vocabularies from training data.

        Creates mappings for species, genus, and family to integer IDs.
        If a normalizer is set, species names are normalized first.
        """
        roles = dataset.roles
        species_df = normalize_species_df(self.normalizer, dataset.species, roles)

        # Build species vocabulary
        self._species_vocab = SpeciesVocab.from_species_data(
            species_df,
            roles.species_id,
        )

        # Build taxonomy vocabulary if available
        if roles.has_taxonomy:
            self._taxonomy_vocab = TaxonomyVocab.from_species_data(
                species_df,
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
        species_df = normalize_species_df(self.normalizer, dataset.species, roles)
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
        agg_df: pl.DataFrame,
        plot_id_col: str,
        k: int,
    ) -> pl.DataFrame:
        """Select k items per plot based on selection mode."""
        return select_top_k_by_mode(agg_df, plot_id_col, k, self.selection)

    def _extract_top_k(
        self,
        species_df: pl.DataFrame,
        roles,
        plot_ids: np.ndarray,
        group_col: str,
        vocab,
        k: int,
        encode_fn: str = "encode",
    ) -> np.ndarray:
        """Extract k items by selection mode and encode to integer IDs."""
        # Determine weight column
        if self.aggregation == "abundance" and roles.has_abundance:
            weight_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1).alias("_weight"))
            weight_col = "_weight"

        # Aggregate by plot and group_col
        agg = (
            df.group_by([roles.species_plot_id, group_col])
            .agg(pl.col(weight_col).sum().alias("_total"))
        )
        selected = self._select_by_mode(agg, roles.species_plot_id, k)

        # Build ID arrays
        n_plots = len(plot_ids)
        n_items = k * 2 if self.selection == "top_bottom" else k
        ids = np.zeros((n_plots, n_items), dtype=np.int64)
        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}

        # Get encoding function
        encoder = getattr(vocab, encode_fn)

        # Vectorized scatter
        s_pids = selected[roles.species_plot_id].to_list()
        s_taxa = selected[group_col].to_list()
        s_ranks = selected["_rank"].to_numpy()
        for i, (pid, taxon) in enumerate(zip(s_pids, s_taxa)):
            if pid in plot_id_to_idx:
                ids[plot_id_to_idx[pid], s_ranks[i]] = encoder(taxon)

        return ids

    def _compute_unknown_fraction(
        self,
        species_df: pl.DataFrame,
        roles,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute fraction of abundance from unknown species."""
        known_species = list(self._species_vocab.species_to_id.keys())
        return compute_unknown_stats(species_df, roles, plot_ids, known_species)

    def state_dict(self) -> dict[str, Any]:
        """Serialize encoder state for saving/checkpointing."""
        return {
            "top_k_species": self.top_k_species,
            "top_k_taxonomy": self.top_k_taxonomy,
            "aggregation": self.aggregation,
            "selection": self.selection,
            "species_vocab": self._species_vocab.species_to_id if self._species_vocab else None,
            "taxonomy_vocab": self._taxonomy_vocab.state_dict() if self._taxonomy_vocab else None,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore encoder state from a saved state dict."""
        if state.get("species_vocab") is not None:
            self._species_vocab = SpeciesVocab(state["species_vocab"])
        if state.get("taxonomy_vocab") is not None:
            self._taxonomy_vocab = TaxonomyVocab.from_state_dict(state["taxonomy_vocab"])
        self._fitted = state.get("fitted", True)


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
