"""Fast species encoding using NumPy/sparse operations (no pandas groupby)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from numba import njit, prange
import mmh3  # MurmurHash3 for fast hashing

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


@njit(parallel=True, cache=True)
def _aggregate_hash_embedding(
    plot_indices: np.ndarray,
    hash_indices: np.ndarray,
    signs: np.ndarray,
    weights: np.ndarray,
    n_plots: int,
    hash_dim: int,
) -> np.ndarray:
    """
    Aggregate weighted hash vectors per plot (parallel Numba implementation).

    Args:
        plot_indices: (n_species_rows,) index of plot for each species row
        hash_indices: (n_species_rows,) hash index for each species
        signs: (n_species_rows,) sign (+1 or -1) for each hash
        weights: (n_species_rows,) weight for each species
        n_plots: Total number of plots
        hash_dim: Dimension of hash embedding

    Returns:
        (n_plots, hash_dim) aggregated hash embedding
    """
    result = np.zeros((n_plots, hash_dim), dtype=np.float32)

    # Process each species row
    for i in prange(len(plot_indices)):
        plot_idx = plot_indices[i]
        hash_idx = hash_indices[i]
        sign = signs[i]
        weight = weights[i]
        result[plot_idx, hash_idx] += sign * weight

    return result


@njit(cache=True)
def _compute_hash_index_and_sign(species_hash: np.int64, hash_dim: int) -> tuple:
    """Compute hash index and sign from species hash."""
    # Use lower bits for index, higher bit for sign (like sklearn FeatureHasher)
    idx = np.abs(species_hash) % hash_dim
    sign = 1 if species_hash >= 0 else -1
    return idx, sign


def _hash_species_batch(species_ids: np.ndarray, hash_dim: int) -> tuple:
    """
    Hash species IDs to indices and signs.

    Returns:
        hash_indices: (n,) array of hash indices
        signs: (n,) array of signs (+1 or -1)
    """
    n = len(species_ids)
    hash_indices = np.zeros(n, dtype=np.int32)
    signs = np.zeros(n, dtype=np.int8)

    for i, sp_id in enumerate(species_ids):
        # MurmurHash3 for fast, uniform hashing
        h = mmh3.hash(f"sp={sp_id}")
        hash_indices[i] = abs(h) % hash_dim
        signs[i] = 1 if h >= 0 else -1

    return hash_indices, signs


@njit(parallel=True, cache=True)
def _aggregate_taxonomy_topk(
    plot_indices: np.ndarray,
    taxon_ids: np.ndarray,
    weights: np.ndarray,
    n_plots: int,
    n_taxa: int,
    top_k: int,
) -> np.ndarray:
    """
    Get top-k taxa per plot by weight (parallel Numba implementation).

    Returns:
        (n_plots, top_k) array of taxon IDs
    """
    # First pass: aggregate weights per (plot, taxon)
    agg = np.zeros((n_plots, n_taxa), dtype=np.float32)
    for i in range(len(plot_indices)):
        agg[plot_indices[i], taxon_ids[i]] += weights[i]

    # Second pass: find top-k per plot
    result = np.zeros((n_plots, top_k), dtype=np.int64)
    for p in prange(n_plots):
        # Simple selection sort for top-k (k is small)
        row = agg[p].copy()
        for k in range(top_k):
            best_idx = 0
            best_val = row[0]
            for j in range(1, n_taxa):
                if row[j] > best_val:
                    best_idx = j
                    best_val = row[j]
            result[p, k] = best_idx
            row[best_idx] = -np.inf  # Mark as used

    return result


class FastSpeciesEncoder:
    """
    Fast species encoder using NumPy/Numba operations.

    Significantly faster than pandas-based encoding for large datasets.
    """

    VALID_NORMALIZATIONS = ("raw", "norm", "log1p")

    def __init__(
        self,
        hash_dim: int = 32,
        top_k: int = 3,
        aggregation: str = "abundance",
        normalization: str = "norm",
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
        self._species_vocab: set[str] = set()
        self._fitted = False

    @property
    def vocab(self) -> Optional[TaxonomyVocab]:
        return self._vocab

    @property
    def n_genera(self) -> int:
        return self._vocab.n_genera if self._vocab else 0

    @property
    def n_families(self) -> int:
        return self._vocab.n_families if self._vocab else 0

    @property
    def n_known_species(self) -> int:
        return len(self._species_vocab)

    def fit(self, dataset: ResolveDataset) -> "FastSpeciesEncoder":
        """Build vocabulary from training data."""
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
        """Encode species composition for all plots (fast implementation)."""
        if not self._fitted:
            raise RuntimeError("FastSpeciesEncoder must be fit before transform")

        roles = dataset.roles
        species_df = dataset.species
        plot_ids = dataset.plot_ids
        n_plots = len(plot_ids)

        # Create plot ID to index mapping
        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}

        # Extract arrays from dataframe (avoid pandas operations in hot path)
        sp_plot_ids = species_df[roles.species_plot_id].values
        sp_species_ids = species_df[roles.species_id].values

        # Get abundances
        if roles.has_abundance:
            abundances = species_df[roles.abundance].fillna(0).values.astype(np.float32)
        else:
            abundances = np.ones(len(species_df), dtype=np.float32)

        # Filter out NA species
        valid_mask = ~species_df[roles.species_id].isna().values
        sp_plot_ids = sp_plot_ids[valid_mask]
        sp_species_ids = sp_species_ids[valid_mask]
        abundances = abundances[valid_mask]

        # Convert plot IDs to indices
        plot_indices = np.array([plot_id_to_idx.get(pid, -1) for pid in sp_plot_ids], dtype=np.int32)
        valid_plot_mask = plot_indices >= 0
        plot_indices = plot_indices[valid_plot_mask]
        sp_species_ids = sp_species_ids[valid_plot_mask]
        abundances = abundances[valid_plot_mask]

        # Compute weights based on normalization
        if self.normalization == "raw":
            weights = abundances
        elif self.normalization == "log1p":
            weights = np.log1p(abundances).astype(np.float32)
        elif self.normalization == "norm":
            # Compute plot totals
            plot_totals = np.zeros(n_plots, dtype=np.float32)
            np.add.at(plot_totals, plot_indices, abundances)
            # Normalize
            weights = abundances / np.maximum(plot_totals[plot_indices], 1e-8)

        # Hash species to indices
        hash_indices, signs = _hash_species_batch(sp_species_ids, self.hash_dim)

        # Aggregate hash embedding (Numba-accelerated)
        hash_emb = _aggregate_hash_embedding(
            plot_indices.astype(np.int64),
            hash_indices.astype(np.int64),
            signs.astype(np.float32),
            weights,
            n_plots,
            self.hash_dim,
        )

        # Build taxonomic IDs if available
        genus_ids = None
        family_ids = None
        if roles.has_taxonomy and self._vocab is not None:
            genus_ids, family_ids = self._build_taxonomy_ids_fast(
                species_df, roles, plot_ids, plot_id_to_idx, weights, valid_mask & valid_plot_mask
            )

        # Compute unknown mass
        unknown_fraction, unknown_count = self._compute_unknown_mass_fast(
            sp_species_ids, plot_indices, abundances, n_plots
        )

        return EncodedSpecies(
            hash_embedding=hash_emb,
            genus_ids=genus_ids,
            family_ids=family_ids,
            plot_ids=plot_ids,
            unknown_fraction=unknown_fraction,
            unknown_count=unknown_count if self.track_unknown_count else None,
        )

    def _compute_unknown_mass_fast(
        self,
        species_ids: np.ndarray,
        plot_indices: np.ndarray,
        abundances: np.ndarray,
        n_plots: int,
    ) -> tuple:
        """Compute unknown species mass per plot (vectorized)."""
        # Mark unknown species
        is_unknown = np.array([sp not in self._species_vocab for sp in species_ids], dtype=np.float32)
        unknown_abundances = abundances * is_unknown

        # Aggregate per plot
        total_abundance = np.zeros(n_plots, dtype=np.float32)
        unknown_abundance = np.zeros(n_plots, dtype=np.float32)
        unknown_counts = np.zeros(n_plots, dtype=np.int32)

        np.add.at(total_abundance, plot_indices, abundances)
        np.add.at(unknown_abundance, plot_indices, unknown_abundances)
        np.add.at(unknown_counts, plot_indices, is_unknown.astype(np.int32))

        # Compute fraction
        unknown_fraction = np.divide(
            unknown_abundance, total_abundance,
            out=np.zeros_like(unknown_abundance),
            where=total_abundance > 0
        ).astype(np.float32)

        return unknown_fraction, unknown_counts

    def _build_taxonomy_ids_fast(
        self,
        species_df,
        roles,
        plot_ids: np.ndarray,
        plot_id_to_idx: dict,
        weights: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple:
        """Build top-k genus and family IDs (fast implementation)."""
        n_plots = len(plot_ids)

        # Get taxonomy columns
        genus_col = species_df[roles.taxonomy_genus].values[valid_mask]
        family_col = species_df[roles.taxonomy_family].values[valid_mask]
        plot_col = species_df[roles.species_plot_id].values[valid_mask]

        # Convert to plot indices
        plot_indices = np.array([plot_id_to_idx.get(pid, -1) for pid in plot_col], dtype=np.int32)
        valid = plot_indices >= 0
        plot_indices = plot_indices[valid]
        genus_col = genus_col[valid]
        family_col = family_col[valid]
        weights_valid = weights[valid] if len(weights) == len(valid) else weights

        # Encode genera and families
        genus_ids_enc = np.array([self._vocab.encode_genus(g) for g in genus_col], dtype=np.int64)
        family_ids_enc = np.array([self._vocab.encode_family(f) for f in family_col], dtype=np.int64)

        # Use Numba aggregation for top-k
        genus_result = _aggregate_taxonomy_topk(
            plot_indices.astype(np.int64),
            genus_ids_enc,
            weights_valid,
            n_plots,
            self._vocab.n_genera,
            self.top_k,
        )

        family_result = _aggregate_taxonomy_topk(
            plot_indices.astype(np.int64),
            family_ids_enc,
            weights_valid,
            n_plots,
            self._vocab.n_families,
            self.top_k,
        )

        return genus_result, family_result
