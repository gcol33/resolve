"""Rank-pool species encoding with additive hierarchical embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from resolve.data.dataset import ResolveDataset
from resolve.encode._pool_base import BasePoolEncoder, extract_ragged_arrays, pad_ragged_encoded
from resolve.encode.normalize import TaxonomyNormalizer


@dataclass
class RankPoolEncodedSpecies:
    """Output of rank-pool species encoding.

    Each plot has a variable number of species. Arrays are ragged (list of per-plot arrays).
    Use pad_rank_pool_encoded() to convert to padded tensors for batched forward passes.
    """

    species_ids: list[np.ndarray]   # per-plot variable-length species ID arrays
    genus_ids: list[np.ndarray]     # per-plot variable-length genus ID arrays
    family_ids: list[np.ndarray]    # per-plot variable-length family ID arrays
    weights: list[np.ndarray]       # per-plot variable-length weight arrays
    plot_ids: np.ndarray            # (n_plots,)
    unknown_fraction: np.ndarray    # (n_plots,) fraction from unknown species
    has_cover: np.ndarray           # (n_plots,) float32, 1.0 if cover info present


def pad_rank_pool_encoded(encoded: RankPoolEncodedSpecies) -> dict[str, np.ndarray]:
    """Pad ragged rank-pool-encoded data to dense arrays for batched forward."""
    result = pad_ragged_encoded(
        encoded.species_ids, encoded.genus_ids, encoded.family_ids, encoded.weights
    )
    result["has_cover"] = encoded.has_cover
    return result


class RankPoolEncoder(BasePoolEncoder):
    """Encodes all species in a plot using shared embedding tables.

    Unlike SpeciesEncoder (hash, fixed-dim) or EmbeddingEncoder (per-position, top-k),
    this encodes ALL species using shared embedding tables with a weighted mean pool.

    Species identity is preserved (no hashing). Taxonomy provides hierarchical
    regularization via additive embeddings (species_embed + genus_embed + family_embed).

    Weighting modes:
        - "binary": w_i = 1 for all species
        - "abundance": w_i = raw abundance
        - "log1p": w_i = log(1 + abundance)
        - "norm": w_i = abundance / sum(abundances)
        - "rank": w_i = 1/rank based on abundance ordering (dense ranking for ties)
    """

    VALID_WEIGHTINGS = ("binary", "abundance", "log1p", "norm", "rank")

    def __init__(
        self,
        weighting: str = "log1p",
        normalizer: Optional[TaxonomyNormalizer] = None,
        min_species_frequency: int = 1,
    ):
        super().__init__(weighting=weighting, normalizer=normalizer,
                         min_species_frequency=min_species_frequency)

    def fit(self, dataset: ResolveDataset) -> RankPoolEncoder:
        """Build species + taxonomy vocabularies from training data."""
        super().fit(dataset)
        return self

    def _compute_weights(
        self, df: pl.DataFrame, abundance_col: str, roles: object
    ) -> pl.DataFrame:
        """Compute per-species weights, including rank weighting."""
        if self.weighting == "rank":
            df = df.with_columns(
                pl.col(abundance_col)
                .rank(method="dense", descending=True)
                .over(roles.species_plot_id)
                .cast(pl.Float32)
                .alias("_dense_rank")
            )
            return df.with_columns(
                (1.0 / pl.col("_dense_rank")).cast(pl.Float32).alias("_weight")
            )
        return super()._compute_weights(df, abundance_col, roles)

    def transform(self, dataset: ResolveDataset) -> RankPoolEncodedSpecies:
        """Encode all plots as variable-length species pools."""
        result, _, plot_ids = self._encode_species_df(dataset)
        n_plots = len(plot_ids)
        all_sp, all_g, all_f, all_w, unknown_fracs = extract_ragged_arrays(result, n_plots)

        roles = dataset.roles
        has_cover = (
            np.ones(n_plots, dtype=np.float32)
            if roles.has_abundance
            else np.zeros(n_plots, dtype=np.float32)
        )

        return RankPoolEncodedSpecies(
            species_ids=all_sp,
            genus_ids=all_g,
            family_ids=all_f,
            weights=all_w,
            plot_ids=plot_ids,
            unknown_fraction=unknown_fracs,
            has_cover=has_cover,
        )
