"""Abstract base class for all species encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from resolve.data.dataset import ResolveDataset


def compute_unknown_stats(
    species_df: pl.DataFrame,
    roles,
    plot_ids: np.ndarray,
    known_species: list,
    include_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute per-plot unknown species fraction (and optionally count).

    Shared helper used by SpeciesEncoder and EmbeddingEncoder.

    Parameters
    ----------
    species_df : pl.DataFrame
        Species occurrence data.
    roles : RoleMapping
        Semantic role mapping.
    plot_ids : np.ndarray
        Ordered plot IDs for result alignment.
    known_species : list
        List of known species identifiers.
    include_count : bool
        If True, also return unknown species count per plot.

    Returns
    -------
    unknown_fraction (n_plots,) if include_count is False,
    else (unknown_fraction, unknown_count) tuple.
    """
    if roles.has_abundance:
        abundance_col = roles.abundance
        df = species_df
    else:
        df = species_df.with_columns(pl.lit(1).alias("_abundance"))
        abundance_col = "_abundance"

    df = df.filter(pl.col(roles.species_id).is_not_null())
    df = df.with_columns(pl.col(abundance_col).fill_null(0).cast(pl.Float64))

    # Match types: known_species may be str (polars path) or int (C++ fast loader)
    col_dtype = df[roles.species_id].dtype
    if col_dtype in (pl.Utf8, pl.String):
        known = [str(s) for s in known_species]
        is_in_expr = pl.col(roles.species_id).is_in(known)
    else:
        known = known_species
        is_in_expr = pl.col(roles.species_id).is_in(known)

    df = df.with_columns(
        is_in_expr.not_().alias("_is_unknown"),
    )
    df = df.with_columns(
        (pl.col(abundance_col) * pl.col("_is_unknown").cast(pl.Float64)).alias("_unknown_abundance"),
    )

    agg_exprs = [
        pl.col(abundance_col).sum().alias("total"),
        pl.col("_unknown_abundance").sum().alias("unknown"),
    ]
    if include_count:
        agg_exprs.append(pl.col("_is_unknown").sum().alias("unknown_count"))

    stats = df.group_by(roles.species_plot_id).agg(agg_exprs)

    plot_ids_df = pl.DataFrame({"_pid": plot_ids, "_order": np.arange(len(plot_ids))})
    result = plot_ids_df.join(
        stats, left_on="_pid", right_on=roles.species_plot_id, how="left"
    ).sort("_order").fill_null(0)

    total = result["total"].to_numpy().astype(np.float64)
    unknown = result["unknown"].to_numpy().astype(np.float64)
    unknown_fraction = np.divide(
        unknown, total, out=np.zeros_like(unknown), where=total > 0
    ).astype(np.float32)

    if include_count:
        return unknown_fraction, result["unknown_count"].to_numpy().astype(np.int32)
    return unknown_fraction


class BaseSpeciesEncoder(ABC):
    """Unified interface for species encoding strategies.

    All species encoders (hash, embedding, bag-of-species, rank-pool)
    implement this interface.

    Subclasses must implement:
        fit(): Build vocabularies/state from training data
        transform(): Encode a dataset using fitted state
        state_dict(): Serialize encoder state for checkpointing
        load_state_dict(): Restore encoder state from checkpoint
    """

    _fitted: bool

    @abstractmethod
    def fit(self, dataset: ResolveDataset) -> BaseSpeciesEncoder:
        """Fit encoder to training data (build vocabularies, etc.)."""
        ...

    @abstractmethod
    def transform(self, dataset: ResolveDataset) -> Any:
        """Encode a dataset using fitted state."""
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Serialize encoder state for saving/checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore encoder state from a saved state dict."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called."""
        return self._fitted
