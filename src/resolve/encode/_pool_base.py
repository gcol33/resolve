"""Shared base class for bag-of-species and rank-pool encoders."""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from resolve.data.dataset import ResolveDataset
from resolve.encode.mixins import TaxonomyEncoderMixin
from resolve.encode.normalize import TaxonomyNormalizer, normalize_species_df
from resolve.encode.vocab import SpeciesVocab, TaxonomyVocab


class BasePoolEncoder(TaxonomyEncoderMixin):
    """Base class for pool-style species encoders (bag, rank-pool).

    Subclasses must define:
        VALID_WEIGHTINGS: tuple of allowed weighting modes
    """

    VALID_WEIGHTINGS: tuple[str, ...] = ()

    def __init__(
        self,
        weighting: str = "log1p",
        normalizer: Optional[TaxonomyNormalizer] = None,
        min_species_frequency: int = 1,
    ):
        if weighting not in self.VALID_WEIGHTINGS:
            raise ValueError(
                f"weighting must be one of {self.VALID_WEIGHTINGS}, got {weighting!r}"
            )
        if min_species_frequency < 1:
            raise ValueError(f"min_species_frequency must be >= 1, got {min_species_frequency}")

        self.weighting = weighting
        self.normalizer = normalizer
        self.min_species_frequency = min_species_frequency

        self._species_vocab: Optional[SpeciesVocab] = None
        self._taxonomy_vocab: Optional[TaxonomyVocab] = None
        self._species_to_genus: dict[str, str] = {}
        self._species_to_family: dict[str, str] = {}
        self._known_species: set[str] = set()
        self._fitted = False

    def fit(self, dataset: ResolveDataset) -> BasePoolEncoder:
        """Build species + taxonomy vocabularies from training data."""
        roles = dataset.roles
        species_df = normalize_species_df(self.normalizer, dataset.species, roles)

        # Build species vocabulary (filtered by frequency)
        self._species_vocab = SpeciesVocab.from_species_data(
            species_df,
            roles.species_id,
            min_count=self.min_species_frequency,
        )
        self._known_species = set(self._species_vocab.species_to_id.keys())

        # Build taxonomy vocabulary + species→genus/family lookups
        if roles.has_taxonomy:
            self._taxonomy_vocab = TaxonomyVocab.from_species_data(
                species_df,
                roles.taxonomy_genus,
                roles.taxonomy_family,
            )
            sp_tax = (
                species_df
                .select(roles.species_id, roles.taxonomy_genus, roles.taxonomy_family)
                .drop_nulls()
                .unique(subset=[roles.species_id])
            )
            sp_ids = sp_tax[roles.species_id].cast(pl.Utf8).to_list()
            genera = sp_tax[roles.taxonomy_genus].cast(pl.Utf8).to_list()
            families = sp_tax[roles.taxonomy_family].cast(pl.Utf8).to_list()
            self._species_to_genus = dict(zip(sp_ids, genera))
            self._species_to_family = dict(zip(sp_ids, families))

        self._fitted = True
        return self

    def _encode_species_df(self, dataset: ResolveDataset) -> tuple[pl.DataFrame, str, np.ndarray]:
        """Shared species encoding: ID mapping, taxonomy, weights, unknown fractions.

        Returns (grouped DataFrame, abundance_col name, plot_ids array).
        The DataFrame has columns: _sp_id, _g_id, _f_id, _weight, _total_abd, _total_unk,
        joined and sorted by plot order.
        """
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit before transform")

        roles = dataset.roles
        species_df = normalize_species_df(self.normalizer, dataset.species, roles)
        plot_ids = dataset.plot_ids

        has_taxonomy = roles.has_taxonomy and self._taxonomy_vocab is not None

        # Determine abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1.0).alias("_abundance"))
            abundance_col = "_abundance"

        # Drop null species and clean abundance
        df = df.filter(pl.col(roles.species_id).is_not_null())
        df = df.with_columns(pl.col(abundance_col).fill_null(0).cast(pl.Float32))

        # --- Pre-encode all IDs vectorized ---
        sp_map = self._species_vocab.species_to_id

        df = df.with_columns(
            pl.col(roles.species_id).cast(pl.Utf8).alias("_sp_str")
        )
        df = df.with_columns(
            pl.col("_sp_str").map_elements(
                lambda x: sp_map.get(x, 0), return_dtype=pl.Int64
            ).alias("_sp_id")
        )

        # Genus/Family IDs
        if has_taxonomy:
            genus_map = self._species_to_genus
            family_map = self._species_to_family
            tax_vocab = self._taxonomy_vocab
            g_to_id = tax_vocab.genus_to_id
            f_to_id = tax_vocab.family_to_id

            df = df.with_columns(
                pl.col("_sp_str").map_elements(
                    lambda x: g_to_id.get(genus_map.get(x), 0), return_dtype=pl.Int64
                ).alias("_g_id"),
                pl.col("_sp_str").map_elements(
                    lambda x: f_to_id.get(family_map.get(x), 0), return_dtype=pl.Int64
                ).alias("_f_id"),
            )
        else:
            df = df.with_columns(
                pl.lit(0).alias("_g_id"),
                pl.lit(0).alias("_f_id"),
            )

        # --- Compute weights vectorized ---
        df = self._compute_weights(df, abundance_col, roles)

        # --- Compute unknown fraction vectorized ---
        df = df.with_columns(
            (pl.col("_sp_id") == 0).alias("_is_unknown")
        )
        df = df.with_columns(
            (pl.col(abundance_col) * pl.col("_is_unknown").cast(pl.Float32)).alias("_unk_abd")
        )

        # --- Single group_by: collect all per-plot data as lists ---
        grouped = df.group_by(roles.species_plot_id).agg(
            pl.col("_sp_id"),
            pl.col("_g_id"),
            pl.col("_f_id"),
            pl.col("_weight"),
            pl.col(abundance_col).sum().alias("_total_abd"),
            pl.col("_unk_abd").sum().alias("_total_unk"),
        )

        # --- Left join to plot_ids for correct ordering ---
        plot_ids_df = pl.DataFrame({"_pid": plot_ids, "_order": np.arange(len(plot_ids))})
        result = plot_ids_df.join(
            grouped, left_on="_pid", right_on=roles.species_plot_id, how="left"
        ).sort("_order")

        return result, abundance_col, plot_ids

    def _compute_weights(
        self, df: pl.DataFrame, abundance_col: str, roles: object
    ) -> pl.DataFrame:
        """Compute per-species weights. Override to add weighting modes."""
        if self.weighting == "binary":
            return df.with_columns(pl.lit(1.0).cast(pl.Float32).alias("_weight"))
        elif self.weighting == "abundance":
            return df.with_columns(pl.col(abundance_col).alias("_weight"))
        elif self.weighting == "log1p":
            return df.with_columns(
                pl.col(abundance_col).cast(pl.Float64).log1p().cast(pl.Float32).alias("_weight")
            )
        else:  # norm
            df = df.with_columns(
                pl.col(abundance_col).sum().over(roles.species_plot_id).alias("_plot_total")
            )
            return df.with_columns(
                pl.when(pl.col("_plot_total") > 0)
                .then(pl.col(abundance_col) / pl.col("_plot_total"))
                .otherwise(pl.lit(1.0).cast(pl.Float32))
                .cast(pl.Float32)
                .alias("_weight")
            )


def extract_ragged_arrays(
    result: pl.DataFrame, n_plots: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Extract ragged arrays and unknown fractions from grouped result.

    Returns (all_sp_ids, all_g_ids, all_f_ids, all_weights, unknown_fracs).
    """
    all_sp_ids = []
    all_g_ids = []
    all_f_ids = []
    all_weights = []
    unknown_fracs = np.zeros(n_plots, dtype=np.float32)

    sp_lists = result["_sp_id"].to_list()
    g_lists = result["_g_id"].to_list()
    f_lists = result["_f_id"].to_list()
    w_lists = result["_weight"].to_list()
    total_abd = result["_total_abd"].to_numpy()
    total_unk = result["_total_unk"].to_numpy()

    for i in range(n_plots):
        if sp_lists[i] is None:
            all_sp_ids.append(np.array([], dtype=np.int32))
            all_g_ids.append(np.array([], dtype=np.int32))
            all_f_ids.append(np.array([], dtype=np.int32))
            all_weights.append(np.array([], dtype=np.float32))
        else:
            all_sp_ids.append(np.array(sp_lists[i], dtype=np.int32))
            all_g_ids.append(np.array(g_lists[i], dtype=np.int32))
            all_f_ids.append(np.array(f_lists[i], dtype=np.int32))
            all_weights.append(np.array(w_lists[i], dtype=np.float32))

    total_abd_f = total_abd.astype(np.float64)
    total_unk_f = total_unk.astype(np.float64)
    total_abd_f = np.nan_to_num(total_abd_f, nan=0.0)
    total_unk_f = np.nan_to_num(total_unk_f, nan=0.0)
    np.divide(total_unk_f, total_abd_f, out=unknown_fracs, where=total_abd_f > 0)

    return all_sp_ids, all_g_ids, all_f_ids, all_weights, unknown_fracs


def pad_ragged_encoded(
    species_ids: list[np.ndarray],
    genus_ids: list[np.ndarray],
    family_ids: list[np.ndarray],
    weights: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Pad ragged species/taxonomy/weight arrays to dense arrays.

    Returns dict with species_ids, genus_ids, family_ids, weights, mask.
    """
    n_plots = len(species_ids)
    if n_plots == 0:
        empty = np.zeros((0, 1), dtype=np.int32)
        return {
            "species_ids": empty,
            "genus_ids": empty,
            "family_ids": empty,
            "weights": np.zeros((0, 1), dtype=np.float32),
            "mask": np.zeros((0, 1), dtype=bool),
        }

    max_sp = max(len(s) for s in species_ids)
    max_sp = max(max_sp, 1)

    sp_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    g_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    f_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    w = np.zeros((n_plots, max_sp), dtype=np.float32)
    mask = np.zeros((n_plots, max_sp), dtype=bool)

    for i in range(n_plots):
        n = len(species_ids[i])
        if n > 0:
            sp_ids[i, :n] = species_ids[i]
            g_ids[i, :n] = genus_ids[i]
            f_ids[i, :n] = family_ids[i]
            w[i, :n] = weights[i]
            mask[i, :n] = True

    return {
        "species_ids": sp_ids,
        "genus_ids": g_ids,
        "family_ids": f_ids,
        "weights": w,
        "mask": mask,
    }
