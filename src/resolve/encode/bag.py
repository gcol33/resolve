"""Bag-of-species encoding with additive hierarchical embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

from resolve.data.dataset import ResolveDataset
from resolve.encode.normalize import TaxonomyNormalizer
from resolve.encode.vocab import SpeciesVocab, TaxonomyVocab


@dataclass
class BagEncodedSpecies:
    """Output of bag-of-species encoding.

    Each plot has a variable number of species. Arrays are ragged (list of per-plot arrays).
    Use pad_bag_encoded() to convert to padded tensors for batched forward passes.
    """

    species_ids: list[np.ndarray]   # per-plot variable-length species ID arrays
    genus_ids: list[np.ndarray]     # per-plot variable-length genus ID arrays
    family_ids: list[np.ndarray]    # per-plot variable-length family ID arrays
    weights: list[np.ndarray]       # per-plot variable-length weight arrays
    plot_ids: np.ndarray            # (n_plots,)
    unknown_fraction: np.ndarray    # (n_plots,) fraction from unknown species


def pad_bag_encoded(bag: BagEncodedSpecies) -> dict[str, np.ndarray]:
    """Pad ragged bag-encoded data to dense arrays for batched forward.

    Returns dict with:
        species_ids: (n_plots, max_species) int64, padded with 0
        genus_ids: (n_plots, max_species) int64, padded with 0
        family_ids: (n_plots, max_species) int64, padded with 0
        weights: (n_plots, max_species) float32, padded with 0
        mask: (n_plots, max_species) bool, True = valid position
    """
    n_plots = len(bag.species_ids)
    if n_plots == 0:
        empty = np.zeros((0, 1), dtype=np.int32)
        return {
            "species_ids": empty,
            "genus_ids": empty,
            "family_ids": empty,
            "weights": np.zeros((0, 1), dtype=np.float32),
            "mask": np.zeros((0, 1), dtype=bool),
        }

    max_sp = max(len(s) for s in bag.species_ids)
    # Ensure at least 1 column
    max_sp = max(max_sp, 1)

    sp_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    g_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    f_ids = np.zeros((n_plots, max_sp), dtype=np.int32)
    w = np.zeros((n_plots, max_sp), dtype=np.float32)
    mask = np.zeros((n_plots, max_sp), dtype=bool)

    for i in range(n_plots):
        n = len(bag.species_ids[i])
        if n > 0:
            sp_ids[i, :n] = bag.species_ids[i]
            g_ids[i, :n] = bag.genus_ids[i]
            f_ids[i, :n] = bag.family_ids[i]
            w[i, :n] = bag.weights[i]
            mask[i, :n] = True

    return {
        "species_ids": sp_ids,
        "genus_ids": g_ids,
        "family_ids": f_ids,
        "weights": w,
        "mask": mask,
    }


class BagOfSpeciesEncoder:
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
    """

    VALID_WEIGHTINGS = ("binary", "abundance", "log1p", "norm")

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

    @property
    def n_species(self) -> int:
        """Number of species in vocab (including unknown at index 0)."""
        return self._species_vocab.n_species if self._species_vocab else 0

    @property
    def n_genera(self) -> int:
        """Number of genera in vocab (including unknown at index 0)."""
        return self._taxonomy_vocab.n_genera if self._taxonomy_vocab else 0

    @property
    def n_families(self) -> int:
        """Number of families in vocab (including unknown at index 0)."""
        return self._taxonomy_vocab.n_families if self._taxonomy_vocab else 0

    def _normalize_species_df(self, species_df: pl.DataFrame, roles) -> pl.DataFrame:
        """Apply taxonomy normalization to species names if normalizer is set."""
        if self.normalizer is None:
            return species_df
        normalized = self.normalizer.normalize_series(species_df[roles.species_id])
        return species_df.with_columns(normalized.alias(roles.species_id))

    def fit(self, dataset: ResolveDataset) -> BagOfSpeciesEncoder:
        """Build species + taxonomy vocabularies from training data."""
        roles = dataset.roles
        species_df = self._normalize_species_df(dataset.species, roles)

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
            # Build species→genus and species→family lookup
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

    def transform(self, dataset: ResolveDataset) -> BagEncodedSpecies:
        """Encode all plots as variable-length species bags.

        Uses vectorized polars group_by instead of per-plot Python loops.
        """
        if not self._fitted:
            raise RuntimeError("BagOfSpeciesEncoder must be fit before transform")

        roles = dataset.roles
        species_df = self._normalize_species_df(dataset.species, roles)
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
        sp_vocab = self._species_vocab
        sp_map = sp_vocab.species_to_id

        # Species IDs
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
        if self.weighting == "binary":
            df = df.with_columns(pl.lit(1.0).cast(pl.Float32).alias("_weight"))
        elif self.weighting == "abundance":
            df = df.with_columns(pl.col(abundance_col).alias("_weight"))
        elif self.weighting == "log1p":
            df = df.with_columns(
                pl.col(abundance_col).cast(pl.Float64).log1p().cast(pl.Float32).alias("_weight")
            )
        else:  # norm
            df = df.with_columns(
                pl.col(abundance_col).sum().over(roles.species_plot_id).alias("_plot_total")
            )
            df = df.with_columns(
                pl.when(pl.col("_plot_total") > 0)
                .then(pl.col(abundance_col) / pl.col("_plot_total"))
                .otherwise(pl.lit(1.0).cast(pl.Float32))
                .cast(pl.Float32)
                .alias("_weight")
            )

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

        # Extract ragged arrays
        n_plots = len(plot_ids)
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

        # Unknown fraction
        total_abd_f = total_abd.astype(np.float64)
        total_unk_f = total_unk.astype(np.float64)
        # Handle NaN from left join (plots with no species)
        total_abd_f = np.nan_to_num(total_abd_f, nan=0.0)
        total_unk_f = np.nan_to_num(total_unk_f, nan=0.0)
        np.divide(total_unk_f, total_abd_f, out=unknown_fracs, where=total_abd_f > 0)

        return BagEncodedSpecies(
            species_ids=all_sp_ids,
            genus_ids=all_g_ids,
            family_ids=all_f_ids,
            weights=all_weights,
            plot_ids=plot_ids,
            unknown_fraction=unknown_fracs,
        )
