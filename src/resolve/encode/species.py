"""Hybrid species encoding: hashing + learned taxonomic embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.feature_extraction import FeatureHasher

from resolve.data.dataset import ResolveDataset
from resolve.encode.normalize import TaxonomyNormalizer
from resolve.encode.vocab import TaxonomyVocab

if TYPE_CHECKING:
    from resolve.data.roles import RoleMapping


@dataclass
class EncodedSpecies:
    """Output of species encoding for a dataset."""

    hash_embedding: Optional[np.ndarray]  # (n_plots, hash_dim) or None for all/presence_absence
    genus_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    family_ids: Optional[np.ndarray]  # (n_plots, top_k) or None
    plot_ids: np.ndarray  # (n_plots,) to match back to dataset
    unknown_fraction: np.ndarray  # (n_plots,) fraction of abundance from unknown species
    unknown_count: Optional[np.ndarray] = None  # (n_plots,) count of unknown species
    species_vector: Optional[np.ndarray] = None  # (n_plots, n_species) for all/presence_absence modes


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
        hash_dim: Dimension of the hashed species embedding (ignored for all/presence_absence)
        top_k: Number of top genera/families to track
        aggregation: How to select top-k taxonomy ("abundance" or "count")
        normalization: How to weight species contributions:
            - "raw": Use abundance values directly
            - "norm": Normalize to sum to 1 per sample (default)
            - "log1p": Apply log(1 + abundance) transformation
        selection: Which species to include:
            - "top": Top-K most abundant/frequent (default, uses hash embedding)
            - "bottom": Bottom-K least abundant/frequent (uses hash embedding)
            - "top_bottom": Top-K + Bottom-K (2K total, uses hash embedding)
            - "all": All species (explicit vector, see representation param)
        representation: How to represent species (only for selection="all"):
            - "abundance": Weighted by abundance (default)
            - "presence_absence": Binary 0/1
        min_species_frequency: For selection="all", only include species in N+ plots

    If taxonomy is absent, only produces hash embedding (or species_vector for selection="all").
    """

    VALID_NORMALIZATIONS = ("raw", "norm", "log1p")
    VALID_SELECTIONS = ("top", "bottom", "top_bottom", "all")
    VALID_REPRESENTATIONS = ("abundance", "presence_absence")

    def __init__(
        self,
        hash_dim: int = 32,
        top_k: int = 3,
        aggregation: str = "abundance",
        normalization: str = "norm",
        track_unknown_count: bool = False,
        selection: str = "top",
        representation: str = "abundance",
        min_species_frequency: int = 1,
        normalizer: Optional[TaxonomyNormalizer] = None,
    ):
        if aggregation not in ("abundance", "count"):
            raise ValueError(f"aggregation must be 'abundance' or 'count', got {aggregation!r}")
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self.VALID_NORMALIZATIONS}, got {normalization!r}"
            )
        if selection not in self.VALID_SELECTIONS:
            raise ValueError(
                f"selection must be one of {self.VALID_SELECTIONS}, got {selection!r}"
            )
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {self.VALID_REPRESENTATIONS}, got {representation!r}"
            )
        if min_species_frequency < 1:
            raise ValueError(f"min_species_frequency must be >= 1, got {min_species_frequency}")

        self.hash_dim = hash_dim
        self.top_k = top_k
        self.aggregation = aggregation
        self.normalization = normalization
        self.track_unknown_count = track_unknown_count
        self.selection = selection
        self.representation = representation
        self.min_species_frequency = min_species_frequency
        self._vocab: Optional[TaxonomyVocab] = None
        self._species_vocab: set[str] = set()  # Known species IDs from training
        self._species_to_idx: dict[str, int] = {}  # For selection="all" mode
        # FeatureHasher with dict input allows explicit weighting
        self.normalizer = normalizer
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

    @property
    def n_species_vector(self) -> int:
        """Number of species in the explicit vector (for all/presence_absence modes)."""
        return len(self._species_to_idx)

    @property
    def uses_explicit_vector(self) -> bool:
        """Whether this encoder uses explicit species vectors (selection='all')."""
        return self.selection == "all"

    @property
    def n_taxonomy_slots(self) -> int:
        """Number of taxonomy slots per plot (2*top_k for top_bottom, else top_k)."""
        return self.top_k * 2 if self.selection == "top_bottom" else self.top_k

    def fit(self, dataset: ResolveDataset) -> SpeciesEncoder:
        """
        Build vocabulary from training data.

        Tracks:
            - Species IDs seen during training (for unknown mass calculation)
            - Taxonomy vocabulary if taxonomy columns are present
            - For all/presence_absence: species-to-index mapping filtered by frequency

        If a normalizer is set, species names are normalized before building vocab.
        """
        roles = dataset.roles

        # Track all species IDs seen during training (after normalization)
        species_col = dataset.species[roles.species_id]
        if self.normalizer is not None:
            species_col = self.normalizer.normalize_series(species_col)
        self._species_vocab = set(species_col.drop_nulls().unique().to_list())

        # For all/presence_absence modes, build filtered species vocabulary
        if self.uses_explicit_vector:
            # Count species occurrences across plots
            species_plot = dataset.species.select(roles.species_plot_id, roles.species_id).unique()
            species_counts = species_plot.group_by(roles.species_id).len()

            # Filter by minimum frequency
            valid_species = (
                species_counts
                .filter(pl.col("len") >= self.min_species_frequency)
                [roles.species_id]
                .to_list()
            )
            self._species_to_idx = {sp: idx for idx, sp in enumerate(sorted(valid_species))}

        if roles.has_taxonomy:
            self._vocab = TaxonomyVocab.from_species_data(
                dataset.species,
                roles.taxonomy_genus,
                roles.taxonomy_family,
            )

        self._fitted = True
        return self

    def _normalize_species_df(self, species_df: pl.DataFrame, roles) -> pl.DataFrame:
        """Apply taxonomy normalization to species names if normalizer is set."""
        if self.normalizer is None:
            return species_df
        normalized = self.normalizer.normalize_series(species_df[roles.species_id])
        return species_df.with_columns(normalized.alias(roles.species_id))

    def transform(self, dataset: ResolveDataset) -> EncodedSpecies:
        """
        Encode species composition for all plots.

        Returns:
            EncodedSpecies with:
                - hash_embedding: weighted species hash (None for all/presence_absence)
                - species_vector: explicit abundance/binary vector (for all/presence_absence)
                - genus_ids/family_ids: top-k taxonomy (if available)
                - unknown_fraction: fraction of abundance from unknown species
                - unknown_count: count of unknown species (if track_unknown_count=True)
        """
        if not self._fitted:
            raise RuntimeError("SpeciesEncoder must be fit before transform")

        roles = dataset.roles
        species_df = self._normalize_species_df(dataset.species, roles)
        plot_ids = dataset.plot_ids

        # Build species representation based on selection mode
        hash_emb = None
        species_vector = None

        if self.uses_explicit_vector:
            # all/presence_absence: build explicit species vector
            species_vector = self._build_species_vector(species_df, roles, plot_ids)
        else:
            # top/bottom/top_bottom: build hash embedding
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
            species_vector=species_vector,
        )

    def _compute_unknown_mass(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute unknown species mass per plot (VECTORIZED).

        Returns:
            unknown_fraction: (n_plots,) fraction of abundance from unknown species
            unknown_count: (n_plots,) count of unknown species per plot
        """
        # Determine abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1).alias("_abundance"))
            abundance_col = "_abundance"

        # Drop rows with null species, fill null abundance
        df = df.filter(pl.col(roles.species_id).is_not_null())
        df = df.with_columns(pl.col(abundance_col).fill_null(0).cast(pl.Float64))

        # Mark unknown species + compute unknown abundance (vectorized)
        known_species = list(self._species_vocab)
        df = df.with_columns(
            pl.col(roles.species_id).is_in(known_species).not_().alias("_is_unknown"),
        )
        df = df.with_columns(
            (pl.col(abundance_col) * pl.col("_is_unknown").cast(pl.Float64)).alias("_unknown_abundance"),
        )

        # Aggregate per plot
        plot_stats = df.group_by(roles.species_plot_id).agg(
            pl.col(abundance_col).sum().alias("total_abundance"),
            pl.col("_unknown_abundance").sum().alias("unknown_abundance"),
            pl.col("_is_unknown").sum().alias("unknown_count"),
        )

        # Left join to plot_ids to ensure correct ordering + fill missing
        plot_ids_df = pl.DataFrame({"_pid": plot_ids, "_order": np.arange(len(plot_ids))})
        result = plot_ids_df.join(
            plot_stats, left_on="_pid", right_on=roles.species_plot_id, how="left"
        ).sort("_order").fill_null(0)

        total = result["total_abundance"].to_numpy().astype(np.float64)
        unknown = result["unknown_abundance"].to_numpy().astype(np.float64)
        unknown_fraction = np.divide(unknown, total, out=np.zeros_like(unknown), where=total > 0).astype(np.float32)
        unknown_count = result["unknown_count"].to_numpy().astype(np.int32)

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
        elif self.normalization == "log1p":
            return np.log1p(abundances).astype(np.float32)
        elif self.normalization == "norm":
            if plot_totals is None:
                # Single-value case, return as-is (will be norm at plot level)
                return abundances.astype(np.float32)
            # Avoid division by zero
            totals = np.where(plot_totals > 0, plot_totals, 1.0)
            return (abundances / totals).astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

    def _build_hash_embedding(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Build hash embedding from species composition (VECTORIZED).

        Uses weighted feature hashing where weights are determined by
        the normalization setting. This implements linear aggregation:
        z_species = Σ w_i * h(species_i)
        """
        # Determine abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1).alias("_abundance"))
            abundance_col = "_abundance"

        # Drop rows with null species
        df = df.filter(pl.col(roles.species_id).is_not_null())

        # Compute weights based on normalization
        if self.normalization == "raw":
            df = df.with_columns(
                pl.col(abundance_col).fill_null(0).cast(pl.Float32).alias("_weight")
            )
        elif self.normalization == "log1p":
            df = df.with_columns(
                pl.col(abundance_col).fill_null(0).cast(pl.Float64).log1p().cast(pl.Float32).alias("_weight")
            )
        elif self.normalization == "norm":
            df = df.with_columns(
                pl.col(abundance_col).fill_null(0).cast(pl.Float64).alias("_abd_clean")
            )
            df = df.with_columns(
                pl.col("_abd_clean").sum().over(roles.species_plot_id).alias("_plot_total")
            )
            df = df.with_columns(
                pl.when(pl.col("_plot_total") > 0)
                .then(pl.col("_abd_clean") / pl.col("_plot_total"))
                .otherwise(pl.col("_abd_clean"))
                .cast(pl.Float32)
                .alias("_weight")
            )

        # Create token strings
        df = df.with_columns(
            (pl.lit("sp=") + pl.col(roles.species_id).cast(pl.Utf8)).alias("_token")
        )

        # Group by plot, collect token-weight pairs as lists
        grouped = df.group_by(roles.species_plot_id).agg(
            pl.col("_token"), pl.col("_weight")
        )

        # Build dict mapping plot_id → {token: weight}
        plot_dict_map = {}
        pids = grouped[roles.species_plot_id].to_list()
        tokens_lists = grouped["_token"].to_list()
        weight_lists = grouped["_weight"].to_list()
        for pid, tokens, weights in zip(pids, tokens_lists, weight_lists):
            plot_dict_map[pid] = dict(zip(tokens, weights))

        # Build list aligned to plot_ids order
        weighted_tokens_list = [plot_dict_map.get(pid, {}) for pid in plot_ids]

        # Hash to fixed dimension (linear aggregation via weighted sum)
        hash_emb = self._hasher.transform(weighted_tokens_list).toarray()
        return hash_emb.astype(np.float32)

    def _build_species_vector(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Build explicit species vector for selection='all' mode.

        Dispatches to appropriate method based on representation.
        """
        if self.representation == "presence_absence":
            return self._build_presence_absence_vector(species_df, roles, plot_ids)
        else:
            return self._build_abundance_vector(species_df, roles, plot_ids)

    def _prepare_species_matrix_indices(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> tuple[pl.DataFrame, int, int, dict]:
        """
        Prepare DataFrame with row/col indices for sparse matrix construction.

        Returns:
            (df, n_plots, n_species, plot_id_to_idx) - filtered df with _row_idx, _col_idx
        """
        n_plots = len(plot_ids)
        n_species = len(self._species_to_idx)

        df = species_df.filter(pl.col(roles.species_id).is_not_null())

        # Map species to column indices (unknown species get -1)
        sp_map = self._species_to_idx
        df = df.with_columns(
            pl.col(roles.species_id).map_elements(
                lambda x: sp_map.get(x, -1), return_dtype=pl.Int64
            ).alias("_col_idx")
        )
        df = df.filter(pl.col("_col_idx") >= 0)

        # Map plots to row indices
        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}
        pid_map = plot_id_to_idx
        df = df.with_columns(
            pl.col(roles.species_plot_id).map_elements(
                lambda x: pid_map.get(x, -1), return_dtype=pl.Int64
            ).alias("_row_idx")
        )
        df = df.filter(pl.col("_row_idx") >= 0)

        return df, n_plots, n_species, plot_id_to_idx

    def _build_presence_absence_vector(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """Build binary presence/absence matrix (n_plots, n_species)."""
        from scipy import sparse

        df, n_plots, n_species, _ = self._prepare_species_matrix_indices(
            species_df, roles, plot_ids
        )

        if len(df) == 0:
            return np.zeros((n_plots, n_species), dtype=np.float32)

        data = np.ones(len(df), dtype=np.float32)
        matrix = sparse.coo_matrix(
            (data, (df["_row_idx"].to_numpy(), df["_col_idx"].to_numpy())),
            shape=(n_plots, n_species),
            dtype=np.float32,
        )
        return matrix.toarray()

    def _build_abundance_vector(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> np.ndarray:
        """Build abundance-weighted species matrix (n_plots, n_species)."""
        from scipy import sparse

        # Get abundance column
        if roles.has_abundance:
            abundance_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1.0).alias("_abundance"))
            abundance_col = "_abundance"
        df = df.with_columns(pl.col(abundance_col).fill_null(0).cast(pl.Float32))

        df, n_plots, n_species, _ = self._prepare_species_matrix_indices(
            df, roles, plot_ids
        )

        if len(df) == 0:
            return np.zeros((n_plots, n_species), dtype=np.float32)

        matrix = sparse.coo_matrix(
            (df[abundance_col].to_numpy(), (df["_row_idx"].to_numpy(), df["_col_idx"].to_numpy())),
            shape=(n_plots, n_species),
            dtype=np.float32,
        ).toarray()

        # Apply normalization
        if self.normalization == "log1p":
            matrix = np.log1p(matrix)
        elif self.normalization == "norm":
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            matrix = matrix / row_sums

        return matrix.astype(np.float32)

    def _select_by_mode(
        self,
        agg_df: pl.DataFrame,
        plot_id_col: str,
        k: int,
        mode: str = None,
    ) -> pl.DataFrame:
        """
        Select k items per plot based on selection mode.

        Args:
            agg_df: DataFrame with plot_id_col and "_total" weight column
            plot_id_col: Name of the plot ID column
            k: Number of items to select per plot
            mode: Selection mode override (defaults to self.selection for
                  top/bottom/top_bottom, or "top" for all/presence_absence)

        Returns:
            DataFrame with selected items and "_rank" column (0 to k-1, or 0 to 2k-1 for top_bottom)
        """
        if mode is None:
            # For all/presence_absence, default taxonomy selection to "top"
            mode = self.selection if self.selection in ("top", "bottom", "top_bottom") else "top"

        # Get the taxonomy column name (second column, after plot_id_col)
        tax_col = [c for c in agg_df.columns if c not in (plot_id_col, "_total")][0]

        if mode == "top":
            ranked = (
                agg_df.sort([plot_id_col, "_total"], descending=[False, True])
                .with_columns(
                    pl.col("_total").cum_count().over(plot_id_col).alias("_rank") - 1
                )
            )
            return ranked.filter(pl.col("_rank") < k)

        elif mode == "bottom":
            ranked = (
                agg_df.sort([plot_id_col, "_total"], descending=[False, False])
                .with_columns(
                    pl.col("_total").cum_count().over(plot_id_col).alias("_rank") - 1
                )
            )
            return ranked.filter(pl.col("_rank") < k)

        else:  # top_bottom
            # Top-k
            top_ranked = (
                agg_df.sort([plot_id_col, "_total"], descending=[False, True])
                .with_columns(
                    pl.col("_total").cum_count().over(plot_id_col).alias("_rank") - 1
                )
            )
            top_selected = top_ranked.filter(pl.col("_rank") < k)

            # Exclude top items from bottom selection via anti-join
            bottom_pool = agg_df.join(
                top_selected.select(plot_id_col, tax_col),
                on=[plot_id_col, tax_col],
                how="anti",
            )
            bottom_ranked = (
                bottom_pool.sort([plot_id_col, "_total"], descending=[False, False])
                .with_columns(
                    pl.col("_total").cum_count().over(plot_id_col).alias("_rank") - 1
                )
            )
            bottom_selected = bottom_ranked.filter(pl.col("_rank") < k).with_columns(
                (pl.col("_rank") + k).alias("_rank")
            )

            return pl.concat([top_selected, bottom_selected])

    def _build_taxonomy_ids(
        self,
        species_df: pl.DataFrame,
        roles: RoleMapping,
        plot_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build top-k genus and family IDs using configured aggregation, normalization, and selection.

        Selection modes:
            - "top": Most abundant/frequent (default)
            - "bottom": Least abundant/frequent (rarest)
            - "top_bottom": Half top + half bottom

        Normalization is applied consistently with hash embedding to ensure
        the same species weighting logic across both pathways.
        """
        # Determine base abundance column
        if self.aggregation == "abundance" and roles.has_abundance:
            abundance_col = roles.abundance
            df = species_df
        else:
            df = species_df.with_columns(pl.lit(1).alias("_count"))
            abundance_col = "_count"

        # Apply normalization to weights (consistent with hash embedding)
        if self.normalization == "log1p":
            df = df.with_columns(
                pl.col(abundance_col).cast(pl.Float64).log1p().alias("_weight")
            )
        elif self.normalization == "norm":
            df = df.with_columns(
                pl.col(abundance_col).cast(pl.Float64).alias("_abd")
            )
            df = df.with_columns(
                pl.col("_abd").sum().over(roles.species_plot_id).alias("_plot_total")
            )
            df = df.with_columns(
                pl.when(pl.col("_plot_total") > 0)
                .then(pl.col("_abd") / pl.col("_plot_total"))
                .otherwise(pl.col("_abd"))
                .alias("_weight")
            )
        else:  # raw
            df = df.with_columns(
                pl.col(abundance_col).cast(pl.Float64).alias("_weight")
            )

        # Aggregate genera by weight
        genus_agg = (
            df.group_by([roles.species_plot_id, roles.taxonomy_genus])
            .agg(pl.col("_weight").sum().alias("_total"))
        )
        genus_selected = self._select_by_mode(genus_agg, roles.species_plot_id, self.top_k)

        # Aggregate families by weight
        family_agg = (
            df.group_by([roles.species_plot_id, roles.taxonomy_family])
            .agg(pl.col("_weight").sum().alias("_total"))
        )
        family_selected = self._select_by_mode(family_agg, roles.species_plot_id, self.top_k)

        # Build ID arrays aligned to plot_ids (vectorized scatter)
        n_plots = len(plot_ids)
        genus_ids = np.zeros((n_plots, self.n_taxonomy_slots), dtype=np.int64)
        family_ids = np.zeros((n_plots, self.n_taxonomy_slots), dtype=np.int64)

        plot_id_to_idx = {pid: i for i, pid in enumerate(plot_ids)}

        # Vectorized: encode + scatter for genus
        g_pids = genus_selected[roles.species_plot_id].to_list()
        g_taxa = genus_selected[roles.taxonomy_genus].to_list()
        g_ranks = genus_selected["_rank"].to_numpy()
        for i, (pid, taxon) in enumerate(zip(g_pids, g_taxa)):
            if pid in plot_id_to_idx:
                genus_ids[plot_id_to_idx[pid], g_ranks[i]] = self._vocab.encode_genus(taxon)

        # Vectorized: encode + scatter for family
        f_pids = family_selected[roles.species_plot_id].to_list()
        f_taxa = family_selected[roles.taxonomy_family].to_list()
        f_ranks = family_selected["_rank"].to_numpy()
        for i, (pid, taxon) in enumerate(zip(f_pids, f_taxa)):
            if pid in plot_id_to_idx:
                family_ids[plot_id_to_idx[pid], f_ranks[i]] = self._vocab.encode_family(taxon)

        return genus_ids, family_ids

    def save(self, path: str) -> None:
        """Save encoder state (vocabulary)."""
        if self._vocab is not None:
            self._vocab.save(path)

    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        self._vocab = TaxonomyVocab.load(path)
        self._fitted = True
