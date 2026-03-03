"""Bag-of-species encoding with additive hierarchical embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from resolve.data.dataset import ResolveDataset
from resolve.encode._pool_base import BasePoolEncoder, extract_ragged_arrays, pad_ragged_encoded
from resolve.encode.normalize import TaxonomyNormalizer


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
    """Pad ragged bag-encoded data to dense arrays for batched forward."""
    return pad_ragged_encoded(bag.species_ids, bag.genus_ids, bag.family_ids, bag.weights)


class BagOfSpeciesEncoder(BasePoolEncoder):
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
        super().__init__(weighting=weighting, normalizer=normalizer,
                         min_species_frequency=min_species_frequency)

    def fit(self, dataset: ResolveDataset) -> BagOfSpeciesEncoder:
        """Build species + taxonomy vocabularies from training data."""
        super().fit(dataset)
        return self

    def transform(self, dataset: ResolveDataset) -> BagEncodedSpecies:
        """Encode all plots as variable-length species bags."""
        result, _, plot_ids = self._encode_species_df(dataset)
        n_plots = len(plot_ids)
        all_sp, all_g, all_f, all_w, unknown_fracs = extract_ragged_arrays(result, n_plots)

        return BagEncodedSpecies(
            species_ids=all_sp,
            genus_ids=all_g,
            family_ids=all_f,
            weights=all_w,
            plot_ids=plot_ids,
            unknown_fraction=unknown_fracs,
        )
