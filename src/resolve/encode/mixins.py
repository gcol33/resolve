"""Shared mixins for species encoders."""

from __future__ import annotations

from typing import Optional

from resolve.encode.vocab import SpeciesVocab, TaxonomyVocab


class TaxonomyEncoderMixin:
    """Mixin providing n_species/n_genera/n_families properties.

    Requires the class to have ``_species_vocab: Optional[SpeciesVocab]``
    and ``_taxonomy_vocab: Optional[TaxonomyVocab]`` attributes.

    Used by BagOfSpeciesEncoder, EmbeddingEncoder, and RankPoolEncoder.
    """

    _species_vocab: Optional[SpeciesVocab]
    _taxonomy_vocab: Optional[TaxonomyVocab]

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
