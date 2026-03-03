"""Abstract base class for all species encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from resolve.data.dataset import ResolveDataset


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
