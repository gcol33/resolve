"""Caching mixin for Trainer.

Handles saving and loading preprocessed tensor data to disk, avoiding
repeated encoding and scaling on subsequent training runs with identical
dataset and configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from resolve.encode.species import SpeciesEncoder

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []

# Cache version — increment when cache format changes
_CACHE_VERSION = 1


class CacheMixin:
    """Mixin providing tensor caching methods for Trainer."""

    def _compute_cache_key(self: Trainer) -> str:
        """Compute a hash key for caching based on dataset and config."""
        # Include dataset fingerprint (convert to strings to handle mixed types)
        plot_ids = sorted(str(x) for x in self.dataset._header[self.dataset._roles.plot_id].unique().to_list())
        species_ids = sorted(str(x) for x in self.dataset._species[self.dataset._roles.species_id].drop_nulls().unique().to_list())

        # Build config dict
        config = {
            "version": _CACHE_VERSION,
            "n_plots": len(plot_ids),
            "n_species": len(species_ids),
            "plot_ids_hash": hashlib.md5(str(plot_ids[:100] + plot_ids[-100:]).encode()).hexdigest()[:8],
            "species_ids_hash": hashlib.md5(str(species_ids[:100] + species_ids[-100:]).encode()).hexdigest()[:8],
            "hash_dim": self.hash_dim,
            "top_k": self.top_k,
            "species_aggregation": self.species_aggregation,
            "species_selection": self.species_selection,
            "species_normalization": self.species_normalization,
            "track_unknown_fraction": self.track_unknown_fraction,
            "track_unknown_count": self.track_unknown_count,
            "targets": sorted(self.dataset.targets.keys()),
        }

        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _cache_path(self: Trainer) -> Optional[Path]:
        """Get path to cache file."""
        if self.cache_dir is None:
            return None
        cache_key = self._compute_cache_key()
        return self.cache_dir / f"preprocessed_{cache_key}.pt"

    def _save_cache(
        self: Trainer,
        train_tensors: tuple[torch.Tensor, ...],
        test_tensors: tuple[torch.Tensor, ...],
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> None:
        """Save preprocessed data to cache."""
        if self.cache_dir is None:
            return

        cache = {
            "train_tensors": train_tensors,
            "test_tensors": test_tensors,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "scalers": self._scalers,
            "target_scalers": {
                k: (v[0].cpu(), v[1].cpu()) for k, v in self._target_scalers.items()
            },
            "species_encoder": {
                "vocab": self._species_encoder._vocab if self._species_encoder else None,
                "species_vocab": self._species_encoder._species_vocab if self._species_encoder else set(),
            },
            "cache_key": self._compute_cache_key(),
        }

        cache_path = self._cache_path()
        torch.save(cache, cache_path)
        print(f"  [Cache saved: {cache_path.name}]")

        # Cleanup old cache files
        self._cleanup_old_caches()

    def _cleanup_old_caches(self: Trainer) -> None:
        """Remove old cache files, keeping only the most recent ones."""
        if self.cache_dir is None or self.max_cache_files <= 0:
            return

        # Find all cache files
        cache_files = list(self.cache_dir.glob("preprocessed_*.pt"))
        if len(cache_files) <= self.max_cache_files:
            return

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)

        # Remove oldest files
        files_to_remove = cache_files[: len(cache_files) - self.max_cache_files]
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"  [Removed old cache: {f.name}]")
            except OSError:
                pass  # Ignore if file can't be deleted

    def _load_cache(self: Trainer) -> Optional[dict]:
        """Load preprocessed data from cache if valid."""
        if self.cache_dir is None:
            return None

        cache_path = self._cache_path()
        if not cache_path.exists():
            return None

        try:
            # Note: weights_only=False is required for sklearn scalers.
            # Only load cache files from trusted sources.
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)

            # Validate cache key matches
            if cache.get("cache_key") != self._compute_cache_key():
                print(f"  [Cache key mismatch, rebuilding...]")
                return None

            print(f"  [Cache loaded: {cache_path.name}]")
            return cache
        except Exception as e:
            print(f"  [Cache load failed: {e}, rebuilding...]")
            return None

    def _restore_from_cache(self: Trainer, cache: dict) -> tuple[tuple, tuple]:
        """Restore state from cache and return tensors."""
        # Restore scalers
        self._scalers = cache["scalers"]
        self._target_scalers = {
            k: (v[0].to(self._device), v[1].to(self._device))
            for k, v in cache["target_scalers"].items()
        }

        # Restore species encoder
        self._species_encoder = SpeciesEncoder(
            hash_dim=self.hash_dim,
            top_k=self.top_k,
            aggregation=self.species_aggregation,
            normalization=self.species_normalization,
            track_unknown_count=self.track_unknown_count,
            selection=self.species_selection,
            representation=self.species_representation,
        )
        enc_state = cache["species_encoder"]
        if enc_state.get("vocab"):
            self._species_encoder._vocab = enc_state["vocab"]
        if enc_state.get("species_vocab"):
            self._species_encoder._species_vocab = enc_state["species_vocab"]
        self._species_encoder._fitted = True

        return cache["train_tensors"], cache["test_tensors"]
