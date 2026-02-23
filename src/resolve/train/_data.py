"""Data preparation mixin for Trainer.

Handles dataset splitting, encoding, tensor construction, and data loader
creation. These methods form the data pipeline that converts a ResolveDataset
into GPU-ready batches for training.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from resolve.data.dataset import ResolveDataset, ResolveSchema
from resolve.encode.embedding import EmbeddingEncoder
from resolve.encode.species import SpeciesEncoder
from resolve.train._loaders import (
    GPUTensorLoader,
    RankPoolBatchDataset,
    _RankPoolPreparedData,
    _rank_pool_collate_fn,
)

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class DataMixin:
    """Mixin providing data preparation methods for Trainer."""

    def _unpack_batch(
        self: Trainer,
        batch: tuple,
        target_names: list[str],
        has_taxonomy: bool,
        data_on_device: bool,
    ) -> tuple:
        """Unpack a batch tuple into model inputs and targets dict.

        Returns:
            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
             targets)
        """
        def _get(i: int) -> torch.Tensor:
            return batch[i] if data_on_device else batch[i].to(self._device, non_blocking=True)

        idx = 0
        continuous = _get(idx); idx += 1

        species_ids = None
        species_vector = None
        pool_genus_ids = None
        pool_family_ids = None
        pool_weights = None
        pool_mask = None
        pool_has_cover = None

        if self.species_encoding == "embed":
            species_ids = _get(idx); idx += 1
        elif self.species_encoding in ("rank_pool", "transformer"):
            species_ids = _get(idx); idx += 1
            if has_taxonomy:
                pool_genus_ids = _get(idx); idx += 1
                pool_family_ids = _get(idx); idx += 1
            pool_weights = _get(idx); idx += 1
            pool_mask = _get(idx); idx += 1
            pool_has_cover = _get(idx); idx += 1
        elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
            species_vector = _get(idx); idx += 1

        if has_taxonomy and self.species_encoding not in ("rank_pool", "transformer"):
            genus_ids = _get(idx); idx += 1
            family_ids = _get(idx); idx += 1
        else:
            genus_ids = None
            family_ids = None

        targets = {}
        for name in target_names:
            targets[name] = _get(idx); idx += 1

        return (
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
            targets,
        )

    def _prepare_data(
        self: Trainer,
        fit_encoder: bool = True,
    ) -> tuple[ResolveDataset, ResolveDataset]:
        """Split and encode data."""
        train_ds, test_ds = self.dataset.split(test_size=0.2)

        if self.species_encoding == "hash":
            # Hash mode: use SpeciesEncoder
            if fit_encoder or self._species_encoder is None or not self._species_encoder._fitted:
                self._species_encoder = SpeciesEncoder(
                    hash_dim=self.hash_dim,
                    top_k=self.top_k,
                    aggregation=self.species_aggregation,
                    normalization=self.species_normalization,
                    track_unknown_count=self.track_unknown_count,
                    selection=self.species_selection,
                    representation=self.species_representation,
                    min_species_frequency=self.min_species_frequency,
                )
                self._species_encoder.fit(train_ds)

                # For all/presence_absence modes, update schema with species vocab size
                if self._species_encoder.uses_explicit_vector:
                    self._schema = replace(
                        self._schema,
                        n_species_vocab=self._species_encoder.n_species_vector,
                        n_genera_vocab=0,
                        n_families_vocab=0,
                    )
        elif self.species_encoding == "embed":
            # Embed mode: use EmbeddingEncoder
            if fit_encoder or self._embedding_encoder is None or not self._embedding_encoder._fitted:
                self._embedding_encoder = EmbeddingEncoder(
                    top_k_species=self.top_k_species,
                    top_k_taxonomy=self.top_k,
                    aggregation=self.species_aggregation,
                    selection=self.species_selection,
                )
                self._embedding_encoder.fit(train_ds)

                # Update schema with vocab sizes for model construction
                self._schema = replace(
                    self._schema,
                    n_species_vocab=self._embedding_encoder.n_species,
                    n_genera_vocab=self._embedding_encoder.n_genera,
                    n_families_vocab=self._embedding_encoder.n_families,
                )
        else:  # rank_pool or transformer mode (both use same data pipeline)
            from resolve.encode.rank_pool import RankPoolEncoder
            # Skip re-fitting if encoder was already fitted on full data during pretraining
            should_fit = (
                (fit_encoder and not getattr(self, '_pretrain_fitted_encoder', False))
                or self._rank_pool_encoder is None
                or not self._rank_pool_encoder._fitted
            )
            if should_fit:
                self._rank_pool_encoder = RankPoolEncoder(
                    weighting=self.species_normalization,
                    min_species_frequency=self.min_species_frequency,
                )
                self._rank_pool_encoder.fit(train_ds)

                # Update schema with vocab sizes
                self._schema = replace(
                    self._schema,
                    n_species_vocab=self._rank_pool_encoder.n_species,
                    n_genera_vocab=self._rank_pool_encoder.n_genera,
                    n_families_vocab=self._rank_pool_encoder.n_families,
                )

        return train_ds, test_ds

    def _build_tensors(
        self: Trainer,
        dataset: ResolveDataset,
        fit_scalers: bool = False,
    ) -> tuple[torch.Tensor, ...] | _RankPoolPreparedData:
        """Convert dataset to tensors (or _RankPoolPreparedData for rank_pool mode)."""
        # Get continuous features
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()

        # Initialize outputs
        species_ids = None
        species_vector = None

        if self.species_encoding == "hash":
            # Hash mode: use SpeciesEncoder
            encoded = self._species_encoder.transform(dataset)

            # Check if using explicit species vector (all/presence_absence)
            if self._species_encoder.uses_explicit_vector:
                # Continuous features WITHOUT hash embedding (separate species_vector input)
                parts = []
                if coords is not None:
                    parts.append(coords)
                if covariates is not None:
                    parts.append(covariates)
                if self.track_unknown_fraction:
                    parts.append(encoded.unknown_fraction.reshape(-1, 1))
                if self.track_unknown_count and encoded.unknown_count is not None:
                    parts.append(encoded.unknown_count.reshape(-1, 1).astype(np.float32))

                species_vector = encoded.species_vector
            else:
                # Standard hash mode: include hash_embedding in continuous
                parts = []
                if coords is not None:
                    parts.append(coords)
                if covariates is not None:
                    parts.append(covariates)
                parts.append(encoded.hash_embedding)
                if self.track_unknown_fraction:
                    parts.append(encoded.unknown_fraction.reshape(-1, 1))
                if self.track_unknown_count and encoded.unknown_count is not None:
                    parts.append(encoded.unknown_count.reshape(-1, 1).astype(np.float32))

            genus_ids = encoded.genus_ids
            family_ids = encoded.family_ids
            unknown_fraction = encoded.unknown_fraction
        elif self.species_encoding == "embed":
            # Embed mode: use EmbeddingEncoder
            embedded = self._embedding_encoder.transform(dataset)

            # Continuous features WITHOUT hash embedding
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            # Include unknown fraction for embed mode too
            if self.track_unknown_fraction:
                parts.append(embedded.unknown_fraction.reshape(-1, 1))

            species_ids = embedded.species_ids
            genus_ids = embedded.genus_ids
            family_ids = embedded.family_ids
            unknown_fraction = embedded.unknown_fraction
        else:  # rank_pool or transformer mode
            pool_encoded = self._rank_pool_encoder.transform(dataset)

            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if self.track_unknown_fraction:
                parts.append(pool_encoded.unknown_fraction.reshape(-1, 1))

            genus_ids = None
            family_ids = None
            unknown_fraction = pool_encoded.unknown_fraction

        continuous = np.hstack(parts) if parts else np.zeros((len(dataset.plot_ids), 0), dtype=np.float32)

        # Scale continuous features
        # Handle missing or incompatible scalers when resuming from checkpoint
        need_fit = fit_scalers
        if not fit_scalers:
            if "continuous" not in self._scalers:
                warnings.warn(
                    "Checkpoint missing 'continuous' scaler - fitting new scaler. "
                    "This may indicate a feature configuration mismatch.",
                    RuntimeWarning,
                )
                need_fit = True
            elif self._scalers["continuous"].n_features_in_ != continuous.shape[1]:
                warnings.warn(
                    f"Scaler dimension mismatch: checkpoint has "
                    f"{self._scalers['continuous'].n_features_in_} features, "
                    f"current data has {continuous.shape[1]}. Fitting new scaler.",
                    RuntimeWarning,
                )
                need_fit = True

        if need_fit:
            self._scalers["continuous"] = StandardScaler()
            continuous = self._scalers["continuous"].fit_transform(continuous)
        else:
            continuous = self._scalers["continuous"].transform(continuous)

        continuous = continuous.astype(np.float32)

        # Get targets
        targets = {}
        for name, cfg in self.model.target_configs.items():
            target_vals = dataset.get_target(name)
            mask = dataset.get_target_mask(name)

            if cfg.task == "regression":
                scaler_key = f"target_{name}"
                # Handle missing target scaler when resuming
                need_fit_target = fit_scalers
                if not fit_scalers and scaler_key not in self._scalers:
                    warnings.warn(
                        f"Checkpoint missing '{scaler_key}' scaler - fitting new scaler.",
                        RuntimeWarning,
                    )
                    need_fit_target = True

                if need_fit_target:
                    self._scalers[scaler_key] = StandardScaler()
                    # Fit on non-null values only
                    self._scalers[scaler_key].fit(target_vals[mask].reshape(-1, 1))
                    # Store scaler params for loss computation
                    scaler = self._scalers[scaler_key]
                    self._target_scalers[name] = (
                        torch.tensor(scaler.mean_[0], dtype=torch.float32, device=self._device),
                        torch.tensor(scaler.scale_[0], dtype=torch.float32, device=self._device),
                    )

                # Transform ALL values (including nulls which become NaN after scaling)
                # The loss function will handle masking during training
                target_scaled = self._scalers[scaler_key].transform(
                    target_vals.reshape(-1, 1)
                )
                targets[name] = target_scaled.flatten().astype(np.float32)
            else:
                # Classification: fill NaN with -1 (will be ignored by CrossEntropyLoss)
                # NaN in float array becomes large negative number when cast to int64
                target_int = np.where(mask, target_vals, -1).astype(np.int64)
                targets[name] = target_int

        # Rank-pool/transformer mode: return _RankPoolPreparedData with ragged arrays (per-batch padding)
        if self.species_encoding in ("rank_pool", "transformer"):
            has_tax = self._schema.has_taxonomy
            return _RankPoolPreparedData(
                continuous=torch.from_numpy(continuous),
                target_tensors=[
                    torch.from_numpy(targets[n]) for n in self.model.target_configs
                ],
                species_ids=pool_encoded.species_ids,
                genus_ids=pool_encoded.genus_ids if has_tax else None,
                family_ids=pool_encoded.family_ids if has_tax else None,
                weights=pool_encoded.weights,
                has_cover=pool_encoded.has_cover,
                has_taxonomy=has_tax,
                n_samples=len(pool_encoded.plot_ids),
            )

        # Build tensor dataset (hash/embed modes)
        tensors = [torch.from_numpy(continuous)]

        # Add species_ids for embed mode (must come before genus/family for consistent unpacking)
        if self.species_encoding == "embed" and species_ids is not None:
            tensors.append(torch.from_numpy(species_ids))

        # Add species_vector for hash mode with all/presence_absence selection
        if self.species_encoding == "hash" and species_vector is not None:
            tensors.append(torch.from_numpy(species_vector))

        if genus_ids is not None:
            tensors.append(torch.from_numpy(genus_ids))
        if family_ids is not None:
            tensors.append(torch.from_numpy(family_ids))

        for name in self.model.target_configs.keys():
            tensors.append(torch.from_numpy(targets[name]))

        return tuple(tensors)

    def _create_loaders(self: Trainer, train_data, test_data) -> None:
        """Create data loaders.

        Accepts either a tuple of tensors (hash/embed modes) or
        _RankPoolPreparedData (rank_pool mode with per-batch padding).
        """
        # Rank-pool mode: per-batch padding via custom Dataset + collate
        # num_workers=0 on Windows: spawn-based multiprocessing adds overhead
        # and can fail serializing ragged numpy arrays. Main-thread collation
        # is equally fast (~4s/ep) since the numpy collate is lightweight.
        if isinstance(train_data, _RankPoolPreparedData):
            n_workers = self.num_workers  # default 0, user can override
            print(f"  Rank-pool mode: per-batch padding, num_workers={n_workers}")
            print(f"  Train: {train_data.n_samples:,} samples (ragged, per-batch padding)")
            print(f"  Test: {test_data.n_samples:,} samples (ragged, per-batch padding)")

            self._train_loader = DataLoader(
                RankPoolBatchDataset(train_data),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=n_workers,
                collate_fn=_rank_pool_collate_fn,
                pin_memory=self._device.type == "cuda" and n_workers > 0,
                persistent_workers=n_workers > 0,
                drop_last=True,
            )
            self._test_loader = DataLoader(
                RankPoolBatchDataset(test_data),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=n_workers,
                collate_fn=_rank_pool_collate_fn,
                pin_memory=self._device.type == "cuda" and n_workers > 0,
                persistent_workers=n_workers > 0,
            )
            self._using_gpu_loader = False
            return

        # Hash/embed modes: standard tensor-based loaders
        train_tensors = train_data
        test_tensors = test_data

        # Debug: verify all tensors have same first dimension
        train_sizes = [t.shape[0] for t in train_tensors]
        test_sizes = [t.shape[0] for t in test_tensors]
        if len(set(train_sizes)) > 1:
            raise ValueError(
                f"Train tensors have mismatched first dimensions: {train_sizes}. "
                f"Tensor shapes: {[t.shape for t in train_tensors]}"
            )
        if len(set(test_sizes)) > 1:
            raise ValueError(
                f"Test tensors have mismatched first dimensions: {test_sizes}. "
                f"Tensor shapes: {[t.shape for t in test_tensors]}"
            )
        print(f"  Train tensors: {train_sizes[0]:,} samples, {len(train_tensors)} tensors")
        print(f"  Test tensors: {test_sizes[0]:,} samples, {len(test_tensors)} tensors")

        if self.gpu_data:
            # Use GPU-resident tensors for maximum throughput
            # This eliminates the DataLoader CPU→GPU bottleneck (~400ms → ~1ms per batch)
            total_size_mb = sum(t.numel() * t.element_size() for t in train_tensors) / 1e6
            total_size_mb += sum(t.numel() * t.element_size() for t in test_tensors) / 1e6
            print(f"  GPU data mode: moving {total_size_mb:.1f} MB to GPU")

            self._train_loader = GPUTensorLoader(
                train_tensors,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                device=self._device,
            )
            self._test_loader = GPUTensorLoader(
                test_tensors,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                device=self._device,
            )
            # Mark that we're using GPU loaders (data already on device)
            self._using_gpu_loader = True
        else:
            # Standard CPU DataLoader with pin_memory for async transfer
            train_ds = TensorDataset(*train_tensors)
            test_ds = TensorDataset(*test_tensors)

            self._train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self._device.type == "cuda",
                persistent_workers=self.num_workers > 0,
                drop_last=True,  # Avoid small final batch overhead
            )
            self._test_loader = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self._device.type == "cuda",
                persistent_workers=self.num_workers > 0,
            )
            self._using_gpu_loader = False
