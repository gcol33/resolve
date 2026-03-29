"""Cross-validation mixin for Trainer.

Provides spatial block cross-validation that fits the encoder once on the
full dataset, then trains a fresh model per fold using pre-computed
spatial train/test splits.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from resolve.data.spatial import SpatialBlockSplitter
from resolve.train._types import CVResult, TrainResult

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class CVMixin:
    """Mixin providing spatial block cross-validation for Trainer."""

    def cross_validate(
        self: Trainer,
        n_splits: int = 10,
        seed: int = 42,
        *,
        block_deg: float | tuple[float, float] | None = None,
        block_km: float | tuple[float, float] | None = None,
        block_ids: np.ndarray | None = None,
        balance: bool = False,
        spatial: bool = True,
        block_size: float | None = None,
    ) -> CVResult:
        """Run spatial block cross-validation.

        Fits the species encoder once on the full dataset (no label leakage
        since encoding is unsupervised), then trains a fresh model per fold.

        Parameters
        ----------
        n_splits : int
            Number of CV folds. Default 10.
        seed : int
            Random seed for block shuffling. Default 42.
        block_deg : float or tuple[float, float] or None
            Block size in degrees (scalar for square, tuple for rectangular).
        block_km : float or tuple[float, float] or None
            Block size in kilometres (converted using mean latitude).
        block_ids : np.ndarray or None
            Pre-assigned 1-D integer block labels (one per plot).
        balance : bool
            If True, use greedy bin-packing to equalise fold sizes.
            If False (default), round-robin block assignment.
        spatial : bool
            If True (default), use spatial block splitting. If False, use
            random splitting (plots shuffled, no spatial structure).
        block_size : float or None
            **Deprecated.** Use *block_deg* instead.

        Returns
        -------
        CVResult
            Per-fold and aggregated metrics.
        """
        # Validate block_ids length if provided
        if block_ids is not None and len(block_ids) != self.dataset.n_plots:
            raise ValueError(
                f"block_ids length ({len(block_ids)}) must match "
                f"dataset.n_plots ({self.dataset.n_plots})"
            )

        coords = self.dataset.get_coordinates()
        if spatial and block_ids is None and coords is None:
            raise ValueError(
                "Spatial CV requires coordinates. Dataset has no coordinate "
                "columns, or set spatial=False for random CV."
            )

        # Build header string
        print(f"\n=== {n_splits}-Fold {'Spatial Block' if spatial else 'Random'} Cross-Validation ===")
        if spatial:
            if block_ids is not None:
                n_unique = len(np.unique(block_ids))
                print(f"  Block mode: block_ids ({n_unique} unique blocks)")
            elif block_km is not None:
                print(f"  Block mode: block_km = {block_km} km")
            elif block_size is not None:
                print(f"  Block mode: block_size = {block_size}° (deprecated)")
            elif block_deg is not None:
                print(f"  Block mode: block_deg = {block_deg}°")
            else:
                print(f"  Block mode: block_deg = 0.1° (default)")
        if balance:
            print(f"  Balance: greedy bin-packing")
        print(f"  Seed: {seed}")
        sys.stdout.flush()

        # Generate fold indices
        if spatial:
            splitter = SpatialBlockSplitter(
                n_splits=n_splits,
                seed=seed,
                block_deg=block_size if block_size is not None else block_deg,
                block_km=block_km,
                block_ids=block_ids,
                balance=balance,
            )
            folds = splitter.split(coords)
        else:
            # Random splitting: shuffle plot indices, assign round-robin
            rng = np.random.default_rng(seed)
            n = self.dataset.n_plots
            perm = rng.permutation(n)
            fold_assignment = np.zeros(n, dtype=np.int32)
            for i, idx in enumerate(perm):
                fold_assignment[idx] = i % n_splits
            all_indices = np.arange(n)
            folds = []
            for k in range(n_splits):
                test_mask = fold_assignment == k
                folds.append((all_indices[~test_mask], all_indices[test_mask]))

        # Report fold sizes
        for k, (train_idx, test_idx) in enumerate(folds):
            print(f"  Fold {k+1}: {len(train_idx):,} train, {len(test_idx):,} test")

        # Fit encoder once on full dataset (unsupervised, no label leakage)
        self._fit_encoder_on_full_dataset()

        fold_results: list[TrainResult] = []
        fold_metrics: list[dict[str, dict[str, float]]] = []
        cv_start = time.time()

        for k, (train_idx, test_idx) in enumerate(folds):
            print(f"\n--- Fold {k+1}/{n_splits} ---")
            sys.stdout.flush()

            train_ds, test_ds = self.dataset.split_by_indices(train_idx, test_idx)

            # Create a fresh trainer for this fold, sharing encoder state
            fold_trainer = self._make_fold_trainer(train_ds, test_ds)
            result = fold_trainer.fit()

            fold_results.append(result)
            fold_metrics.append(result.final_metrics)

            # Log fold result
            metric_str = " | ".join(
                f"{name}: {result.final_metrics[name].get('band_25', result.final_metrics[name].get('accuracy', 0)):.2%}"
                for name in result.final_metrics
            )
            print(f"  Fold {k+1} result: {metric_str} (best epoch {result.best_epoch})")

        cv_time = time.time() - cv_start

        # Aggregate metrics across folds
        mean_metrics, std_metrics = self._aggregate_fold_metrics(fold_metrics)

        print(f"\n=== CV Summary ({cv_time:.1f}s) ===")
        for target, metrics in mean_metrics.items():
            parts = []
            for metric, value in metrics.items():
                std = std_metrics[target].get(metric, 0.0)
                parts.append(f"{metric}={value:.4f}+/-{std:.4f}")
            print(f"  {target}: {' | '.join(parts)}")

        return CVResult(
            fold_results=fold_results,
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            n_folds=n_splits,
        )

    def _fit_encoder_on_full_dataset(self: Trainer) -> None:
        """Fit the species encoder on the full dataset (all plots).

        This is unsupervised (species hashing/embedding vocabularies),
        so there is no label leakage.
        """
        from resolve.encode.species import SpeciesEncoder
        from resolve.encode.embedding import EmbeddingEncoder

        if self.species_encoding == "hash":
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
            self._species_encoder.fit(self.dataset)
            if self._species_encoder.uses_explicit_vector:
                self._schema = replace(
                    self._schema,
                    n_species_vocab=self._species_encoder.n_species_vector,
                    n_genera_vocab=0,
                    n_families_vocab=0,
                )
        elif self.species_encoding == "embed":
            self._embedding_encoder = EmbeddingEncoder(
                top_k_species=self.top_k_species,
                top_k_taxonomy=self.top_k,
                aggregation=self.species_aggregation,
                selection=self.species_selection,
            )
            self._embedding_encoder.fit(self.dataset)
            self._schema = replace(
                self._schema,
                n_species_vocab=self._embedding_encoder.n_species,
                n_genera_vocab=self._embedding_encoder.n_genera,
                n_families_vocab=self._embedding_encoder.n_families,
            )
        elif self.species_encoding == "trait_net":
            # TraitNet doesn't need a species encoder — traits are provided directly
            pass
        else:  # rank_pool or transformer
            from resolve.encode.rank_pool import RankPoolEncoder
            self._rank_pool_encoder = RankPoolEncoder(
                weighting=self.species_normalization,
                min_species_frequency=self.min_species_frequency,
            )
            self._rank_pool_encoder.fit(self.dataset)
            self._schema = replace(
                self._schema,
                n_species_vocab=self._rank_pool_encoder.n_species,
                n_genera_vocab=self._rank_pool_encoder.n_genera,
                n_families_vocab=self._rank_pool_encoder.n_families,
            )

        # Mark encoder as pre-fitted so _prepare_data doesn't re-fit
        self._pretrain_fitted_encoder = True
        print(f"  Encoder fitted on full dataset ({self.dataset.n_plots:,} plots)")

    def _make_fold_trainer(
        self: Trainer,
        train_ds,
        test_ds,
    ) -> Trainer:
        """Create a fresh Trainer for one CV fold.

        Shares the encoder state from the parent trainer but creates
        a fresh model, optimizer, and training state.
        """
        from resolve.train.trainer import Trainer

        # Create new trainer with same config but the train split as dataset
        fold_trainer = Trainer(
            dataset=train_ds,
            species_encoding=self.species_encoding,
            hash_dim=self.hash_dim,
            species_embed_dim=self.species_embed_dim,
            top_k=self.top_k,
            top_k_species=self.top_k_species,
            hidden_dims=list(self.hidden_dims),
            genus_emb_dim=self.genus_emb_dim,
            family_emb_dim=self.family_emb_dim,
            dropout=self.dropout,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr=self.lr,
            weight_decay=self.weight_decay,
            checkpoint_dir=None,  # No checkpointing for CV folds
            cache_dir=None,  # No caching for CV folds
            loss_config="mae",  # Overridden by phases below
            phases=self.phases,
            phase_boundaries=self.phase_boundaries,
            device=str(self._device),
            use_amp=self.use_amp,
            compile_model=self.compile_model,
            prefetch_data=self.prefetch_data,
            gpu_data=self._gpu_data_setting,
            species_aggregation=self.species_aggregation,
            species_selection=self.species_selection,
            species_representation=self.species_representation,
            min_species_frequency=self.min_species_frequency,
            cover_dropout=self.cover_dropout,
            n_attention_layers=self.n_attention_layers,
            n_heads=self.n_heads,
            transformer_ff_dim=self.transformer_ff_dim,
            transformer_pooling=self.transformer_pooling,
            transformer_dropout=self.transformer_dropout,
            pretrain_epochs=0,  # No pretraining in CV folds
            pretrain_mask_prob=self.pretrain_mask_prob,
            pretrain_lr=self.pretrain_lr,
            pretrain_all_data=False,
            categorical_embed_dim=self.categorical_embed_dim,
            label_smoothing=self.label_smoothing,
            class_weights=self.class_weights,
            ema_decay=self.ema_decay,
            species_dropout=self.species_dropout,
            head_hidden_dims=self.head_hidden_dims,
            stratified_split=self.stratified_split,
            trait_net_config=self.trait_net_config,
            traits=self.traits,
            verbose=self.verbose,
        )

        # Copy the pre-fitted encoder so _prepare_data won't re-fit
        fold_trainer._species_encoder = self._species_encoder
        fold_trainer._embedding_encoder = self._embedding_encoder
        fold_trainer._rank_pool_encoder = self._rank_pool_encoder
        fold_trainer._pretrain_fitted_encoder = True
        fold_trainer._schema = self._schema

        # Override _prepare_data to use the provided test_ds instead of
        # doing a random split
        _original_prepare = fold_trainer._prepare_data

        def _cv_prepare_data(fit_encoder: bool = True):
            # Just return the pre-split datasets; encoder is already fitted
            return train_ds, test_ds

        fold_trainer._prepare_data = _cv_prepare_data

        return fold_trainer

    @staticmethod
    def _aggregate_fold_metrics(
        fold_metrics: list[dict[str, dict[str, float]]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Compute mean and std of metrics across folds."""
        if not fold_metrics:
            return {}, {}

        # Collect all target/metric keys from first fold
        targets = list(fold_metrics[0].keys())
        mean_metrics: dict[str, dict[str, float]] = {}
        std_metrics: dict[str, dict[str, float]] = {}

        for target in targets:
            metric_keys = list(fold_metrics[0][target].keys())
            mean_metrics[target] = {}
            std_metrics[target] = {}
            for metric in metric_keys:
                values = [
                    fm[target][metric]
                    for fm in fold_metrics
                    if target in fm and metric in fm[target]
                ]
                if values:
                    mean_metrics[target][metric] = float(np.mean(values))
                    std_metrics[target][metric] = float(np.std(values))

        return mean_metrics, std_metrics
