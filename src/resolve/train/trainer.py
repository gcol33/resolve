"""Trainer: training orchestration for ResolveModel."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from resolve.data.dataset import ResolveDataset, ResolveSchema
from resolve.encode.species import SpeciesEncoder
from resolve.encode.embedding import EmbeddingEncoder
from resolve.model.resolve import ResolveModel
from resolve.train.loss import MultiTaskLoss, PhaseConfig

# Preset loss configurations
LOSS_PRESETS = {
    "mae": {1: PhaseConfig(mae=1.0)},
    "combined": {1: PhaseConfig(mae=0.80, smape=0.15, band=0.05)},
    "smape": {1: PhaseConfig(mae=0.5, smape=0.5)},
}
from resolve.train.metrics import compute_metrics

# Cache version - increment when cache format changes
_CACHE_VERSION = 1


@dataclass
class TrainResult:
    """Results from training."""

    best_epoch: int
    final_metrics: dict[str, dict[str, float]]
    history: dict[str, list[float]] = field(default_factory=dict)
    resumed_from_epoch: Optional[int] = None
    train_time: float = 0.0  # Total training time in seconds


class Trainer:
    """
    Trains ResolveModel with phased loss schedule.

    Minimal usage:
        trainer = Trainer(dataset)
        trainer.fit()
        predictions = trainer.predict(dataset)

    Handles:
        - Model construction from dataset schema
        - Data preprocessing (encoding, scaling)
        - Training loop with early stopping
        - Checkpointing
        - Evaluation and prediction
    """

    def __init__(
        self,
        dataset: ResolveDataset,
        # Species encoding mode
        species_encoding: str = "hash",  # "hash" or "embed"
        # Model architecture
        hash_dim: int = 32,
        species_embed_dim: int = 32,  # Embedding dim per species (embed mode)
        top_k: int = 5,
        top_k_species: int = 10,  # Number of top species for embed mode
        hidden_dims: Optional[list[int]] = None,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        dropout: float = 0.3,
        # Training
        batch_size: int = 32768,  # Larger batch for better GPU utilization
        num_workers: int = 0,  # 0 is safest for TensorDataset on Windows
        max_epochs: int = 500,
        patience: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Checkpointing
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_every: int = 50,
        resume: bool = True,
        reset_patience: bool = False,  # Reset early stopping counter on resume
        # Caching
        cache_dir: Optional[str | Path] = None,  # Cache preprocessed tensors for faster restarts
        max_cache_files: int = 5,  # Max number of cache files to keep (oldest deleted first)
        # Loss configuration
        loss_config: str = "mae",  # "mae", "combined", "smape" or custom PhaseConfig dict
        # Advanced (deprecated - use loss_config instead)
        phases: Optional[dict[int, PhaseConfig]] = None,
        phase_boundaries: Optional[list[int]] = None,
        device: str = "auto",
        use_amp: bool = True,
        compile_model: bool = False,  # Use torch.compile() for potential speedup
        species_aggregation: str = "abundance",
        species_selection: str = "top",  # "top", "bottom", "top_bottom", or "all"
        species_representation: str = "abundance",  # "abundance" or "presence_absence" (only for selection="all")
        min_species_frequency: int = 1,  # For selection="all": minimum plot frequency
    ):
        self.dataset = dataset

        # Validate species_encoding
        if species_encoding not in ("hash", "embed"):
            raise ValueError(f"species_encoding must be 'hash' or 'embed', got {species_encoding!r}")
        self.species_encoding = species_encoding

        self.hash_dim = hash_dim
        self.species_embed_dim = species_embed_dim
        self.top_k = top_k
        self.top_k_species = top_k_species
        self.hidden_dims = hidden_dims if hidden_dims is not None else [2048, 1024, 512, 256, 128, 64]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.genus_emb_dim = genus_emb_dim
        self.family_emb_dim = family_emb_dim
        self.dropout = dropout

        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay

        # Resolve loss configuration
        if phases is not None:
            # Explicit phases override loss_config
            self.phases = phases
        elif isinstance(loss_config, str):
            if loss_config not in LOSS_PRESETS:
                raise ValueError(f"Unknown loss_config: {loss_config!r}. Use one of {list(LOSS_PRESETS.keys())}")
            self.phases = LOSS_PRESETS[loss_config]
        else:
            # Assume loss_config is a dict of PhaseConfig
            self.phases = loss_config
        self.phase_boundaries = phase_boundaries
        self.species_aggregation = species_aggregation
        self.species_selection = species_selection
        self.species_representation = species_representation
        self.min_species_frequency = min_species_frequency
        self.compile_model = compile_model
        self.max_grad_norm = 1.0  # Gradient clipping

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.reset_patience = reset_patience
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_files = max_cache_files
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Read species encoding config from dataset
        self.species_normalization = dataset.species_normalization
        self.track_unknown_fraction = dataset.track_unknown_fraction
        self.track_unknown_count = dataset.track_unknown_count

        # Device selection
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Enable cudnn benchmark for faster training on CUDA
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # AMP (only on CUDA)
        self.use_amp = use_amp and self._device.type == "cuda"

        # Store schema for later use (will be modified for embed mode in fit())
        self._schema = dataset.schema

        # Model will be built in fit() after vocab is ready
        self.model: Optional[ResolveModel] = None

        # Components to be initialized in fit()
        self._species_encoder: Optional[SpeciesEncoder] = None
        self._embedding_encoder: Optional[EmbeddingEncoder] = None
        self._scalers: dict[str, StandardScaler] = {}
        self._target_scalers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._optimizer: Optional[AdamW] = None
        self._scheduler: Optional[OneCycleLR] = None
        self._loss_fn: Optional[MultiTaskLoss] = None
        self._grad_scaler: Optional[GradScaler] = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def n_params(self) -> int:
        """
        Get total number of trainable parameters.

        If model has been built (after fit()), returns actual count.
        Otherwise, builds a temporary model to compute the count.
        """
        if self.model is not None:
            return sum(p.numel() for p in self.model.parameters())

        # Build temporary model to count params
        uses_explicit_vector = self.species_selection in ("all", "presence_absence")
        temp_model = ResolveModel(
            schema=self._schema,
            targets=self.dataset.targets,
            species_encoding=self.species_encoding,
            hash_dim=self.hash_dim,
            species_embed_dim=self.species_embed_dim,
            genus_emb_dim=self.genus_emb_dim,
            family_emb_dim=self.family_emb_dim,
            top_k=self.top_k,
            top_k_species=self.top_k_species,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            uses_explicit_vector=uses_explicit_vector,
        )
        return sum(p.numel() for p in temp_model.parameters())

    def _prepare_data(self, fit_encoder: bool = True) -> tuple[ResolveDataset, ResolveDataset]:
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
                    self._schema = ResolveSchema(
                        n_plots=self._schema.n_plots,
                        n_species=self._schema.n_species,
                        n_continuous=self._schema.n_continuous,
                        has_coordinates=self._schema.has_coordinates,
                        has_abundance=self._schema.has_abundance,
                        has_taxonomy=self._schema.has_taxonomy,
                        n_genera=self._schema.n_genera,
                        n_families=self._schema.n_families,
                        targets=self._schema.targets,
                        covariate_names=self._schema.covariate_names,
                        species_normalization=self._schema.species_normalization,
                        track_unknown_fraction=self._schema.track_unknown_fraction,
                        track_unknown_count=self._schema.track_unknown_count,
                        n_species_vocab=self._species_encoder.n_species_vector,
                        n_genera_vocab=0,
                        n_families_vocab=0,
                    )
        else:  # embed mode
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
                self._schema = ResolveSchema(
                    n_plots=self._schema.n_plots,
                    n_species=self._schema.n_species,
                    n_continuous=self._schema.n_continuous,
                    has_coordinates=self._schema.has_coordinates,
                    has_abundance=self._schema.has_abundance,
                    has_taxonomy=self._schema.has_taxonomy,
                    n_genera=self._schema.n_genera,
                    n_families=self._schema.n_families,
                    targets=self._schema.targets,
                    covariate_names=self._schema.covariate_names,
                    species_normalization=self._schema.species_normalization,
                    track_unknown_fraction=self._schema.track_unknown_fraction,
                    track_unknown_count=self._schema.track_unknown_count,
                    n_species_vocab=self._embedding_encoder.n_species,
                    n_genera_vocab=self._embedding_encoder.n_genera,
                    n_families_vocab=self._embedding_encoder.n_families,
                )

        return train_ds, test_ds

    def _build_tensors(
        self,
        dataset: ResolveDataset,
        fit_scalers: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Convert dataset to tensors."""
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
        else:  # embed mode
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

        continuous = np.hstack(parts) if parts else np.zeros((len(dataset.plot_ids), 0), dtype=np.float32)

        # Scale continuous features
        # Handle missing or incompatible scalers when resuming from checkpoint
        need_fit = fit_scalers
        if not fit_scalers:
            if "continuous" not in self._scalers:
                import warnings
                warnings.warn(
                    "Checkpoint missing 'continuous' scaler - fitting new scaler. "
                    "This may indicate a feature configuration mismatch.",
                    RuntimeWarning,
                )
                need_fit = True
            elif self._scalers["continuous"].n_features_in_ != continuous.shape[1]:
                import warnings
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
                    import warnings
                    warnings.warn(
                        f"Checkpoint missing '{scaler_key}' scaler - fitting new scaler.",
                        RuntimeWarning,
                    )
                    need_fit_target = True

                if need_fit_target:
                    self._scalers[scaler_key] = StandardScaler()
                    target_scaled = self._scalers[scaler_key].fit_transform(
                        target_vals[mask].reshape(-1, 1)
                    )
                    # Store scaler params for loss computation
                    scaler = self._scalers[scaler_key]
                    self._target_scalers[name] = (
                        torch.tensor(scaler.mean_[0], dtype=torch.float32, device=self._device),
                        torch.tensor(scaler.scale_[0], dtype=torch.float32, device=self._device),
                    )
                else:
                    target_scaled = self._scalers[scaler_key].transform(
                        target_vals.reshape(-1, 1)
                    )
                targets[name] = target_scaled.flatten().astype(np.float32)
            else:
                targets[name] = target_vals.astype(np.int64)

        # Build tensor dataset
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

    def _create_loaders(
        self,
        train_tensors: tuple[torch.Tensor, ...],
        test_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Create data loaders."""
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

    # --- Caching methods ---

    def _compute_cache_key(self) -> str:
        """Compute a hash key for caching based on dataset and config."""
        # Include dataset fingerprint (convert to strings to handle mixed types)
        plot_ids = sorted(str(x) for x in self.dataset._header[self.dataset._roles.plot_id].unique())
        species_ids = sorted(str(x) for x in self.dataset._species[self.dataset._roles.species_id].dropna().unique())

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

    def _cache_path(self) -> Optional[Path]:
        """Get path to cache file."""
        if self.cache_dir is None:
            return None
        cache_key = self._compute_cache_key()
        return self.cache_dir / f"preprocessed_{cache_key}.pt"

    def _save_cache(
        self,
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

    def _cleanup_old_caches(self) -> None:
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

    def _load_cache(self) -> Optional[dict]:
        """Load preprocessed data from cache if valid."""
        if self.cache_dir is None:
            return None

        cache_path = self._cache_path()
        if not cache_path.exists():
            return None

        try:
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

    def _restore_from_cache(self, cache: dict) -> tuple[tuple, tuple]:
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

    # --- Checkpoint methods ---

    def _checkpoint_path(self) -> Optional[Path]:
        """Get path to checkpoint file."""
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "checkpoint.pt"

    def _progress_path(self) -> Optional[Path]:
        """Get path to progress JSON file (human-readable)."""
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "progress.json"

    def save_checkpoint(
        self,
        epoch: int,
        best_epoch: int,
        best_metric: float,
        epochs_without_improvement: int,
        history: dict,
    ) -> None:
        """Save training checkpoint for resume."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            # Training state
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            # Model state
            "model_state_dict": self.model.state_dict(),
            "best_state": self._best_state if hasattr(self, "_best_state") else None,
            # Optimizer state
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "grad_scaler_state_dict": self._grad_scaler.state_dict() if self._grad_scaler else None,
            # Data state
            "scalers": self._scalers,
            "target_scalers": {
                k: (v[0].cpu(), v[1].cpu()) for k, v in self._target_scalers.items()
            },
            # Species encoder state
            "species_encoder": {
                "vocab": self._species_encoder._vocab if self._species_encoder else None,
                "species_vocab": self._species_encoder._species_vocab if self._species_encoder else set(),
            },
            # Config (for validation on resume)
            "config": {
                "hash_dim": self.hash_dim,
                "top_k": self.top_k,
                "hidden_dims": self.hidden_dims,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
            },
        }

        # Save checkpoint
        torch.save(checkpoint, self._checkpoint_path())

        # Save human-readable progress
        progress = {
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "best_epoch": best_epoch,
            "best_metric": float(best_metric),
            "epochs_without_improvement": epochs_without_improvement,
            "patience": self.patience,
            "progress_pct": round(100 * epoch / self.max_epochs, 1),
        }
        # Add latest metrics from history
        if history.get("test_loss"):
            progress["latest_test_loss"] = float(history["test_loss"][-1])
        with open(self._progress_path(), "w") as f:
            json.dump(progress, f, indent=2)

        print(f"  [Checkpoint saved: epoch {epoch}, best={best_metric:.2%}]")

    def load_checkpoint(self) -> Optional[dict]:
        """Load checkpoint if exists and resume=True."""
        if not self.resume or self.checkpoint_dir is None:
            return None

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return None

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Validate config matches
        saved_config = checkpoint.get("config", {})
        if saved_config.get("hash_dim") != self.hash_dim:
            print(f"  Warning: hash_dim mismatch ({saved_config.get('hash_dim')} vs {self.hash_dim})")
            return None
        if saved_config.get("hidden_dims") != self.hidden_dims:
            print(f"  Warning: hidden_dims mismatch, starting fresh")
            return None

        return checkpoint

    def _restore_scalers_from_checkpoint(self, checkpoint: dict) -> None:
        """Restore scalers and species encoder from checkpoint (before building tensors)."""
        # Restore scalers
        if checkpoint.get("scalers"):
            self._scalers = checkpoint["scalers"]
        if checkpoint.get("target_scalers"):
            self._target_scalers = {
                k: (v[0].to(self._device), v[1].to(self._device))
                for k, v in checkpoint["target_scalers"].items()
            }

        # Restore species encoder state
        if checkpoint.get("species_encoder"):
            enc_state = checkpoint["species_encoder"]
            # Create encoder if not exists
            if self._species_encoder is None:
                self._species_encoder = SpeciesEncoder(
                    hash_dim=self.hash_dim,
                    top_k=self.top_k,
                    aggregation=self.species_aggregation,
                    normalization=self.species_normalization,
                    track_unknown_count=self.track_unknown_count,
                    selection=self.species_selection,
                    representation=self.species_representation,
                )
            if enc_state.get("vocab"):
                self._species_encoder._vocab = enc_state["vocab"]
            if enc_state.get("species_vocab"):
                self._species_encoder._species_vocab = enc_state["species_vocab"]
            self._species_encoder._fitted = True

    def _restore_from_checkpoint(self, checkpoint: dict) -> tuple[int, int, float, int, dict]:
        """Restore training state from checkpoint (model, optimizer, etc.)."""
        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("best_state"):
            self._best_state = checkpoint["best_state"]

        # Restore optimizer (but NOT scheduler - we'll recreate it for remaining epochs)
        if checkpoint.get("optimizer_state_dict") and self._optimizer:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # NOTE: Scheduler state NOT restored - OneCycleLR doesn't support extending total_steps.
        # We'll recreate the scheduler for remaining epochs after this method returns.
        if checkpoint.get("grad_scaler_state_dict") and self._grad_scaler:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        # Note: Scalers already restored by _restore_scalers_from_checkpoint (called earlier)

        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_metric = checkpoint["best_metric"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]

        print(f"  Resumed from epoch {epoch} (best={best_metric:.2%} at epoch {best_epoch})")

        return epoch, best_epoch, best_metric, epochs_without_improvement, history

    def fit(self) -> TrainResult:
        """
        Train the model.

        Automatically resumes from checkpoint if available and resume=True.
        Saves checkpoints every `checkpoint_every` epochs if checkpoint_dir is set.

        Returns:
            TrainResult with metrics and history
        """
        # Check for existing checkpoint before data prep
        checkpoint = self.load_checkpoint()
        resumed_from_epoch = None

        # Try to load from cache first
        t_prep_start = time.time()
        data_cache = self._load_cache()

        if data_cache is not None:
            # Restore from cache
            train_tensors, test_tensors = self._restore_from_cache(data_cache)
            print(f"  Data loaded from cache in {time.time() - t_prep_start:.1f}s")
        else:
            # If resuming from checkpoint, load scalers and encoder BEFORE building tensors
            if checkpoint is not None:
                self._restore_scalers_from_checkpoint(checkpoint)

            # Prepare data fresh (skip encoder fitting if already restored from checkpoint)
            train_ds, test_ds = self._prepare_data(fit_encoder=(checkpoint is None))

            # Build model now that schema (with vocab sizes for embed mode) is ready
            if self.model is None:
                uses_explicit_vector = (
                    self.species_encoding == "hash" and
                    self._species_encoder is not None and
                    self._species_encoder.uses_explicit_vector
                )
                # Get actual taxonomy slot count (2*top_k for top_bottom mode)
                n_taxonomy_slots = (
                    self._species_encoder.n_taxonomy_slots
                    if self._species_encoder else self.top_k
                )
                self.model = ResolveModel(
                    schema=self._schema,
                    targets=self.dataset.targets,
                    species_encoding=self.species_encoding,
                    hash_dim=self.hash_dim,
                    species_embed_dim=self.species_embed_dim,
                    genus_emb_dim=self.genus_emb_dim,
                    family_emb_dim=self.family_emb_dim,
                    top_k=n_taxonomy_slots,
                    top_k_species=self.top_k_species,
                    hidden_dims=self.hidden_dims,
                    dropout=self.dropout,
                    uses_explicit_vector=uses_explicit_vector,
                )

            train_tensors = self._build_tensors(train_ds, fit_scalers=(checkpoint is None))
            test_tensors = self._build_tensors(test_ds, fit_scalers=False)
            print(f"  Data prepared in {time.time() - t_prep_start:.1f}s")

            # Save to cache for next time
            if self.cache_dir:
                self._save_cache(
                    train_tensors,
                    test_tensors,
                    train_indices=np.array([]),  # Could store actual indices if needed
                    test_indices=np.array([]),
                )

        self._create_loaders(train_tensors, test_tensors)

        # Build model now that schema (with vocab sizes for embed mode) is ready
        if self.model is None:
            uses_explicit_vector = (
                self.species_encoding == "hash" and
                self._species_encoder is not None and
                self._species_encoder.uses_explicit_vector
            )
            # Get actual taxonomy slot count (2*top_k for top_bottom mode)
            n_taxonomy_slots = (
                self._species_encoder.n_taxonomy_slots
                if self._species_encoder else self.top_k
            )
            self.model = ResolveModel(
                schema=self._schema,
                targets=self.dataset.targets,
                species_encoding=self.species_encoding,
                hash_dim=self.hash_dim,
                species_embed_dim=self.species_embed_dim,
                genus_emb_dim=self.genus_emb_dim,
                family_emb_dim=self.family_emb_dim,
                top_k=n_taxonomy_slots,
                top_k_species=self.top_k_species,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                uses_explicit_vector=uses_explicit_vector,
            )

        # Move model to device
        self.model.to(self._device)

        # Compile model for potential speedup (PyTorch 2.0+)
        compiled = False
        if self.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                compiled = True
            except Exception as e:
                print(f"  Warning: torch.compile failed ({e}), using eager mode")
        if compiled:
            print(f"Training on: {self._device} (AMP: {self.use_amp}, batch_size: {self.batch_size}, compiled: True)")
        else:
            print(f"Training on: {self._device} (AMP: {self.use_amp}, batch_size: {self.batch_size})")

        # Setup optimizer and scheduler
        self._optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Calculate total steps for scheduler
        # If resuming, we'll recreate the scheduler for remaining epochs
        steps_per_epoch = len(self._train_loader)
        total_steps = self.max_epochs * steps_per_epoch
        self._scheduler = OneCycleLR(
            self._optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self._steps_per_epoch = steps_per_epoch  # Store for resume logic

        # Setup AMP gradient scaler
        if self.use_amp:
            self._grad_scaler = GradScaler()

        # Setup loss
        self._loss_fn = MultiTaskLoss(
            self.model.target_configs,
            phases=self.phases,
            phase_boundaries=self.phase_boundaries,
        )

        # Initialize training state
        start_epoch = 0
        best_metric = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = {"train_loss": [], "test_loss": []}

        # Restore from checkpoint if available
        if checkpoint is not None:
            start_epoch, best_epoch, best_metric, epochs_without_improvement, history = \
                self._restore_from_checkpoint(checkpoint)
            start_epoch += 1  # Continue from next epoch
            resumed_from_epoch = start_epoch - 1

            # Reset patience counter if requested (allows continuing after early stop)
            if self.reset_patience:
                print(f"  Resetting patience counter (was {epochs_without_improvement})")
                epochs_without_improvement = 0

            # Check if max_epochs was increased
            saved_max = checkpoint.get("config", {}).get("max_epochs", self.max_epochs)
            if self.max_epochs > saved_max:
                print(f"  max_epochs increased: {saved_max} -> {self.max_epochs}")

            # Recreate scheduler for remaining epochs
            # OneCycleLR doesn't support extending total_steps, so we create a fresh one
            remaining_epochs = self.max_epochs - start_epoch
            if remaining_epochs > 0:
                remaining_steps = remaining_epochs * self._steps_per_epoch
                self._scheduler = OneCycleLR(
                    self._optimizer,
                    max_lr=self.lr,
                    total_steps=remaining_steps,
                    pct_start=0.1,
                    anneal_strategy="cos",
                )
                print(f"  Scheduler recreated for {remaining_epochs} remaining epochs ({remaining_steps} steps)")

        target_names = list(self.model.target_configs.keys())
        has_taxonomy = self.model.schema.has_taxonomy

        train_start_time = time.time()
        for epoch in range(start_epoch, self.max_epochs):
            # Train
            train_loss = self._train_epoch(epoch, target_names, has_taxonomy)
            history["train_loss"].append(train_loss)

            # Evaluate
            test_loss, metrics = self._eval_epoch(epoch, target_names, has_taxonomy)
            history["test_loss"].append(test_loss)

            # Track best by first regression target's band_25 or classification accuracy
            first_target = target_names[0]
            cfg = self.model.target_configs[first_target]
            if cfg.task == "regression":
                current_metric = metrics[first_target]["band_25"]
            else:
                current_metric = metrics[first_target]["accuracy"]

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            # Log progress
            phase = self._loss_fn.phased_loss.get_phase(epoch)
            metric_str = " | ".join(
                f"{name}: {metrics[name].get('band_25', metrics[name].get('accuracy', 0)):.2%}"
                for name in target_names
            )
            print(
                f"Epoch {epoch:3d} [P{phase}] | "
                f"train={train_loss:.4f} test={test_loss:.4f} | {metric_str}"
            )

            # Save checkpoint periodically (always after epoch 1, then every checkpoint_every)
            if self.checkpoint_dir and (epoch == 0 or (epoch + 1) % self.checkpoint_every == 0):
                self.save_checkpoint(epoch, best_epoch, best_metric, epochs_without_improvement, history)

            # Early stopping
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                # Save final checkpoint
                if self.checkpoint_dir:
                    self.save_checkpoint(epoch, best_epoch, best_metric, epochs_without_improvement, history)
                break

        # Restore best model
        self.model.load_state_dict(self._best_state)
        train_time = time.time() - train_start_time

        # Final evaluation
        _, final_metrics = self._eval_epoch(best_epoch, target_names, has_taxonomy)

        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint(epoch, best_epoch, best_metric, epochs_without_improvement, history)

        return TrainResult(
            best_epoch=best_epoch,
            final_metrics=final_metrics,
            history=history,
            resumed_from_epoch=resumed_from_epoch,
            train_time=train_time,
        )

    def _train_epoch(
        self,
        epoch: int,
        target_names: list[str],
        has_taxonomy: bool,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in self._train_loader:
            # Unpack batch - use non_blocking=True with pin_memory for async transfer
            idx = 0
            continuous = batch[idx].to(self._device, non_blocking=True)
            idx += 1

            # species_ids only present in embed mode
            # species_vector for hash mode with all/presence_absence selection
            species_ids = None
            species_vector = None
            if self.species_encoding == "embed":
                species_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
            elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
                species_vector = batch[idx].to(self._device, non_blocking=True)
                idx += 1

            if has_taxonomy:
                genus_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
                family_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx].to(self._device, non_blocking=True)
                idx += 1

            # Reshape targets for loss
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            # Forward + backward with optional AMP
            self._optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                    loss, _ = self._loss_fn(
                        predictions, targets, epoch, self._target_scalers
                    )
                self._grad_scaler.scale(loss).backward()
                # Unscale before gradient clipping
                self._grad_scaler.unscale_(self._optimizer)
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                loss, _ = self._loss_fn(
                    predictions, targets, epoch, self._target_scalers
                )
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optimizer.step()

            # Step scheduler after optimizer (fixes PyTorch warning)
            self._scheduler.step()

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected at batch, skipping...")
                continue

            total_loss += loss.item() * continuous.size(0)

        return total_loss / len(self._train_loader.dataset)

    @torch.no_grad()
    def _eval_epoch(
        self,
        epoch: int,
        target_names: list[str],
        has_taxonomy: bool,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0

        all_preds = {name: [] for name in target_names}
        all_targets = {name: [] for name in target_names}

        for batch in self._test_loader:
            idx = 0
            continuous = batch[idx].to(self._device, non_blocking=True)
            idx += 1

            # species_ids only present in embed mode
            # species_vector for hash mode with all/presence_absence selection
            species_ids = None
            species_vector = None
            if self.species_encoding == "embed":
                species_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
            elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
                species_vector = batch[idx].to(self._device, non_blocking=True)
                idx += 1

            if has_taxonomy:
                genus_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
                family_ids = batch[idx].to(self._device, non_blocking=True)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx].to(self._device, non_blocking=True)
                idx += 1

            # Use AMP for faster eval inference
            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
            else:
                predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)

            # Reshape for loss
            targets_for_loss = {}
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets_for_loss[name] = targets[name].unsqueeze(-1)
                else:
                    targets_for_loss[name] = targets[name]

            loss, _ = self._loss_fn(
                predictions, targets_for_loss, epoch, self._target_scalers
            )
            total_loss += loss.item() * continuous.size(0)

            # Collect predictions
            for name in target_names:
                cfg = self.model.target_configs[name]
                pred = predictions[name]

                if cfg.task == "regression":
                    # Inverse scale
                    scaler = self._scalers[f"target_{name}"]
                    pred_np = pred.cpu().numpy()
                    pred_np = scaler.inverse_transform(pred_np).flatten()
                    target_np = scaler.inverse_transform(
                        targets[name].cpu().numpy().reshape(-1, 1)
                    ).flatten()
                else:
                    pred_np = pred.argmax(dim=-1).cpu().numpy()
                    target_np = targets[name].cpu().numpy()

                all_preds[name].append(pred_np)
                all_targets[name].append(target_np)

        avg_loss = total_loss / len(self._test_loader.dataset)

        # Compute metrics
        metrics = {}
        for name in target_names:
            cfg = self.model.target_configs[name]
            pred = np.concatenate(all_preds[name])
            target = np.concatenate(all_targets[name])
            metrics[name] = compute_metrics(pred, target, cfg.task, cfg.transform)

        return avg_loss, metrics

    def save(self, path: str | Path) -> None:
        """Save model, encoder, and scalers."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "schema": self.model.schema,
            "target_configs": self.model.target_configs,
            "hash_dim": self.model.hash_dim,
            "top_k": self.model.top_k,
            "hidden_dims": self.model.hidden_dims,
            "genus_emb_dim": self.model.genus_emb_dim,
            "family_emb_dim": self.model.family_emb_dim,
            "dropout": self.model.dropout,
            "scalers": self._scalers,
            "vocab": self._species_encoder.vocab if self._species_encoder else None,
            "species_aggregation": self._species_encoder.aggregation if self._species_encoder else "abundance",
            "species_selection": self._species_encoder.selection if self._species_encoder else "top",
            "species_representation": self._species_encoder.representation if self._species_encoder else "abundance",
            "species_normalization": self._species_encoder.normalization if self._species_encoder else "norm",
            "track_unknown_fraction": self.track_unknown_fraction,
            "track_unknown_count": self._species_encoder.track_unknown_count if self._species_encoder else False,
            "species_vocab": self._species_encoder._species_vocab if self._species_encoder else set(),
            "uses_explicit_vector": self.model.uses_explicit_vector,
        }
        torch.save(state, path)

    @torch.no_grad()
    def predict(
        self,
        dataset: ResolveDataset,
        output_space: str = "raw",
        confidence_threshold: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """
        Predict on a dataset.

        Args:
            dataset: ResolveDataset to predict on
            output_space: "raw" (original scale) or "transformed" (model scale)
            confidence_threshold: Minimum confidence for predictions (0-1).
                Predictions below threshold are set to NaN.
                Default 0 means all predictions are kept (gap-fill everything).

                Confidence semantics:
                - Regression: confidence = 1 - unknown_fraction, where unknown_fraction
                  is the proportion of species abundance not seen during training.
                  This reflects coverage of the species space, not statistical uncertainty.
                - Classification: confidence = max softmax probability across classes.

                These values are heuristic and intended for filtering/diagnostics,
                not formal uncertainty quantification.

        Returns:
            Dict mapping target name to predictions array
        """
        if self._species_encoder is None:
            raise RuntimeError("Trainer must be fit before predict")

        self.model.eval()

        # Encode and scale
        encoded = self._species_encoder.transform(dataset)
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()

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
        continuous = np.hstack(parts)
        continuous = self._scalers["continuous"].transform(continuous).astype(np.float32)

        # To tensors
        continuous_t = torch.from_numpy(continuous).to(self._device)
        genus_t = None
        family_t = None
        if encoded.genus_ids is not None:
            genus_t = torch.from_numpy(encoded.genus_ids).to(self._device)
            family_t = torch.from_numpy(encoded.family_ids).to(self._device)

        # Forward
        preds_raw = self.model(continuous_t, genus_t, family_t)

        # Compute confidence per sample (1 - unknown_fraction for regression)
        confidence = 1.0 - encoded.unknown_fraction

        # Post-process
        predictions = {}
        for name, pred in preds_raw.items():
            cfg = self.model.target_configs[name]
            if cfg.task == "regression":
                pred_np = pred.cpu().numpy()
                scaler = self._scalers[f"target_{name}"]
                pred_np = scaler.inverse_transform(pred_np).flatten()
                if cfg.transform == "log1p" and output_space == "raw":
                    pred_np = np.expm1(pred_np)
                # Apply confidence threshold
                pred_np = np.where(confidence >= confidence_threshold, pred_np, np.nan)
                predictions[name] = pred_np
            else:
                # Classification: use max softmax probability as confidence
                probs = torch.softmax(pred, dim=-1)
                class_confidence = probs.max(dim=-1).values.cpu().numpy()
                pred_np = pred.argmax(dim=-1).cpu().numpy().astype(np.float64)
                # Apply confidence threshold
                pred_np = np.where(class_confidence >= confidence_threshold, pred_np, np.nan)
                predictions[name] = pred_np

        return predictions

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> tuple[ResolveModel, SpeciesEncoder, dict]:
        """
        Load model from checkpoint.

        Returns:
            (model, species_encoder, scalers)
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        track_unknown_count = state.get("track_unknown_count", False)
        uses_explicit_vector = state.get("uses_explicit_vector", False)

        model = ResolveModel(
            schema=state["schema"],
            targets=state["target_configs"],
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
            hidden_dims=state.get("hidden_dims"),
            genus_emb_dim=state.get("genus_emb_dim", 8),
            family_emb_dim=state.get("family_emb_dim", 8),
            dropout=state.get("dropout", 0.3),
            track_unknown_count=track_unknown_count,
            uses_explicit_vector=uses_explicit_vector,
        )
        model.load_state_dict(state["model_state_dict"])

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        model.to(dev)

        encoder = SpeciesEncoder(
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
            aggregation=state.get("species_aggregation", "abundance"),
            normalization=state.get("species_normalization", "norm"),
            track_unknown_count=track_unknown_count,
            selection=state.get("species_selection", "top"),
            representation=state.get("species_representation", "abundance"),
        )
        if state["vocab"] is not None:
            encoder._vocab = state["vocab"]
        # Restore species vocabulary for unknown mass calculation
        encoder._species_vocab = state.get("species_vocab", set())
        encoder._fitted = True

        return model, encoder, state["scalers"]
