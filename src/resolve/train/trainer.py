"""Trainer: training orchestration for ResolveModel."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
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
from resolve.train._loaders import (
    CUDAPrefetcher,
    GPUTensorLoader,
    RankPoolBatchDataset,
    _RankPoolPreparedData,
    _rank_pool_collate_fn,
)
from resolve.train._types import ProfileResult, Timer, TrainResult

# Preset loss configurations
LOSS_PRESETS = {
    "mae": {1: PhaseConfig(mae=1.0)},
    "combined": {1: PhaseConfig(mae=0.80, smape=0.15, band=0.05)},
    "smape": {1: PhaseConfig(mae=0.5, smape=0.5)},
}
from resolve.train.metrics import compute_metrics

# Cache version - increment when cache format changes
_CACHE_VERSION = 1


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
        species_encoding: str = "hash",
        # Model architecture
        hash_dim: int = 32,
        species_embed_dim: int = 32,
        top_k: int = 5,
        top_k_species: int = 10,
        hidden_dims: Optional[list[int]] = None,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        dropout: float = 0.3,
        # Training
        batch_size: int = 32768,
        num_workers: int = 0,
        max_epochs: int = 500,
        patience: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Checkpointing
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_every: int = 50,
        resume: bool = True,
        reset_patience: bool = False,
        # Caching
        cache_dir: Optional[str | Path] = None,
        max_cache_files: int = 5,
        # Loss configuration
        loss_config: str = "mae",
        # Advanced (deprecated - use loss_config instead)
        phases: Optional[dict[int, PhaseConfig]] = None,
        phase_boundaries: Optional[list[int]] = None,
        device: str = "auto",
        use_amp: bool = True,
        compile_model: bool = False,
        prefetch_data: Optional[bool] = None,
        gpu_data: Optional[bool] = None,
        species_aggregation: str = "abundance",
        species_selection: str = "top",
        species_representation: str = "abundance",
        min_species_frequency: int = 1,
        cover_dropout: float = 0.0,
        # Transformer-specific (species_encoding="transformer")
        n_attention_layers: int = 0,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_pooling: str = "attention",
        transformer_dropout: float = 0.1,
        pretrain_epochs: int = 0,
        pretrain_mask_prob: float = 0.15,
        pretrain_lr: float = 1e-4,
        # v7: label smoothing, class weights, EMA, deeper head
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        ema_decay: float = 0.0,
        head_hidden_dims: Optional[list[int]] = None,
        verbose: int = 1,
    ):
        """
        Initialize trainer for RESOLVE models.

        The trainer automatically constructs the model from the dataset schema.
        Call fit() to train and predict() to make predictions.

        Args:
            dataset: Training dataset containing plots and species data.

            species_encoding: How to encode species composition.
                - "hash": Feature hashing for fixed-dim embedding (default, faster)
                - "embed": Learned embeddings per species (more expressive)

            hash_dim: Dimension of hashed species embedding (hash mode only).
            species_embed_dim: Embedding dimension per species (embed mode only).
            top_k: Number of top genera/families to track for taxonomy embeddings.
            top_k_species: Number of top species for embed mode.
            hidden_dims: List of hidden layer dimensions. Default: [2048, 1024, 512, 256, 128, 64].
            genus_emb_dim: Embedding dimension for genus (if taxonomy available).
            family_emb_dim: Embedding dimension for family (if taxonomy available).
            dropout: Dropout rate for regularization. Must be in [0, 1).

            batch_size: Training batch size. Larger values improve GPU utilization.
            num_workers: DataLoader workers. 0 is safest on Windows.
            max_epochs: Maximum training epochs before stopping.
            patience: Early stopping patience (epochs without improvement).
            lr: Learning rate for AdamW optimizer.
            weight_decay: L2 regularization weight.

            checkpoint_dir: Directory to save training checkpoints. If None, no checkpoints.
            checkpoint_every: Save checkpoint every N epochs.
            resume: If True, resume from existing checkpoint in checkpoint_dir.
            reset_patience: If True, reset early stopping counter when resuming.

            cache_dir: Directory to cache preprocessed tensors. Speeds up restarts.
            max_cache_files: Maximum cache files to keep (oldest deleted first).

            loss_config: Loss function preset.
                - "mae": Pure MAE loss (default, most stable)
                - "combined": 80% MAE + 15% SMAPE + 5% band accuracy
                - "smape": 50% MAE + 50% SMAPE

            device: Compute device. "auto" selects CUDA if available, else CPU.
            use_amp: Use automatic mixed precision on CUDA (faster, less memory).
            compile_model: Use torch.compile() for potential speedup (experimental).
            prefetch_data: Use async data prefetching on CUDA.
                - None (default): Auto-enable for batch_size >= 16384
                - True: Always enable
                - False: Always disable
                Only beneficial with very large batch sizes where GPU compute
                time is long enough to hide the stream synchronization overhead.

            gpu_data: Store training data on GPU for faster batch sampling.
                - None (default): Auto-enable on CUDA (eliminates DataLoader bottleneck)
                - True: Always enable (requires GPU with enough VRAM)
                - False: Always disable (use CPU DataLoader)
                Provides 10-20x speedup by avoiding CPU→GPU transfer overhead.
                Typical VRAM usage: ~500 MB for 1.5M samples.

            species_aggregation: How to aggregate species for top-k selection.
                - "abundance": Weight by abundance (default)
                - "count": Count occurrences

            species_selection: Which species to include in encoding.
                - "top": Top-K most abundant (default, uses hash embedding)
                - "bottom": Bottom-K least abundant
                - "top_bottom": Top-K + Bottom-K (2K total)
                - "all": All species (explicit vector, see species_representation)

            species_representation: How to represent species (only for selection="all").
                - "abundance": Weighted by abundance (default)
                - "presence_absence": Binary 0/1

            min_species_frequency: For selection="all", only include species in N+ plots.

            verbose: Verbosity level.
                - 0: Silent (no output)
                - 1: Normal progress (default)
                - 2: Debug (batch-level statistics)

        Raises:
            ValueError: If any parameter is invalid.

        Example:
            >>> trainer = Trainer(dataset, loss_config="mae")
            >>> result = trainer.fit()
            >>> predictions = trainer.predict(test_dataset)
        """
        self.dataset = dataset

        # === Parameter Validation ===
        # Species encoding
        if species_encoding not in ("hash", "embed", "rank_pool", "transformer"):
            raise ValueError(f"species_encoding must be 'hash', 'embed', 'rank_pool', or 'transformer', got {species_encoding!r}")
        self.species_encoding = species_encoding

        # Dimension parameters
        if hash_dim < 1:
            raise ValueError(f"hash_dim must be >= 1, got {hash_dim}")
        if species_embed_dim < 1:
            raise ValueError(f"species_embed_dim must be >= 1, got {species_embed_dim}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k_species < 1:
            raise ValueError(f"top_k_species must be >= 1, got {top_k_species}")
        if genus_emb_dim < 1:
            raise ValueError(f"genus_emb_dim must be >= 1, got {genus_emb_dim}")
        if family_emb_dim < 1:
            raise ValueError(f"family_emb_dim must be >= 1, got {family_emb_dim}")

        # Training parameters
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        # Species selection mode
        valid_selections = ("top", "bottom", "top_bottom", "all")
        if species_selection not in valid_selections:
            raise ValueError(f"species_selection must be one of {valid_selections}, got {species_selection!r}")

        # Species representation mode
        valid_representations = ("abundance", "presence_absence")
        if species_representation not in valid_representations:
            raise ValueError(f"species_representation must be one of {valid_representations}, got {species_representation!r}")

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
        self.cover_dropout = cover_dropout
        self.n_attention_layers = n_attention_layers
        self.n_heads = n_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_pooling = transformer_pooling
        self.transformer_dropout = transformer_dropout
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_mask_prob = pretrain_mask_prob
        self.pretrain_lr = pretrain_lr
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.ema_decay = ema_decay
        self.head_hidden_dims = head_hidden_dims
        self.compile_model = compile_model
        # Auto-enable prefetch for large batch sizes (16K+)
        if prefetch_data is None:
            self.prefetch_data = batch_size >= 16384
        else:
            self.prefetch_data = prefetch_data
        # GPU data will be resolved after device is known (below)
        self._gpu_data_setting = gpu_data
        self.max_grad_norm = 1.0  # Gradient clipping
        self.verbose = verbose

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

        # GPU data storage: auto-enable on CUDA (eliminates DataLoader bottleneck)
        if self._gpu_data_setting is None:
            self.gpu_data = self._device.type == "cuda"
        else:
            self.gpu_data = self._gpu_data_setting and self._device.type == "cuda"

        # Store schema for later use (will be modified for embed mode in fit())
        self._schema = dataset.schema

        # Model will be built in fit() after vocab is ready
        self.model: Optional[ResolveModel] = None

        # Components to be initialized in fit()
        self._species_encoder: Optional[SpeciesEncoder] = None
        self._embedding_encoder: Optional[EmbeddingEncoder] = None
        self._rank_pool_encoder = None  # Optional[RankPoolEncoder]
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

        temp_model = self._build_model()
        return sum(p.numel() for p in temp_model.parameters())

    def _build_model(self) -> ResolveModel:
        """Build ResolveModel from current trainer config.

        Computes uses_explicit_vector and n_taxonomy_slots from encoder state.
        All model construction sites should use this to stay DRY.
        """
        uses_explicit_vector = (
            self.species_encoding == "hash"
            and self._species_encoder is not None
            and self._species_encoder.uses_explicit_vector
        )
        n_taxonomy_slots = (
            self._species_encoder.n_taxonomy_slots
            if self._species_encoder else self.top_k
        )
        return ResolveModel(
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
            cover_dropout=self.cover_dropout,
            n_attention_layers=self.n_attention_layers,
            n_heads=self.n_heads,
            transformer_ff_dim=self.transformer_ff_dim,
            transformer_pooling=self.transformer_pooling,
            transformer_dropout=self.transformer_dropout,
            head_hidden_dims=self.head_hidden_dims,
        )

    def _unpack_batch(
        self,
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
        else:  # rank_pool or transformer mode (both use same data pipeline)
            from resolve.encode.rank_pool import RankPoolEncoder
            if fit_encoder or self._rank_pool_encoder is None or not self._rank_pool_encoder._fitted:
                self._rank_pool_encoder = RankPoolEncoder(
                    weighting=self.species_normalization,
                    min_species_frequency=self.min_species_frequency,
                )
                self._rank_pool_encoder.fit(train_ds)

                # Update schema with vocab sizes
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
                    n_species_vocab=self._rank_pool_encoder.n_species,
                    n_genera_vocab=self._rank_pool_encoder.n_genera,
                    n_families_vocab=self._rank_pool_encoder.n_families,
                )

        return train_ds, test_ds

    def _build_tensors(
        self,
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

    def _create_loaders(self, train_data, test_data) -> None:
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

    # --- Caching methods ---

    def _compute_cache_key(self) -> str:
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
                "species_encoding": self.species_encoding,
                "species_selection": self.species_selection,
                "species_representation": self.species_representation,
                "genus_emb_dim": self.genus_emb_dim,
                "family_emb_dim": self.family_emb_dim,
                "n_attention_layers": self.n_attention_layers,
                "n_heads": self.n_heads,
                "transformer_ff_dim": self.transformer_ff_dim,
                "transformer_pooling": self.transformer_pooling,
                "transformer_dropout": self.transformer_dropout,
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
        # Note: weights_only=False is required for sklearn scalers and encoder state.
        # Only load checkpoint files from trusted sources.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Validate config matches - architecture parameters must match exactly
        saved_config = checkpoint.get("config", {})
        config_mismatches = []

        # Critical architecture parameters
        if saved_config.get("hash_dim") != self.hash_dim:
            config_mismatches.append(
                f"hash_dim: checkpoint={saved_config.get('hash_dim')}, current={self.hash_dim}"
            )
        if saved_config.get("hidden_dims") != self.hidden_dims:
            config_mismatches.append(
                f"hidden_dims: checkpoint={saved_config.get('hidden_dims')}, current={self.hidden_dims}"
            )
        if saved_config.get("top_k") != self.top_k:
            config_mismatches.append(
                f"top_k: checkpoint={saved_config.get('top_k')}, current={self.top_k}"
            )
        if saved_config.get("species_encoding") != self.species_encoding:
            config_mismatches.append(
                f"species_encoding: checkpoint={saved_config.get('species_encoding')}, current={self.species_encoding}"
            )
        if saved_config.get("species_selection") != self.species_selection:
            config_mismatches.append(
                f"species_selection: checkpoint={saved_config.get('species_selection')}, current={self.species_selection}"
            )

        if config_mismatches:
            print("  Warning: Cannot resume - configuration mismatch:")
            for mismatch in config_mismatches:
                print(f"    - {mismatch}")
            print("  Starting fresh training run.")
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

        # Restore species encoder state (hash mode only; rank_pool/embed use different encoders)
        if checkpoint.get("species_encoder") and self.species_encoding == "hash":
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

    def pretrain(self) -> None:
        """Run masked species pretraining (v6) on the transformer encoder.

        Trains the encoder with BERT-style masked language modelling over species
        tokens. Uses a separate MaskedSpeciesHead (discarded after pretraining).

        Must be called BEFORE fit(). Requires species_encoding="transformer" and
        pretrain_epochs > 0.

        The method:
          1. Prepares data (same pipeline as rank_pool)
          2. Builds model if not already built
          3. Runs MLM pretraining loop
          4. Discards the MaskedSpeciesHead, keeps encoder weights
        """
        if self.species_encoding != "transformer":
            raise ValueError("pretrain() requires species_encoding='transformer'")
        if self.pretrain_epochs < 1:
            raise ValueError("pretrain() requires pretrain_epochs >= 1")

        from resolve.model.pretrain import MaskedSpeciesHead, MaskedSpeciesCollateWrapper

        print("\n=== Masked Species Pretraining (v6) ===")
        print(f"  Epochs: {self.pretrain_epochs}")
        print(f"  Mask prob: {self.pretrain_mask_prob}")
        print(f"  LR: {self.pretrain_lr}")

        # Prepare data (same as fit, but only need train split)
        train_ds, _ = self._prepare_data(fit_encoder=True)

        # Build model if not done yet
        if self.model is None:
            self.model = self._build_model()

        self.model.to(self._device)

        # Build tensors and data loader with MLM masking
        train_tensors = self._build_tensors(train_ds, fit_scalers=True)
        has_taxonomy = self._schema.has_taxonomy

        # Create masking collate wrapper
        from resolve.train.trainer import _rank_pool_collate_fn, RankPoolBatchDataset
        mlm_collate = MaskedSpeciesCollateWrapper(
            base_collate_fn=_rank_pool_collate_fn,
            n_species=self._schema.n_species_vocab,
            mask_prob=self.pretrain_mask_prob,
            has_taxonomy=has_taxonomy,
        )

        pretrain_loader = DataLoader(
            RankPoolBatchDataset(train_tensors),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=mlm_collate,
            drop_last=True,
        )

        # Build MLM head (will be discarded after pretraining)
        encoder = self.model.encoder
        mlm_head = MaskedSpeciesHead(encoder.d_model, self._schema.n_species_vocab)
        mlm_head.to(self._device)

        # Optimizer for encoder + MLM head only (not task heads)
        pretrain_params = list(encoder.parameters()) + list(mlm_head.parameters())
        optimizer = AdamW(pretrain_params, lr=self.pretrain_lr, weight_decay=self.weight_decay)

        # AMP scaler
        grad_scaler = GradScaler() if self.use_amp else None

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        for epoch in range(1, self.pretrain_epochs + 1):
            encoder.train()
            mlm_head.train()
            total_loss = 0.0
            n_batches = 0

            for batch in pretrain_loader:
                # Unpack batch: (continuous, masked_sp, [g, f,] w, mask, has_cover, *targets, mlm_mask, mlm_targets)
                idx = 0
                continuous = batch[idx].to(self._device, non_blocking=True); idx += 1
                species_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                if has_taxonomy:
                    pool_genus_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                    pool_family_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                else:
                    pool_genus_ids = None
                    pool_family_ids = None
                pool_weights = batch[idx].to(self._device, non_blocking=True); idx += 1
                pool_mask = batch[idx].to(self._device, non_blocking=True); idx += 1
                pool_has_cover = batch[idx].to(self._device, non_blocking=True); idx += 1

                # Skip targets (we don't need them for pretraining)
                n_targets = len(self.model.target_configs)
                idx += n_targets

                mlm_mask = batch[idx].to(self._device, non_blocking=True); idx += 1
                mlm_targets = batch[idx].to(self._device, non_blocking=True); idx += 1

                # Forward through encoder to get token-level representations
                # We need the pre-pooling token embeddings, not the pooled output
                # Re-run embedding + transformer without pooling
                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast(device_type="cuda"):
                        token_embs = self._get_pretrain_tokens(
                            encoder, continuous, species_ids, pool_genus_ids,
                            pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                            mlm_mask,
                        )
                        # Extract masked positions and predict
                        masked_embs = token_embs[mlm_mask]  # (N_masked, d_model)
                        logits = mlm_head(masked_embs)  # (N_masked, n_species)
                        targets = mlm_targets[mlm_mask]  # (N_masked,)
                        loss = loss_fn(logits, targets)

                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    token_embs = self._get_pretrain_tokens(
                        encoder, continuous, species_ids, pool_genus_ids,
                        pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                        mlm_mask,
                    )
                    masked_embs = token_embs[mlm_mask]
                    logits = mlm_head(masked_embs)
                    targets = mlm_targets[mlm_mask]
                    loss = loss_fn(logits, targets)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if self.verbose >= 1:
                print(f"  Pretrain epoch {epoch}/{self.pretrain_epochs}: MLM loss = {avg_loss:.4f}")

        # Discard MLM head, keep encoder weights
        del mlm_head
        print("  Pretraining complete. MLM head discarded, encoder weights retained.")

    @staticmethod
    def _get_pretrain_tokens(
        encoder, continuous, species_ids, genus_ids, family_ids,
        weights, mask, has_cover, mlm_mask,
    ) -> torch.Tensor:
        """Get token-level embeddings from PlotEncoderTransformer (pre-pooling).

        Runs the embedding + self-attention layers but skips pooling and MLP,
        returning per-token representations for MLM prediction.
        """
        batch_size = continuous.shape[0]

        if has_cover is None:
            has_cover = torch.ones(batch_size, device=continuous.device)

        if mask is None:
            mask = species_ids != 0

        # Additive token embeddings
        tokens = encoder.species_embedding(species_ids)
        if encoder.has_taxonomy and genus_ids is not None and family_ids is not None:
            tokens = tokens + encoder.genus_embedding(genus_ids) + encoder.family_embedding(family_ids)
        if weights is not None:
            tokens = tokens + encoder.weight_proj(weights.unsqueeze(-1))

        # Apply mask embedding at MLM positions
        tokens = tokens.clone()
        tokens[mlm_mask] = encoder.mask_embedding

        # Self-attention
        padding_mask = ~mask
        if encoder.transformer_encoder is not None:
            tokens = encoder.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        return tokens

    def fit(self) -> TrainResult:
        """
        Train the model.

        Automatically resumes from checkpoint if available and resume=True.
        Saves checkpoints every `checkpoint_every` epochs if checkpoint_dir is set.

        Returns:
            TrainResult with metrics and history
        """
        # Suppress harmless PyTorch warning about scheduler step order on first batch
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=".*lr_scheduler.step\\(\\) before optimizer.step\\(\\).*",
            category=UserWarning,
        )

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
                self.model = self._build_model()

            train_tensors = self._build_tensors(train_ds, fit_scalers=(checkpoint is None))
            test_tensors = self._build_tensors(test_ds, fit_scalers=False)
            print(f"  Data prepared in {time.time() - t_prep_start:.1f}s")

            # Save to cache for next time (skip rank_pool mode - ragged data isn't cacheable)
            if self.cache_dir and not isinstance(train_tensors, _RankPoolPreparedData):
                self._save_cache(
                    train_tensors,
                    test_tensors,
                    train_indices=np.array([]),  # Could store actual indices if needed
                    test_indices=np.array([]),
                )

        self._create_loaders(train_tensors, test_tensors)

        # Build model now that schema (with vocab sizes for embed mode) is ready
        if self.model is None:
            self.model = self._build_model()

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
        # Ensure at least 1 step per epoch (handles batch_size > dataset_size)
        steps_per_epoch = max(1, len(self._train_loader))
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
            label_smoothing=self.label_smoothing,
            class_weights=self.class_weights,
        )

        # Initialize EMA state (exponential moving average of model weights)
        self._ema_state = None
        if self.ema_decay > 0:
            self._ema_state = {k: v.clone() for k, v in self.model.state_dict().items()}

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
        epoch_times = []  # Track epoch durations for ETA
        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()
            # Train
            train_loss = self._train_epoch(epoch, target_names, has_taxonomy)
            history["train_loss"].append(train_loss)

            # Evaluate (swap to EMA weights if active, swap back after)
            if self._ema_state is not None:
                _train_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.model.load_state_dict(self._ema_state)

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
                # Save EMA state as best when EMA is active, otherwise save current model
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            # Swap back to training weights after eval
            if self._ema_state is not None:
                self.model.load_state_dict(_train_state)

            # Track epoch time and compute ETA
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            avg_epoch_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])  # Rolling avg of last 10
            remaining_epochs = self.max_epochs - epoch - 1
            eta_seconds = remaining_epochs * avg_epoch_time

            # Format ETA
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            # Log progress
            phase = self._loss_fn.phased_loss.get_phase(epoch)
            metric_str = " | ".join(
                f"{name}: {metrics[name].get('band_25', metrics[name].get('accuracy', 0)):.2%}"
                for name in target_names
            )
            print(
                f"Epoch {epoch:3d} [P{phase}] | "
                f"train={train_loss:.4f} test={test_loss:.4f} | {metric_str} | "
                f"{epoch_time:.1f}s/ep, ETA {eta_str}"
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

    def profile(
        self,
        n_batches: int = 50,
        warmup_batches: int = 5,
        save_trace: bool = False,
        trace_dir: Optional[str | Path] = None,
    ) -> ProfileResult:
        """
        Profile training performance to identify bottlenecks.

        Runs a small number of training batches with detailed timing,
        breaking down time spent in forward pass, backward pass,
        optimizer step, and data loading.

        Args:
            n_batches: Number of batches to profile (after warmup).
            warmup_batches: Number of warmup batches to run first (not timed).
            save_trace: If True, save detailed Chrome trace for analysis.
            trace_dir: Directory to save trace files. Default: ./profiles/

        Returns:
            ProfileResult with timing breakdown.

        Example:
            >>> trainer = Trainer(dataset)
            >>> # Prepare data first (or call fit() for a few epochs)
            >>> result = trainer.profile(n_batches=100)
            >>> print(result)
            === Training Profile ===
            Total time:      1234.5 ms
              Forward:       456.7 ms (37.0%)
              Backward:      567.8 ms (46.0%)
              ...
        """
        # Ensure model and data are ready
        if self.model is None or self._train_loader is None:
            # Do data prep without full training
            print("Preparing data for profiling...")
            checkpoint = self.load_checkpoint()

            t_prep_start = time.time()
            data_cache = self._load_cache()

            if data_cache is not None:
                train_tensors, test_tensors = self._restore_from_cache(data_cache)
            else:
                if checkpoint is not None:
                    self._restore_scalers_from_checkpoint(checkpoint)
                train_ds, test_ds = self._prepare_data(fit_encoder=(checkpoint is None))

                if self.model is None:
                    self.model = self._build_model()

                train_tensors = self._build_tensors(train_ds, fit_scalers=(checkpoint is None))
                test_tensors = self._build_tensors(test_ds, fit_scalers=False)

            self._create_loaders(train_tensors, test_tensors)

            if self.model is None:
                self.model = self._build_model()

            self.model.to(self._device)
            print(f"  Data prepared in {time.time() - t_prep_start:.1f}s")

        # Setup optimizer if not already
        if self._optimizer is None:
            self._optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Setup loss if not already
        if self._loss_fn is None:
            self._loss_fn = MultiTaskLoss(
                self.model.target_configs,
                phases=self.phases,
                phase_boundaries=self.phase_boundaries,
                label_smoothing=self.label_smoothing,
                class_weights=self.class_weights,
            )

        # Setup AMP
        if self.use_amp and self._grad_scaler is None:
            self._grad_scaler = GradScaler()

        # Track GPU memory
        gpu_memory_peak = 0.0
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        self.model.train()
        timer = Timer()
        target_names = list(self.model.target_configs.keys())
        has_taxonomy = self.model.schema.has_taxonomy

        # Determine if data is already on GPU (from GPUTensorLoader or CUDAPrefetcher)
        use_gpu_loader = getattr(self, "_using_gpu_loader", False)
        use_prefetch = self.prefetch_data and self._device.type == "cuda" and not use_gpu_loader
        data_on_device = use_gpu_loader or use_prefetch

        # Choose loader
        if use_prefetch:
            loader = CUDAPrefetcher(self._train_loader, self._device)
        else:
            loader = self._train_loader

        total_samples = 0
        batch_count = 0

        # Warmup (not timed)
        print(f"Warming up ({warmup_batches} batches)...")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= warmup_batches:
                break
            idx = 0
            continuous = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
            idx += 1

            species_ids = None
            species_vector = None
            if self.species_encoding == "embed":
                species_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
            elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
                species_vector = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1

            if has_taxonomy:
                genus_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
                family_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1

            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            self._optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                    loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
                self._grad_scaler.scale(loss).backward()
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
                loss.backward()
                self._optimizer.step()

        # Profile batches
        print(f"Profiling ({n_batches} batches)...")
        # Re-initialize loader for profiling
        if use_prefetch:
            loader = CUDAPrefetcher(self._train_loader, self._device)
        else:
            loader = self._train_loader

        timer.start("total")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            # Data loading timing
            timer.start("data")
            idx = 0
            continuous = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
            idx += 1

            species_ids = None
            species_vector = None
            if self.species_encoding == "embed":
                species_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
            elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
                species_vector = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1

            if has_taxonomy:
                genus_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
                family_ids = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx] if data_on_device else batch[idx].to(self._device, non_blocking=True)
                idx += 1

            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)
            timer.stop("data")

            # Forward pass
            self._optimizer.zero_grad(set_to_none=True)
            timer.start("forward")
            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                    loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            else:
                predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            timer.stop("forward")

            # Backward pass
            timer.start("backward")
            if self.use_amp:
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            timer.stop("backward")

            # Optimizer step
            timer.start("optimizer")
            if self.use_amp:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optimizer.step()
            timer.stop("optimizer")

            total_samples += continuous.size(0)
            batch_count += 1

        timer.stop("total")

        # Get GPU memory stats
        if self._device.type == "cuda":
            gpu_memory_peak = torch.cuda.max_memory_allocated(self._device) / (1024 * 1024)  # MB

        # Build result
        total_time = timer.get("total")
        result = ProfileResult(
            total_time_ms=total_time,
            forward_time_ms=timer.get("forward"),
            backward_time_ms=timer.get("backward"),
            optimizer_time_ms=timer.get("optimizer"),
            data_time_ms=timer.get("data"),
            n_batches=batch_count,
            avg_batch_time_ms=total_time / batch_count if batch_count > 0 else 0,
            samples_per_second=total_samples / (total_time / 1000) if total_time > 0 else 0,
            gpu_memory_peak_mb=gpu_memory_peak,
        )

        # Optional: save torch.profiler trace
        if save_trace:
            trace_path = Path(trace_dir) if trace_dir else Path("./profiles")
            trace_path.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path / f"profile_{time.strftime('%Y%m%d_%H%M%S')}.json"

            try:
                from torch.profiler import profile as torch_profile, ProfilerActivity

                print(f"Saving detailed trace to {trace_file}...")
                with torch_profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as prof:
                    # Run a few batches for trace
                    loader = CUDAPrefetcher(self._train_loader, self._device) if use_prefetch else self._train_loader
                    for batch_idx, batch in enumerate(loader):
                        if batch_idx >= 10:
                            break
                        idx = 0
                        continuous = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                        idx += 1
                        species_ids = None
                        species_vector = None
                        if self.species_encoding == "embed":
                            species_ids = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                            idx += 1
                        elif self.species_encoding == "hash" and self._species_encoder.uses_explicit_vector:
                            species_vector = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                            idx += 1
                        if has_taxonomy:
                            genus_ids = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                            idx += 1
                            family_ids = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                            idx += 1
                        else:
                            genus_ids = None
                            family_ids = None
                        targets = {}
                        for name in target_names:
                            targets[name] = batch[idx] if use_prefetch else batch[idx].to(self._device, non_blocking=True)
                            idx += 1
                        for name in target_names:
                            cfg = self.model.target_configs[name]
                            if cfg.task == "regression":
                                targets[name] = targets[name].unsqueeze(-1)

                        self._optimizer.zero_grad(set_to_none=True)
                        if self.use_amp:
                            with autocast(device_type="cuda"):
                                predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
                            self._grad_scaler.scale(loss).backward()
                            self._grad_scaler.step(self._optimizer)
                            self._grad_scaler.update()
                        else:
                            predictions = self.model(continuous, genus_ids, family_ids, species_ids, species_vector)
                            loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
                            loss.backward()
                            self._optimizer.step()

                prof.export_chrome_trace(str(trace_file))
                result = ProfileResult(
                    total_time_ms=result.total_time_ms,
                    forward_time_ms=result.forward_time_ms,
                    backward_time_ms=result.backward_time_ms,
                    optimizer_time_ms=result.optimizer_time_ms,
                    data_time_ms=result.data_time_ms,
                    n_batches=result.n_batches,
                    avg_batch_time_ms=result.avg_batch_time_ms,
                    samples_per_second=result.samples_per_second,
                    gpu_memory_peak_mb=result.gpu_memory_peak_mb,
                    detailed_trace_path=str(trace_file),
                )
            except ImportError:
                print("  Warning: torch.profiler not available, skipping trace")

        return result

    def _train_epoch(
        self,
        epoch: int,
        target_names: list[str],
        has_taxonomy: bool,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        nan_batch_count = 0
        total_batches = len(self._train_loader)

        # Debug: track batch statistics
        batch_losses = [] if self.verbose >= 2 else None
        grad_norms = [] if self.verbose >= 2 else None

        # Determine if data is already on GPU (from GPUTensorLoader or CUDAPrefetcher)
        use_gpu_loader = getattr(self, "_using_gpu_loader", False)
        use_prefetch = self.prefetch_data and self._device.type == "cuda" and not use_gpu_loader
        data_on_device = use_gpu_loader or use_prefetch

        # Choose loader: GPU loader is already fast, prefetcher wraps CPU loader
        if use_prefetch:
            loader = CUDAPrefetcher(self._train_loader, self._device)
        else:
            loader = self._train_loader

        for batch_idx, batch in enumerate(loader):
            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
             targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            # Reshape targets for loss
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            # Forward + backward with optional AMP
            self._optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(
                        continuous, genus_ids, family_ids, species_ids, species_vector,
                        pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                        pool_weights=pool_weights, pool_mask=pool_mask,
                        pool_has_cover=pool_has_cover,
                    )
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
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                )
                loss, _ = self._loss_fn(
                    predictions, targets, epoch, self._target_scalers
                )
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optimizer.step()

            # Step scheduler after optimizer (fixes PyTorch warning)
            self._scheduler.step()

            # EMA update: exponential moving average of model weights
            if self._ema_state is not None:
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        self._ema_state[k].lerp_(v, 1.0 - self.ema_decay)

            # Check for NaN loss
            if torch.isnan(loss):
                nan_batch_count += 1
                if nan_batch_count == 1 and self.verbose >= 1:
                    print(f"  WARNING: NaN loss detected, skipping batch...")
                continue

            batch_loss = loss.item()
            total_loss += batch_loss * continuous.size(0)

            # Debug: collect batch statistics
            if self.verbose >= 2:
                batch_losses.append(batch_loss)
                # Compute gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)

        # Report NaN statistics if any occurred
        if nan_batch_count > 0:
            nan_pct = 100 * nan_batch_count / total_batches
            if self.verbose >= 1:
                print(f"  WARNING: NaN loss in {nan_batch_count}/{total_batches} batches ({nan_pct:.1f}%)")
            if nan_pct > 50:
                raise RuntimeError(
                    f"Training unstable: NaN loss in {nan_pct:.1f}% of batches. "
                    "Try reducing learning rate or checking data for invalid values."
                )

        # Debug: print batch-level diagnostics
        if self.verbose >= 2 and batch_losses:
            import statistics
            print(f"    [Debug] Batch losses: min={min(batch_losses):.4f}, max={max(batch_losses):.4f}, "
                  f"mean={statistics.mean(batch_losses):.4f}, std={statistics.stdev(batch_losses) if len(batch_losses) > 1 else 0:.4f}")
            print(f"    [Debug] Grad norms: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, "
                  f"mean={statistics.mean(grad_norms):.4f}")

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

        # Check if data is already on GPU (from GPUTensorLoader)
        data_on_device = getattr(self, "_using_gpu_loader", False)

        for batch in self._test_loader:
            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
             targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            # Use AMP for faster eval inference
            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(
                        continuous, genus_ids, family_ids, species_ids, species_vector,
                        pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                        pool_weights=pool_weights, pool_mask=pool_mask,
                        pool_has_cover=pool_has_cover,
                    )
            else:
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                )

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
        """
        Save model, encoder, and scalers to file.

        Raises:
            RuntimeError: If trainer has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot save: model has not been built yet. "
                "Call trainer.fit() before trainer.save()."
            )
        # Check appropriate encoder based on mode
        if self.species_encoding == "hash" and self._species_encoder is None:
            raise RuntimeError(
                "Cannot save: species encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )
        if self.species_encoding == "embed" and self._embedding_encoder is None:
            raise RuntimeError(
                "Cannot save: embedding encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )
        if self.species_encoding in ("rank_pool", "transformer") and self._rank_pool_encoder is None:
            raise RuntimeError(
                "Cannot save: rank_pool encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "schema": self.model.schema,
            "target_configs": self.model.target_configs,
            "species_encoding": self.species_encoding,
            "hash_dim": self.model.hash_dim,
            "top_k": self.model.top_k,
            "hidden_dims": self.model.hidden_dims,
            "genus_emb_dim": self.model.genus_emb_dim,
            "family_emb_dim": self.model.family_emb_dim,
            "dropout": self.model.dropout,
            "scalers": self._scalers,
            "track_unknown_fraction": self.track_unknown_fraction,
            "uses_explicit_vector": self.model.uses_explicit_vector,
            "head_hidden_dims": self.head_hidden_dims,
        }

        # Save encoder-specific state
        if self.species_encoding == "hash" and self._species_encoder:
            state["vocab"] = self._species_encoder.vocab
            state["species_aggregation"] = self._species_encoder.aggregation
            state["species_selection"] = self._species_encoder.selection
            state["species_representation"] = self._species_encoder.representation
            state["species_normalization"] = self._species_encoder.normalization
            state["track_unknown_count"] = self._species_encoder.track_unknown_count
            state["species_vocab"] = self._species_encoder._species_vocab
            state["species_to_idx"] = self._species_encoder._species_to_idx
            # Save normalizer if present
            if self._species_encoder.normalizer is not None:
                state["normalizer"] = self._species_encoder.normalizer
        elif self.species_encoding == "embed" and self._embedding_encoder:
            state["species_vocab_obj"] = self._embedding_encoder._species_vocab
            state["taxonomy_vocab_obj"] = self._embedding_encoder._taxonomy_vocab
            state["species_aggregation"] = self._embedding_encoder.aggregation
            state["species_selection"] = self._embedding_encoder.selection
            state["top_k_species"] = self._embedding_encoder.top_k_species
            state["top_k_taxonomy"] = self._embedding_encoder.top_k_taxonomy
            if self._embedding_encoder.normalizer is not None:
                state["normalizer"] = self._embedding_encoder.normalizer
        elif self.species_encoding in ("rank_pool", "transformer") and self._rank_pool_encoder:
            state["species_vocab_obj"] = self._rank_pool_encoder._species_vocab
            state["taxonomy_vocab_obj"] = self._rank_pool_encoder._taxonomy_vocab
            state["species_to_genus"] = self._rank_pool_encoder._species_to_genus
            state["species_to_family"] = self._rank_pool_encoder._species_to_family
            state["species_normalization"] = self._rank_pool_encoder.weighting
            state["min_species_frequency"] = self._rank_pool_encoder.min_species_frequency
            if self._rank_pool_encoder.normalizer is not None:
                state["normalizer"] = self._rank_pool_encoder.normalizer
            # Save transformer-specific params
            if self.species_encoding == "transformer":
                state["n_attention_layers"] = self.n_attention_layers
                state["n_heads"] = self.n_heads
                state["transformer_ff_dim"] = self.transformer_ff_dim
                state["transformer_pooling"] = self.transformer_pooling
                state["transformer_dropout"] = self.transformer_dropout

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

        Raises:
            RuntimeError: If trainer has not been fitted yet.
            ValueError: If output_space or confidence_threshold is invalid.
        """
        encoder_ready = (
            (self.species_encoding == "hash" and self._species_encoder is not None) or
            (self.species_encoding == "embed" and self._embedding_encoder is not None) or
            (self.species_encoding in ("rank_pool", "transformer") and self._rank_pool_encoder is not None)
        )
        if not encoder_ready or self.model is None:
            raise RuntimeError(
                "Cannot predict: trainer has not been fitted yet. "
                "Call trainer.fit() before trainer.predict()."
            )

        if output_space not in ("raw", "transformed"):
            raise ValueError(f"output_space must be 'raw' or 'transformed', got {output_space!r}")

        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {confidence_threshold}")

        self.model.eval()

        # Encode species based on mode
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()

        species_ids_t = None
        species_vector_t = None
        genus_t = None
        family_t = None
        # Rank-pool mode tensors
        pool_genus_ids_t = None
        pool_family_ids_t = None
        pool_weights_t = None
        pool_mask_t = None
        pool_has_cover_t = None

        if self.species_encoding == "embed":
            embedded = self._embedding_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if self.track_unknown_fraction:
                parts.append(embedded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = embedded.unknown_fraction

            species_ids_t = torch.from_numpy(embedded.species_ids).to(self._device)
            if embedded.genus_ids is not None:
                genus_t = torch.from_numpy(embedded.genus_ids).to(self._device)
                family_t = torch.from_numpy(embedded.family_ids).to(self._device)

        elif self.species_encoding in ("rank_pool", "transformer"):
            pool_encoded = self._rank_pool_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if self.track_unknown_fraction:
                parts.append(pool_encoded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = pool_encoded.unknown_fraction

            # Pad rank-pool data for batched forward (global padding OK for one-shot inference)
            from resolve.encode.rank_pool import pad_rank_pool_encoded
            padded = pad_rank_pool_encoded(pool_encoded)
            species_ids_t = torch.from_numpy(padded["species_ids"]).long().to(self._device)
            if self.model.schema.has_taxonomy:
                pool_genus_ids_t = torch.from_numpy(padded["genus_ids"]).long().to(self._device)
                pool_family_ids_t = torch.from_numpy(padded["family_ids"]).long().to(self._device)
            pool_weights_t = torch.from_numpy(padded["weights"]).to(self._device)
            pool_mask_t = torch.from_numpy(padded["mask"]).to(self._device)
            pool_has_cover_t = torch.from_numpy(padded["has_cover"]).to(self._device)

        else:
            # Hash mode
            encoded = self._species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if self._species_encoder.uses_explicit_vector:
                species_vector_t = torch.from_numpy(encoded.species_vector).to(self._device)
            else:
                parts.append(encoded.hash_embedding)
            if self.track_unknown_fraction:
                parts.append(encoded.unknown_fraction.reshape(-1, 1))
            if self.track_unknown_count and encoded.unknown_count is not None:
                parts.append(encoded.unknown_count.reshape(-1, 1).astype(np.float32))
            unknown_fraction = encoded.unknown_fraction

            if encoded.genus_ids is not None:
                genus_t = torch.from_numpy(encoded.genus_ids).to(self._device)
                family_t = torch.from_numpy(encoded.family_ids).to(self._device)

        continuous = np.hstack(parts) if parts else np.zeros((len(dataset.plot_ids), 0), dtype=np.float32)
        continuous = self._scalers["continuous"].transform(continuous).astype(np.float32)

        # To tensors
        continuous_t = torch.from_numpy(continuous).to(self._device)

        # Forward pass (dispatch based on encoding mode)
        if self.species_encoding == "transformer":
            # Batched forward to avoid OOM from O(n^2) attention over full dataset
            n = continuous_t.shape[0]
            pred_chunks = {name: [] for name in self.model.target_configs}
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                chunk_preds = self.model(
                    continuous_t[start:end], genus_ids=None, family_ids=None,
                    species_ids=species_ids_t[start:end], species_vector=None,
                    pool_genus_ids=pool_genus_ids_t[start:end] if pool_genus_ids_t is not None else None,
                    pool_family_ids=pool_family_ids_t[start:end] if pool_family_ids_t is not None else None,
                    pool_weights=pool_weights_t[start:end],
                    pool_mask=pool_mask_t[start:end],
                    pool_has_cover=pool_has_cover_t[start:end],
                )
                for name, pred in chunk_preds.items():
                    pred_chunks[name].append(pred)
            preds_raw = {name: torch.cat(chunks) for name, chunks in pred_chunks.items()}
        elif self.species_encoding == "rank_pool":
            preds_raw = self.model(
                continuous_t, genus_ids=None, family_ids=None,
                species_ids=species_ids_t, species_vector=None,
                pool_genus_ids=pool_genus_ids_t, pool_family_ids=pool_family_ids_t,
                pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                pool_has_cover=pool_has_cover_t,
            )
        else:
            preds_raw = self.model(
                continuous_t, genus_t, family_t,
                species_ids=species_ids_t, species_vector=species_vector_t,
            )

        # Compute confidence per sample (1 - unknown_fraction for regression)
        confidence = 1.0 - unknown_fraction

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
    def load(cls, path: str | Path, device: str = "auto") -> tuple[ResolveModel, SpeciesEncoder | EmbeddingEncoder, dict]:
        """
        Load model from checkpoint.

        Dispatches encoder creation based on species_encoding saved in checkpoint:
        - "hash": creates SpeciesEncoder
        - "embed": creates EmbeddingEncoder with restored vocabs
        - "rank_pool": creates RankPoolEncoder with restored vocabs

        Returns:
            (model, species_encoder, scalers)

        Security Note:
            This method uses pickle deserialization (weights_only=False) to load
            sklearn scalers and encoder state. Only load model files from trusted sources.
        """
        # Note: weights_only=False is required for sklearn scalers and encoder state.
        # Only load model files from trusted sources.
        state = torch.load(path, map_location="cpu", weights_only=False)

        species_encoding = state.get("species_encoding", "hash")
        track_unknown_count = state.get("track_unknown_count", False)
        uses_explicit_vector = state.get("uses_explicit_vector", False)

        model = ResolveModel(
            schema=state["schema"],
            targets=state["target_configs"],
            species_encoding=species_encoding,
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
            top_k_species=state.get("top_k_species", 10),
            hidden_dims=state.get("hidden_dims"),
            genus_emb_dim=state.get("genus_emb_dim", 8),
            family_emb_dim=state.get("family_emb_dim", 8),
            dropout=state.get("dropout", 0.3),
            track_unknown_count=track_unknown_count,
            uses_explicit_vector=uses_explicit_vector,
            n_attention_layers=state.get("n_attention_layers", 0),
            n_heads=state.get("n_heads", 4),
            transformer_ff_dim=state.get("transformer_ff_dim", 256),
            transformer_pooling=state.get("transformer_pooling", "attention"),
            transformer_dropout=state.get("transformer_dropout", 0.1),
            head_hidden_dims=state.get("head_hidden_dims"),
        )
        model.load_state_dict(state["model_state_dict"])

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        model.to(dev)

        # Dispatch encoder creation based on species_encoding
        if species_encoding == "embed":
            encoder = EmbeddingEncoder(
                top_k_species=state.get("top_k_species", 10),
                top_k_taxonomy=state.get("top_k_taxonomy", 3),
                aggregation=state.get("species_aggregation", "abundance"),
                selection=state.get("species_selection", "top"),
            )
            encoder._species_vocab = state.get("species_vocab_obj")
            encoder._taxonomy_vocab = state.get("taxonomy_vocab_obj")
            encoder._fitted = True
        elif species_encoding in ("rank_pool", "transformer"):
            from resolve.encode.rank_pool import RankPoolEncoder
            encoder = RankPoolEncoder(
                weighting=state.get("species_normalization", "log1p"),
                min_species_frequency=state.get("min_species_frequency", 1),
            )
            encoder._species_vocab = state.get("species_vocab_obj")
            encoder._taxonomy_vocab = state.get("taxonomy_vocab_obj")
            encoder._species_to_genus = state.get("species_to_genus", {})
            encoder._species_to_family = state.get("species_to_family", {})
            encoder._fitted = True
        else:
            # Hash mode (default)
            encoder = SpeciesEncoder(
                hash_dim=state["hash_dim"],
                top_k=state["top_k"],
                aggregation=state.get("species_aggregation", "abundance"),
                normalization=state.get("species_normalization", "norm"),
                track_unknown_count=track_unknown_count,
                selection=state.get("species_selection", "top"),
                representation=state.get("species_representation", "abundance"),
            )
            if state.get("vocab") is not None:
                encoder._vocab = state["vocab"]
            encoder._species_vocab = state.get("species_vocab", set())
            encoder._species_to_idx = state.get("species_to_idx", {})
            encoder._fitted = True

        # Restore normalizer for all modes (if saved in checkpoint)
        normalizer = state.get("normalizer")
        if normalizer is not None:
            encoder.normalizer = normalizer

        return model, encoder, state["scalers"]
