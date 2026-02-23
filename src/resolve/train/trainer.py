"""Trainer: training orchestration for ResolveModel."""

from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from resolve.data.dataset import ResolveDataset, ResolveSchema
from resolve.encode.embedding import EmbeddingEncoder
from resolve.encode.species import SpeciesEncoder
from resolve.model.resolve import ResolveModel
from resolve.train._loaders import CUDAPrefetcher, _RankPoolPreparedData
from resolve.train._types import TrainResult
from resolve.train.loss import MultiTaskLoss, PhaseConfig
from resolve.train.metrics import compute_metrics

# Mixin imports
from resolve.train._cache import CacheMixin
from resolve.train._checkpoint import CheckpointMixin
from resolve.train._data import DataMixin
from resolve.train._persistence import PersistenceMixin
from resolve.train._pretrain import PretrainMixin
from resolve.train._profiling import ProfilingMixin

# Preset loss configurations
LOSS_PRESETS = {
    "mae": {1: PhaseConfig(mae=1.0)},
    "combined": {1: PhaseConfig(mae=0.80, smape=0.15, band=0.05)},
    "smape": {1: PhaseConfig(mae=0.5, smape=0.5)},
}


class Trainer(
    DataMixin,
    CacheMixin,
    CheckpointMixin,
    PretrainMixin,
    ProfilingMixin,
    PersistenceMixin,
):
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
        self._scalers: dict[str, object] = {}
        self._target_scalers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._train_loader = None
        self._test_loader = None
        self._optimizer: Optional[AdamW] = None
        self._scheduler: Optional[OneCycleLR] = None
        self._loss_fn: Optional[MultiTaskLoss] = None
        self._grad_scaler: GradScaler | None = None

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

    def fit(self) -> TrainResult:
        """
        Train the model.

        Automatically resumes from checkpoint if available and resume=True.
        Saves checkpoints every `checkpoint_every` epochs if checkpoint_dir is set.

        Returns:
            TrainResult with metrics and history
        """
        # Suppress harmless PyTorch warning about scheduler step order on first batch
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

        # GradScaler with enabled=False is a no-op passthrough (no AMP branch needed)
        self._grad_scaler = GradScaler(enabled=self.use_amp)

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

            ctx = autocast(device_type="cuda") if self.use_amp else nullcontext()
            with ctx:
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
            self._grad_scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()

            # Step scheduler after optimizer (fixes PyTorch warning)
            self._scheduler.step()

            # EMA update: exponential moving average of model weights
            if self._ema_state is not None:
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        if v.is_floating_point():
                            self._ema_state[k].lerp_(v, 1.0 - self.ema_decay)
                        else:
                            self._ema_state[k].copy_(v)

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

            # AMP autocast wraps forward + loss to avoid float16 overflow
            # in log-softmax for classification targets
            ctx = autocast(device_type="cuda") if self.use_amp else nullcontext()
            with ctx:
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
