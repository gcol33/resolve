"""Trainer: training orchestration for ResolveModel."""

from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

from resolve._predict_utils import postprocess_predictions
from resolve.constants import (
    DEFAULT_HIDDEN_DIMS,
    ETA_WINDOW,
    MAX_GRAD_NORM,
    NAN_THRESHOLD_PCT,
    PREFETCH_BATCH_THRESHOLD,
    SCHEDULER_PCT_START,
)
from resolve.data.dataset import ResolveDataset, ResolveSchema
from resolve.encode.embedding import EmbeddingEncoder
from resolve.encode.species import SpeciesEncoder
from resolve.model.resolve import ResolveModel
from resolve.train._loaders import CUDAPrefetcher, _RankPoolPreparedData
from resolve.train._types import (
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    TrainResult,
    TrainingConfig,
)
from resolve.train.loss import MultiTaskLoss, PhaseConfig
from resolve.train.metrics import compute_metrics

# Mixin imports
from resolve.train._cache import CacheMixin
from resolve.train._checkpoint import CheckpointMixin
from resolve.train._cv import CVMixin
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


def _apply_species_dropout(
    dropout_prob: float,
    species_encoding: str,
    schema: ResolveSchema,
    continuous: torch.Tensor,
    species_vector: torch.Tensor | None,
    pool_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply species dropout augmentation during training.

    Randomly zeros out species positions with probability ``dropout_prob``.
    For hash mode, zeros columns of the hash embedding within the continuous
    tensor.  For explicit-vector hash mode, zeros positions in
    ``species_vector``.  For rank_pool/transformer, sets random mask
    positions to False so that those species are ignored by pooling.

    Returns the (possibly modified) continuous, species_vector, and pool_mask.
    """
    if species_encoding == "hash":
        if species_vector is not None:
            # Explicit vector mode: dropout individual species positions
            mask = torch.rand(species_vector.shape[1], device=species_vector.device) > dropout_prob
            species_vector = species_vector * mask.unsqueeze(0)
        else:
            # Standard hash mode: dropout hash embedding columns in continuous tensor
            # Hash embedding starts after coords + covariates (= schema.n_continuous)
            n_base = schema.n_continuous
            n_total = continuous.shape[1]
            if n_total > n_base:
                # Determine hash embedding end: it occupies a contiguous block
                # after the base features. We mask per-sample (row-wise) to
                # avoid zeroing the same columns for every sample.
                hash_cols = n_total - n_base  # includes hash_dim + unknown features
                col_mask = torch.rand(
                    continuous.shape[0], hash_cols, device=continuous.device
                ) > dropout_prob
                continuous = continuous.clone()
                continuous[:, n_base:] = continuous[:, n_base:] * col_mask
    elif species_encoding in ("rank_pool", "transformer"):
        if pool_mask is not None:
            # Zero out random species by setting mask positions to False
            drop = torch.rand_like(pool_mask.float()) < dropout_prob
            pool_mask = pool_mask & ~drop

    return continuous, species_vector, pool_mask


class Trainer(
    DataMixin,
    CacheMixin,
    CheckpointMixin,
    CVMixin,
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
        hidden_dims: list[int] | None = None,
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
        lr_scheduler: str = "onecycle",
        lr_factor: float = 0.1,
        lr_patience: int = 5,
        # Checkpointing
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 50,
        resume: bool = True,
        reset_patience: bool = False,
        # Caching
        cache_dir: str | Path | None = None,
        max_cache_files: int = 5,
        # Loss configuration
        loss_config: str = "mae",
        # Advanced (deprecated - use loss_config instead)
        phases: dict[int, PhaseConfig] | None = None,
        phase_boundaries: list[int] | None = None,
        device: str = "auto",
        use_amp: bool = True,
        compile_model: bool = False,
        prefetch_data: bool | None = None,
        gpu_data: bool | None = None,
        species_aggregation: str = "abundance",
        species_selection: str = "top",
        species_representation: str = "abundance",
        rank_pool_weighting: str | None = None,
        min_species_frequency: int = 1,
        cover_dropout: float = 0.0,
        stratified_split: bool = False,
        # Transformer-specific (species_encoding="transformer")
        n_attention_layers: int = 0,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_pooling: str = "attention",
        transformer_dropout: float = 0.1,
        pretrain_epochs: int = 0,
        pretrain_mask_prob: float = 0.15,
        pretrain_lr: float = 1e-4,
        pretrain_all_data: bool = False,
        # Categorical features
        categorical_embed_dim: int = 8,
        # v7: label smoothing, class weights, EMA, deeper head
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        ema_decay: float = 0.0,
        species_dropout: float = 0.0,
        head_hidden_dims: list[int] | None = None,
        # Advanced architecture (C++ backend only for non-MLP)
        encoder_architecture: str = "mlp",
        # MoE
        moe_routing: str = "none",
        n_experts: int = 4,
        expert_hidden_dims: list[int] | None = None,
        moe_top_k: int = 2,
        moe_noise_std: float = 0.1,
        moe_aux_loss_weight: float = 0.01,
        # Architecture sub-configs
        ft_transformer_config: dict | None = None,
        tabnet_config: dict | None = None,
        saint_config: dict | None = None,
        gnn_config: dict | None = None,
        excelformer_config: dict | None = None,
        # TraitNet configuration
        trait_net_config: dict | None = None,
        traits: torch.Tensor | None = None,
        verbose: int = 1,
        # rank_pool species cap: max species per plot kept per plot before padding.
        # None (default) = auto-compute p99 of species counts across all plots.
        # Set to a positive int to override. 0 or negative = no cap (use actual max).
        rank_pool_species_cap: int | None = None,
        # Grouped config objects (alternative to individual kwargs)
        model_config: ModelConfig | None = None,
        training_config: TrainingConfig | None = None,
        data_config: DataConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
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
                - "rank_pool": Variable-length species lists with weighted pooling
                - "transformer": Transformer-based encoder with attention pooling

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
            lr_scheduler: Learning rate scheduler type.
                - "onecycle": OneCycleLR with cosine annealing (default). Steps per batch.
                - "plateau": ReduceLROnPlateau. Reduces LR when validation loss plateaus.
                - "cosine": CosineAnnealingLR. Anneals LR to 0 over max_epochs.
                - "none": Constant learning rate.
            lr_factor: Factor to reduce LR by for plateau scheduler (default 0.1).
            lr_patience: Epochs to wait before reducing LR for plateau scheduler (default 5).

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

            rank_pool_weighting: Per-species weighting for the rank-pool / transformer
                encoder. Only used when species_encoding is "rank_pool" or "transformer".
                - None (default): falls back to ``species_aggregation`` for back-compat;
                  if that is not a valid weighting, the encoder default ("log1p") is used.
                - "binary": w_i = 1 for all species
                - "abundance": w_i = raw abundance
                - "log1p": w_i = log(1 + abundance)
                - "norm": w_i = abundance / sum(abundances)
                - "rank": w_i = 1/rank based on abundance ordering

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

        # === Merge grouped config objects ===
        # Config dataclass values serve as defaults; explicit kwargs always win.
        # We detect "not explicitly passed" via matching the function-signature default.
        if model_config is not None:
            _mc = model_config
            if species_encoding == "hash": species_encoding = _mc.species_encoding
            if hash_dim == 32: hash_dim = _mc.hash_dim
            if species_embed_dim == 32: species_embed_dim = _mc.species_embed_dim
            if top_k == 5: top_k = _mc.top_k
            if top_k_species == 10: top_k_species = _mc.top_k_species
            if hidden_dims is None: hidden_dims = _mc.hidden_dims
            if genus_emb_dim == 8: genus_emb_dim = _mc.genus_emb_dim
            if family_emb_dim == 8: family_emb_dim = _mc.family_emb_dim
            if dropout == 0.3: dropout = _mc.dropout
            if head_hidden_dims is None: head_hidden_dims = _mc.head_hidden_dims
            if n_attention_layers == 0: n_attention_layers = _mc.n_attention_layers
            if n_heads == 4: n_heads = _mc.n_heads
            if transformer_ff_dim == 256: transformer_ff_dim = _mc.transformer_ff_dim
            if transformer_pooling == "attention": transformer_pooling = _mc.transformer_pooling
            if transformer_dropout == 0.1: transformer_dropout = _mc.transformer_dropout
            if encoder_architecture == "mlp": encoder_architecture = _mc.encoder_architecture
            if moe_routing == "none": moe_routing = _mc.moe_routing
            if n_experts == 4: n_experts = _mc.n_experts
            if expert_hidden_dims is None: expert_hidden_dims = _mc.expert_hidden_dims
            if moe_top_k == 2: moe_top_k = _mc.moe_top_k
            if moe_noise_std == 0.1: moe_noise_std = _mc.moe_noise_std
            if moe_aux_loss_weight == 0.01: moe_aux_loss_weight = _mc.moe_aux_loss_weight
            if ft_transformer_config is None: ft_transformer_config = _mc.ft_transformer_config
            if tabnet_config is None: tabnet_config = _mc.tabnet_config
            if saint_config is None: saint_config = _mc.saint_config
            if gnn_config is None: gnn_config = _mc.gnn_config
            if excelformer_config is None: excelformer_config = _mc.excelformer_config

        if training_config is not None:
            _tc = training_config
            if batch_size == 32768: batch_size = _tc.batch_size
            if num_workers == 0: num_workers = _tc.num_workers
            if max_epochs == 500: max_epochs = _tc.max_epochs
            if patience == 50: patience = _tc.patience
            if lr == 1e-3: lr = _tc.lr
            if weight_decay == 1e-4: weight_decay = _tc.weight_decay
            if lr_scheduler == "onecycle": lr_scheduler = _tc.lr_scheduler
            if lr_factor == 0.1: lr_factor = _tc.lr_factor
            if lr_patience == 5: lr_patience = _tc.lr_patience
            if loss_config == "mae": loss_config = _tc.loss_config
            if device == "auto": device = _tc.device
            if use_amp is True: use_amp = _tc.use_amp
            if compile_model is False: compile_model = _tc.compile_model
            if prefetch_data is None: prefetch_data = _tc.prefetch_data
            if gpu_data is None: gpu_data = _tc.gpu_data
            if label_smoothing == 0.0: label_smoothing = _tc.label_smoothing
            if class_weights is None: class_weights = _tc.class_weights
            if ema_decay == 0.0: ema_decay = _tc.ema_decay
            if species_dropout == 0.0: species_dropout = _tc.species_dropout
            if verbose == 1: verbose = _tc.verbose

        if data_config is not None:
            _dc = data_config
            if species_aggregation == "abundance": species_aggregation = _dc.species_aggregation
            if species_selection == "top": species_selection = _dc.species_selection
            if species_representation == "abundance": species_representation = _dc.species_representation
            if rank_pool_weighting is None: rank_pool_weighting = _dc.rank_pool_weighting
            if min_species_frequency == 1: min_species_frequency = _dc.min_species_frequency
            if cover_dropout == 0.0: cover_dropout = _dc.cover_dropout
            if stratified_split is False: stratified_split = _dc.stratified_split
            if categorical_embed_dim == 8: categorical_embed_dim = _dc.categorical_embed_dim
            if pretrain_epochs == 0: pretrain_epochs = _dc.pretrain_epochs
            if pretrain_mask_prob == 0.15: pretrain_mask_prob = _dc.pretrain_mask_prob
            if pretrain_lr == 1e-4: pretrain_lr = _dc.pretrain_lr
            if pretrain_all_data is False: pretrain_all_data = _dc.pretrain_all_data

        if checkpoint_config is not None:
            _cc = checkpoint_config
            if checkpoint_dir is None: checkpoint_dir = _cc.checkpoint_dir
            if checkpoint_every == 50: checkpoint_every = _cc.checkpoint_every
            if resume is True: resume = _cc.resume
            if reset_patience is False: reset_patience = _cc.reset_patience
            if cache_dir is None: cache_dir = _cc.cache_dir
            if max_cache_files == 5: max_cache_files = _cc.max_cache_files

        # === Parameter Validation ===
        # Species encoding
        if species_encoding not in ("hash", "embed", "rank_pool", "transformer", "trait_net"):
            raise ValueError(f"species_encoding must be 'hash', 'embed', 'rank_pool', 'transformer', or 'trait_net', got {species_encoding!r}")
        self.species_encoding = species_encoding

        # Encoder architecture
        valid_architectures = ("mlp", "ft_transformer", "tabnet", "saint", "gnn", "excelformer")
        if encoder_architecture not in valid_architectures:
            raise ValueError(f"encoder_architecture must be one of {valid_architectures}, got '{encoder_architecture}'")
        self.encoder_architecture = encoder_architecture

        # MoE routing
        valid_moe_routing = ("none", "soft", "topk")
        if moe_routing not in valid_moe_routing:
            raise ValueError(f"moe_routing must be one of {valid_moe_routing}, got '{moe_routing}'")
        self.moe_routing = moe_routing
        self.n_experts = n_experts
        self.expert_hidden_dims = expert_hidden_dims
        self.moe_top_k = moe_top_k
        self.moe_noise_std = moe_noise_std
        self.moe_aux_loss_weight = moe_aux_loss_weight

        # Architecture sub-configs
        self.ft_transformer_config = ft_transformer_config
        self.tabnet_config = tabnet_config
        self.saint_config = saint_config
        self.gnn_config = gnn_config
        self.excelformer_config = excelformer_config
        self.trait_net_config = trait_net_config
        self.traits = traits

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

        # Scheduler parameters
        valid_schedulers = ("onecycle", "plateau", "cosine", "none")
        if lr_scheduler not in valid_schedulers:
            raise ValueError(f"lr_scheduler must be one of {valid_schedulers}, got {lr_scheduler!r}")
        if lr_factor <= 0 or lr_factor >= 1:
            raise ValueError(f"lr_factor must be in (0, 1), got {lr_factor}")
        if lr_patience < 1:
            raise ValueError(f"lr_patience must be >= 1, got {lr_patience}")

        # Species selection mode
        valid_selections = ("top", "bottom", "top_bottom", "all")
        if species_selection not in valid_selections:
            raise ValueError(f"species_selection must be one of {valid_selections}, got {species_selection!r}")

        # Species representation mode
        valid_representations = ("abundance", "presence_absence")
        if species_representation not in valid_representations:
            raise ValueError(f"species_representation must be one of {valid_representations}, got {species_representation!r}")

        # Rank-pool weighting (only used when species_encoding in {"rank_pool", "transformer"})
        valid_rank_pool_weightings = ("binary", "abundance", "log1p", "norm", "rank")
        if rank_pool_weighting is not None and rank_pool_weighting not in valid_rank_pool_weightings:
            raise ValueError(
                f"rank_pool_weighting must be one of {valid_rank_pool_weightings} or None, "
                f"got {rank_pool_weighting!r}"
            )

        self.hash_dim = hash_dim
        self.species_embed_dim = species_embed_dim
        self.top_k = top_k
        self.top_k_species = top_k_species
        self.hidden_dims = hidden_dims if hidden_dims is not None else list(DEFAULT_HIDDEN_DIMS)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.genus_emb_dim = genus_emb_dim
        self.family_emb_dim = family_emb_dim
        self.dropout = dropout

        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

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
        self.rank_pool_weighting = rank_pool_weighting
        self.min_species_frequency = min_species_frequency
        self.cover_dropout = cover_dropout
        self.stratified_split = stratified_split
        self.categorical_embed_dim = categorical_embed_dim
        self.n_attention_layers = n_attention_layers
        self.n_heads = n_heads
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_pooling = transformer_pooling
        self.transformer_dropout = transformer_dropout
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_mask_prob = pretrain_mask_prob
        self.pretrain_lr = pretrain_lr
        self.pretrain_all_data = pretrain_all_data
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.ema_decay = ema_decay
        # Species dropout: randomly zero out species positions during training
        if not 0 <= species_dropout < 1:
            raise ValueError(f"species_dropout must be in [0, 1), got {species_dropout}")
        self.species_dropout = species_dropout
        self.rank_pool_species_cap = rank_pool_species_cap
        self.head_hidden_dims = head_hidden_dims
        self.compile_model = compile_model
        # Auto-enable prefetch for large batch sizes (16K+)
        if prefetch_data is None:
            self.prefetch_data = batch_size >= PREFETCH_BATCH_THRESHOLD
        else:
            self.prefetch_data = prefetch_data
        # GPU data will be resolved after device is known (below)
        self._gpu_data_setting = gpu_data
        self.max_grad_norm = MAX_GRAD_NORM
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
        self.model: ResolveModel | None = None

        # Components to be initialized in fit()
        self._species_encoder: SpeciesEncoder | None = None
        self._embedding_encoder: EmbeddingEncoder | None = None
        self._rank_pool_encoder = None  # RankPoolEncoder | None
        self._pretrain_fitted_encoder = False
        self._scalers: dict[str, object] = {}
        self._target_scalers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._categorical_vocabs: dict[str, object] = {}
        self._train_loader = None
        self._test_loader = None
        self._optimizer: AdamW | None = None
        self._scheduler: OneCycleLR | ReduceLROnPlateau | CosineAnnealingLR | None = None
        self._loss_fn: MultiTaskLoss | None = None
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
        from dataclasses import replace as _replace

        uses_explicit_vector = (
            self.species_encoding == "hash"
            and self._species_encoder is not None
            and self._species_encoder.uses_explicit_vector
        )
        n_taxonomy_slots = (
            self._species_encoder.n_taxonomy_slots
            if self._species_encoder else self.top_k
        )

        # Update schema with categorical embed dim and vocab sizes from built vocabs
        schema = self._schema
        if schema.has_categoricals:
            updates = {"categorical_embed_dim": self.categorical_embed_dim}
            if self._categorical_vocabs:
                updates["categorical_vocab_sizes"] = {
                    name: vocab.n_categories
                    for name, vocab in self._categorical_vocabs.items()
                }
            schema = _replace(schema, **updates)

        return ResolveModel(
            schema=schema,
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
            n_experts=self.n_experts if self.moe_routing != "none" else 0,
            expert_hidden_dims=self.expert_hidden_dims or [256, 128],
            moe_routing=self.moe_routing,
            moe_top_k=self.moe_top_k,
            moe_noise_std=self.moe_noise_std,
            encoder_architecture=self.encoder_architecture,
            ft_transformer_config=self.ft_transformer_config,
            tabnet_config=self.tabnet_config,
            saint_config=self.saint_config,
            gnn_config=self.gnn_config,
            excelformer_config=self.excelformer_config,
            trait_net_config=self.trait_net_config,
            traits=self.traits,
        )

    def _ensure_model(self) -> ResolveModel:
        """Build model if not already built, and return it."""
        if self.model is None:
            self.model = self._build_model()
        return self.model

    def fit(self) -> TrainResult:
        """
        Train the model.

        Automatically resumes from checkpoint if available and resume=True.
        Saves checkpoints every `checkpoint_every` epochs if checkpoint_dir is set.

        Returns:
            TrainResult with metrics and history
        """
        # Note: we no longer suppress the PyTorch scheduler ordering warning.
        # Instead, scheduler.step() is only called when optimizer.step()
        # actually executed (i.e., GradScaler did not skip due to inf grads).

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
            self._ensure_model()

            train_tensors = self._build_tensors(train_ds, fit_scalers=(checkpoint is None))
            test_tensors = self._build_tensors(test_ds, fit_scalers=False)
            print(f"  Data prepared in {time.time() - t_prep_start:.1f}s")

            # Save to cache for next time (rank_pool now returns pre-padded dense tensors, also cacheable)
            if self.cache_dir and not isinstance(train_tensors, _RankPoolPreparedData):
                self._save_cache(
                    train_tensors,
                    test_tensors,
                    train_indices=np.array([]),  # Could store actual indices if needed
                    test_indices=np.array([]),
                )

        self._create_loaders(train_tensors, test_tensors)

        # Build model now that schema (with vocab sizes for embed mode) is ready
        self._ensure_model()

        # Move model to device
        self.model.to(self._device)

        # Compile model for potential speedup (PyTorch 2.0+)
        compiled = False
        if self.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                compiled = True
            except RuntimeError as e:
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

        # Setup learning rate scheduler
        steps_per_epoch = max(1, len(self._train_loader))
        self._steps_per_epoch = steps_per_epoch
        self._scheduler = self._create_scheduler(self.max_epochs, steps_per_epoch)

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

            # Restore or recreate scheduler
            remaining_epochs = self.max_epochs - start_epoch
            if remaining_epochs > 0:
                sched_state = checkpoint.get("scheduler_state_dict")
                saved_max = checkpoint.get("config", {}).get("max_epochs", self.max_epochs)

                if self.lr_scheduler in ("onecycle", "cosine"):
                    if sched_state and saved_max == self.max_epochs:
                        # Same total epochs — restore state dict for exact LR continuation
                        self._scheduler.load_state_dict(sched_state)
                        lr_now = self._optimizer.param_groups[0]["lr"]
                        print(f"  Scheduler restored from checkpoint (lr={lr_now:.2e})")
                    else:
                        # max_epochs changed — recreate for remaining epochs
                        self._scheduler = self._create_scheduler(remaining_epochs, self._steps_per_epoch)
                        print(f"  Scheduler recreated for {remaining_epochs} remaining epochs")
                elif self.lr_scheduler == "plateau":
                    if sched_state and self._scheduler is not None:
                        self._scheduler.load_state_dict(sched_state)
                        print(f"  Scheduler restored (plateau, lr={self._optimizer.param_groups[0]['lr']:.2e})")

        # Initialize EMA state (exponential moving average of model weights)
        # If restored from checkpoint, _ema_state is already set; otherwise init from model
        if not hasattr(self, "_ema_state") or self._ema_state is None:
            self._ema_state = None
            if self.ema_decay > 0:
                self._ema_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        target_names = list(self.model.target_configs.keys())
        has_taxonomy = self.model.schema.has_taxonomy

        train_start_time = time.time()
        epoch_times = []  # Track epoch durations for ETA
        print(f"Starting training loop: epochs {start_epoch} to {self.max_epochs - 1}", flush=True)
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

            # Step per-epoch schedulers
            if self._scheduler is not None:
                if self.lr_scheduler == "plateau":
                    self._scheduler.step(test_loss)
                elif self.lr_scheduler == "cosine":
                    self._scheduler.step()

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
            recent = epoch_times[-ETA_WINDOW:]
            avg_epoch_time = sum(recent) / len(recent)
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
            # Include LR in log for non-constant schedulers
            lr_str = ""
            if self.lr_scheduler != "none" and self._optimizer is not None:
                current_lr = self._optimizer.param_groups[0]["lr"]
                lr_str = f" lr={current_lr:.2e}"
            print(
                f"Epoch {epoch:3d} [P{phase}] | "
                f"train={train_loss:.4f} test={test_loss:.4f} | {metric_str} | "
                f"{epoch_time:.1f}s/ep, ETA {eta_str}{lr_str}",
                flush=True,
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

    def _create_scheduler(
        self, n_epochs: int, steps_per_epoch: int
    ) -> OneCycleLR | ReduceLROnPlateau | CosineAnnealingLR | None:
        """Create LR scheduler based on self.lr_scheduler config."""
        if self.lr_scheduler == "onecycle":
            total_steps = n_epochs * steps_per_epoch
            return OneCycleLR(
                self._optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=SCHEDULER_PCT_START,
                anneal_strategy="cos",
            )
        elif self.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                self._optimizer,
                mode="min",
                factor=self.lr_factor,
                patience=self.lr_patience,
            )
        elif self.lr_scheduler == "cosine":
            return CosineAnnealingLR(self._optimizer, T_max=n_epochs)
        else:  # "none"
            return None

    # ------------------------------------------------------------------
    # Learning rate range test (Smith 2017)
    # ------------------------------------------------------------------

    def lr_range_test(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        n_steps: int = 100,
        smooth_factor: float = 0.05,
    ) -> dict:
        """Run learning rate range test (Smith 2017).

        Trains for *n_steps* mini-batches with an exponentially increasing
        learning rate from *start_lr* to *end_lr*.  Returns a dict with
        ``lrs``, ``losses`` (smoothed), and ``suggested_lr``.

        The suggested LR is one decade below the LR at which the minimum
        smoothed loss was observed — a conservative rule that keeps you
        on the descending side of the loss curve.

        Usage::

            result = trainer.lr_range_test()
            suggested_lr = result['suggested_lr']
            # Plot: plt.plot(result['lrs'], result['losses']); plt.xscale('log')

        Parameters
        ----------
        start_lr : float
            Starting learning rate (default 1e-7).
        end_lr : float
            Final learning rate (default 10.0).
        n_steps : int
            Number of training steps (default 100).
        smooth_factor : float
            Exponential smoothing factor for the loss (default 0.05).
            Lower = smoother.

        Returns
        -------
        dict
            ``lrs`` — list of learning rates at each step.
            ``losses`` — list of smoothed losses at each step.
            ``suggested_lr`` — recommended learning rate.
        """
        if start_lr <= 0:
            raise ValueError(f"start_lr must be > 0, got {start_lr}")
        if end_lr <= start_lr:
            raise ValueError(f"end_lr must be > start_lr, got end_lr={end_lr}, start_lr={start_lr}")
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")
        if not 0 < smooth_factor < 1:
            raise ValueError(f"smooth_factor must be in (0, 1), got {smooth_factor}")

        # Ensure data and model are ready
        if self._train_loader is None:
            train_ds, test_ds = self._prepare_data()
            self._ensure_model()
            train_tensors = self._build_tensors(train_ds, fit_scalers=True)
            test_tensors = self._build_tensors(test_ds, fit_scalers=False)
            self._create_loaders(train_tensors, test_tensors)
        self._ensure_model()
        self.model.to(self._device)

        # Save model state so we can restore after the test
        model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Temporary optimizer at start_lr
        tmp_optimizer = AdamW(
            self.model.parameters(), lr=start_lr, weight_decay=self.weight_decay
        )
        tmp_scaler = GradScaler(enabled=self.use_amp)

        # Temporary loss function (use epoch 0 phase)
        target_names = list(self.dataset.targets.keys())
        has_taxonomy = self._schema.has_taxonomy
        if self._loss_fn is None:
            self._loss_fn = MultiTaskLoss(
                self.model.target_configs,
                phases=self.phases,
                phase_boundaries=self.phase_boundaries,
            )

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_loss = 0.0
        best_loss = float("inf")
        self.model.train()

        # Build an infinite batch iterator
        batch_iter = iter(self._train_loader)
        use_gpu_loader = getattr(self, "_using_gpu_loader", False)
        data_on_device = use_gpu_loader

        for step in range(n_steps):
            # Compute LR for this step (exponential schedule)
            lr = start_lr * (end_lr / start_lr) ** (step / n_steps)
            for pg in tmp_optimizer.param_groups:
                pg["lr"] = lr

            # Get next batch (wrap around if loader is exhausted)
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(self._train_loader)
                batch = next(batch_iter)

            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask,
             pool_has_cover, categorical_ids, targets) = self._unpack_batch(
                batch, target_names, has_taxonomy, data_on_device
            )
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            # Forward + backward
            tmp_optimizer.zero_grad(set_to_none=True)
            ctx = autocast(device_type="cuda") if self.use_amp else nullcontext()
            with ctx:
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                    categorical_ids=categorical_ids,
                )
                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)

            tmp_scaler.scale(loss).backward()
            tmp_scaler.unscale_(tmp_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            tmp_scaler.step(tmp_optimizer)
            tmp_scaler.update()

            # Smooth loss
            raw_loss = loss.item()
            if step == 0:
                smoothed_loss = raw_loss
            else:
                smoothed_loss = smooth_factor * raw_loss + (1 - smooth_factor) * smoothed_loss

            lrs.append(lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Divergence guard: stop if loss exceeds 4x minimum
            if smoothed_loss > 4 * best_loss:
                break

        # Restore original model state
        self.model.load_state_dict(model_state)

        # Suggested LR: 1/10 of the LR at minimum loss (stay on descending slope)
        if losses:
            min_idx = int(np.argmin(losses))
            suggested_lr = lrs[min_idx] / 10.0
        else:
            suggested_lr = start_lr

        if self.verbose >= 1:
            print(
                f"LR range test: {len(lrs)} steps, "
                f"min loss {best_loss:.4f} at lr={lrs[int(np.argmin(losses))]:.2e}, "
                f"suggested lr={suggested_lr:.2e}"
            )

        return {"lrs": lrs, "losses": losses, "suggested_lr": suggested_lr}

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    @staticmethod
    def grid_search(
        dataset: ResolveDataset,
        param_grid: dict[str, list],
        n_cv_folds: int = 3,
        **fixed_params,
    ) -> list[dict]:
        """Run grid search over hyperparameter combinations.

        Iterates over every combination of values in *param_grid*, creates
        a ``Trainer`` for each, runs ``cross_validate(n_splits=n_cv_folds)``,
        and collects results.  Combinations are sorted by the primary
        metric (band_25 for regression, accuracy for classification),
        best first.

        Parameters
        ----------
        dataset : ResolveDataset
            Training dataset.
        param_grid : dict[str, list]
            Mapping from ``Trainer`` parameter names to lists of values.
            Example: ``{"lr": [1e-3, 1e-4], "hidden_dims": [[256, 128], [512, 256]]}``
        n_cv_folds : int
            Number of CV folds per combination (default 3).
        **fixed_params
            Fixed parameters forwarded to every ``Trainer``.

        Returns
        -------
        list[dict]
            One entry per combination, sorted best-first. Each dict has:
            ``params`` — the hyperparameter dict for this run,
            ``mean_metrics`` — mean across folds,
            ``std_metrics`` — std across folds,
            ``cv_result`` — the full ``CVResult`` object.
        """
        import itertools

        # Validate param_grid
        if not param_grid:
            raise ValueError("param_grid must be a non-empty dict")
        for key, values in param_grid.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(
                    f"param_grid['{key}'] must be a non-empty list, "
                    f"got {type(values).__name__}"
                )

        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))
        n_combos = len(combos)
        print(f"\n=== Grid Search: {n_combos} combinations x {n_cv_folds} folds ===")

        results: list[dict] = []

        for i, values in enumerate(combos):
            combo = dict(zip(keys, values))
            params = {**fixed_params, **combo}
            print(f"\n--- Combination {i + 1}/{n_combos}: {combo} ---")

            trainer = Trainer(dataset=dataset, **params)
            cv_result = trainer.cross_validate(n_splits=n_cv_folds)

            results.append({
                "params": combo,
                "mean_metrics": cv_result.mean_metrics,
                "std_metrics": cv_result.std_metrics,
                "cv_result": cv_result,
            })

        # Sort by primary metric (first target's band_25 or accuracy), descending
        def _sort_key(entry: dict) -> float:
            mean = entry["mean_metrics"]
            first_target = next(iter(mean))
            m = mean[first_target]
            return m.get("band_25", m.get("accuracy", 0.0))

        results.sort(key=_sort_key, reverse=True)

        # Print summary
        print(f"\n=== Grid Search Results (best first) ===")
        for rank, entry in enumerate(results, 1):
            mean = entry["mean_metrics"]
            first_target = next(iter(mean))
            m = mean[first_target]
            primary = m.get("band_25", m.get("accuracy", 0.0))
            std = entry["std_metrics"][first_target]
            primary_std = std.get("band_25", std.get("accuracy", 0.0))
            print(f"  {rank}. {entry['params']} -> {primary:.4f} +/- {primary_std:.4f}")

        return results

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
             categorical_ids, targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            # Species dropout augmentation: randomly zero out species positions
            if self.species_dropout > 0:
                continuous, species_vector, pool_mask = _apply_species_dropout(
                    self.species_dropout, self.species_encoding, self._schema,
                    continuous, species_vector, pool_mask,
                )

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
                    categorical_ids=categorical_ids,
                )
                loss, _ = self._loss_fn(
                    predictions, targets, epoch, self._target_scalers
                )

            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Track whether optimizer.step() actually executes.
            # GradScaler skips the step when it finds inf/NaN gradients;
            # in that case we must NOT advance the LR scheduler.
            scale_before = self._grad_scaler.get_scale()
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
            optimizer_stepped = self._grad_scaler.get_scale() >= scale_before

            # Step per-batch schedulers (OneCycleLR) only after a real optimizer step
            if self._scheduler is not None and self.lr_scheduler == "onecycle" and optimizer_stepped:
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
            if nan_pct > NAN_THRESHOLD_PCT:
                raise RuntimeError(
                    f"Training unstable: NaN loss in {nan_pct:.1f}% of batches. "
                    "Try reducing learning rate or checking data for invalid values."
                )

        # Debug: print batch-level diagnostics
        if self.verbose >= 2 and batch_losses:
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
             categorical_ids, targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            # AMP autocast wraps forward + loss to avoid float16 overflow
            # in log-softmax for classification targets
            ctx = autocast(device_type="cuda") if self.use_amp else nullcontext()
            with ctx:
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                    categorical_ids=categorical_ids,
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
            metrics[name] = compute_metrics(
                pred, target, cfg.task, cfg.transform, num_classes=cfg.num_classes,
            )

        return avg_loss, metrics

    @torch.no_grad()
    def _batched_forward(
        self,
        continuous_t: torch.Tensor,
        kwargs: dict,
    ) -> dict:
        n = continuous_t.shape[0]
        bs = max(int(self.batch_size), 1)
        pred_chunks: dict = {name: [] for name in self.model.target_configs}
        for start in range(0, n, bs):
            end = min(start + bs, n)
            sliced = {
                k: (v[start:end] if isinstance(v, torch.Tensor) else v)
                for k, v in kwargs.items()
            }
            chunk_preds = self.model(continuous_t[start:end], **sliced)
            for name, pred in chunk_preds.items():
                pred_chunks[name].append(pred)
        return {name: torch.cat(chunks) for name, chunks in pred_chunks.items()}

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

        # Build categorical IDs for prediction
        categorical_ids_t = None
        if self._schema.has_categoricals and self._categorical_vocabs:
            cat_data = dataset.get_categoricals()
            if cat_data is not None:
                cat_arrays = []
                for cat_name in self._schema.categorical_names:
                    vocab = self._categorical_vocabs[cat_name]
                    cat_arrays.append(vocab.encode_array(cat_data[cat_name]))
                categorical_ids_t = torch.from_numpy(
                    np.stack(cat_arrays, axis=1)
                ).to(self._device)

        # Forward pass — always batched to avoid OOM on large test sets.
        # rank_pool/transformer materialize [B, S, D] activations; hash/embed
        # similarly scale linearly in B. One unified path for all modes.
        if self.species_encoding in ("rank_pool", "transformer"):
            forward_kwargs = dict(
                genus_ids=None, family_ids=None, species_vector=None,
                species_ids=species_ids_t,
                pool_genus_ids=pool_genus_ids_t,
                pool_family_ids=pool_family_ids_t,
                pool_weights=pool_weights_t,
                pool_mask=pool_mask_t,
                pool_has_cover=pool_has_cover_t,
                categorical_ids=categorical_ids_t,
            )
        else:
            forward_kwargs = dict(
                genus_ids=genus_t, family_ids=family_t,
                species_ids=species_ids_t, species_vector=species_vector_t,
                categorical_ids=categorical_ids_t,
            )

        preds_raw = self._batched_forward(continuous_t, forward_kwargs)

        # Post-process
        predictions, _ = postprocess_predictions(
            preds_raw, self.model.target_configs, self._scalers,
            unknown_fraction, output_space, confidence_threshold,
        )

        return predictions
