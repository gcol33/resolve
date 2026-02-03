"""
High-level Trainer class that wraps the C++ core.

Provides a dataset-first API matching the paper's expected interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Any

import torch

from resolve_core import (
    Trainer as _CoreTrainer,
    Predictor as _CorePredictor,
    ResolveModel as _CoreModel,
    ModelConfig as _CoreModelConfig,
    TrainConfig as _CoreTrainConfig,
    TrainResult,
    ResolvePredictions,
    Scalers,
    LossConfigMode,
    MoERoutingType,
    ActivationType,
    NormLayerType,
    NetworkDiagnostics,
    CalibrationResult,
    ResidualAnalysis,
    CrossValidationResult,
)
from .dataset import ResolveDataset

__all__ = ["Trainer"]

# Valid loss configuration modes
_VALID_LOSS_CONFIGS = {"mae", "smape", "combined"}

# Valid MoE routing types
_VALID_MOE_ROUTING = {"none", "soft", "topk"}

# Valid activation functions
_VALID_ACTIVATIONS = {
    "relu", "leaky_relu", "gelu", "silu", "swish",
    "tanh", "mish", "elu", "selu", "softplus", "prelu"
}

# Valid normalization layers
_VALID_NORMALIZATIONS = {"batchnorm", "layernorm", "groupnorm", "rmsnorm", "none"}

# Activation string to enum mapping
_ACTIVATION_MAP = {
    "relu": ActivationType.ReLU,
    "leaky_relu": ActivationType.LeakyReLU,
    "gelu": ActivationType.GELU,
    "silu": ActivationType.SiLU,
    "swish": ActivationType.SiLU,  # Alias
    "tanh": ActivationType.Tanh,
    "mish": ActivationType.Mish,
    "elu": ActivationType.ELU,
    "selu": ActivationType.SELU,
    "softplus": ActivationType.Softplus,
    "prelu": ActivationType.PReLU,
}

# Normalization string to enum mapping
_NORMALIZATION_MAP = {
    "batchnorm": NormLayerType.BatchNorm,
    "layernorm": NormLayerType.LayerNorm,
    "groupnorm": NormLayerType.GroupNorm,
    "rmsnorm": NormLayerType.RMSNorm,
    "none": NormLayerType.None_,
}


def _build_model_config(
    dataset: ResolveDataset,
    hash_dim: int,
    top_k: int,
    top_k_species: int,
    species_embed_dim: int,
    genus_emb_dim: int,
    family_emb_dim: int,
    hidden_dims: list[int],
    dropout: float,
    # MoE parameters
    moe_routing: str,
    n_experts: int,
    expert_hidden_dims: list[int],
    moe_top_k: int,
    moe_noise_std: float,
    moe_aux_loss_weight: float,
    # Architecture parameters
    activation: str,
    normalization: str,
    norm_groups: int,
    use_residual: bool,
    leaky_relu_slope: float,
    elu_alpha: float,
    # Head architecture
    head_hidden_dims: list[int],
    head_activation: str,
    head_dropout: float,
) -> _CoreModelConfig:
    """Build ModelConfig from training parameters."""
    config = _CoreModelConfig()
    config.species_encoding = dataset._core.config.species_encoding
    # Sync dimensions with dataset - the dataset determines actual tensor dimensions
    # hash_dim and n_taxonomy_slots MUST match what the dataset created
    config.hash_dim = dataset._core.config.hash_dim
    config.n_taxonomy_slots = dataset._core.config.top_k
    # These are model-specific (embedding sizes, architecture)
    config.species_embed_dim = species_embed_dim
    config.genus_emb_dim = genus_emb_dim
    config.family_emb_dim = family_emb_dim
    config.top_k = top_k
    config.top_k_species = top_k_species
    config.hidden_dims = hidden_dims
    config.dropout = dropout

    # MoE configuration
    moe_map = {
        "none": MoERoutingType.None_,
        "soft": MoERoutingType.Soft,
        "topk": MoERoutingType.TopK,
    }
    config.moe_routing = moe_map.get(moe_routing.lower(), MoERoutingType.None_)
    config.n_experts = n_experts
    config.expert_hidden_dims = expert_hidden_dims
    config.moe_top_k = moe_top_k
    config.moe_noise_std = moe_noise_std
    config.moe_aux_loss_weight = moe_aux_loss_weight

    # Architecture configuration
    config.activation = _ACTIVATION_MAP.get(activation.lower(), ActivationType.GELU)
    config.normalization = _NORMALIZATION_MAP.get(normalization.lower(), NormLayerType.BatchNorm)
    config.norm_groups = norm_groups
    config.use_residual = use_residual
    config.leaky_relu_slope = leaky_relu_slope
    config.elu_alpha = elu_alpha

    # Head architecture configuration
    config.head_hidden_dims = head_hidden_dims
    config.head_activation = _ACTIVATION_MAP.get(head_activation.lower(), ActivationType.GELU)
    config.head_dropout = head_dropout

    return config


def _build_train_config(
    batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    phase_boundaries: Optional[list[int]],
    loss_config: str,
    use_amp: bool,
) -> _CoreTrainConfig:
    """Build TrainConfig from training parameters."""
    config = _CoreTrainConfig()
    config.batch_size = batch_size
    config.max_epochs = max_epochs
    config.patience = patience
    config.lr = lr
    config.weight_decay = weight_decay
    if phase_boundaries:
        config.phase_boundaries = phase_boundaries

    # Map loss config string to enum
    loss_map = {
        "mae": LossConfigMode.MAE,
        "smape": LossConfigMode.SMAPE,
        "combined": LossConfigMode.Combined,
    }
    config.loss_config = loss_map.get(loss_config.lower(), LossConfigMode.Combined)

    # Automatic Mixed Precision
    config.use_amp = use_amp
    return config


class Trainer:
    """
    High-level trainer for RESOLVE models.

    Takes a dataset as first argument and automatically creates the model.

    Example:
        trainer = Trainer(
            dataset,
            hash_dim=64,
            top_k=10,
            hidden_dims=[512, 256, 128, 64],
            max_epochs=100,
            patience=10,
        )
        results = trainer.fit()
        predictions = trainer.predict(test_dataset)
    """

    def __init__(
        self,
        dataset: ResolveDataset,
        # Model configuration
        hash_dim: int = 32,
        top_k: int = 3,
        top_k_species: int = 10,
        species_embed_dim: int = 16,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
        # MoE configuration
        moe_routing: str = "none",
        n_experts: int = 4,
        expert_hidden_dims: list[int] = None,
        moe_top_k: int = 2,
        moe_noise_std: float = 0.1,
        moe_aux_loss_weight: float = 0.01,
        # Architecture configuration
        activation: str = "gelu",
        normalization: str = "batchnorm",
        norm_groups: int = 32,
        use_residual: bool = False,
        leaky_relu_slope: float = 0.01,
        elu_alpha: float = 1.0,
        # Head architecture
        head_hidden_dims: list[int] = None,
        head_activation: str = "gelu",
        head_dropout: float = 0.0,
        # Training configuration
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        test_size: float = 0.2,
        seed: int = 42,
        loss_config: str = "combined",
        phase_boundaries: list[int] = None,
        # Checkpointing
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 0,
        resume: bool = False,
        # Device
        device: str = "cuda",
        # Automatic Mixed Precision (disabled by default - minimal benefit for MLPs)
        use_amp: bool = False,
    ):
        """
        Create a trainer for the given dataset.

        Args:
            dataset: ResolveDataset to train on
            hash_dim: Dimension for species hash embedding
            top_k: Number of top genera/families for taxonomy
            top_k_species: Number of top species for embed mode
            species_embed_dim: Embedding dimension for species (embed mode)
            genus_emb_dim: Embedding dimension for genus
            family_emb_dim: Embedding dimension for family
            hidden_dims: MLP hidden layer dimensions
            dropout: Dropout rate
            moe_routing: MoE routing type ("none", "soft", or "topk")
            n_experts: Number of expert networks (default 4)
            expert_hidden_dims: Expert MLP architecture (default [256, 128])
            moe_top_k: For TopK routing, experts per sample (default 2)
            moe_noise_std: Noise for load balancing during training (default 0.1)
            moe_aux_loss_weight: Weight for auxiliary load balancing loss (default 0.01)
            activation: Activation function for MLP layers (default "gelu")
            normalization: Normalization layer type (default "batchnorm")
            norm_groups: Number of groups for GroupNorm (default 32)
            use_residual: Whether to use residual connections (default False)
            leaky_relu_slope: Negative slope for LeakyReLU (default 0.01)
            elu_alpha: Alpha parameter for ELU (default 1.0)
            head_hidden_dims: Hidden dimensions for prediction heads (default None = single layer)
            head_activation: Activation function for head layers (default "gelu")
            head_dropout: Dropout rate for head layers (default 0.0)
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            batch_size: Training batch size
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            test_size: Fraction of data for validation
            seed: Random seed for reproducibility
            loss_config: Loss function ("mae", "smape", or "combined")
            phase_boundaries: Epoch boundaries for phased training
            checkpoint_dir: Directory for saving checkpoints
            checkpoint_every: Save checkpoint every N epochs (0 = only at end)
            resume: Whether to resume from checkpoint
            device: Device to train on ("cuda" or "cpu")
            use_amp: Enable Automatic Mixed Precision (default False, CUDA only).
                Note: AMP provides minimal benefit for MLP-based models.
        """
        # Validate inputs
        if not isinstance(dataset, ResolveDataset):
            raise TypeError(f"dataset must be a ResolveDataset, got {type(dataset).__name__}")
        if hash_dim < 1:
            raise ValueError(f"hash_dim must be positive, got {hash_dim}")
        if top_k < 1:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        if patience < 1:
            raise ValueError(f"patience must be positive, got {patience}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if loss_config.lower() not in _VALID_LOSS_CONFIGS:
            raise ValueError(f"loss_config must be one of {_VALID_LOSS_CONFIGS}, got '{loss_config}'")
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{device}'")
        if moe_routing.lower() not in _VALID_MOE_ROUTING:
            raise ValueError(f"moe_routing must be one of {_VALID_MOE_ROUTING}, got '{moe_routing}'")
        if n_experts < 1:
            raise ValueError(f"n_experts must be positive, got {n_experts}")
        if moe_top_k < 1 or moe_top_k > n_experts:
            raise ValueError(f"moe_top_k must be between 1 and n_experts ({n_experts}), got {moe_top_k}")
        if activation.lower() not in _VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {_VALID_ACTIVATIONS}, got '{activation}'")
        if normalization.lower() not in _VALID_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {_VALID_NORMALIZATIONS}, got '{normalization}'")
        if norm_groups < 1:
            raise ValueError(f"norm_groups must be positive, got {norm_groups}")
        if leaky_relu_slope <= 0:
            raise ValueError(f"leaky_relu_slope must be positive, got {leaky_relu_slope}")
        if elu_alpha <= 0:
            raise ValueError(f"elu_alpha must be positive, got {elu_alpha}")
        if head_activation.lower() not in _VALID_ACTIVATIONS:
            raise ValueError(f"head_activation must be one of {_VALID_ACTIVATIONS}, got '{head_activation}'")
        if not 0 <= head_dropout < 1:
            raise ValueError(f"head_dropout must be between 0 and 1, got {head_dropout}")

        self._dataset = dataset
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_every = checkpoint_every
        self._resume = resume
        self._device = device
        self._test_size = test_size
        self._seed = seed

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        if expert_hidden_dims is None:
            expert_hidden_dims = [256, 128]

        if head_hidden_dims is None:
            head_hidden_dims = []

        # Build configs using factory functions
        model_config = _build_model_config(
            dataset, hash_dim, top_k, top_k_species,
            species_embed_dim, genus_emb_dim, family_emb_dim,
            hidden_dims, dropout,
            moe_routing, n_experts, expert_hidden_dims,
            moe_top_k, moe_noise_std, moe_aux_loss_weight,
            activation, normalization, norm_groups,
            use_residual, leaky_relu_slope, elu_alpha,
            head_hidden_dims, head_activation, head_dropout,
        )
        train_config = _build_train_config(
            batch_size, max_epochs, patience, lr, weight_decay,
            phase_boundaries, loss_config, use_amp
        )

        self._model_config = model_config
        self._train_config = train_config

        # Create model from schema
        schema = dataset.schema
        self._model = _CoreModel(schema, model_config)

        # Move to device
        if device == "cuda":
            self._model.to("cuda")

        # Create core trainer
        self._core = _CoreTrainer(self._model, train_config)

        # Prepare data
        self._core.prepare_data(dataset._core, test_size, seed)

        # Handle resume
        if resume and self._checkpoint_dir and (self._checkpoint_dir / "checkpoint.pt").exists():
            self._core = _CoreTrainer.load(
                str(self._checkpoint_dir / "checkpoint.pt"),
                device  # Pass string, not torch.device - bindings expect string
            )

    def fit(self) -> TrainResult:
        """
        Train the model.

        Returns:
            TrainResult with training history and metrics
        """
        result = self._core.fit()

        # Save checkpoint if directory specified
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._core.save(str(self._checkpoint_dir / "checkpoint.pt"))

        return result

    def create_predictor(self, device: Optional[str] = None) -> _CorePredictor:
        """
        Create a predictor from the trained model.

        Args:
            device: Device to run prediction on (default: trainer's device)

        Returns:
            Predictor instance for running inference
        """
        if device is None:
            device = self._device
        return _CorePredictor(
            self._core.model,
            self._core.scalers,
            device,  # Pass string, not torch.device
        )

    def predict(
        self,
        dataset: Optional[ResolveDataset] = None,
        return_latent: bool = False,
    ) -> ResolvePredictions:
        """
        Run prediction on a dataset.

        Args:
            dataset: Dataset to predict on (default: training dataset)
            return_latent: Whether to return latent representations

        Returns:
            ResolvePredictions with predictions and optional latent vectors
        """
        predictor = self.create_predictor()

        if dataset is None:
            dataset = self._dataset

        return predictor.predict_dataset(dataset._core, return_latent)

    def save(self, path: Union[str, Path]):
        """Save trainer state to file."""
        self._core.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "Trainer":
        """Load trainer from file."""
        # Load core trainer
        core = _CoreTrainer.load(str(path), device)

        # Create wrapper without full init
        trainer = cls.__new__(cls)
        trainer._core = core
        trainer._model = core.model
        trainer._model_config = core.model.config
        trainer._train_config = core.config
        trainer._checkpoint_dir = None
        trainer._checkpoint_every = 0
        trainer._resume = False
        trainer._device = device
        trainer._dataset = None

        return trainer

    @property
    def model(self):
        """Access the underlying model."""
        return self._model

    @property
    def scalers(self) -> Scalers:
        """Access the data scalers."""
        return self._core.scalers

    def n_parameters(self) -> int:
        """Count total trainable parameters."""
        return self._model.n_parameters()

    @property
    def n_params(self) -> int:
        """Alias for n_parameters() as a property."""
        return self.n_parameters()

    def compute_diagnostics(self) -> NetworkDiagnostics:
        """
        Compute network health diagnostics.

        Analyzes the model for common training issues:
        - Dead neurons (never activate)
        - Saturated neurons (always at extremes)
        - Activation statistics per layer

        Returns:
            NetworkDiagnostics with per-layer and overall health metrics
        """
        return self._core.compute_diagnostics()

    def compute_calibration(
        self,
        target_name: str,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """
        Compute calibration curve for a classification target.

        Calibration measures how well predicted probabilities match
        actual frequencies. A well-calibrated model predicts 70% confidence
        for samples that are correct 70% of the time.

        Args:
            target_name: Name of the classification target
            n_bins: Number of probability bins (default 10)

        Returns:
            CalibrationResult with bins, ECE, and MCE metrics
        """
        return self._core.compute_calibration(target_name, n_bins)

    def compute_residuals(self, target_name: str) -> ResidualAnalysis:
        """
        Compute residual analysis for a regression target.

        Analyzes prediction errors (residuals = actual - predicted):
        - Mean and std of residuals
        - Skewness and kurtosis
        - Quantiles (5th, 25th, 50th, 75th, 95th)

        Args:
            target_name: Name of the regression target

        Returns:
            ResidualAnalysis with residual statistics and raw values
        """
        return self._core.compute_residuals(target_name)

    def cross_validate(
        self,
        n_folds: int = 5,
        seed: int = 42,
    ) -> CrossValidationResult:
        """
        Perform k-fold cross-validation.

        Splits the full dataset into k folds, trains on k-1 folds,
        and evaluates on the held-out fold. Repeats for all folds
        and aggregates metrics.

        Args:
            n_folds: Number of cross-validation folds (default 5)
            seed: Random seed for fold assignment (default 42)

        Returns:
            CrossValidationResult with mean/std metrics across folds
            and per-fold TrainResult objects

        Example:
            cv_result = trainer.cross_validate(n_folds=5)
            print(f"R² = {cv_result.mean_metrics['area']['r_squared']:.3f} "
                  f"± {cv_result.std_metrics['area']['r_squared']:.3f}")
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        return self._core.cross_validate(n_folds, seed)
