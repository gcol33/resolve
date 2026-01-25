"""
High-level Trainer class that wraps the C++ core.

Provides a dataset-first API matching the paper's expected interface.
"""

from pathlib import Path
from typing import Optional, Union

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
)
from .dataset import ResolveDataset


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
        """
        self._dataset = dataset
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_every = checkpoint_every
        self._resume = resume
        self._device = device
        self._test_size = test_size
        self._seed = seed

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Build model config
        model_config = _CoreModelConfig()
        model_config.species_encoding = dataset._core.config.species_encoding
        model_config.hash_dim = hash_dim
        model_config.species_embed_dim = species_embed_dim
        model_config.genus_emb_dim = genus_emb_dim
        model_config.family_emb_dim = family_emb_dim
        model_config.top_k = top_k
        model_config.top_k_species = top_k_species
        model_config.hidden_dims = hidden_dims
        model_config.dropout = dropout

        # Build train config
        train_config = _CoreTrainConfig()
        train_config.batch_size = batch_size
        train_config.max_epochs = max_epochs
        train_config.patience = patience
        train_config.lr = lr
        train_config.weight_decay = weight_decay
        if phase_boundaries:
            train_config.phase_boundaries = phase_boundaries

        # Map loss config string to enum
        loss_map = {
            "mae": LossConfigMode.MAE,
            "smape": LossConfigMode.SMAPE,
            "combined": LossConfigMode.Combined,
        }
        train_config.loss_config = loss_map.get(loss_config.lower(), LossConfigMode.Combined)

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
                torch.device(device)
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
        device = torch.device(self._device)
        predictor = _CorePredictor(
            self._core.model,
            self._core.scalers,
            device,
        )

        if dataset is None:
            dataset = self._dataset

        return predictor.predict(
            dataset._core.coordinates,
            dataset._core.covariates,
            dataset._core.hash_embedding,
            dataset._core.genus_ids,
            dataset._core.family_ids,
            return_latent,
        )

    def save(self, path: Union[str, Path]):
        """Save trainer state to file."""
        self._core.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "Trainer":
        """Load trainer from file."""
        # Load core trainer
        core = _CoreTrainer.load(str(path), torch.device(device))

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
