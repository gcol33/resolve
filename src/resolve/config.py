"""
Configuration dataclasses for RESOLVE.

Provides typed, documented configuration objects for training.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class TrainerConfig:
    """
    Configuration for RESOLVE Trainer.

    Groups all training parameters into a single, reusable config object.

    Example:
        config = TrainerConfig(
            hash_dim=64,
            top_k=10,
            hidden_dims=[512, 256, 128, 64],
            max_epochs=1000,
            patience=50,
        )
        trainer = Trainer(dataset, **config.to_kwargs())

        # Or use preset configs
        trainer = Trainer(dataset, **LARGE_MODEL.to_kwargs())
    """

    # Species encoding
    hash_dim: int = 64
    top_k: int = 10
    top_k_species: int = 10
    species_embed_dim: int = 16

    # Taxonomy embeddings
    genus_emb_dim: int = 8
    family_emb_dim: int = 8

    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout: float = 0.1

    # Training
    max_epochs: int = 1000
    patience: int = 50
    batch_size: int = 16384
    lr: float = 1e-3
    weight_decay: float = 1e-5
    test_size: float = 0.2
    seed: int = 42
    loss_config: str = "combined"
    phase_boundaries: Optional[list[int]] = None

    # Checkpointing
    checkpoint_dir: Optional[Union[str, Path]] = None
    checkpoint_every: int = 50
    resume: bool = True

    # Device
    device: str = "cuda"

    def to_kwargs(self) -> dict:
        """
        Convert config to kwargs dict for Trainer.__init__().

        Returns:
            Dict of keyword arguments for Trainer constructor.
        """
        kwargs = {
            "hash_dim": self.hash_dim,
            "top_k": self.top_k,
            "top_k_species": self.top_k_species,
            "species_embed_dim": self.species_embed_dim,
            "genus_emb_dim": self.genus_emb_dim,
            "family_emb_dim": self.family_emb_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "test_size": self.test_size,
            "seed": self.seed,
            "loss_config": self.loss_config,
            "device": self.device,
        }

        if self.phase_boundaries is not None:
            kwargs["phase_boundaries"] = self.phase_boundaries

        if self.checkpoint_dir is not None:
            kwargs["checkpoint_dir"] = self.checkpoint_dir
            kwargs["checkpoint_every"] = self.checkpoint_every
            kwargs["resume"] = self.resume

        return kwargs

    def with_checkpoint(self, checkpoint_dir: Union[str, Path]) -> "TrainerConfig":
        """
        Return a copy of this config with checkpointing enabled.

        Args:
            checkpoint_dir: Directory for saving checkpoints.

        Returns:
            New TrainerConfig with checkpoint settings.
        """
        from dataclasses import replace
        return replace(self, checkpoint_dir=checkpoint_dir)

    def to_trainer_kwargs(self, dataset) -> dict:
        """
        Convert config to kwargs dict for Trainer, including dataset.

        Args:
            dataset: ResolveDataset instance.

        Returns:
            Dict of keyword arguments for Trainer constructor.
        """
        kwargs = self.to_kwargs()
        kwargs["dataset"] = dataset
        return kwargs


# Preset configurations
TINY_MODEL = TrainerConfig(
    hidden_dims=[128, 64],
)

SMALL_MODEL = TrainerConfig(
    hidden_dims=[256, 128, 64],
)

MEDIUM_MODEL = TrainerConfig(
    hidden_dims=[512, 256, 128, 64],
)

LARGE_MODEL = TrainerConfig(
    hidden_dims=[1024, 512, 256, 128, 64],
)

XL_MODEL = TrainerConfig(
    hidden_dims=[2048, 1024, 512, 256, 128, 64],
)

MAX_MODEL = TrainerConfig(
    hidden_dims=[2048, 1024, 512, 256, 128, 64, 32],
)
