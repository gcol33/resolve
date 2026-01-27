"""
RESOLVE - Species composition-based prediction.

High-level Python API for training and inference.

Example:
    from resolve import ResolveDataset, Trainer

    dataset = ResolveDataset.from_csv(
        header="header.csv",
        species="species.csv",
        roles={"plot_id": "plot_id", "species_id": "species"},
        targets={"area": {"column": "area", "task": "regression"}},
    )

    trainer = Trainer(dataset, hash_dim=64, hidden_dims=[256, 128])
    results = trainer.fit()
"""

import os
os.add_dll_directory(os.path.join(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"), "bin"))
import torch  # noqa: E402
from .dataset import ResolveDataset, RoleMapping, TargetConfig
from .trainer import Trainer
from .config import TrainerConfig

__version__ = "0.1.0"

__all__ = [
    "ResolveDataset",
    "Trainer",
    "RoleMapping",
    "TargetConfig",
    "TrainerConfig",
]
