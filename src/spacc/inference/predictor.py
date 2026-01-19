"""Predictor: inference interface for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from spacc.data.dataset import SpaccDataset
from spacc.encode.species import SpeciesEncoder
from spacc.model.spacc import SpaccModel
from spacc.train.trainer import Trainer


@dataclass
class SpaccPredictions:
    """Container for model predictions."""

    predictions: dict[str, np.ndarray]
    plot_ids: np.ndarray
    latent: Optional[np.ndarray] = None

    def __getitem__(self, target: str) -> np.ndarray:
        """Get predictions for a target."""
        return self.predictions[target]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        df = pd.DataFrame({"plot_id": self.plot_ids})
        for name, pred in self.predictions.items():
            df[name] = pred
        return df

    def to_csv(self, path: str | Path) -> None:
        """Save predictions to CSV."""
        self.to_dataframe().to_csv(path, index=False)


class Predictor:
    """
    Inference interface for trained Spacc models.

    Loads a saved model and predicts on new datasets.
    """

    def __init__(
        self,
        model: SpaccModel,
        species_encoder: SpeciesEncoder,
        scalers: dict,
        device: str = "auto",
    ):
        self.model = model
        self.species_encoder = species_encoder
        self.scalers = scalers

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.model.to(self._device)
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Predictor:
        """Load predictor from saved checkpoint."""
        model, encoder, scalers = Trainer.load(path, device)
        return cls(model, encoder, scalers, device)

    @torch.no_grad()
    def predict(
        self,
        dataset: SpaccDataset,
        return_latent: bool = False,
    ) -> SpaccPredictions:
        """
        Predict on a dataset.

        Args:
            dataset: SpaccDataset to predict on
            return_latent: If True, also return latent representations

        Returns:
            SpaccPredictions with results for all targets
        """
        # Encode species
        encoded = self.species_encoder.transform(dataset)

        # Get continuous features
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()

        if covariates is not None:
            continuous = np.hstack([coords, covariates, encoded.hash_embedding])
        else:
            continuous = np.hstack([coords, encoded.hash_embedding])

        # Scale
        continuous = self.scalers["continuous"].transform(continuous).astype(np.float32)

        # Convert to tensors
        continuous_t = torch.from_numpy(continuous).to(self._device)

        if encoded.genus_ids is not None:
            genus_t = torch.from_numpy(encoded.genus_ids).to(self._device)
            family_t = torch.from_numpy(encoded.family_ids).to(self._device)
        else:
            genus_t = None
            family_t = None

        # Forward pass
        predictions_raw = self.model(continuous_t, genus_t, family_t)

        # Get latent if requested
        latent = None
        if return_latent:
            latent = self.model.get_latent(continuous_t, genus_t, family_t)
            latent = latent.cpu().numpy()

        # Post-process predictions
        predictions = {}
        for name, pred in predictions_raw.items():
            cfg = self.model.target_configs[name]

            if cfg.task == "regression":
                # Inverse scale
                pred_np = pred.cpu().numpy()
                scaler = self.scalers[f"target_{name}"]
                pred_np = scaler.inverse_transform(pred_np).flatten()

                # Inverse transform (e.g., expm1 for log1p)
                if cfg.transform == "log1p":
                    pred_np = np.expm1(pred_np)

                predictions[name] = pred_np
            else:
                # Classification: return class indices
                predictions[name] = pred.argmax(dim=-1).cpu().numpy()

        return SpaccPredictions(
            predictions=predictions,
            plot_ids=dataset.plot_ids,
            latent=latent,
        )

    def get_embeddings(self, dataset: SpaccDataset) -> np.ndarray:
        """
        Get latent embeddings for all plots.

        Useful for visualization and interpretation.
        """
        result = self.predict(dataset, return_latent=True)
        return result.latent

    def get_genus_embeddings(self) -> np.ndarray:
        """
        Get learned genus embedding weights.

        Returns:
            (n_genera, genus_emb_dim) array
        """
        if not self.model.encoder.has_taxonomy:
            raise ValueError("Model has no taxonomy embeddings")

        # Get first genus embedding layer weights
        return self.model.encoder.genus_embeddings[0].weight.detach().cpu().numpy()

    def get_family_embeddings(self) -> np.ndarray:
        """
        Get learned family embedding weights.

        Returns:
            (n_families, family_emb_dim) array
        """
        if not self.model.encoder.has_taxonomy:
            raise ValueError("Model has no taxonomy embeddings")

        return self.model.encoder.family_embeddings[0].weight.detach().cpu().numpy()
