"""TaskHead: prediction head for a single target."""

from typing import Literal, Optional

import numpy as np
import torch
from torch import nn


class TaskHead(nn.Module):
    """
    Prediction head for a single target.

    Supports:
        - Regression: linear output
        - Classification: linear + softmax

    Handles target transforms (log1p) for proper inverse transformation.
    """

    def __init__(
        self,
        latent_dim: int,
        task: Literal["regression", "classification"],
        num_classes: Optional[int] = None,
        transform: Optional[Literal["log1p"]] = None,
    ):
        super().__init__()

        self.task = task
        self.transform = transform
        self.num_classes = num_classes

        if task == "regression":
            self.head = nn.Linear(latent_dim, 1)
        else:
            if num_classes is None:
                raise ValueError("num_classes required for classification")
            self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            latent: (batch, latent_dim)

        Returns:
            For regression: (batch, 1) predictions
            For classification: (batch, num_classes) logits
        """
        return self.head(latent)

    def predict(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict with inverse transform applied.

        Args:
            latent: (batch, latent_dim)

        Returns:
            For regression: (batch,) predictions in original scale
            For classification: (batch,) predicted class indices
        """
        out = self.forward(latent)

        if self.task == "regression":
            out = out.squeeze(-1)
            if self.transform == "log1p":
                out = torch.expm1(out)
        else:
            out = out.argmax(dim=-1)

        return out

    def inverse_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transform to raw predictions.

        Only applicable for regression with transform.
        """
        if self.task == "regression" and self.transform == "log1p":
            return torch.expm1(predictions)
        return predictions

    def inverse_transform_numpy(self, predictions: np.ndarray) -> np.ndarray:
        """Numpy version of inverse transform."""
        if self.task == "regression" and self.transform == "log1p":
            return np.expm1(predictions)
        return predictions
