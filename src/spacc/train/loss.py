"""Loss functions for Spacc training."""

from typing import Optional

import torch
from torch import nn


class PhasedLoss:
    """
    Phased loss for regression targets.

    Phases:
        - Phase 1: MAE only (stable learning)
        - Phase 2: MAE + SMAPE (relative accuracy)
        - Phase 3: MAE + SMAPE + band penalty (target accuracy)

    For classification targets, uses CrossEntropy throughout.
    """

    def __init__(
        self,
        phase_boundaries: tuple[int, int] = (100, 300),
        smape_weight_p2: float = 0.2,
        smape_weight_p3: float = 0.15,
        band_weight_p3: float = 0.05,
        band_threshold: float = 0.25,
        eps: float = 1e-8,
    ):
        self.phase_boundaries = phase_boundaries
        self.smape_weight_p2 = smape_weight_p2
        self.smape_weight_p3 = smape_weight_p3
        self.band_weight_p3 = band_weight_p3
        self.band_threshold = band_threshold
        self.eps = eps

        self._mae = nn.L1Loss()
        self._ce = nn.CrossEntropyLoss()

    def get_phase(self, epoch: int) -> int:
        """Get current training phase (1, 2, or 3)."""
        if epoch < self.phase_boundaries[0]:
            return 1
        elif epoch < self.phase_boundaries[1]:
            return 2
        else:
            return 3

    def regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        scaler_mean: Optional[torch.Tensor] = None,
        scaler_scale: Optional[torch.Tensor] = None,
        transform: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute phased regression loss.

        Args:
            pred: (batch, 1) predictions (scaled)
            target: (batch, 1) targets (scaled)
            epoch: current epoch for phase determination
            scaler_mean: target scaler mean for inverse transform
            scaler_scale: target scaler scale for inverse transform
            transform: "log1p" or None
        """
        phase = self.get_phase(epoch)
        mae_loss = self._mae(pred, target)

        if phase == 1:
            return mae_loss

        # Phases 2 and 3 need original scale for SMAPE
        if scaler_mean is not None and scaler_scale is not None:
            pred_orig = pred * scaler_scale + scaler_mean
            target_orig = target * scaler_scale + scaler_mean
            # Only apply expm1 if log1p transform was used
            if transform == "log1p":
                pred_raw = torch.expm1(pred_orig)
                target_raw = torch.expm1(target_orig)
            else:
                pred_raw = pred_orig
                target_raw = target_orig
        else:
            pred_raw = pred
            target_raw = target

        # SMAPE
        abs_diff = torch.abs(pred_raw - target_raw)
        denominator = (torch.abs(pred_raw) + torch.abs(target_raw)) / 2 + self.eps
        smape = (abs_diff / denominator).mean()

        if phase == 2:
            return (1 - self.smape_weight_p2) * mae_loss + self.smape_weight_p2 * smape

        # Phase 3: add band penalty
        rel_error = abs_diff / (torch.abs(target_raw) + self.eps)
        band_violation = torch.relu(rel_error - self.band_threshold)
        band_loss = band_violation.mean()

        total_weight = 1 - self.smape_weight_p3 - self.band_weight_p3
        return (
            total_weight * mae_loss
            + self.smape_weight_p3 * smape
            + self.band_weight_p3 * band_loss
        )

    def classification_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            pred: (batch, num_classes) logits
            target: (batch,) integer labels
        """
        return self._ce(pred, target)


class MultiTaskLoss:
    """
    Combines losses across multiple targets.

    Applies appropriate loss function per task type
    and weights by target configuration.
    """

    def __init__(
        self,
        target_configs: dict,
        phase_boundaries: tuple[int, int] = (100, 300),
    ):
        self.target_configs = target_configs
        self.phased_loss = PhasedLoss(phase_boundaries=phase_boundaries)

    def __call__(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        epoch: int,
        scalers: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute combined multi-task loss.

        Args:
            predictions: {target_name: pred_tensor}
            targets: {target_name: target_tensor}
            epoch: current epoch
            scalers: optional {target_name: (mean, scale)} for regression

        Returns:
            total_loss, {target_name: individual_loss}
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for name, cfg in self.target_configs.items():
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            if cfg.task == "regression":
                scaler_mean = None
                scaler_scale = None
                if scalers and name in scalers:
                    scaler_mean, scaler_scale = scalers[name]

                loss = self.phased_loss.regression_loss(
                    pred, target, epoch, scaler_mean, scaler_scale, cfg.transform
                )
            else:
                loss = self.phased_loss.classification_loss(pred, target)

            losses[name] = loss
            total = total + cfg.weight * loss

        return total, losses
