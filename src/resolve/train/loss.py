"""Loss functions for RESOLVE training."""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@dataclass
class PhaseConfig:
    """
    Configuration for loss components in a single training phase.

    All weights should sum to 1.0 for interpretability, but this is not enforced.

    Available loss components:
        - mae: Mean Absolute Error (L1) - robust to outliers
        - mse: Mean Squared Error (L2) - penalizes large errors more
        - huber: Huber loss - combines MAE/MSE, smooth transition at delta
        - smape: Symmetric Mean Absolute Percentage Error - relative accuracy
        - band: Band penalty - penalizes predictions outside threshold

    Example:
        # Phase with 80% MAE and 20% SMAPE
        phase = PhaseConfig(mae=0.8, smape=0.2)
    """

    mae: float = 0.0
    mse: float = 0.0
    huber: float = 0.0
    smape: float = 0.0
    band: float = 0.0

    # Huber delta (transition point between L1 and L2)
    huber_delta: float = 1.0
    # Band threshold (relative error threshold for penalty)
    band_threshold: float = 0.25

    def __post_init__(self):
        total = self.mae + self.mse + self.huber + self.smape + self.band
        if total == 0:
            raise ValueError("At least one loss component must have non-zero weight")

    @property
    def needs_original_scale(self) -> bool:
        """Whether this phase needs original-scale values (for SMAPE/band)."""
        return self.smape > 0 or self.band > 0


class PhasedLoss:
    """
    Phased loss for regression targets with configurable loss components.

    Users configure which loss functions to use in each phase via PhaseConfig.
    Supports any number of phases (1, 2, 3, ...).

    For classification targets, uses CrossEntropy throughout.

    Example:
        # Single phase (no boundaries needed)
        loss = PhasedLoss(
            phases={1: PhaseConfig(mae=0.8, smape=0.15, band=0.05)}
        )

        # Two phases
        loss = PhasedLoss(
            phase_boundaries=[100],
            phases={
                1: PhaseConfig(mae=1.0),
                2: PhaseConfig(mae=0.7, smape=0.3),
            }
        )

        # Four phases
        loss = PhasedLoss(
            phase_boundaries=[50, 150, 300],
            phases={
                1: PhaseConfig(mse=1.0),
                2: PhaseConfig(mse=0.8, smape=0.2),
                3: PhaseConfig(mse=0.6, smape=0.3, band=0.1),
                4: PhaseConfig(mse=0.5, smape=0.3, band=0.2),
            }
        )
    """

    def __init__(
        self,
        phases: Optional[dict[int, PhaseConfig]] = None,
        phase_boundaries: Optional[list[int]] = None,
        eps: float = 1e-8,
    ):
        """
        Args:
            phases: Dict mapping phase number (1, 2, ...) to PhaseConfig.
                    If None, uses single-phase default with MAE only.
            phase_boundaries: List of epoch thresholds for phase transitions.
                              Length must be len(phases) - 1.
                              Example: [100, 300] for 3 phases.
                              If None and phases has 1 entry, no boundaries needed.
            eps: Small constant for numerical stability
        """
        # Default: single phase with MAE only
        if phases is None:
            phases = {1: PhaseConfig(mae=1.0)}

        self.phases = phases
        self.num_phases = len(phases)

        # Validate phase_boundaries
        if phase_boundaries is None:
            phase_boundaries = []
        self.phase_boundaries = list(phase_boundaries)

        expected_boundaries = self.num_phases - 1
        if len(self.phase_boundaries) != expected_boundaries:
            raise ValueError(
                f"phases has {self.num_phases} entries, so phase_boundaries must have "
                f"{expected_boundaries} entries, got {len(self.phase_boundaries)}"
            )

        # Validate phases dict has consecutive keys starting at 1
        expected_keys = set(range(1, self.num_phases + 1))
        if set(phases.keys()) != expected_keys:
            raise ValueError(
                f"phases keys must be consecutive integers starting at 1: {expected_keys}, "
                f"got {set(phases.keys())}"
            )

        self.eps = eps
        self._mae = nn.L1Loss()
        self._mse = nn.MSELoss()
        self._ce = nn.CrossEntropyLoss()

    def get_phase(self, epoch: int) -> int:
        """Get current training phase (1-indexed)."""
        for i, boundary in enumerate(self.phase_boundaries):
            if epoch < boundary:
                return i + 1
        return self.num_phases

    def _compute_loss_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_raw: torch.Tensor,
        target_raw: torch.Tensor,
        config: PhaseConfig,
    ) -> torch.Tensor:
        """Compute weighted sum of loss components based on config."""
        # Mask for valid (non-NaN, non-Inf) values
        valid_mask = torch.isfinite(pred) & torch.isfinite(target)
        if not valid_mask.any():
            # Return zero loss that maintains gradient chain
            return (pred * 0.0).sum()

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        # Collect loss components
        components = []

        # MAE (scaled space)
        if config.mae > 0:
            components.append(config.mae * self._mae(pred_valid, target_valid))

        # MSE (scaled space)
        if config.mse > 0:
            components.append(config.mse * self._mse(pred_valid, target_valid))

        # Huber (scaled space)
        if config.huber > 0:
            huber = nn.HuberLoss(delta=config.huber_delta)
            components.append(config.huber * huber(pred_valid, target_valid))

        # SMAPE (original scale)
        if config.smape > 0:
            pred_raw_valid = pred_raw[valid_mask]
            target_raw_valid = target_raw[valid_mask]
            abs_diff = torch.abs(pred_raw_valid - target_raw_valid)
            denominator = (torch.abs(pred_raw_valid) + torch.abs(target_raw_valid)) / 2 + self.eps
            # Clamp denominator to avoid very small values
            denominator = torch.clamp(denominator, min=self.eps * 10)
            smape = (abs_diff / denominator).mean()
            components.append(config.smape * smape)

        # Band penalty (original scale)
        if config.band > 0:
            pred_raw_valid = pred_raw[valid_mask]
            target_raw_valid = target_raw[valid_mask]
            abs_diff = torch.abs(pred_raw_valid - target_raw_valid)
            # Clamp denominator to avoid division by zero
            rel_error = abs_diff / torch.clamp(torch.abs(target_raw_valid), min=self.eps * 10)
            band_violation = torch.relu(rel_error - config.band_threshold)
            band_loss = band_violation.mean()
            components.append(config.band * band_loss)

        # Sum all components (will maintain gradient)
        if components:
            return sum(components)
        else:
            # Fallback: return zero loss that maintains gradient chain
            return (pred * 0.0).sum()

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
        config = self.phases.get(phase, self.phases[max(self.phases.keys())])

        # Compute original-scale values if needed for SMAPE/band
        if config.needs_original_scale and scaler_mean is not None and scaler_scale is not None:
            pred_orig = pred * scaler_scale + scaler_mean
            target_orig = target * scaler_scale + scaler_mean
            if transform == "log1p":
                pred_raw = torch.expm1(pred_orig)
                target_raw = torch.expm1(target_orig)
            else:
                pred_raw = pred_orig
                target_raw = target_orig
        else:
            pred_raw = pred
            target_raw = target

        return self._compute_loss_components(pred, target, pred_raw, target_raw, config)

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

    Optimized for common single-target regression case.
    """

    def __init__(
        self,
        target_configs: dict,
        phases: Optional[dict[int, PhaseConfig]] = None,
        phase_boundaries: Optional[list[int]] = None,
    ):
        self.target_configs = target_configs
        self.phased_loss = PhasedLoss(phases=phases, phase_boundaries=phase_boundaries)

        # Pre-compute target info for fast path
        self._target_names = list(target_configs.keys())
        self._target_cfgs = [target_configs[n] for n in self._target_names]
        self._n_targets = len(self._target_names)

        # Check if we can use fast path (single regression target, MAE-only, single phase)
        self._use_fast_path = (
            self._n_targets == 1
            and self._target_cfgs[0].task == "regression"
            and self.phased_loss.num_phases == 1
            and self.phased_loss.phases[1].mae == 1.0
            and self.phased_loss.phases[1].mse == 0.0
            and self.phased_loss.phases[1].huber == 0.0
            and self.phased_loss.phases[1].smape == 0.0
            and self.phased_loss.phases[1].band == 0.0
        )

        # Cache loss function for fast path
        self._mae = nn.L1Loss()
        self._ce = nn.CrossEntropyLoss()

        # Cache current epoch/phase to avoid repeated lookups
        self._cached_epoch = -1
        self._cached_phase = 1
        self._cached_config = self.phased_loss.phases[1]

    def _update_phase_cache(self, epoch: int) -> None:
        """Update cached phase config if epoch changed."""
        if epoch != self._cached_epoch:
            self._cached_epoch = epoch
            self._cached_phase = self.phased_loss.get_phase(epoch)
            self._cached_config = self.phased_loss.phases.get(
                self._cached_phase, self.phased_loss.phases[max(self.phased_loss.phases.keys())]
            )

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
        # Fast path: single regression target with MAE-only loss
        if self._use_fast_path:
            name = self._target_names[0]
            pred = predictions[name]
            target = targets[name]
            loss = self._mae(pred, target)
            return loss, {name: loss}

        # Standard path with phase handling
        self._update_phase_cache(epoch)

        losses = {}
        total = None

        for i, name in enumerate(self._target_names):
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]
            cfg = self._target_cfgs[i]

            if cfg.task == "regression":
                scaler_mean = None
                scaler_scale = None
                if scalers and name in scalers:
                    scaler_mean, scaler_scale = scalers[name]

                loss = self.phased_loss.regression_loss(
                    pred, target, epoch, scaler_mean, scaler_scale, cfg.transform
                )
            else:
                loss = self._ce(pred, target)

            losses[name] = loss
            weighted = cfg.weight * loss
            if total is None:
                total = weighted
            else:
                total = total + weighted

        return total if total is not None else pred.new_tensor(0.0), losses
