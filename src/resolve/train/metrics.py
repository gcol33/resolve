"""Metrics for Spacc evaluation."""

from typing import Optional

import numpy as np


def band_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.25,
    eps: float = 1e-8,
) -> float:
    """
    Compute fraction of predictions within ±threshold of target.

    For log1p transformed values, first converts back to original scale.

    Args:
        pred: predictions (may be log-transformed)
        target: targets (may be log-transformed)
        threshold: relative error threshold (e.g., 0.25 for ±25%)
        eps: small constant for numerical stability

    Returns:
        Fraction of predictions within band
    """
    rel_error = np.abs(pred - target) / (np.abs(target) + eps)
    return float((rel_error <= threshold).mean())


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.abs(pred - target).mean())


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(((pred - target) ** 2).mean()))


def smape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric mean absolute percentage error."""
    numerator = np.abs(pred - target)
    denominator = (np.abs(pred) + np.abs(target)) / 2 + eps
    return float((numerator / denominator).mean())


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """Classification accuracy."""
    return float((pred == target).mean())


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    task: str,
    transform: Optional[str] = None,
) -> dict[str, float]:
    """
    Compute all relevant metrics for a target.

    Args:
        pred: predictions
        target: targets
        task: "regression" or "classification"
        transform: "log1p" or None (for regression inverse transform)

    Returns:
        Dictionary of metric names to values
    """
    if task == "regression":
        # Apply inverse transform for interpretable metrics
        if transform == "log1p":
            # Clamp to prevent overflow in expm1 (safe range for float64: ~±709)
            pred_orig = np.expm1(np.clip(pred, -700, 700))
            target_orig = np.expm1(np.clip(target, -700, 700))
        else:
            pred_orig = pred
            target_orig = target

        return {
            "mae": mae(pred_orig, target_orig),
            "rmse": rmse(pred_orig, target_orig),
            "smape": smape(pred_orig, target_orig),
            "band_25": band_accuracy(pred_orig, target_orig, 0.25),
            "band_50": band_accuracy(pred_orig, target_orig, 0.50),
            "band_75": band_accuracy(pred_orig, target_orig, 0.75),
        }
    else:
        return {
            "accuracy": accuracy(pred, target),
        }
