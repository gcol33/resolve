"""Metrics for RESOLVE evaluation."""

from typing import Optional

import numpy as np


def r_squared(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination (R²).

    Args:
        pred: predictions array
        target: target array

    Returns:
        R² value. 1.0 is perfect prediction, 0.0 means predicting the mean,
        negative values mean worse than predicting the mean.
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: predicted class labels (integer array)
        target: true class labels (integer array)
        n_classes: number of classes

    Returns:
        (n_classes, n_classes) confusion matrix where cm[i, j] is the count
        of samples with true class i predicted as class j.
    """
    pred_int = pred.astype(np.int64)
    target_int = target.astype(np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(target_int, pred_int):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> dict[str, float]:
    """Compute precision, recall, F1 per class from a confusion matrix.

    Args:
        cm: (n_classes, n_classes) confusion matrix from ``confusion_matrix()``.

    Returns:
        Dictionary with per-class precision/recall/F1 and macro averages.
    """
    n_classes = cm.shape[0]
    metrics: dict[str, float] = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"class_{c}_precision"] = precision
        metrics[f"class_{c}_recall"] = recall
        metrics[f"class_{c}_f1"] = f1

    metrics["macro_precision"] = sum(
        metrics[f"class_{c}_precision"] for c in range(n_classes)
    ) / n_classes
    metrics["macro_recall"] = sum(
        metrics[f"class_{c}_recall"] for c in range(n_classes)
    ) / n_classes
    metrics["macro_f1"] = sum(
        metrics[f"class_{c}_f1"] for c in range(n_classes)
    ) / n_classes
    return metrics


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
    num_classes: Optional[int] = None,
) -> dict[str, float]:
    """
    Compute all relevant metrics for a target.

    Args:
        pred: predictions (class labels for classification, values for regression)
        target: targets
        task: "regression" or "classification"
        transform: "log1p" or None (for regression inverse transform)
        num_classes: number of classes (required for classification)

    Returns:
        Dictionary of metric names to values. For classification, includes
        per-class precision/recall/F1 and macro averages.
    """
    if task == "regression":
        # Apply inverse transform for interpretable metrics
        if transform == "log1p":
            # Clamp to prevent overflow in expm1
            # Upper bound 88 gives exp(88) ≈ 1.6e38, safely within float64
            # For context: log1p(1e9 m²) ≈ 20.7, so 88 is very conservative
            pred_orig = np.expm1(np.clip(pred, -88, 88))
            target_orig = np.expm1(np.clip(target, -88, 88))
        else:
            pred_orig = pred
            target_orig = target

        return {
            "mae": mae(pred_orig, target_orig),
            "rmse": rmse(pred_orig, target_orig),
            "r_squared": r_squared(pred_orig, target_orig),
            "smape": smape(pred_orig, target_orig),
            "band_25": band_accuracy(pred_orig, target_orig, 0.25),
            "band_50": band_accuracy(pred_orig, target_orig, 0.50),
            "band_75": band_accuracy(pred_orig, target_orig, 0.75),
        }
    else:
        if num_classes is None:
            raise ValueError("num_classes is required for classification metrics")
        cm = confusion_matrix(pred, target, num_classes)
        class_metrics = per_class_metrics(cm)
        result: dict[str, float] = {"accuracy": accuracy(pred, target)}
        result.update(class_metrics)
        return result
