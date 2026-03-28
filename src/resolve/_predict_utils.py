"""Shared prediction post-processing for Predictor and Trainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["postprocess_predictions"]


def postprocess_predictions(
    predictions_raw: dict[str, torch.Tensor],
    target_configs: dict[str, Any],
    scalers: dict[str, Any],
    unknown_fraction: np.ndarray,
    output_space: str,
    confidence_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Post-process raw model outputs into final predictions.

    Handles inverse scaling, inverse transforms (expm1 for log1p),
    confidence computation, and threshold filtering for both regression
    and classification targets.

    Args:
        predictions_raw: Dict mapping target name to raw prediction tensors.
        target_configs: Dict mapping target name to config with .task and .transform.
        scalers: Dict of fitted scalers; keys like "target_{name}".
        unknown_fraction: Per-sample unknown species fraction (n_samples,).
        output_space: "raw" to inverse-transform, "transformed" to keep model scale.
        confidence_threshold: Minimum confidence; predictions below are set to NaN.

    Returns:
        Tuple of (predictions dict, confidence dict), both mapping
        target name to numpy arrays of shape (n_samples,).
    """
    regression_confidence = 1.0 - unknown_fraction
    predictions = {}
    confidence = {}

    for name, pred in predictions_raw.items():
        cfg = target_configs[name]

        if cfg.task == "regression":
            pred_np = pred.cpu().numpy()
            scaler = scalers[f"target_{name}"]
            pred_np = scaler.inverse_transform(pred_np).flatten()

            if cfg.transform == "log1p" and output_space == "raw":
                pred_np = np.expm1(pred_np)

            pred_np = np.where(regression_confidence >= confidence_threshold, pred_np, np.nan)
            predictions[name] = pred_np
            confidence[name] = regression_confidence
        else:
            probs = torch.softmax(pred, dim=-1)
            class_confidence = probs.max(dim=-1).values.cpu().numpy()
            pred_np = pred.argmax(dim=-1).cpu().numpy().astype(np.float64)
            pred_np = np.where(class_confidence >= confidence_threshold, pred_np, np.nan)
            predictions[name] = pred_np
            confidence[name] = class_confidence

    return predictions, confidence
