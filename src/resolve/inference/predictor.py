"""Predictor: inference interface for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch

from resolve.data.dataset import ResolveDataset
from resolve.encode.species import SpeciesEncoder
from resolve.encode.embedding import EmbeddingEncoder
from resolve.model.resolve import ResolveModel
from resolve.train.trainer import Trainer


@dataclass
class ResolvePredictions:
    """Container for model predictions."""

    predictions: dict[str, np.ndarray]
    plot_ids: np.ndarray
    latent: Optional[np.ndarray] = None
    confidence: Optional[dict[str, np.ndarray]] = None

    def __getitem__(self, target: str) -> np.ndarray:
        """Get predictions for a target."""
        return self.predictions[target]

    def to_polars(self) -> pl.DataFrame:
        """Convert predictions to polars DataFrame."""
        data = {"plot_id": self.plot_ids}
        data.update(self.predictions)
        return pl.DataFrame(data)

    def to_csv(self, path: str | Path) -> None:
        """Save predictions to CSV."""
        self.to_polars().write_csv(path)


class Predictor:
    """
    Inference interface for trained RESOLVE models.

    Loads a saved model and predicts on new datasets.
    """

    def __init__(
        self,
        model: ResolveModel,
        species_encoder: SpeciesEncoder | EmbeddingEncoder,
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
        dataset: ResolveDataset,
        return_latent: bool = False,
        output_space: str = "raw",
        confidence_threshold: float = 0.0,
    ) -> ResolvePredictions:
        """
        Predict on a dataset.

        Args:
            dataset: ResolveDataset to predict on
            return_latent: If True, also return latent representations
            output_space: Output space for regression predictions.
                "raw" (default): inverse-transform predictions to original scale
                "transformed": keep predictions in transformed space (e.g., log1p)
            confidence_threshold: Minimum confidence for predictions (0-1).
                Predictions below threshold are set to NaN.
                Default 0 means all predictions are kept (gap-fill everything).

                Confidence semantics:
                - Regression: confidence = 1 - unknown_fraction, where unknown_fraction
                  is the proportion of species abundance not seen during training.
                  This reflects coverage of the species space, not statistical uncertainty.
                - Classification: confidence = max softmax probability across classes.

                These values are heuristic and intended for filtering/diagnostics,
                not formal uncertainty quantification.

        Returns:
            ResolvePredictions with results for all targets
        """
        if output_space not in ("raw", "transformed"):
            raise ValueError(f"output_space must be 'raw' or 'transformed', got {output_space!r}")

        # Get continuous features
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()
        schema = self.model.schema

        species_ids_t = None
        species_vector_t = None
        genus_t = None
        family_t = None
        # Rank-pool mode tensors
        pool_genus_ids_t = None
        pool_family_ids_t = None
        pool_weights_t = None
        pool_mask_t = None
        pool_has_cover_t = None

        encoding = self.model.species_encoding

        if encoding == "embed":
            embedded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if schema.track_unknown_fraction:
                parts.append(embedded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = embedded.unknown_fraction

            species_ids_t = torch.from_numpy(embedded.species_ids).to(self._device)
            if embedded.genus_ids is not None:
                genus_t = torch.from_numpy(embedded.genus_ids).to(self._device)
                family_t = torch.from_numpy(embedded.family_ids).to(self._device)

        elif encoding == "rank_pool":
            pool_encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if schema.track_unknown_fraction:
                parts.append(pool_encoded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = pool_encoded.unknown_fraction

            from resolve.encode.rank_pool import pad_rank_pool_encoded
            padded = pad_rank_pool_encoded(pool_encoded)
            species_ids_t = torch.from_numpy(padded["species_ids"]).long().to(self._device)
            pool_genus_ids_t = torch.from_numpy(padded["genus_ids"]).long().to(self._device)
            pool_family_ids_t = torch.from_numpy(padded["family_ids"]).long().to(self._device)
            pool_weights_t = torch.from_numpy(padded["weights"]).to(self._device)
            pool_mask_t = torch.from_numpy(padded["mask"]).to(self._device)
            pool_has_cover_t = torch.from_numpy(padded["has_cover"]).to(self._device)

        else:
            # Hash mode
            encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords)
            if covariates is not None:
                parts.append(covariates)
            if self.model.uses_explicit_vector:
                species_vector_t = torch.from_numpy(encoded.species_vector).to(self._device)
            else:
                parts.append(encoded.hash_embedding)
            if schema.track_unknown_fraction:
                parts.append(encoded.unknown_fraction.reshape(-1, 1))
            if schema.track_unknown_count and encoded.unknown_count is not None:
                parts.append(encoded.unknown_count.reshape(-1, 1).astype(np.float32))
            unknown_fraction = encoded.unknown_fraction

            if encoded.genus_ids is not None:
                genus_t = torch.from_numpy(encoded.genus_ids).to(self._device)
                family_t = torch.from_numpy(encoded.family_ids).to(self._device)

        continuous = np.hstack(parts) if parts else np.zeros((len(dataset.plot_ids), 0), dtype=np.float32)

        # Scale
        continuous = self.scalers["continuous"].transform(continuous).astype(np.float32)

        # Convert to tensors
        continuous_t = torch.from_numpy(continuous).to(self._device)

        # Forward pass (dispatch based on encoding mode)
        if encoding == "rank_pool":
            predictions_raw = self.model(
                continuous_t, genus_ids=None, family_ids=None,
                species_ids=species_ids_t, species_vector=None,
                pool_genus_ids=pool_genus_ids_t, pool_family_ids=pool_family_ids_t,
                pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                pool_has_cover=pool_has_cover_t,
            )
        else:
            predictions_raw = self.model(
                continuous_t, genus_t, family_t,
                species_ids=species_ids_t, species_vector=species_vector_t,
            )

        # Get latent if requested
        latent = None
        if return_latent:
            if encoding == "rank_pool":
                latent = self.model.get_latent(
                    continuous_t, genus_ids=None, family_ids=None,
                    species_ids=species_ids_t, species_vector=None,
                    pool_genus_ids=pool_genus_ids_t, pool_family_ids=pool_family_ids_t,
                    pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                    pool_has_cover=pool_has_cover_t,
                )
            else:
                latent = self.model.get_latent(
                    continuous_t, genus_t, family_t,
                    species_ids=species_ids_t, species_vector=species_vector_t,
                )
            latent = latent.cpu().numpy()

        # Compute confidence (1 - unknown_fraction for regression)
        regression_confidence = 1.0 - unknown_fraction

        # Post-process predictions
        predictions = {}
        confidence = {}
        for name, pred in predictions_raw.items():
            cfg = self.model.target_configs[name]

            if cfg.task == "regression":
                # Inverse scale
                pred_np = pred.cpu().numpy()
                scaler = self.scalers[f"target_{name}"]
                pred_np = scaler.inverse_transform(pred_np).flatten()

                # Inverse transform (e.g., expm1 for log1p) unless user wants transformed space
                if cfg.transform == "log1p" and output_space == "raw":
                    pred_np = np.expm1(pred_np)

                # Apply confidence threshold
                pred_np = np.where(regression_confidence >= confidence_threshold, pred_np, np.nan)
                predictions[name] = pred_np
                confidence[name] = regression_confidence
            else:
                # Classification: use max softmax probability as confidence
                probs = torch.softmax(pred, dim=-1)
                class_confidence = probs.max(dim=-1).values.cpu().numpy()
                pred_np = pred.argmax(dim=-1).cpu().numpy().astype(np.float64)
                # Apply confidence threshold
                pred_np = np.where(class_confidence >= confidence_threshold, pred_np, np.nan)
                predictions[name] = pred_np
                confidence[name] = class_confidence

        return ResolvePredictions(
            predictions=predictions,
            plot_ids=dataset.plot_ids,
            latent=latent,
            confidence=confidence,
        )

    def get_embeddings(self, dataset: ResolveDataset) -> np.ndarray:
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
