"""Predictor: inference interface for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import polars as pl
import torch

from resolve._predict_utils import postprocess_predictions
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
        scalers: dict[str, Any],
        device: str = "auto",
        categorical_vocabs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.species_encoder = species_encoder
        self.scalers = scalers
        self.categorical_vocabs = categorical_vocabs or {}

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.model.to(self._device)
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> Predictor:
        """Load predictor from saved checkpoint."""
        model, encoder, scalers, categorical_vocabs = Trainer.load(path, device)
        return cls(model, encoder, scalers, device, categorical_vocabs=categorical_vocabs)

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

        # Build categorical IDs for prediction
        categorical_ids_t = None
        if self.categorical_vocabs and schema.has_categoricals:
            cat_data = dataset.get_categoricals()
            if cat_data is not None:
                cat_arrays = []
                for cat_name in schema.categorical_names:
                    vocab = self.categorical_vocabs[cat_name]
                    cat_arrays.append(vocab.encode_array(cat_data[cat_name]))
                categorical_ids_t = torch.from_numpy(
                    np.stack(cat_arrays, axis=1)
                ).to(self._device)

        # Forward pass (dispatch based on encoding mode)
        if encoding == "rank_pool":
            predictions_raw = self.model(
                continuous_t, genus_ids=None, family_ids=None,
                species_ids=species_ids_t, species_vector=None,
                pool_genus_ids=pool_genus_ids_t, pool_family_ids=pool_family_ids_t,
                pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                pool_has_cover=pool_has_cover_t,
                categorical_ids=categorical_ids_t,
            )
        else:
            predictions_raw = self.model(
                continuous_t, genus_t, family_t,
                species_ids=species_ids_t, species_vector=species_vector_t,
                categorical_ids=categorical_ids_t,
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
                    categorical_ids=categorical_ids_t,
                )
            else:
                latent = self.model.get_latent(
                    continuous_t, genus_t, family_t,
                    species_ids=species_ids_t, species_vector=species_vector_t,
                    categorical_ids=categorical_ids_t,
                )
            latent = latent.cpu().numpy()

        # Post-process predictions
        predictions, confidence = postprocess_predictions(
            predictions_raw, self.model.target_configs, self.scalers,
            unknown_fraction, output_space, confidence_threshold,
        )

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
            (n_genera, genus_emb_dim) array. Averaged across positions for
            per-rank encoders (hash, sparse).
        """
        weights = self.model.get_genus_weights()
        if weights is None:
            raise ValueError("Model has no genus embeddings")
        return weights.detach().cpu().numpy()

    def get_family_embeddings(self) -> np.ndarray:
        """
        Get learned family embedding weights.

        Returns:
            (n_families, family_emb_dim) array. Averaged across positions for
            per-rank encoders (hash, sparse).
        """
        weights = self.model.get_family_weights()
        if weights is None:
            raise ValueError("Model has no family embeddings")
        return weights.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Model export (encoder-only)
    # ------------------------------------------------------------------
    #
    # These methods export only the encoder (latent extraction), not the
    # full multi-head model.  The encoder maps raw features to a fixed-
    # length latent vector; task heads can be trivially re-attached in
    # the target runtime.  This sidesteps TorchScript/ONNX limitations
    # with dict-returning forward methods.
    #
    # Supported encoding modes: hash (default), hash+explicit_vector,
    # embed.  rank_pool and transformer modes have variable-length
    # optional inputs that cannot be cleanly traced; use
    # torch.jit.script or export the full Python model instead.
    # ------------------------------------------------------------------

    def _compute_n_continuous(self) -> int:
        """Derive n_continuous as seen by the encoder from the model schema.

        This mirrors the computation in ``ResolveModel.__init__`` so that
        example inputs have the correct feature dimension.
        """
        schema = self.model.schema
        encoding = self.model.species_encoding
        n_coords = 2 if schema.has_coordinates else 0
        n_unknown = 0
        if schema.track_unknown_fraction:
            n_unknown += 1
        if schema.track_unknown_count:
            n_unknown += 1
        n_cat_embed = len(self.model._categorical_names) * self.model.categorical_embed_dim

        if encoding == "hash" and not self.model.uses_explicit_vector:
            return n_coords + len(schema.covariate_names) + self.model.hash_dim + n_unknown + n_cat_embed
        else:
            # embed, rank_pool, transformer, hash+explicit_vector
            return n_coords + len(schema.covariate_names) + n_unknown + n_cat_embed

    def _make_example_input(self, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
        """Create dummy tensors matching the encoder's expected input signature.

        Args:
            batch_size: Number of example rows.

        Returns:
            Tuple of tensors suitable for ``torch.jit.trace`` or
            ``torch.onnx.export`` of ``self.model.encoder``.

        Raises:
            ValueError: If the model's encoding mode cannot be traced.
        """
        schema = self.model.schema
        encoding = self.model.species_encoding
        device = self._device

        if encoding in ("rank_pool", "transformer"):
            raise ValueError(
                f"Export is not supported for species_encoding={encoding!r}. "
                "rank_pool and transformer encoders have variable-length optional "
                "inputs that cannot be cleanly traced. Use torch.jit.script or "
                "export the full Python model instead."
            )

        n_continuous = self._compute_n_continuous()

        if encoding == "hash" and not self.model.uses_explicit_vector:
            # PlotEncoder.forward(continuous, genus_ids, family_ids)
            continuous = torch.randn(batch_size, n_continuous, device=device)
            if self.model.encoder.has_taxonomy:
                genus_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                family_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                return (continuous, genus_ids, family_ids)
            return (continuous,)

        elif encoding == "hash" and self.model.uses_explicit_vector:
            # PlotEncoderSparse.forward(continuous, species_abundances, genus_ids, family_ids)
            n_species = schema.n_species_vocab
            continuous = torch.randn(batch_size, n_continuous, device=device)
            species_vector = torch.zeros(batch_size, n_species, device=device)
            if self.model.encoder.has_taxonomy:
                genus_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                family_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                return (continuous, species_vector, genus_ids, family_ids)
            return (continuous, species_vector)

        elif encoding == "embed":
            # PlotEncoderEmbed.forward(continuous, species_ids, genus_ids, family_ids)
            continuous = torch.randn(batch_size, n_continuous, device=device)
            species_ids = torch.zeros(batch_size, self.model.top_k_species, dtype=torch.long, device=device)
            if self.model.encoder.has_taxonomy:
                genus_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                family_ids = torch.zeros(batch_size, self.model.top_k, dtype=torch.long, device=device)
                return (continuous, species_ids, genus_ids, family_ids)
            return (continuous, species_ids)

        raise ValueError(f"Unknown species_encoding: {encoding!r}")

    def _get_export_names(self) -> tuple[list[str], list[str], dict[str, dict[int, str]]]:
        """Return (input_names, output_names, dynamic_axes) for ONNX export.

        Adapts to the model's encoding mode and taxonomy availability.
        """
        encoding = self.model.species_encoding
        has_taxonomy = self.model.encoder.has_taxonomy

        input_names: list[str] = ["continuous"]
        dynamic_axes: dict[str, dict[int, str]] = {"continuous": {0: "batch"}}

        if encoding == "hash" and self.model.uses_explicit_vector:
            input_names.append("species_vector")
            dynamic_axes["species_vector"] = {0: "batch"}
        elif encoding == "embed":
            input_names.append("species_ids")
            dynamic_axes["species_ids"] = {0: "batch"}

        if has_taxonomy:
            input_names.extend(["genus_ids", "family_ids"])
            dynamic_axes["genus_ids"] = {0: "batch"}
            dynamic_axes["family_ids"] = {0: "batch"}

        output_names = ["latent"]
        dynamic_axes["latent"] = {0: "batch"}

        return input_names, output_names, dynamic_axes

    @torch.no_grad()
    def export_torchscript(self, path: str | Path) -> None:
        """Export the encoder as TorchScript for portable deployment.

        Traces the encoder (latent extraction) only, not the full multi-head
        model.  The traced module maps raw features to the latent vector;
        task heads are simple linear layers that can be re-attached in the
        target runtime.

        Args:
            path: Output file path (e.g. ``"encoder.pt"``).

        Raises:
            ValueError: If the encoding mode cannot be traced
                (rank_pool, transformer).
        """
        self.model.eval()
        example = self._make_example_input()
        traced = torch.jit.trace(self.model.encoder, example)
        torch.jit.save(traced, str(path))

    @torch.no_grad()
    def export_onnx(self, path: str | Path, opset_version: int = 14) -> None:
        """Export the encoder as ONNX for cross-platform deployment.

        Exports the encoder (latent extraction) only, not the full multi-head
        model.  Dynamic batch axes are configured so the exported model
        accepts variable batch sizes at inference time.

        Args:
            path: Output file path (e.g. ``"encoder.onnx"``).
            opset_version: ONNX opset version (default 14).

        Raises:
            ValueError: If the encoding mode cannot be traced
                (rank_pool, transformer).
        """
        if opset_version < 11:
            raise ValueError(f"opset_version must be >= 11, got {opset_version}")

        self.model.eval()
        example = self._make_example_input()
        input_names, output_names, dynamic_axes = self._get_export_names()

        torch.onnx.export(
            self.model.encoder,
            example,
            str(path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    @torch.no_grad()
    def predict_batched(
        self,
        dataset: ResolveDataset,
        batch_size: int = 32768,
        output_space: str = "raw",
        confidence_threshold: float = 0.0,
    ) -> ResolvePredictions:
        """Predict on a dataset in batches, concatenating results.

        Useful for large datasets that don't fit in GPU memory at once.

        Args:
            dataset: ResolveDataset to predict on.
            batch_size: Number of samples per batch.
            output_space: "raw" or "transformed".
            confidence_threshold: Minimum confidence for predictions.

        Returns:
            ResolvePredictions with concatenated results from all batches.
        """
        all_predictions: dict[str, list[np.ndarray]] = {}
        all_confidence: dict[str, list[np.ndarray]] = {}

        for batch_result in self.predict_generator(
            dataset, batch_size=batch_size,
            output_space=output_space, confidence_threshold=confidence_threshold,
        ):
            for name, pred in batch_result.predictions.items():
                all_predictions.setdefault(name, []).append(pred)
            if batch_result.confidence:
                for name, conf in batch_result.confidence.items():
                    all_confidence.setdefault(name, []).append(conf)

        predictions = {k: np.concatenate(v) for k, v in all_predictions.items()}
        confidence = {k: np.concatenate(v) for k, v in all_confidence.items()} or None

        return ResolvePredictions(
            predictions=predictions,
            plot_ids=dataset.plot_ids,
            latent=None,
            confidence=confidence,
        )

    @torch.no_grad()
    def predict_generator(
        self,
        dataset: ResolveDataset,
        batch_size: int = 32768,
        output_space: str = "raw",
        confidence_threshold: float = 0.0,
    ) -> Generator[ResolvePredictions, None, None]:
        """Predict on a dataset in batches, yielding per-batch results.

        Memory-efficient streaming prediction for very large datasets.

        Args:
            dataset: ResolveDataset to predict on.
            batch_size: Number of samples per batch.
            output_space: "raw" or "transformed".
            confidence_threshold: Minimum confidence for predictions.

        Yields:
            ResolvePredictions for each batch.
        """
        if output_space not in ("raw", "transformed"):
            raise ValueError(f"output_space must be 'raw' or 'transformed', got {output_space!r}")

        # --- Prepare all data once ---
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()
        schema = self.model.schema
        encoding = self.model.species_encoding
        n_plots = len(dataset.plot_ids)

        # Encode species (full dataset)
        species_ids_all = None
        species_vector_all = None
        genus_all = None
        family_all = None
        pool_genus_ids_all = None
        pool_family_ids_all = None
        pool_weights_all = None
        pool_mask_all = None
        pool_has_cover_all = None
        unknown_fraction = np.zeros(n_plots, dtype=np.float32)

        if encoding == "embed":
            embedded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None: parts.append(coords)
            if covariates is not None: parts.append(covariates)
            if schema.track_unknown_fraction:
                parts.append(embedded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = embedded.unknown_fraction
            species_ids_all = embedded.species_ids
            if embedded.genus_ids is not None:
                genus_all = embedded.genus_ids
                family_all = embedded.family_ids

        elif encoding == "rank_pool":
            pool_encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None: parts.append(coords)
            if covariates is not None: parts.append(covariates)
            if schema.track_unknown_fraction:
                parts.append(pool_encoded.unknown_fraction.reshape(-1, 1))
            unknown_fraction = pool_encoded.unknown_fraction
            from resolve.encode.rank_pool import pad_rank_pool_encoded
            padded = pad_rank_pool_encoded(pool_encoded)
            species_ids_all = padded["species_ids"]
            pool_genus_ids_all = padded["genus_ids"]
            pool_family_ids_all = padded["family_ids"]
            pool_weights_all = padded["weights"]
            pool_mask_all = padded["mask"]
            pool_has_cover_all = padded["has_cover"]

        else:
            encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None: parts.append(coords)
            if covariates is not None: parts.append(covariates)
            if self.model.uses_explicit_vector:
                species_vector_all = encoded.species_vector
            else:
                parts.append(encoded.hash_embedding)
            if schema.track_unknown_fraction:
                parts.append(encoded.unknown_fraction.reshape(-1, 1))
            if schema.track_unknown_count and encoded.unknown_count is not None:
                parts.append(encoded.unknown_count.reshape(-1, 1).astype(np.float32))
            unknown_fraction = encoded.unknown_fraction
            if encoded.genus_ids is not None:
                genus_all = encoded.genus_ids
                family_all = encoded.family_ids

        continuous_all = np.hstack(parts) if parts else np.zeros((n_plots, 0), dtype=np.float32)
        continuous_all = self.scalers["continuous"].transform(continuous_all).astype(np.float32)

        # Categorical IDs
        categorical_ids_all = None
        if self.categorical_vocabs and schema.has_categoricals:
            cat_data = dataset.get_categoricals()
            if cat_data is not None:
                cat_arrays = []
                for cat_name in schema.categorical_names:
                    vocab = self.categorical_vocabs[cat_name]
                    cat_arrays.append(vocab.encode_array(cat_data[cat_name]))
                categorical_ids_all = np.stack(cat_arrays, axis=1)

        # --- Yield per-batch ---
        def _to_device(arr: np.ndarray | None, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
            if arr is None:
                return None
            return torch.from_numpy(arr).to(dtype).to(self._device)

        for start in range(0, n_plots, batch_size):
            end = min(start + batch_size, n_plots)
            sl = slice(start, end)

            continuous_t = _to_device(continuous_all[sl])
            species_ids_t = _to_device(species_ids_all[sl] if species_ids_all is not None else None, torch.long)
            species_vector_t = _to_device(species_vector_all[sl] if species_vector_all is not None else None)
            genus_t = _to_device(genus_all[sl] if genus_all is not None else None, torch.long)
            family_t = _to_device(family_all[sl] if family_all is not None else None, torch.long)
            pool_genus_t = _to_device(pool_genus_ids_all[sl] if pool_genus_ids_all is not None else None, torch.long)
            pool_family_t = _to_device(pool_family_ids_all[sl] if pool_family_ids_all is not None else None, torch.long)
            pool_weights_t = _to_device(pool_weights_all[sl] if pool_weights_all is not None else None)
            pool_mask_t = _to_device(pool_mask_all[sl] if pool_mask_all is not None else None)
            pool_has_cover_t = _to_device(pool_has_cover_all[sl] if pool_has_cover_all is not None else None)
            cat_ids_t = _to_device(categorical_ids_all[sl] if categorical_ids_all is not None else None, torch.long)

            # Forward
            if encoding == "rank_pool":
                preds_raw = self.model(
                    continuous_t, genus_ids=None, family_ids=None,
                    species_ids=species_ids_t, species_vector=None,
                    pool_genus_ids=pool_genus_t, pool_family_ids=pool_family_t,
                    pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                    pool_has_cover=pool_has_cover_t,
                    categorical_ids=cat_ids_t,
                )
            else:
                preds_raw = self.model(
                    continuous_t, genus_t, family_t,
                    species_ids=species_ids_t, species_vector=species_vector_t,
                    categorical_ids=cat_ids_t,
                )

            # Post-process using shared helper
            batch_unk = unknown_fraction[sl]
            predictions, confidence = postprocess_predictions(
                preds_raw, self.model.target_configs, self.scalers,
                batch_unk, output_space, confidence_threshold,
            )

            yield ResolvePredictions(
                predictions=predictions,
                plot_ids=dataset.plot_ids[sl],
                latent=None,
                confidence=confidence,
            )
