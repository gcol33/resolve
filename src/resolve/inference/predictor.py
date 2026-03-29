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
            # Trace with fixed max_species padding (default 100)
            max_species = getattr(self, '_export_max_species', 100)
            n_continuous = self._compute_n_continuous()
            n_species = schema.n_species_vocab or 100
            n_genera = schema.n_genera_vocab or 20
            n_families = schema.n_families_vocab or 10
            continuous = torch.randn(batch_size, n_continuous, device=device)
            species_ids = torch.randint(0, n_species, (batch_size, max_species), device=device)
            genus_ids = torch.randint(0, n_genera, (batch_size, max_species), device=device)
            family_ids = torch.randint(0, n_families, (batch_size, max_species), device=device)
            weights = torch.rand(batch_size, max_species, device=device)
            mask = torch.ones(batch_size, max_species, dtype=torch.bool, device=device)
            has_cover = torch.ones(batch_size, device=device)
            return (continuous, species_ids, genus_ids, family_ids, weights, mask, has_cover)

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

        if encoding in ("rank_pool", "transformer"):
            for name in ["species_ids", "genus_ids", "family_ids", "weights", "mask", "has_cover"]:
                input_names.append(name)
                dynamic_axes[name] = {0: "batch"}
        elif encoding == "hash" and self.model.uses_explicit_vector:
            input_names.append("species_vector")
            dynamic_axes["species_vector"] = {0: "batch"}
        elif encoding == "embed":
            input_names.append("species_ids")
            dynamic_axes["species_ids"] = {0: "batch"}

        if encoding not in ("rank_pool", "transformer") and has_taxonomy:
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

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _prepare_forward_tensors(
        self,
        dataset: ResolveDataset,
        n_samples: int | None = None,
    ) -> tuple[
        torch.Tensor,                       # continuous_t
        torch.Tensor | None,                # genus_t
        torch.Tensor | None,                # family_t
        torch.Tensor | None,                # species_ids_t
        torch.Tensor | None,                # species_vector_t
        torch.Tensor | None,                # pool_genus_ids_t
        torch.Tensor | None,                # pool_family_ids_t
        torch.Tensor | None,                # pool_weights_t
        torch.Tensor | None,                # pool_mask_t
        torch.Tensor | None,                # pool_has_cover_t
        torch.Tensor | None,                # categorical_ids_t
        list[str],                           # feature_names for the continuous block
    ]:
        """Prepare tensors for a forward pass, optionally subsampled.

        Returns all tensors needed by the model's forward method plus a list
        of human-readable feature names for each column of the continuous
        tensor (used by feature-importance methods).
        """
        n_total = len(dataset.plot_ids)
        if n_samples is not None and n_samples < n_total:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_total, size=n_samples, replace=False)
            indices.sort()
        else:
            indices = np.arange(n_total)

        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()
        schema = self.model.schema
        encoding = self.model.species_encoding

        # Build feature name list in the same order as columns are stacked
        feature_names: list[str] = []

        species_ids_t = None
        species_vector_t = None
        genus_t = None
        family_t = None
        pool_genus_ids_t = None
        pool_family_ids_t = None
        pool_weights_t = None
        pool_mask_t = None
        pool_has_cover_t = None

        if encoding == "embed":
            embedded = self.species_encoder.transform(dataset)
            parts: list[np.ndarray] = []
            if coords is not None:
                parts.append(coords[indices])
                feature_names.extend(["lat", "lon"])
            if covariates is not None:
                parts.append(covariates[indices])
                feature_names.extend(schema.covariate_names)
            if schema.track_unknown_fraction:
                parts.append(embedded.unknown_fraction[indices].reshape(-1, 1))
                feature_names.append("unknown_fraction")
            species_ids_t = torch.from_numpy(embedded.species_ids[indices]).to(self._device)
            if embedded.genus_ids is not None:
                genus_t = torch.from_numpy(embedded.genus_ids[indices]).to(self._device)
                family_t = torch.from_numpy(embedded.family_ids[indices]).to(self._device)

        elif encoding == "rank_pool":
            pool_encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords[indices])
                feature_names.extend(["lat", "lon"])
            if covariates is not None:
                parts.append(covariates[indices])
                feature_names.extend(schema.covariate_names)
            if schema.track_unknown_fraction:
                parts.append(pool_encoded.unknown_fraction[indices].reshape(-1, 1))
                feature_names.append("unknown_fraction")
            from resolve.encode.rank_pool import pad_rank_pool_encoded
            padded = pad_rank_pool_encoded(pool_encoded)
            species_ids_t = torch.from_numpy(padded["species_ids"][indices]).long().to(self._device)
            pool_genus_ids_t = torch.from_numpy(padded["genus_ids"][indices]).long().to(self._device)
            pool_family_ids_t = torch.from_numpy(padded["family_ids"][indices]).long().to(self._device)
            pool_weights_t = torch.from_numpy(padded["weights"][indices]).to(self._device)
            pool_mask_t = torch.from_numpy(padded["mask"][indices]).to(self._device)
            pool_has_cover_t = torch.from_numpy(padded["has_cover"][indices]).to(self._device)

        else:
            encoded = self.species_encoder.transform(dataset)
            parts = []
            if coords is not None:
                parts.append(coords[indices])
                feature_names.extend(["lat", "lon"])
            if covariates is not None:
                parts.append(covariates[indices])
                feature_names.extend(schema.covariate_names)
            if self.model.uses_explicit_vector:
                species_vector_t = torch.from_numpy(encoded.species_vector[indices]).to(self._device)
            else:
                parts.append(encoded.hash_embedding[indices])
                feature_names.extend([f"hash_{i}" for i in range(encoded.hash_embedding.shape[1])])
            if schema.track_unknown_fraction:
                parts.append(encoded.unknown_fraction[indices].reshape(-1, 1))
                feature_names.append("unknown_fraction")
            if schema.track_unknown_count and encoded.unknown_count is not None:
                parts.append(encoded.unknown_count[indices].reshape(-1, 1).astype(np.float32))
                feature_names.append("unknown_count")
            if encoded.genus_ids is not None:
                genus_t = torch.from_numpy(encoded.genus_ids[indices]).to(self._device)
                family_t = torch.from_numpy(encoded.family_ids[indices]).to(self._device)

        continuous = np.hstack(parts) if parts else np.zeros((len(indices), 0), dtype=np.float32)
        continuous = self.scalers["continuous"].transform(continuous).astype(np.float32)
        continuous_t = torch.from_numpy(continuous).to(self._device)

        # Categorical IDs
        categorical_ids_t = None
        if self.categorical_vocabs and schema.has_categoricals:
            cat_data = dataset.get_categoricals()
            if cat_data is not None:
                cat_arrays = []
                for cat_name in schema.categorical_names:
                    vocab = self.categorical_vocabs[cat_name]
                    arr = vocab.encode_array(cat_data[cat_name])
                    cat_arrays.append(arr[indices])
                categorical_ids_t = torch.from_numpy(
                    np.stack(cat_arrays, axis=1)
                ).to(self._device)

        return (
            continuous_t, genus_t, family_t, species_ids_t, species_vector_t,
            pool_genus_ids_t, pool_family_ids_t, pool_weights_t, pool_mask_t,
            pool_has_cover_t, categorical_ids_t, feature_names,
        )

    def _forward_for_target(
        self,
        target: str,
        continuous_t: torch.Tensor,
        genus_t: torch.Tensor | None,
        family_t: torch.Tensor | None,
        species_ids_t: torch.Tensor | None,
        species_vector_t: torch.Tensor | None,
        pool_genus_ids_t: torch.Tensor | None,
        pool_family_ids_t: torch.Tensor | None,
        pool_weights_t: torch.Tensor | None,
        pool_mask_t: torch.Tensor | None,
        pool_has_cover_t: torch.Tensor | None,
        categorical_ids_t: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run forward pass and return the raw output for a single target."""
        encoding = self.model.species_encoding
        if encoding == "rank_pool":
            preds = self.model(
                continuous_t, genus_ids=None, family_ids=None,
                species_ids=species_ids_t, species_vector=None,
                pool_genus_ids=pool_genus_ids_t, pool_family_ids=pool_family_ids_t,
                pool_weights=pool_weights_t, pool_mask=pool_mask_t,
                pool_has_cover=pool_has_cover_t,
                categorical_ids=categorical_ids_t,
            )
        else:
            preds = self.model(
                continuous_t, genus_t, family_t,
                species_ids=species_ids_t, species_vector=species_vector_t,
                categorical_ids=categorical_ids_t,
            )
        return preds[target]

    def compute_feature_importance(
        self,
        dataset: ResolveDataset,
        target: str,
        n_samples: int = 1000,
        method: str = "gradient",
    ) -> dict[str, float]:
        """Compute feature importance for continuous input features.

        Two methods are available:

        * ``"gradient"``: mean absolute gradient of the target loss w.r.t.
          each continuous input feature.  Fast (single forward + backward),
          but only captures first-order sensitivity.
        * ``"permutation"``: drop in metric when each feature column is
          randomly shuffled.  Model-agnostic and captures non-linear effects,
          but requires one forward pass per feature.

        Feature importance is computed over the *continuous* input block
        (coordinates, covariates, hash embedding, unknown fraction, etc.).
        Discrete inputs (genus/family IDs, species IDs, categorical IDs) are
        held fixed and not attributed.

        Args:
            dataset: ResolveDataset to draw samples from.
            target: Name of the target head to attribute.
            n_samples: Number of samples to use (randomly sub-sampled if
                the dataset is larger).
            method: ``"gradient"`` or ``"permutation"``.

        Returns:
            Dictionary mapping feature name to importance score (always
            non-negative; higher means more important).

        Raises:
            ValueError: If *target* is not a valid target name, or *method*
                is not one of the supported methods.
        """
        if target not in self.model.target_configs:
            valid = list(self.model.target_configs.keys())
            raise ValueError(f"Unknown target {target!r}, must be one of {valid}")
        if method not in ("gradient", "permutation"):
            raise ValueError(f"method must be 'gradient' or 'permutation', got {method!r}")

        (
            continuous_t, genus_t, family_t, species_ids_t, species_vector_t,
            pool_genus_ids_t, pool_family_ids_t, pool_weights_t, pool_mask_t,
            pool_has_cover_t, categorical_ids_t, feature_names,
        ) = self._prepare_forward_tensors(dataset, n_samples)

        if continuous_t.shape[1] == 0:
            return {}

        if method == "gradient":
            return self._gradient_importance(
                target, continuous_t, genus_t, family_t, species_ids_t,
                species_vector_t, pool_genus_ids_t, pool_family_ids_t,
                pool_weights_t, pool_mask_t, pool_has_cover_t,
                categorical_ids_t, feature_names,
            )
        else:
            return self._permutation_importance(
                target, continuous_t, genus_t, family_t, species_ids_t,
                species_vector_t, pool_genus_ids_t, pool_family_ids_t,
                pool_weights_t, pool_mask_t, pool_has_cover_t,
                categorical_ids_t, feature_names,
            )

    def _gradient_importance(
        self,
        target: str,
        continuous_t: torch.Tensor,
        genus_t: torch.Tensor | None,
        family_t: torch.Tensor | None,
        species_ids_t: torch.Tensor | None,
        species_vector_t: torch.Tensor | None,
        pool_genus_ids_t: torch.Tensor | None,
        pool_family_ids_t: torch.Tensor | None,
        pool_weights_t: torch.Tensor | None,
        pool_mask_t: torch.Tensor | None,
        pool_has_cover_t: torch.Tensor | None,
        categorical_ids_t: torch.Tensor | None,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Gradient-based feature importance.

        Enables gradients on the continuous input, runs a forward pass,
        computes a scalar loss (MSE for regression, cross-entropy surrogate
        for classification), then backpropagates.  Importance per feature is
        the mean absolute gradient across samples.
        """
        cfg = self.model.target_configs[target]

        # Detach + clone continuous so we can enable gradients
        cont = continuous_t.detach().clone().requires_grad_(True)

        # Forward pass (model must be in eval mode but we need gradients)
        self.model.eval()
        output = self._forward_for_target(
            target, cont, genus_t, family_t, species_ids_t, species_vector_t,
            pool_genus_ids_t, pool_family_ids_t, pool_weights_t, pool_mask_t,
            pool_has_cover_t, categorical_ids_t,
        )

        # Compute a scalar loss to backpropagate
        if cfg.task == "regression":
            # Use sum of squared outputs as a proxy (no targets needed --
            # we want sensitivity of the output to each input).
            loss = (output ** 2).sum()
        else:
            # For classification, use the log-sum-exp (total logit magnitude)
            # as a differentiable scalar that captures sensitivity.
            loss = output.logsumexp(dim=-1).sum()

        loss.backward()

        # Importance = mean |grad| per feature across samples
        grad = cont.grad  # (n_samples, n_features)
        importance = grad.abs().mean(dim=0).cpu().numpy()

        result = {}
        for i, name in enumerate(feature_names):
            result[name] = float(importance[i])
        return result

    def _permutation_importance(
        self,
        target: str,
        continuous_t: torch.Tensor,
        genus_t: torch.Tensor | None,
        family_t: torch.Tensor | None,
        species_ids_t: torch.Tensor | None,
        species_vector_t: torch.Tensor | None,
        pool_genus_ids_t: torch.Tensor | None,
        pool_family_ids_t: torch.Tensor | None,
        pool_weights_t: torch.Tensor | None,
        pool_mask_t: torch.Tensor | None,
        pool_has_cover_t: torch.Tensor | None,
        categorical_ids_t: torch.Tensor | None,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Permutation-based feature importance.

        Computes a baseline metric on the unperturbed data, then for each
        feature shuffles that column and measures the metric drop.

        Uses MAE for regression targets and accuracy for classification.
        Importance is ``baseline_metric - shuffled_metric`` (positive means
        shuffling hurt performance).
        """
        cfg = self.model.target_configs[target]
        n_features = continuous_t.shape[1]

        # We need ground-truth targets -- but the user hasn't provided them.
        # Use the model's own predictions as a pseudo-baseline: importance
        # then measures self-consistency degradation under permutation.
        # This is the standard "prediction-stability" variant of permutation
        # importance (used when ground truth is unavailable at inference time).
        with torch.no_grad():
            baseline_output = self._forward_for_target(
                target, continuous_t, genus_t, family_t, species_ids_t,
                species_vector_t, pool_genus_ids_t, pool_family_ids_t,
                pool_weights_t, pool_mask_t, pool_has_cover_t,
                categorical_ids_t,
            )
            if cfg.task == "regression":
                baseline_pred = baseline_output.cpu().numpy().flatten()
            else:
                baseline_pred = baseline_output.argmax(dim=-1).cpu().numpy()

        rng = np.random.default_rng(42)
        result: dict[str, float] = {}

        for col_idx in range(n_features):
            # Shuffle one column
            shuffled = continuous_t.clone()
            perm = torch.from_numpy(
                rng.permutation(shuffled.shape[0])
            ).to(self._device)
            shuffled[:, col_idx] = shuffled[perm, col_idx]

            with torch.no_grad():
                shuffled_output = self._forward_for_target(
                    target, shuffled, genus_t, family_t, species_ids_t,
                    species_vector_t, pool_genus_ids_t, pool_family_ids_t,
                    pool_weights_t, pool_mask_t, pool_has_cover_t,
                    categorical_ids_t,
                )

            if cfg.task == "regression":
                shuffled_pred = shuffled_output.cpu().numpy().flatten()
                # Importance = mean absolute deviation from baseline prediction
                importance = float(np.abs(baseline_pred - shuffled_pred).mean())
            else:
                shuffled_pred = shuffled_output.argmax(dim=-1).cpu().numpy()
                # Importance = fraction of predictions that changed
                importance = float((baseline_pred != shuffled_pred).mean())

            name = feature_names[col_idx] if col_idx < len(feature_names) else f"feature_{col_idx}"
            result[name] = importance

        return result
