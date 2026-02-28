"""Persistence mixin for Trainer.

Handles saving trained models to disk and loading them back for inference.
The save/load pair captures all state needed to reconstruct a Trainer for
prediction: model weights, encoder vocabularies, and feature scalers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from resolve.encode.embedding import EmbeddingEncoder
from resolve.encode.species import SpeciesEncoder
from resolve.model.resolve import ResolveModel

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class PersistenceMixin:
    """Mixin providing model save/load methods for Trainer."""

    def save(self: Trainer, path: Union[str, Path]) -> None:
        """Save model, encoder, and scalers to file.

        Raises:
            RuntimeError: If trainer has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot save: model has not been built yet. "
                "Call trainer.fit() before trainer.save()."
            )
        # Check appropriate encoder based on mode
        if self.species_encoding == "hash" and self._species_encoder is None:
            raise RuntimeError(
                "Cannot save: species encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )
        if self.species_encoding == "embed" and self._embedding_encoder is None:
            raise RuntimeError(
                "Cannot save: embedding encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )
        if self.species_encoding in ("rank_pool", "transformer") and self._rank_pool_encoder is None:
            raise RuntimeError(
                "Cannot save: rank_pool encoder not initialized. "
                "Call trainer.fit() before trainer.save()."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "schema": self.model.schema,
            "target_configs": self.model.target_configs,
            "species_encoding": self.species_encoding,
            "hash_dim": self.model.hash_dim,
            "top_k": self.model.top_k,
            "hidden_dims": self.model.hidden_dims,
            "genus_emb_dim": self.model.genus_emb_dim,
            "family_emb_dim": self.model.family_emb_dim,
            "dropout": self.model.dropout,
            "scalers": self._scalers,
            "track_unknown_fraction": self.track_unknown_fraction,
            "uses_explicit_vector": self.model.uses_explicit_vector,
            "head_hidden_dims": self.head_hidden_dims,
            "categorical_vocabs": self._categorical_vocabs if hasattr(self, "_categorical_vocabs") else {},
            "categorical_embed_dim": self.categorical_embed_dim if hasattr(self, "categorical_embed_dim") else 8,
            "species_embed_dim": self.model.species_embed_dim,
        }

        # Save encoder-specific state
        if self.species_encoding == "hash" and self._species_encoder:
            state["vocab"] = self._species_encoder.vocab
            state["species_aggregation"] = self._species_encoder.aggregation
            state["species_selection"] = self._species_encoder.selection
            state["species_representation"] = self._species_encoder.representation
            state["species_normalization"] = self._species_encoder.normalization
            state["track_unknown_count"] = self._species_encoder.track_unknown_count
            state["species_vocab"] = self._species_encoder._species_vocab
            state["species_to_idx"] = self._species_encoder._species_to_idx
            # Save normalizer if present
            if self._species_encoder.normalizer is not None:
                state["normalizer"] = self._species_encoder.normalizer
        elif self.species_encoding == "embed" and self._embedding_encoder:
            state["species_vocab_obj"] = self._embedding_encoder._species_vocab
            state["taxonomy_vocab_obj"] = self._embedding_encoder._taxonomy_vocab
            state["species_aggregation"] = self._embedding_encoder.aggregation
            state["species_selection"] = self._embedding_encoder.selection
            state["top_k_species"] = self._embedding_encoder.top_k_species
            state["top_k_taxonomy"] = self._embedding_encoder.top_k_taxonomy
            if self._embedding_encoder.normalizer is not None:
                state["normalizer"] = self._embedding_encoder.normalizer
        elif self.species_encoding in ("rank_pool", "transformer") and self._rank_pool_encoder:
            state["species_vocab_obj"] = self._rank_pool_encoder._species_vocab
            state["taxonomy_vocab_obj"] = self._rank_pool_encoder._taxonomy_vocab
            state["species_to_genus"] = self._rank_pool_encoder._species_to_genus
            state["species_to_family"] = self._rank_pool_encoder._species_to_family
            state["species_normalization"] = self._rank_pool_encoder.weighting
            state["min_species_frequency"] = self._rank_pool_encoder.min_species_frequency
            if self._rank_pool_encoder.normalizer is not None:
                state["normalizer"] = self._rank_pool_encoder.normalizer
            # Save transformer-specific params
            if self.species_encoding == "transformer":
                state["n_attention_layers"] = self.n_attention_layers
                state["n_heads"] = self.n_heads
                state["transformer_ff_dim"] = self.transformer_ff_dim
                state["transformer_pooling"] = self.transformer_pooling
                state["transformer_dropout"] = self.transformer_dropout

        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "auto",
    ) -> tuple[ResolveModel, Union[SpeciesEncoder, EmbeddingEncoder], dict, dict]:
        """Load model from checkpoint.

        Dispatches encoder creation based on species_encoding saved in checkpoint:
        - "hash": creates SpeciesEncoder
        - "embed": creates EmbeddingEncoder with restored vocabs
        - "rank_pool"/"transformer": creates RankPoolEncoder with restored vocabs

        Returns:
            (model, species_encoder, scalers, categorical_vocabs)

        Security Note:
            This method uses pickle deserialization (weights_only=False) to load
            sklearn scalers and encoder state. Only load model files from trusted sources.
        """
        # Note: weights_only=False is required for sklearn scalers and encoder state.
        # Only load model files from trusted sources.
        state = torch.load(path, map_location="cpu", weights_only=False)

        species_encoding = state.get("species_encoding", "hash")
        track_unknown_count = state.get("track_unknown_count", False)
        uses_explicit_vector = state.get("uses_explicit_vector", False)

        # Restore categorical info into schema from saved state.
        # Use vocab sizes from the saved schema (which matches model weights)
        # rather than recomputing from vocab.n_categories, because the vocab
        # is built from the training split only and may have fewer categories
        # than the full dataset used to create the model.
        schema = state["schema"]
        categorical_vocabs = state.get("categorical_vocabs", {})
        if categorical_vocabs:
            from dataclasses import replace as _replace
            categorical_embed_dim = state.get("categorical_embed_dim", 8)
            updates = {
                "categorical_names": list(categorical_vocabs.keys()),
                "categorical_embed_dim": categorical_embed_dim,
            }
            # Only set vocab sizes if the schema doesn't already have them
            if not schema.categorical_vocab_sizes:
                updates["categorical_vocab_sizes"] = {
                    name: v.n_categories for name, v in categorical_vocabs.items()
                }
            schema = _replace(schema, **updates)

        # Infer species_embed_dim from saved weights if not stored explicitly
        species_embed_dim = state.get("species_embed_dim", 32)
        if species_embed_dim == 32:
            # Check if saved weights indicate a different embed dim
            sd = state["model_state_dict"]
            for key in ("encoder.species_embedding.weight",):
                if key in sd:
                    species_embed_dim = sd[key].shape[1]
                    break

        model = ResolveModel(
            schema=schema,
            targets=state["target_configs"],
            species_encoding=species_encoding,
            species_embed_dim=species_embed_dim,
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
            top_k_species=state.get("top_k_species", 10),
            hidden_dims=state.get("hidden_dims"),
            genus_emb_dim=state.get("genus_emb_dim", 8),
            family_emb_dim=state.get("family_emb_dim", 8),
            dropout=state.get("dropout", 0.3),
            track_unknown_count=track_unknown_count,
            uses_explicit_vector=uses_explicit_vector,
            n_attention_layers=state.get("n_attention_layers", 0),
            n_heads=state.get("n_heads", 4),
            transformer_ff_dim=state.get("transformer_ff_dim", 256),
            transformer_pooling=state.get("transformer_pooling", "attention"),
            transformer_dropout=state.get("transformer_dropout", 0.1),
            head_hidden_dims=state.get("head_hidden_dims"),
        )
        model.load_state_dict(state["model_state_dict"])

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        model.to(dev)

        # Dispatch encoder creation based on species_encoding
        if species_encoding == "embed":
            encoder = EmbeddingEncoder(
                top_k_species=state.get("top_k_species", 10),
                top_k_taxonomy=state.get("top_k_taxonomy", 3),
                aggregation=state.get("species_aggregation", "abundance"),
                selection=state.get("species_selection", "top"),
            )
            encoder._species_vocab = state.get("species_vocab_obj")
            encoder._taxonomy_vocab = state.get("taxonomy_vocab_obj")
            encoder._fitted = True
        elif species_encoding in ("rank_pool", "transformer"):
            from resolve.encode.rank_pool import RankPoolEncoder
            encoder = RankPoolEncoder(
                weighting=state.get("species_normalization", "log1p"),
                min_species_frequency=state.get("min_species_frequency", 1),
            )
            encoder._species_vocab = state.get("species_vocab_obj")
            encoder._taxonomy_vocab = state.get("taxonomy_vocab_obj")
            encoder._species_to_genus = state.get("species_to_genus", {})
            encoder._species_to_family = state.get("species_to_family", {})
            encoder._fitted = True
        else:
            # Hash mode (default)
            encoder = SpeciesEncoder(
                hash_dim=state["hash_dim"],
                top_k=state["top_k"],
                aggregation=state.get("species_aggregation", "abundance"),
                normalization=state.get("species_normalization", "norm"),
                track_unknown_count=track_unknown_count,
                selection=state.get("species_selection", "top"),
                representation=state.get("species_representation", "abundance"),
            )
            if state.get("vocab") is not None:
                encoder._vocab = state["vocab"]
            encoder._species_vocab = state.get("species_vocab", set())
            encoder._species_to_idx = state.get("species_to_idx", {})
            encoder._fitted = True

        # Restore normalizer for all modes (if saved in checkpoint)
        normalizer = state.get("normalizer")
        if normalizer is not None:
            encoder.normalizer = normalizer

        return model, encoder, state["scalers"], categorical_vocabs
