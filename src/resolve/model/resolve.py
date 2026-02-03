"""ResolveModel: full model composing encoder and task heads."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from resolve.data.dataset import ResolveSchema
from resolve.data.roles import TargetConfig
from resolve.model.encoder import PlotEncoder, PlotEncoderEmbed, PlotEncoderSparse
from resolve.model.head import TaskHead


class ResolveModel(nn.Module):
    """
    Full RESOLVE model: shared encoder with multiple task heads.

    The encoder processes:
        - Continuous features (coords + covariates + hash embedding)
        - Taxonomic IDs (genus, family) via learned embeddings

    Each target gets its own prediction head.

    Species encoding modes:
        - "hash" (default): Feature hashing for species, learnable embeddings for taxonomy
        - "embed": Learnable embeddings for species AND taxonomy (requires vocab in schema)

    When uses_explicit_vector=True with hash encoding, species are passed as an explicit
    (n_plots, n_species) vector instead of being hashed. This enables "all" and
    "presence_absence" selection modes.
    """

    def __init__(
        self,
        schema: ResolveSchema,
        targets: dict[str, TargetConfig],
        species_encoding: Literal["hash", "embed"] = "hash",
        hash_dim: int = 32,
        species_embed_dim: int = 32,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        top_k: int = 3,
        top_k_species: int = 10,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        track_unknown_count: bool = None,  # Deprecated: read from schema
        uses_explicit_vector: bool = False,  # For hash mode with all/presence_absence selection
    ):
        super().__init__()

        if species_encoding not in ("hash", "embed"):
            raise ValueError(f"species_encoding must be 'hash' or 'embed', got {species_encoding!r}")

        self._schema = schema
        self._targets = targets
        self.species_encoding = species_encoding
        self.uses_explicit_vector = uses_explicit_vector
        self.hash_dim = hash_dim
        self.species_embed_dim = species_embed_dim
        self.top_k = top_k
        self.top_k_species = top_k_species
        self.hidden_dims = hidden_dims if hidden_dims is not None else [2048, 1024, 512, 256, 128, 64]
        self.genus_emb_dim = genus_emb_dim
        self.family_emb_dim = family_emb_dim
        self.dropout = dropout

        # Number of base continuous features (coords + covariates)
        n_coords = 2 if schema.has_coordinates else 0
        n_unknown_features = 0
        if schema.track_unknown_fraction:
            n_unknown_features += 1
        if schema.track_unknown_count:
            n_unknown_features += 1

        if species_encoding == "hash" and not uses_explicit_vector:
            # Hash mode: continuous includes hash_dim
            n_continuous = n_coords + len(schema.covariate_names) + hash_dim + n_unknown_features

            # Build hash-based encoder
            self.encoder = PlotEncoder(
                n_continuous=n_continuous,
                n_genera=schema.n_genera + 1 if schema.has_taxonomy else 0,
                n_families=schema.n_families + 1 if schema.has_taxonomy else 0,
                genus_emb_dim=genus_emb_dim,
                family_emb_dim=family_emb_dim,
                top_k=top_k,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        elif species_encoding == "hash" and uses_explicit_vector:
            # Explicit vector mode (all/presence_absence): continuous does NOT include species info
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features

            # Validate vocab sizes are present
            if schema.n_species_vocab == 0:
                raise ValueError(
                    "uses_explicit_vector=True requires n_species_vocab > 0 in schema. "
                    "Use SpeciesEncoder with selection='all' or 'presence_absence'."
                )

            # Build sparse encoder (for explicit species vector input)
            self.encoder = PlotEncoderSparse(
                n_continuous=n_continuous,
                n_species=schema.n_species_vocab,
                species_embed_dim=species_embed_dim,
                n_genera=schema.n_genera + 1 if schema.has_taxonomy else 0,
                n_families=schema.n_families + 1 if schema.has_taxonomy else 0,
                genus_emb_dim=genus_emb_dim,
                family_emb_dim=family_emb_dim,
                top_k=top_k,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        else:  # embed mode
            # Embed mode: continuous does NOT include hash embedding
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features

            # Validate vocab sizes are present
            if schema.n_species_vocab == 0:
                raise ValueError(
                    "species_encoding='embed' requires n_species_vocab > 0 in schema. "
                    "Use EmbeddingEncoder to build vocab and set schema.n_species_vocab."
                )

            # Build embedding-based encoder
            self.encoder = PlotEncoderEmbed(
                n_continuous=n_continuous,
                n_species=schema.n_species_vocab,
                n_genera=schema.n_genera_vocab if schema.n_genera_vocab > 0 else schema.n_genera + 1,
                n_families=schema.n_families_vocab if schema.n_families_vocab > 0 else schema.n_families + 1,
                species_embed_dim=species_embed_dim,
                genus_emb_dim=genus_emb_dim,
                family_emb_dim=family_emb_dim,
                top_k_species=top_k_species,
                top_k_taxonomy=top_k,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        # Build task heads
        self.heads = nn.ModuleDict()
        for name, cfg in targets.items():
            self.heads[name] = TaskHead(
                latent_dim=self.encoder.latent_dim,
                task=cfg.task,
                num_classes=cfg.num_classes,
                transform=cfg.transform,
            )

    @property
    def schema(self) -> ResolveSchema:
        return self._schema

    @property
    def target_configs(self) -> dict[str, TargetConfig]:
        return self._targets

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    def forward(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for all targets.

        Args:
            continuous: (batch, n_continuous)
            genus_ids: (batch, top_k) optional, used in all modes with taxonomy
            family_ids: (batch, top_k) optional, used in all modes with taxonomy
            species_ids: (batch, top_k_species) optional, only for embed mode
            species_vector: (batch, n_species) optional, for all/presence_absence selection

        Returns:
            Dict mapping target name to predictions
        """
        if self.species_encoding == "embed":
            latent = self.encoder(continuous, species_ids, genus_ids, family_ids)
        elif self.uses_explicit_vector:
            latent = self.encoder(continuous, species_vector, genus_ids, family_ids)
        else:  # hash
            latent = self.encoder(continuous, genus_ids, family_ids)
        return {name: head(latent) for name, head in self.heads.items()}

    def forward_single(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        target: str = None,
    ) -> torch.Tensor:
        """Forward pass for a single target."""
        if self.species_encoding == "embed":
            latent = self.encoder(continuous, species_ids, genus_ids, family_ids)
        elif self.uses_explicit_vector:
            latent = self.encoder(continuous, species_vector, genus_ids, family_ids)
        else:  # hash
            latent = self.encoder(continuous, genus_ids, family_ids)
        return self.heads[target](latent)

    def get_latent(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent representation without task heads."""
        if self.species_encoding == "embed":
            return self.encoder(continuous, species_ids, genus_ids, family_ids)
        elif self.uses_explicit_vector:
            return self.encoder(continuous, species_vector, genus_ids, family_ids)
        else:  # hash
            return self.encoder(continuous, genus_ids, family_ids)
