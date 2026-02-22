"""ResolveModel: full model composing encoder and task heads."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from resolve.data.dataset import ResolveSchema
from resolve.data.roles import TargetConfig
from resolve.model.encoder import PlotEncoder, PlotEncoderEmbed, PlotEncoderSparse, PlotEncoderRankPool, PlotEncoderTransformer
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
        - "rank_pool": Rank-pool encoding with shared additive hierarchical embeddings
                       (species + genus + family), weighted mean pool across all species,
                       with optional cover dropout and has_cover flag
        - "transformer": Transformer encoder over species tokens with attention pooling.
                         Reuses rank_pool data pipeline. Supports v4 (attention pool only),
                         v5 (self-attention + attention pool), v6 (masked pretraining).

    When uses_explicit_vector=True with hash encoding, species are passed as an explicit
    (n_plots, n_species) vector instead of being hashed. This enables "all" and
    "presence_absence" selection modes.
    """

    def __init__(
        self,
        schema: ResolveSchema,
        targets: dict[str, TargetConfig],
        species_encoding: Literal["hash", "embed", "rank_pool", "transformer"] = "hash",
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
        cover_dropout: float = 0.0,  # For rank_pool/transformer mode: randomly drop cover info
        # Transformer-specific params
        n_attention_layers: int = 0,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_pooling: str = "attention",
        transformer_dropout: float = 0.1,
        # Classification head
        head_hidden_dims: Optional[list[int]] = None,
    ):
        super().__init__()

        if species_encoding not in ("hash", "embed", "rank_pool", "transformer"):
            raise ValueError(f"species_encoding must be 'hash', 'embed', 'rank_pool', or 'transformer', got {species_encoding!r}")

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
        elif species_encoding == "embed":
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
        elif species_encoding == "transformer":
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features

            if schema.n_species_vocab == 0:
                raise ValueError(
                    "species_encoding='transformer' requires n_species_vocab > 0 in schema. "
                    "Use RankPoolEncoder to build vocab and set schema.n_species_vocab."
                )

            self.encoder = PlotEncoderTransformer(
                n_continuous=n_continuous,
                n_species=schema.n_species_vocab,
                n_genera=schema.n_genera_vocab if schema.n_genera_vocab > 0 else 0,
                n_families=schema.n_families_vocab if schema.n_families_vocab > 0 else 0,
                d_model=species_embed_dim,
                n_heads=n_heads,
                n_attention_layers=n_attention_layers,
                transformer_ff_dim=transformer_ff_dim,
                transformer_pooling=transformer_pooling,
                transformer_dropout=transformer_dropout,
                hidden_dims=hidden_dims,
                dropout=dropout,
                cover_dropout=cover_dropout,
            )
        else:  # rank_pool mode
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features

            if schema.n_species_vocab == 0:
                raise ValueError(
                    "species_encoding='rank_pool' requires n_species_vocab > 0 in schema. "
                    "Use RankPoolEncoder to build vocab and set schema.n_species_vocab."
                )

            self.encoder = PlotEncoderRankPool(
                n_continuous=n_continuous,
                n_species=schema.n_species_vocab,
                n_genera=schema.n_genera_vocab if schema.n_genera_vocab > 0 else 0,
                n_families=schema.n_families_vocab if schema.n_families_vocab > 0 else 0,
                species_embed_dim=species_embed_dim,
                genus_embed_dim=genus_emb_dim,
                family_embed_dim=family_emb_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                cover_dropout=cover_dropout,
            )

        # Build task heads
        self.head_hidden_dims = head_hidden_dims
        self.heads = nn.ModuleDict()
        for name, cfg in targets.items():
            self.heads[name] = TaskHead(
                latent_dim=self.encoder.latent_dim,
                task=cfg.task,
                num_classes=cfg.num_classes,
                transform=cfg.transform,
                head_hidden_dims=head_hidden_dims if cfg.task == "classification" else None,
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

    def _get_latent(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        pool_genus_ids: Optional[torch.Tensor] = None,
        pool_family_ids: Optional[torch.Tensor] = None,
        pool_weights: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pool_has_cover: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal: compute latent representation for any encoding mode."""
        if self.species_encoding in ("rank_pool", "transformer"):
            return self.encoder(
                continuous, species_ids,
                genus_ids=pool_genus_ids, family_ids=pool_family_ids,
                weights=pool_weights, mask=pool_mask,
                has_cover=pool_has_cover,
            )
        elif self.species_encoding == "embed":
            return self.encoder(continuous, species_ids, genus_ids, family_ids)
        elif self.uses_explicit_vector:
            return self.encoder(continuous, species_vector, genus_ids, family_ids)
        else:  # hash
            return self.encoder(continuous, genus_ids, family_ids)

    def forward(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        pool_genus_ids: Optional[torch.Tensor] = None,
        pool_family_ids: Optional[torch.Tensor] = None,
        pool_weights: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pool_has_cover: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for all targets.

        Args:
            continuous: (batch, n_continuous)
            genus_ids: (batch, top_k) optional, for hash/embed modes with taxonomy
            family_ids: (batch, top_k) optional, for hash/embed modes with taxonomy
            species_ids: (batch, top_k_species) optional, for embed/rank_pool mode
            species_vector: (batch, n_species) optional, for hash all/presence_absence
            pool_genus_ids: (batch, max_species) optional, for rank_pool mode
            pool_family_ids: (batch, max_species) optional, for rank_pool mode
            pool_weights: (batch, max_species) optional, for rank_pool mode
            pool_mask: (batch, max_species) optional, for rank_pool mode
            pool_has_cover: (batch,) optional, for rank_pool mode

        Returns:
            Dict mapping target name to predictions
        """
        latent = self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
        )
        return {name: head(latent) for name, head in self.heads.items()}

    def forward_single(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        target: str = None,
        pool_genus_ids: Optional[torch.Tensor] = None,
        pool_family_ids: Optional[torch.Tensor] = None,
        pool_weights: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pool_has_cover: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single target."""
        latent = self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
        )
        return self.heads[target](latent)

    def get_latent(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        pool_genus_ids: Optional[torch.Tensor] = None,
        pool_family_ids: Optional[torch.Tensor] = None,
        pool_weights: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pool_has_cover: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent representation without task heads."""
        return self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
        )
