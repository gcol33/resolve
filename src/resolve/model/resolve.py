"""ResolveModel: full model composing encoder and task heads."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from resolve.data.dataset import ResolveSchema
from resolve.data.roles import TargetConfig
from resolve.model.encoder import PlotEncoder
from resolve.model.head import TaskHead


class ResolveModel(nn.Module):
    """
    Full RESOLVE model: shared encoder with multiple task heads.

    The encoder processes:
        - Continuous features (coords + covariates + hash embedding)
        - Taxonomic IDs (genus, family) via learned embeddings

    Each target gets its own prediction head.
    """

    def __init__(
        self,
        schema: ResolveSchema,
        targets: dict[str, TargetConfig],
        hash_dim: int = 32,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        top_k: int = 3,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        track_unknown_count: bool = False,
    ):
        super().__init__()

        self._schema = schema
        self._targets = targets
        self.hash_dim = hash_dim
        self.top_k = top_k
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128, 64]
        self.genus_emb_dim = genus_emb_dim
        self.family_emb_dim = family_emb_dim
        self.dropout = dropout
        self.track_unknown_count = track_unknown_count

        # Number of continuous features:
        # coordinates (if present) + covariates + hash_dim + unknown_fraction + (optional) unknown_count
        n_coords = 2 if schema.has_coordinates else 0
        n_unknown_features = 2 if track_unknown_count else 1  # fraction always, count optional
        n_continuous = n_coords + len(schema.covariate_names) + hash_dim + n_unknown_features

        # Build encoder
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
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for all targets.

        Args:
            continuous: (batch, n_continuous)
            genus_ids: (batch, top_k) optional
            family_ids: (batch, top_k) optional

        Returns:
            Dict mapping target name to predictions
        """
        latent = self.encoder(continuous, genus_ids, family_ids)
        return {name: head(latent) for name, head in self.heads.items()}

    def forward_single(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        target: str = None,
    ) -> torch.Tensor:
        """Forward pass for a single target."""
        latent = self.encoder(continuous, genus_ids, family_ids)
        return self.heads[target](latent)

    def get_latent(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent representation without task heads."""
        return self.encoder(continuous, genus_ids, family_ids)
