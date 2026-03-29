"""TraitNet: trait-environment interaction encoder.

Uses bilinear interaction between environmental features and species traits
to produce a plot-level latent representation.

Reference: C++ implementation in src/core/include/resolve/attention.hpp
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearInteraction(nn.Module):
    """Bilinear interaction between environment and trait encodings."""

    def __init__(self, env_dim: int, trait_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, env_dim, trait_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, env: torch.Tensor, traits: torch.Tensor) -> torch.Tensor:
        # env: (B, env_dim), traits: (S, trait_dim)
        # weight: (O, env_dim, trait_dim)
        # result: (B, S, O)
        # env @ W: (B, O, trait_dim) via einsum
        env_proj = torch.einsum("be,oet->bot", env, self.weight)  # (B, O, trait_dim)
        # (B, O, trait_dim) @ (trait_dim, S) -> (B, O, S)
        interaction = torch.matmul(env_proj, traits.T)  # (B, O, S)
        return interaction.permute(0, 2, 1)  # (B, S, O)


def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers = []
    prev = input_dim
    for dim in hidden_dims:
        layers.extend([
            nn.Linear(prev, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        prev = dim
    return nn.Sequential(*layers)


class PlotEncoderTraitNet(nn.Module):
    """Trait-environment interaction encoder.

    Takes environmental features (coords + covariates) and a static species-trait
    matrix, computes bilinear interactions, and produces a latent representation.

    Unlike hash/embed/rank_pool encoders, TraitNet does not use species occurrence
    data directly. Instead, it learns how environmental conditions interact with
    species traits to predict plot-level outcomes.

    Args:
        n_env_features: Number of environmental input features.
        n_trait_features: Number of trait features per species.
        n_species: Number of species in the trait matrix.
        env_hidden_dim: Hidden dimension for environment encoder.
        trait_hidden_dim: Hidden dimension for trait encoder.
        interaction_dim: Dimension of bilinear interaction output.
        n_layers: Number of MLP layers in env/trait encoders.
        dropout: Dropout rate.
        hidden_dims: MLP backbone dimensions after interaction pooling.
        traits: Optional (n_species, n_trait_features) tensor. If provided,
            registered as a buffer. Otherwise, must be passed to forward().
    """

    def __init__(
        self,
        n_env_features: int,
        n_trait_features: int,
        n_species: int,
        env_hidden_dim: int = 128,
        trait_hidden_dim: int = 64,
        interaction_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        hidden_dims: list[int] | None = None,
        traits: torch.Tensor | None = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_species = n_species
        self.n_env_features = n_env_features
        self.n_trait_features = n_trait_features

        # Environment encoder
        env_layers = [env_hidden_dim] * n_layers
        self.env_encoder = _build_mlp(n_env_features, env_layers, dropout)
        env_out_dim = env_layers[-1] if env_layers else n_env_features

        # Trait encoder
        trait_layers = [trait_hidden_dim] * n_layers
        self.trait_encoder = _build_mlp(n_trait_features, trait_layers, dropout)
        trait_out_dim = trait_layers[-1] if trait_layers else n_trait_features

        # Bilinear interaction
        self.interaction = BilinearInteraction(env_out_dim, trait_out_dim, interaction_dim)

        # Pool + project: (B, interaction_dim) -> MLP backbone
        backbone_input = interaction_dim
        layers = []
        prev = backbone_input
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = dim
        self.mlp = nn.Sequential(*layers)

        self._latent_dim = hidden_dims[-1] if hidden_dims else interaction_dim

        # Store traits as buffer if provided
        if traits is not None:
            self.register_buffer("_traits", traits)
        else:
            self._traits = None

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def set_traits(self, traits: torch.Tensor) -> None:
        """Set the species trait matrix."""
        # Remove existing attribute if it was set as a plain attr (not a buffer)
        if "_traits" in self.__dict__:
            del self._traits
        self.register_buffer("_traits", traits)

    def forward(
        self,
        continuous: torch.Tensor,
        traits: torch.Tensor | None = None,
        **kwargs,  # Accept and ignore extra args for compatibility
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            continuous: (batch, n_env_features) environmental features.
            traits: (n_species, n_trait_features) species traits. Uses stored
                buffer if not provided.

        Returns:
            (batch, latent_dim) latent representation.
        """
        if traits is None:
            traits = self._traits
        if traits is None:
            raise ValueError("No traits tensor provided or stored. Call set_traits() first.")

        # Move traits to same device as input
        if traits.device != continuous.device:
            traits = traits.to(continuous.device)

        # Encode
        env_enc = self.env_encoder(continuous)     # (B, env_hidden_dim)
        trait_enc = self.trait_encoder(traits)     # (S, trait_hidden_dim)

        # Bilinear interaction
        inter = self.interaction(env_enc, trait_enc)  # (B, S, interaction_dim)

        # Mean pool over species
        pooled = inter.mean(dim=1)  # (B, interaction_dim)

        # MLP backbone
        return self.mlp(pooled)  # (B, latent_dim)
