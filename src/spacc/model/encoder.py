"""PlotEncoder: shared encoder for plot features."""

from typing import Optional

import torch
from torch import nn


class PlotEncoder(nn.Module):
    """
    Encodes plot features into a shared latent representation.

    Architecture:
        - Learned embeddings for top-k genera and families (if available)
        - Concatenate: continuous features + hash embedding + taxonomic embeddings
        - Feed through MLP with BatchNorm, GELU, Dropout

    Inputs:
        - continuous: (batch, n_continuous) coordinates + covariates + hash embedding
        - genus_ids: (batch, top_k) integer IDs, optional
        - family_ids: (batch, top_k) integer IDs, optional

    Output:
        - latent: (batch, latent_dim)
    """

    def __init__(
        self,
        n_continuous: int,
        n_genera: int = 0,
        n_families: int = 0,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        top_k: int = 3,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        self.top_k = top_k
        self.has_taxonomy = n_genera > 0 and n_families > 0

        # Taxonomic embeddings (if available)
        if self.has_taxonomy:
            self.genus_embeddings = nn.ModuleList([
                nn.Embedding(n_genera, genus_emb_dim) for _ in range(top_k)
            ])
            self.family_embeddings = nn.ModuleList([
                nn.Embedding(n_families, family_emb_dim) for _ in range(top_k)
            ])
            taxonomy_dim = top_k * genus_emb_dim + top_k * family_emb_dim
        else:
            self.genus_embeddings = None
            self.family_embeddings = None
            taxonomy_dim = 0

        # MLP
        input_dim = n_continuous + taxonomy_dim
        dims = [input_dim] + hidden_dims

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.latent_dim = hidden_dims[-1]

    def forward(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            continuous: (batch, n_continuous) continuous features
            genus_ids: (batch, top_k) genus integer IDs
            family_ids: (batch, top_k) family integer IDs

        Returns:
            latent: (batch, latent_dim)
        """
        parts = [continuous]

        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            # Embed each of the top-k genera
            genus_embs = []
            for k in range(self.top_k):
                emb = self.genus_embeddings[k](genus_ids[:, k])
                genus_embs.append(emb)

            # Embed each of the top-k families
            family_embs = []
            for k in range(self.top_k):
                emb = self.family_embeddings[k](family_ids[:, k])
                family_embs.append(emb)

            parts.extend(genus_embs)
            parts.extend(family_embs)

        x = torch.cat(parts, dim=1)
        latent = self.mlp(x)
        return latent
