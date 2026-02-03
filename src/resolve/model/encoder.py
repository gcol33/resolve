"""PlotEncoder: shared encoder for plot features."""

from typing import Literal, Optional

import torch
from torch import nn


# Valid species encoding modes
SPECIES_ENCODING_MODES = ("hash", "embed")


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
            hidden_dims = [2048, 1024, 512, 256, 128, 64]

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
        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            # Optimized: vectorized embedding lookups with stack + reshape
            # Instead of loop + list + cat, we stack embeddings and flatten
            genus_embs = torch.stack(
                [emb(genus_ids[:, k]) for k, emb in enumerate(self.genus_embeddings)],
                dim=1
            ).flatten(start_dim=1)  # (batch, top_k * emb_dim)

            family_embs = torch.stack(
                [emb(family_ids[:, k]) for k, emb in enumerate(self.family_embeddings)],
                dim=1
            ).flatten(start_dim=1)  # (batch, top_k * emb_dim)

            x = torch.cat([continuous, genus_embs, family_embs], dim=1)
        else:
            x = continuous

        latent = self.mlp(x)
        return latent


class PlotEncoderEmbed(nn.Module):
    """
    Encodes plot features using learnable embeddings for species.

    Unlike PlotEncoder which expects hash embeddings in the continuous features,
    this encoder uses learnable embeddings for top-k species, genera, and families.

    Architecture:
        - Learned embeddings for top-k species (position-aware)
        - Learned embeddings for top-k genera and families (if available)
        - Concatenate: continuous features + all embeddings
        - Feed through MLP with BatchNorm, GELU, Dropout

    Inputs:
        - continuous: (batch, n_continuous) coordinates + covariates (NO hash embedding)
        - species_ids: (batch, top_k_species) integer IDs
        - genus_ids: (batch, top_k_taxonomy) integer IDs, optional
        - family_ids: (batch, top_k_taxonomy) integer IDs, optional

    Output:
        - latent: (batch, latent_dim)
    """

    def __init__(
        self,
        n_continuous: int,
        n_species: int,
        n_genera: int = 0,
        n_families: int = 0,
        species_embed_dim: int = 32,
        genus_emb_dim: int = 8,
        family_emb_dim: int = 8,
        top_k_species: int = 10,
        top_k_taxonomy: int = 3,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256, 128, 64]

        self.top_k_species = top_k_species
        self.top_k_taxonomy = top_k_taxonomy
        self.has_taxonomy = n_genera > 0 and n_families > 0

        # Species embeddings (one table per position)
        self.species_embeddings = nn.ModuleList([
            nn.Embedding(n_species, species_embed_dim, padding_idx=0)
            for _ in range(top_k_species)
        ])
        species_dim = top_k_species * species_embed_dim

        # Taxonomic embeddings (if available)
        if self.has_taxonomy:
            self.genus_embeddings = nn.ModuleList([
                nn.Embedding(n_genera, genus_emb_dim, padding_idx=0)
                for _ in range(top_k_taxonomy)
            ])
            self.family_embeddings = nn.ModuleList([
                nn.Embedding(n_families, family_emb_dim, padding_idx=0)
                for _ in range(top_k_taxonomy)
            ])
            taxonomy_dim = top_k_taxonomy * genus_emb_dim + top_k_taxonomy * family_emb_dim
        else:
            self.genus_embeddings = None
            self.family_embeddings = None
            taxonomy_dim = 0

        # MLP
        input_dim = n_continuous + species_dim + taxonomy_dim
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
        species_ids: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            continuous: (batch, n_continuous) continuous features (coords + covariates)
            species_ids: (batch, top_k_species) species integer IDs
            genus_ids: (batch, top_k_taxonomy) genus integer IDs
            family_ids: (batch, top_k_taxonomy) family integer IDs

        Returns:
            latent: (batch, latent_dim)
        """
        # Optimized: vectorized embedding lookups with stack + flatten
        species_embs = torch.stack(
            [emb(species_ids[:, k]) for k, emb in enumerate(self.species_embeddings)],
            dim=1
        ).flatten(start_dim=1)  # (batch, top_k_species * emb_dim)

        # Embed taxonomy if available
        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            genus_embs = torch.stack(
                [emb(genus_ids[:, k]) for k, emb in enumerate(self.genus_embeddings)],
                dim=1
            ).flatten(start_dim=1)

            family_embs = torch.stack(
                [emb(family_ids[:, k]) for k, emb in enumerate(self.family_embeddings)],
                dim=1
            ).flatten(start_dim=1)

            x = torch.cat([continuous, species_embs, genus_embs, family_embs], dim=1)
        else:
            x = torch.cat([continuous, species_embs], dim=1)

        latent = self.mlp(x)
        return latent


class PlotEncoderSparse(nn.Module):
    """
    Encodes plot features using explicit species abundance vectors.

    Unlike hash encoding (fixed-dim compression) or embed encoding (learnable
    per-species embeddings), this takes the raw species abundance matrix directly
    and learns a linear projection to a species embedding space.

    Architecture:
        - Linear projection from species abundances to species embedding
        - Optional: Learned embeddings for top-k genera and families
        - Concatenate: continuous features + species embedding + taxonomic embeddings
        - Feed through MLP with BatchNorm, GELU, Dropout

    Inputs:
        - continuous: (batch, n_continuous) coordinates + covariates
        - species_abundances: (batch, n_species) normalized abundances
        - genus_ids: (batch, top_k) integer IDs, optional
        - family_ids: (batch, top_k) integer IDs, optional

    Output:
        - latent: (batch, latent_dim)

    Note: This encoder is best suited for moderate species pools (<5k species).
    For very large pools, consider hash encoding for efficiency.
    """

    def __init__(
        self,
        n_continuous: int,
        n_species: int,
        species_embed_dim: int = 64,
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
            hidden_dims = [2048, 1024, 512, 256, 128, 64]

        self.n_species = n_species
        self.top_k = top_k
        self.has_taxonomy = n_genera > 0 and n_families > 0

        # Linear projection from species abundances to embedding space
        # This learns species-specific weights
        self.species_projection = nn.Linear(n_species, species_embed_dim)

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
        input_dim = n_continuous + species_embed_dim + taxonomy_dim
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
        species_abundances: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            continuous: (batch, n_continuous) continuous features
            species_abundances: (batch, n_species) species abundance vector
            genus_ids: (batch, top_k) genus integer IDs
            family_ids: (batch, top_k) family integer IDs

        Returns:
            latent: (batch, latent_dim)
        """
        # Project species abundances to embedding space
        species_emb = self.species_projection(species_abundances)

        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            # Optimized: vectorized embedding lookups with stack + flatten
            genus_embs = torch.stack(
                [emb(genus_ids[:, k]) for k, emb in enumerate(self.genus_embeddings)],
                dim=1
            ).flatten(start_dim=1)

            family_embs = torch.stack(
                [emb(family_ids[:, k]) for k, emb in enumerate(self.family_embeddings)],
                dim=1
            ).flatten(start_dim=1)

            x = torch.cat([continuous, species_emb, genus_embs, family_embs], dim=1)
        else:
            x = torch.cat([continuous, species_emb], dim=1)

        latent = self.mlp(x)
        return latent
