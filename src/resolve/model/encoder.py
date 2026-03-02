"""PlotEncoder: shared encoder for plot features."""

import math
from typing import Literal, Optional

import torch
from torch import nn

from resolve.csrc.fused_embed_linear import fused_embed_concat_linear


# Valid species encoding modes
SPECIES_ENCODING_MODES = ("hash", "embed", "rank_pool", "transformer")


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
) -> tuple[nn.Sequential, int]:
    """Build MLP with BatchNorm, GELU, Dropout. Returns (mlp, latent_dim)."""
    dims = [input_dim] + hidden_dims
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers), hidden_dims[-1]


def _build_taxonomy_modulelists(
    n_genera: int,
    n_families: int,
    genus_emb_dim: int,
    family_emb_dim: int,
    top_k: int,
    padding_idx: int | None = None,
) -> tuple[bool, nn.ModuleList | None, nn.ModuleList | None, int]:
    """Build per-position taxonomy embedding tables.

    Returns (has_taxonomy, genus_embeddings, family_embeddings, taxonomy_dim).
    """
    has_taxonomy = n_genera > 0 and n_families > 0
    if has_taxonomy:
        genus_embeddings = nn.ModuleList([
            nn.Embedding(n_genera, genus_emb_dim, padding_idx=padding_idx)
            for _ in range(top_k)
        ])
        family_embeddings = nn.ModuleList([
            nn.Embedding(n_families, family_emb_dim, padding_idx=padding_idx)
            for _ in range(top_k)
        ])
        taxonomy_dim = top_k * genus_emb_dim + top_k * family_emb_dim
    else:
        genus_embeddings = None
        family_embeddings = None
        taxonomy_dim = 0
    return has_taxonomy, genus_embeddings, family_embeddings, taxonomy_dim


def _embed_taxonomy_modulelists(
    genus_embeddings: nn.ModuleList | None,
    family_embeddings: nn.ModuleList | None,
    genus_ids: torch.Tensor | None,
    family_ids: torch.Tensor | None,
    has_taxonomy: bool,
) -> torch.Tensor | None:
    """Compute taxonomy embeddings via stack+flatten. Returns (batch, taxonomy_dim) or None."""
    if not has_taxonomy or genus_ids is None or family_ids is None:
        return None
    genus_embs = torch.stack(
        [emb(genus_ids[:, k]) for k, emb in enumerate(genus_embeddings)],
        dim=1,
    ).flatten(start_dim=1)
    family_embs = torch.stack(
        [emb(family_ids[:, k]) for k, emb in enumerate(family_embeddings)],
        dim=1,
    ).flatten(start_dim=1)
    return torch.cat([genus_embs, family_embs], dim=1)


def _apply_cover_dropout(
    training: bool,
    cover_dropout: float,
    batch_size: int,
    device: torch.device,
    weights: torch.Tensor | None,
    mask: torch.Tensor | None,
    species_ids: torch.Tensor,
    has_cover: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Apply cover dropout during training. Returns (weights, has_cover)."""
    if training and cover_dropout > 0 and weights is not None:
        drop_mask = torch.rand(batch_size, device=device) < cover_dropout
        if drop_mask.any():
            weights = weights.clone()
            has_cover = has_cover.clone()
            if mask is not None:
                weights[drop_mask] = mask[drop_mask].float()
            else:
                weights[drop_mask] = (species_ids[drop_mask] != 0).float()
            has_cover[drop_mask] = 0.0
    return weights, has_cover


def _get_modulelist_weights(
    embeddings: nn.ModuleList | None,
    has_taxonomy: bool,
) -> torch.Tensor | None:
    """Average embedding weights across position slots in a ModuleList.

    Used by PlotEncoder, PlotEncoderEmbed, PlotEncoderSparse for
    get_genus_weights() and get_family_weights().
    """
    if not has_taxonomy or embeddings is None:
        return None
    return torch.stack([emb.weight for emb in embeddings], dim=0).mean(0)


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
        self.has_taxonomy, self.genus_embeddings, self.family_embeddings, taxonomy_dim = (
            _build_taxonomy_modulelists(n_genera, n_families, genus_emb_dim, family_emb_dim, top_k)
        )
        self.mlp, self.latent_dim = _build_mlp(n_continuous + taxonomy_dim, hidden_dims, dropout)

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
            # Fused embed + concat + first linear: avoids materializing
            # the intermediate (batch, D_cont + K*G + K*F) tensor
            first_linear_out = fused_embed_concat_linear(
                continuous, genus_ids, family_ids,
                list(self.genus_embeddings), list(self.family_embeddings),
                self.mlp[0],  # first nn.Linear in the Sequential
            )
            return self.mlp[1:](first_linear_out)
        return self.mlp(continuous)

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights averaged across positions."""
        return _get_modulelist_weights(self.genus_embeddings, self.has_taxonomy)

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights averaged across positions."""
        return _get_modulelist_weights(self.family_embeddings, self.has_taxonomy)


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

        # Species embeddings (one table per position)
        self.species_embeddings = nn.ModuleList([
            nn.Embedding(n_species, species_embed_dim, padding_idx=0)
            for _ in range(top_k_species)
        ])
        species_dim = top_k_species * species_embed_dim

        self.has_taxonomy, self.genus_embeddings, self.family_embeddings, taxonomy_dim = (
            _build_taxonomy_modulelists(
                n_genera, n_families, genus_emb_dim, family_emb_dim, top_k_taxonomy, padding_idx=0,
            )
        )
        self.mlp, self.latent_dim = _build_mlp(
            n_continuous + species_dim + taxonomy_dim, hidden_dims, dropout,
        )

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

        taxonomy_emb = _embed_taxonomy_modulelists(
            self.genus_embeddings, self.family_embeddings,
            genus_ids, family_ids, self.has_taxonomy,
        )
        parts = [continuous, species_embs]
        if taxonomy_emb is not None:
            parts.append(taxonomy_emb)
        return self.mlp(torch.cat(parts, dim=1))

    def get_species_weights(self) -> torch.Tensor:
        """Get species embedding weights averaged across positions."""
        return torch.stack([emb.weight for emb in self.species_embeddings], dim=0).mean(0)

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights averaged across positions."""
        return _get_modulelist_weights(self.genus_embeddings, self.has_taxonomy)

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights averaged across positions."""
        return _get_modulelist_weights(self.family_embeddings, self.has_taxonomy)


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

        # Linear projection from species abundances to embedding space
        self.species_projection = nn.Linear(n_species, species_embed_dim)

        self.has_taxonomy, self.genus_embeddings, self.family_embeddings, taxonomy_dim = (
            _build_taxonomy_modulelists(n_genera, n_families, genus_emb_dim, family_emb_dim, top_k)
        )
        self.mlp, self.latent_dim = _build_mlp(
            n_continuous + species_embed_dim + taxonomy_dim, hidden_dims, dropout,
        )

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

        taxonomy_emb = _embed_taxonomy_modulelists(
            self.genus_embeddings, self.family_embeddings,
            genus_ids, family_ids, self.has_taxonomy,
        )
        parts = [continuous, species_emb]
        if taxonomy_emb is not None:
            parts.append(taxonomy_emb)
        return self.mlp(torch.cat(parts, dim=1))

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights averaged across positions."""
        return _get_modulelist_weights(self.genus_embeddings, self.has_taxonomy)

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights averaged across positions."""
        return _get_modulelist_weights(self.family_embeddings, self.has_taxonomy)


class PlotEncoderRankPool(nn.Module):
    """Rank-pool species encoder with additive hierarchical embeddings.

    For each species in a plot:
        repr_i = species_embed(sp_id) + genus_embed(genus_id) + family_embed(family_id)

    Pool across all species (weighted mean, masked for padding):
        plot_repr = weighted_mean(repr_i, weights=w_i)

    Concatenate with continuous features + has_cover flag and feed through MLP.

    Cover dropout: during training, randomly replaces rank weights with uniform
    (binary) for a fraction of the batch, setting has_cover=0 for those samples.
    This teaches the model to handle both cover-ordered and unordered species lists.

    Inputs:
        - continuous: (batch, n_continuous) coordinates + covariates
        - species_ids: (batch, max_species) padded integer IDs
        - genus_ids: (batch, max_species) padded integer IDs
        - family_ids: (batch, max_species) padded integer IDs
        - weights: (batch, max_species) abundance weights
        - mask: (batch, max_species) bool, True = valid position
        - has_cover: (batch,) float scalar, 1.0 if cover info present

    Output:
        - latent: (batch, latent_dim)
    """

    def __init__(
        self,
        n_continuous: int,
        n_species: int,
        n_genera: int = 0,
        n_families: int = 0,
        species_embed_dim: int = 64,
        genus_embed_dim: int = 16,
        family_embed_dim: int = 16,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        cover_dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256, 128, 64]

        self.has_taxonomy = n_genera > 0 and n_families > 0
        self.cover_dropout = cover_dropout

        # Single shared embedding table per taxonomic level
        self.species_embedding = nn.Embedding(n_species, species_embed_dim, padding_idx=0)

        if self.has_taxonomy:
            self.genus_embedding = nn.Embedding(n_genera, genus_embed_dim, padding_idx=0)
            self.family_embedding = nn.Embedding(n_families, family_embed_dim, padding_idx=0)
            embed_dim = species_embed_dim + genus_embed_dim + family_embed_dim
        else:
            self.genus_embedding = None
            self.family_embedding = None
            embed_dim = species_embed_dim

        # +1 for has_cover flag
        self.mlp, self.latent_dim = _build_mlp(n_continuous + embed_dim + 1, hidden_dims, dropout)

    def forward(
        self,
        continuous: torch.Tensor,
        species_ids: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        has_cover: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with weighted mean pooling over variable-length species.

        Args:
            continuous: (batch, n_continuous) continuous features
            species_ids: (batch, max_species) padded species integer IDs
            genus_ids: (batch, max_species) padded genus integer IDs
            family_ids: (batch, max_species) padded family integer IDs
            weights: (batch, max_species) abundance weights (pre-normalized or raw)
            mask: (batch, max_species) bool, True = valid species position
            has_cover: (batch,) float, 1.0 if cover info is present

        Returns:
            latent: (batch, latent_dim)
        """
        batch_size = continuous.shape[0]

        # Default has_cover to 1.0 if not provided
        if has_cover is None:
            has_cover = torch.ones(batch_size, device=continuous.device)

        weights, has_cover = _apply_cover_dropout(
            self.training, self.cover_dropout, batch_size, continuous.device,
            weights, mask, species_ids, has_cover,
        )

        # Species embeddings: (batch, max_sp, d_sp)
        sp_emb = self.species_embedding(species_ids)

        # Additive hierarchical embedding
        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            g_emb = self.genus_embedding(genus_ids)    # (batch, max_sp, d_g)
            f_emb = self.family_embedding(family_ids)  # (batch, max_sp, d_f)
            combined = torch.cat([sp_emb, g_emb, f_emb], dim=-1)  # (batch, max_sp, d_total)
        else:
            combined = sp_emb

        # Weighted mean pool (masked)
        if mask is not None:
            mask_float = mask.float()  # (batch, max_sp)
        else:
            mask_float = (species_ids != 0).float()

        if weights is not None:
            w = weights * mask_float  # zero out padding positions
        else:
            w = mask_float

        # Normalize weights to sum to 1 per sample (avoid div by zero)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w_normed = w / w_sum  # (batch, max_sp)

        # Weighted sum: (batch, max_sp, 1) * (batch, max_sp, d_total) -> sum -> (batch, d_total)
        pooled = (combined * w_normed.unsqueeze(-1)).sum(dim=1)

        # Concatenate with continuous + has_cover flag and feed through MLP
        has_cover_col = has_cover.unsqueeze(-1)  # (batch, 1)
        x = torch.cat([continuous, pooled, has_cover_col], dim=1)
        return self.mlp(x)

    def get_species_weights(self) -> torch.Tensor:
        """Get species embedding weights."""
        return self.species_embedding.weight

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights."""
        if not self.has_taxonomy:
            return None
        return self.genus_embedding.weight

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights."""
        if not self.has_taxonomy:
            return None
        return self.family_embedding.weight


class PlotEncoderTransformer(nn.Module):
    """Transformer species encoder with attention pooling.

    Reuses the same data pipeline as PlotEncoderRankPool (same forward signature,
    same collate_fn, same batch unpacking). Only the encoder architecture changes.

    Variants:
        - v4 (n_attention_layers=0): Additive embeddings + attention pooling only.
          Species don't interact; learned query attends over independent tokens.
        - v5 (n_attention_layers>=1): Self-attention layers + attention pooling.
          Species interact through self-attention before pooling.
        - v6: Same as v5 but with masked species pretraining (handled externally
          by pretrain.py; this class provides mask_embedding for masking).

    Embedding scheme (additive, all in d_model space):
        token_i = species_emb(sp_id) + genus_emb(g_id) + family_emb(f_id) + weight_proj(cover)

    Pooling:
        - "attention": Learned query + cross-attention over token sequence.
        - "cls": Learnable CLS token prepended before transformer, extracted after.

    Inputs/output: identical to PlotEncoderRankPool.
    """

    def __init__(
        self,
        n_continuous: int,
        n_species: int,
        n_genera: int = 0,
        n_families: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        n_attention_layers: int = 0,
        transformer_ff_dim: int = 256,
        transformer_pooling: str = "attention",
        transformer_dropout: float = 0.1,
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.3,
        cover_dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512]

        if transformer_pooling not in ("attention", "cls"):
            raise ValueError(
                f"transformer_pooling must be 'attention' or 'cls', got {transformer_pooling!r}"
            )

        self.d_model = d_model
        self.n_attention_layers = n_attention_layers
        self.transformer_pooling = transformer_pooling
        self.has_taxonomy = n_genera > 0 and n_families > 0
        self.cover_dropout = cover_dropout

        # --- Embedding layers (all d_model-dimensional, additive) ---
        self.species_embedding = nn.Embedding(n_species, d_model, padding_idx=0)

        if self.has_taxonomy:
            self.genus_embedding = nn.Embedding(n_genera, d_model, padding_idx=0)
            self.family_embedding = nn.Embedding(n_families, d_model, padding_idx=0)
        else:
            self.genus_embedding = None
            self.family_embedding = None

        # Project scalar cover weight to d_model
        self.weight_proj = nn.Linear(1, d_model, bias=False)

        # Learned mask embedding for v6 pretraining (replaces masked species tokens)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))

        # Init embeddings with std=0.02 (BERT convention for additive scheme)
        self._init_embeddings()

        # --- Self-attention (v5+, when n_attention_layers >= 1) ---
        if n_attention_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=transformer_ff_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=transformer_dropout,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_attention_layers
            )
        else:
            self.transformer_encoder = None

        # --- Pooling ---
        if transformer_pooling == "attention":
            # Learned query for cross-attention pooling
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=transformer_dropout, batch_first=True
            )
            self.pool_norm = nn.LayerNorm(d_model)
        else:  # cls
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # MLP after pooling: +1 for has_cover flag
        self.mlp, self.latent_dim = _build_mlp(n_continuous + d_model + 1, hidden_dims, dropout)

    def _init_embeddings(self) -> None:
        """Initialize all embeddings with std=0.02 (BERT convention)."""
        nn.init.normal_(self.species_embedding.weight, std=0.02)
        # Re-zero padding index
        with torch.no_grad():
            self.species_embedding.weight[0].zero_()

        if self.has_taxonomy:
            nn.init.normal_(self.genus_embedding.weight, std=0.02)
            nn.init.normal_(self.family_embedding.weight, std=0.02)
            with torch.no_grad():
                self.genus_embedding.weight[0].zero_()
                self.family_embedding.weight[0].zero_()

        nn.init.normal_(self.weight_proj.weight, std=0.02)
        nn.init.normal_(self.mask_embedding, std=0.02)

    def forward(
        self,
        continuous: torch.Tensor,
        species_ids: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        has_cover: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with transformer encoding over variable-length species.

        Args:
            continuous: (batch, n_continuous) continuous features
            species_ids: (batch, max_species) padded species integer IDs
            genus_ids: (batch, max_species) padded genus integer IDs
            family_ids: (batch, max_species) padded family integer IDs
            weights: (batch, max_species) abundance weights
            mask: (batch, max_species) bool, True = valid species position
            has_cover: (batch,) float, 1.0 if cover info is present
            masked_positions: (batch, max_species) bool, True = masked for MLM.
                Used during v6 pretraining to replace tokens with mask_embedding.

        Returns:
            latent: (batch, latent_dim)
        """
        batch_size = continuous.shape[0]

        # Default has_cover to 1.0 if not provided
        if has_cover is None:
            has_cover = torch.ones(batch_size, device=continuous.device)

        weights, has_cover = _apply_cover_dropout(
            self.training, self.cover_dropout, batch_size, continuous.device,
            weights, mask, species_ids, has_cover,
        )

        # Build mask if not provided
        if mask is None:
            mask = species_ids != 0  # (batch, max_sp)

        # --- Additive token embeddings ---
        # Species: (batch, max_sp, d_model)
        tokens = self.species_embedding(species_ids)

        # Taxonomy (additive in same d_model space)
        if self.has_taxonomy and genus_ids is not None and family_ids is not None:
            tokens = tokens + self.genus_embedding(genus_ids) + self.family_embedding(family_ids)

        # Cover weight projection: (batch, max_sp, 1) -> (batch, max_sp, d_model)
        if weights is not None:
            w_proj = self.weight_proj(weights.unsqueeze(-1))  # (batch, max_sp, d_model)
            tokens = tokens + w_proj

        # Apply mask embedding for MLM pretraining (v6)
        if masked_positions is not None:
            tokens = tokens.clone()
            tokens[masked_positions] = self.mask_embedding

        # --- Self-attention + Pooling ---
        # PyTorch TransformerEncoder: src_key_padding_mask True = IGNORE
        padding_mask = ~mask  # invert: True=padding (ignore)

        if self.transformer_pooling == "attention":
            # Self-attention first, then cross-attention pooling
            if self.transformer_encoder is not None:
                tokens = self.transformer_encoder(
                    tokens, src_key_padding_mask=padding_mask
                )
            query = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, d_model)
            pooled, _ = self.pool_attn(
                query, tokens, tokens,
                key_padding_mask=padding_mask,
            )  # (B, 1, d_model)
            pooled = self.pool_norm(pooled.squeeze(1))  # (B, d_model)
        else:  # cls
            # Prepend CLS token BEFORE transformer, then extract position 0
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+max_sp, d_model)
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            padding_mask = torch.cat([cls_pad, padding_mask], dim=1)

            if self.transformer_encoder is not None:
                tokens = self.transformer_encoder(
                    tokens, src_key_padding_mask=padding_mask
                )
            pooled = tokens[:, 0, :]  # (B, d_model)

        # --- MLP: cat([continuous, pooled, has_cover]) -> latent ---
        has_cover_col = has_cover.unsqueeze(-1)  # (batch, 1)
        x = torch.cat([continuous, pooled, has_cover_col], dim=1)
        return self.mlp(x)

    def get_species_weights(self) -> torch.Tensor:
        """Get species embedding weights."""
        return self.species_embedding.weight

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights."""
        if not self.has_taxonomy:
            return None
        return self.genus_embedding.weight

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights."""
        if not self.has_taxonomy:
            return None
        return self.family_embedding.weight
