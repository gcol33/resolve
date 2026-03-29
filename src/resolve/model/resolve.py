"""ResolveModel: full model composing encoder and task heads."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from resolve.data.dataset import ResolveSchema
from resolve.data.roles import TargetConfig
from resolve.model.encoder import PlotEncoder, PlotEncoderEmbed, PlotEncoderSparse, PlotEncoderRankPool, PlotEncoderTransformer
from resolve.model.experts import MixtureOfExperts
from resolve.model.head import TaskHead
from resolve.model.trait_net import PlotEncoderTraitNet


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
        - "trait_net": Trait-environment interaction network. Uses bilinear interaction
                       between environmental features and a static species-trait matrix.
                       Does not use species occurrence data directly.

    When uses_explicit_vector=True with hash encoding, species are passed as an explicit
    (n_plots, n_species) vector instead of being hashed. This enables "all" and
    "presence_absence" selection modes.
    """

    def __init__(
        self,
        schema: ResolveSchema,
        targets: dict[str, TargetConfig],
        species_encoding: Literal["hash", "embed", "rank_pool", "transformer", "trait_net"] = "hash",
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
        # MoE configuration
        n_experts: int = 0,
        expert_hidden_dims: Optional[list[int]] = None,
        moe_routing: str = "soft",
        moe_top_k: int = 2,
        moe_noise_std: float = 0.1,
        # Advanced architecture (requires C++ backend for non-MLP)
        encoder_architecture: str = "mlp",
        # Architecture sub-configs (dicts passed to C++ backend)
        ft_transformer_config: Optional[dict] = None,
        tabnet_config: Optional[dict] = None,
        saint_config: Optional[dict] = None,
        gnn_config: Optional[dict] = None,
        excelformer_config: Optional[dict] = None,
        # TraitNet configuration
        trait_net_config: Optional[dict] = None,
        traits: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if species_encoding not in ("hash", "embed", "rank_pool", "transformer", "trait_net"):
            raise ValueError(f"species_encoding must be 'hash', 'embed', 'rank_pool', 'transformer', or 'trait_net', got {species_encoding!r}")

        valid_architectures = ("mlp", "ft_transformer", "tabnet", "saint", "gnn", "excelformer")
        if encoder_architecture not in valid_architectures:
            raise ValueError(f"encoder_architecture must be one of {valid_architectures}, got '{encoder_architecture}'")

        if encoder_architecture != "mlp":
            try:
                import resolve_core  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    f"encoder_architecture='{encoder_architecture}' requires the C++ backend "
                    f"(resolve_core). Install resolve-core or use encoder_architecture='mlp'."
                )
            # Advanced architectures (ft_transformer, tabnet, saint, gnn, excelformer)
            # are implemented only in the C++ backend. This Python ResolveModel still
            # builds a standard MLP encoder as a structural placeholder so that the
            # nn.Module graph is valid, but the actual architecture is applied when
            # the C++ Trainer constructs its own ResolveModel from ModelConfig with
            # encoder_architecture set to the requested value. The Python Trainer
            # reads self.encoder_architecture and the sub-config dicts to populate
            # the resolve_core.ModelConfig before handing off to the C++ training
            # loop. See resolve_core.EncoderArchitecture for the C++ enum mapping.

        self.encoder_architecture = encoder_architecture
        self.ft_transformer_config = ft_transformer_config
        self.tabnet_config = tabnet_config
        self.saint_config = saint_config
        self.gnn_config = gnn_config
        self.excelformer_config = excelformer_config
        self.trait_net_config = trait_net_config
        self.traits = traits

        self._schema = schema
        self._targets = targets
        self.species_encoding = species_encoding
        self.uses_explicit_vector = uses_explicit_vector
        self.hash_dim = hash_dim
        self.uses_moe = n_experts > 0
        self.species_embed_dim = species_embed_dim
        self.top_k = top_k
        self.top_k_species = top_k_species
        self.hidden_dims = hidden_dims if hidden_dims is not None else [2048, 1024, 512, 256, 128, 64]
        self.genus_emb_dim = genus_emb_dim
        self.family_emb_dim = family_emb_dim
        self.dropout = dropout

        # Build categorical embedding tables (plot-level features like ecoregion, country)
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_embed_dim = schema.categorical_embed_dim
        self._categorical_names = list(schema.categorical_names or [])
        n_categorical_embed = 0
        for cat_name in self._categorical_names:
            vocab_size = schema.categorical_vocab_sizes[cat_name]
            self.categorical_embeddings[cat_name] = nn.Embedding(
                vocab_size, schema.categorical_embed_dim, padding_idx=0,
            )
            n_categorical_embed += schema.categorical_embed_dim

        # Number of base continuous features (coords + covariates)
        n_coords = 2 if schema.has_coordinates else 0
        n_unknown_features = 0
        if schema.track_unknown_fraction:
            n_unknown_features += 1
        if schema.track_unknown_count:
            n_unknown_features += 1

        if species_encoding == "hash" and not uses_explicit_vector:
            # Hash mode: continuous includes hash_dim
            n_continuous = n_coords + len(schema.covariate_names) + hash_dim + n_unknown_features + n_categorical_embed

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
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features + n_categorical_embed

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
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features + n_categorical_embed

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
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features + n_categorical_embed

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
        elif species_encoding == "trait_net":
            if traits is None:
                raise ValueError(
                    "species_encoding='trait_net' requires a traits tensor. "
                    "Pass traits=(n_species, n_traits) to ResolveModel."
                )
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features + n_categorical_embed
            cfg = trait_net_config or {}
            self.encoder = PlotEncoderTraitNet(
                n_env_features=n_continuous,
                n_trait_features=traits.shape[1],
                n_species=schema.n_species_vocab or traits.shape[0],
                env_hidden_dim=cfg.get("env_dim", 128),
                trait_hidden_dim=cfg.get("trait_dim", 64),
                interaction_dim=cfg.get("interaction_dim", 256),
                n_layers=cfg.get("n_layers", 2),
                dropout=dropout,
                hidden_dims=hidden_dims,
                traits=traits,
            )
        else:  # rank_pool mode
            n_continuous = n_coords + len(schema.covariate_names) + n_unknown_features + n_categorical_embed

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

        # Optional MoE layer after encoder
        if self.uses_moe:
            if expert_hidden_dims is None:
                expert_hidden_dims = [256, 128]
            encoder_latent = self.encoder.latent_dim
            moe_output_dim = encoder_latent  # Preserve dimension
            self.moe_layer = MixtureOfExperts(
                input_dim=encoder_latent,
                expert_hidden_dims=expert_hidden_dims,
                output_dim=moe_output_dim,
                n_experts=n_experts,
                routing=moe_routing,
                top_k=moe_top_k,
                noise_std=moe_noise_std,
                dropout=dropout,
            )
            effective_latent_dim = moe_output_dim
        else:
            self.moe_layer = None
            effective_latent_dim = self.encoder.latent_dim

        # Build task heads
        self.head_hidden_dims = head_hidden_dims
        self.heads = nn.ModuleDict()
        for name, cfg in targets.items():
            self.heads[name] = TaskHead(
                latent_dim=effective_latent_dim,
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
        if self.moe_layer is not None:
            return self.moe_layer.output_dim
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
        categorical_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal: compute latent representation for any encoding mode."""
        # --- Input validation ---
        if continuous.dim() != 2:
            raise ValueError(
                f"continuous must be 2D (batch, features), got {continuous.dim()}D"
            )
        batch_size = continuous.shape[0]

        def _check_batch(name: str, tensor: torch.Tensor | None) -> None:
            if tensor is not None and tensor.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} != continuous batch size {batch_size}"
                )

        _check_batch("genus_ids", genus_ids)
        _check_batch("family_ids", family_ids)
        _check_batch("species_ids", species_ids)
        _check_batch("species_vector", species_vector)
        _check_batch("pool_genus_ids", pool_genus_ids)
        _check_batch("pool_family_ids", pool_family_ids)
        _check_batch("pool_weights", pool_weights)
        _check_batch("pool_mask", pool_mask)
        _check_batch("categorical_ids", categorical_ids)

        if self.species_encoding in ("rank_pool", "transformer") and species_ids is None:
            raise ValueError(
                f"species_ids is required for species_encoding='{self.species_encoding}'"
            )

        if pool_mask is not None and species_ids is not None:
            if pool_mask.shape != species_ids.shape:
                raise ValueError(
                    f"pool_mask shape {pool_mask.shape} must match species_ids shape {species_ids.shape}"
                )

        # Embed categorical features and concatenate to continuous
        if len(self.categorical_embeddings) > 0:
            if categorical_ids is not None:
                cat_embeds = []
                for i, cat_name in enumerate(self._categorical_names):
                    cat_embeds.append(self.categorical_embeddings[cat_name](categorical_ids[:, i]))
                continuous = torch.cat([continuous] + cat_embeds, dim=-1)
            else:
                # No categorical IDs provided: pad with zeros to match expected input dim
                n_cat_dims = len(self._categorical_names) * self.categorical_embed_dim
                zeros = torch.zeros(continuous.shape[0], n_cat_dims, device=continuous.device)
                continuous = torch.cat([continuous, zeros], dim=-1)

        if self.species_encoding == "trait_net":
            latent = self.encoder(continuous)
        elif self.species_encoding in ("rank_pool", "transformer"):
            latent = self.encoder(
                continuous, species_ids,
                genus_ids=pool_genus_ids, family_ids=pool_family_ids,
                weights=pool_weights, mask=pool_mask,
                has_cover=pool_has_cover,
            )
        elif self.species_encoding == "embed":
            latent = self.encoder(continuous, species_ids, genus_ids, family_ids)
        elif self.uses_explicit_vector:
            latent = self.encoder(continuous, species_vector, genus_ids, family_ids)
        else:  # hash
            latent = self.encoder(continuous, genus_ids, family_ids)

        if self.moe_layer is not None:
            latent = self.moe_layer.forward_simple(latent)
        return latent

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
        categorical_ids: Optional[torch.Tensor] = None,
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
            categorical_ids: (batch, n_categoricals) optional, integer IDs for categorical features

        Returns:
            Dict mapping target name to predictions
        """
        latent = self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
            categorical_ids=categorical_ids,
        )
        return {name: head(latent) for name, head in self.heads.items()}

    def forward_single(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        target: Optional[str] = None,
        pool_genus_ids: Optional[torch.Tensor] = None,
        pool_family_ids: Optional[torch.Tensor] = None,
        pool_weights: Optional[torch.Tensor] = None,
        pool_mask: Optional[torch.Tensor] = None,
        pool_has_cover: Optional[torch.Tensor] = None,
        categorical_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single target."""
        if target is None:
            if len(self.heads) == 1:
                target = next(iter(self.heads))
            else:
                raise ValueError(
                    f"target must be specified when model has multiple heads: "
                    f"{list(self.heads.keys())}"
                )
        latent = self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
            categorical_ids=categorical_ids,
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
        categorical_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get latent representation without task heads."""
        return self._get_latent(
            continuous, genus_ids, family_ids, species_ids, species_vector,
            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
            categorical_ids=categorical_ids,
        )

    def get_genus_weights(self) -> torch.Tensor | None:
        """Get genus embedding weights averaged across positions."""
        if hasattr(self.encoder, 'get_genus_weights'):
            return self.encoder.get_genus_weights()
        return None

    def get_family_weights(self) -> torch.Tensor | None:
        """Get family embedding weights averaged across positions."""
        if hasattr(self.encoder, 'get_family_weights'):
            return self.encoder.get_family_weights()
        return None

    def get_species_weights(self) -> torch.Tensor | None:
        """Get species embedding weights averaged across positions (embed mode only)."""
        if hasattr(self.encoder, 'get_species_weights'):
            return self.encoder.get_species_weights()
        return None

    def optimize_for_inference(self) -> None:
        """Optimize model for inference using torch.compile and BN fusion.

        Applies:
            1. Batch norm folding (fuses Linear+BN into single Linear)
            2. torch.compile() for kernel fusion (if available)

        Call this after loading a trained model and before predict().
        """
        self.eval()
        self._fuse_batch_norms()
        try:
            self.encoder = torch.compile(self.encoder)
        except RuntimeError:
            import logging
            logging.getLogger("resolve.model").warning(
                "torch.compile unavailable, skipping kernel fusion"
            )

    @torch.no_grad()
    def _fuse_batch_norms(self) -> None:
        """Fuse Linear+BatchNorm1d pairs into single Linear layers."""
        for module in self.modules():
            if not isinstance(module, nn.Sequential):
                continue
            # Find Linear+BN pairs
            fuse_pairs: list[tuple[int, int]] = []
            for i in range(len(module) - 1):
                if isinstance(module[i], nn.Linear) and isinstance(module[i + 1], nn.BatchNorm1d):
                    fuse_pairs.append((i, i + 1))

            # Fuse in reverse order to preserve indices
            for lin_idx, bn_idx in reversed(fuse_pairs):
                linear = module[lin_idx]
                bn = module[bn_idx]
                # Compute fused parameters
                # BN: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
                # Fused: y = x * (bn.weight / sqrt(var + eps)) + (bn.bias - bn.running_mean * bn.weight / sqrt(var + eps))
                std = torch.sqrt(bn.running_var + bn.eps)
                scale = bn.weight / std
                # Fuse into linear: W_new = scale * W, b_new = scale * b + bn.bias - scale * bn.running_mean
                linear.weight.mul_(scale.unsqueeze(1))
                if linear.bias is not None:
                    linear.bias.mul_(scale).add_(bn.bias - scale * bn.running_mean)
                else:
                    linear.bias = nn.Parameter(bn.bias - scale * bn.running_mean)
                # Replace BN with identity
                module[bn_idx] = nn.Identity()
