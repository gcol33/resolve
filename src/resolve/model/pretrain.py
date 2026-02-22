"""Masked species pretraining (v6) for PlotEncoderTransformer.

BERT-style masked language modelling over species tokens:
  - 15% of valid species positions are selected for masking
  - Of those: 80% replaced with mask_embedding, 10% random species, 10% kept
  - Cross-entropy loss on masked positions only

Usage:
    1. Build PlotEncoderTransformer with n_attention_layers >= 1
    2. Call Trainer.pretrain() which uses MaskedSpeciesHead + mask_species_batch
    3. After pretraining, encoder weights are kept; MaskedSpeciesHead is discarded
    4. Call Trainer.fit() for supervised fine-tuning
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedSpeciesHead(nn.Module):
    """Linear projection from d_model to species vocabulary for MLM.

    Discarded after pretraining; only used to compute cross-entropy
    on masked species positions.
    """

    def __init__(self, d_model: int, n_species: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_species, bias=False)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Project token embeddings to species logits.

        Args:
            token_embeddings: (N_masked, d_model) embeddings at masked positions

        Returns:
            logits: (N_masked, n_species)
        """
        return self.proj(token_embeddings)


def mask_species_batch(
    species_ids: torch.Tensor,
    mask: torch.Tensor,
    n_species: int,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply BERT-style masking to a batch of species IDs.

    For each valid (non-padding) species position, with probability mask_prob:
      - 80%: replace with a sentinel value (0, handled by mask_embedding in encoder)
      - 10%: replace with random species ID
      - 10%: keep original (model must predict from context)

    Args:
        species_ids: (batch, max_sp) original species IDs
        mask: (batch, max_sp) bool, True = valid position
        n_species: vocabulary size (for random replacement)
        mask_prob: fraction of valid positions to mask

    Returns:
        masked_ids: (batch, max_sp) species IDs with masking applied
        mlm_mask: (batch, max_sp) bool, True = position was selected for prediction
        mlm_targets: (batch, max_sp) original species IDs at masked positions
            (only meaningful where mlm_mask is True)
    """
    masked_ids = species_ids.clone()
    mlm_targets = species_ids.clone()

    # Select positions to mask (only valid, non-padding positions)
    rand = torch.rand_like(mask, dtype=torch.float)
    mlm_mask = mask & (rand < mask_prob)

    # Of selected positions: 80% -> mask token (id=0, encoder replaces with mask_embedding)
    # 10% -> random species, 10% -> keep original
    rand_action = torch.rand_like(mlm_mask, dtype=torch.float)

    # 80%: replace with 0 (will be replaced by mask_embedding in forward)
    replace_mask = mlm_mask & (rand_action < 0.8)
    masked_ids[replace_mask] = 0

    # 10%: replace with random species (IDs from 1 to n_species-1)
    random_mask = mlm_mask & (rand_action >= 0.8) & (rand_action < 0.9)
    random_ids = torch.randint(1, n_species, masked_ids.shape, device=masked_ids.device)
    masked_ids[random_mask] = random_ids[random_mask]

    # 10%: keep original (no action needed)

    return masked_ids, mlm_mask, mlm_targets


class MaskedSpeciesCollateWrapper:
    """Wraps rank_pool collate_fn to add MLM masking to each batch.

    The wrapper intercepts collated batches and applies mask_species_batch
    to the species_ids tensor. The mlm_mask and mlm_targets are appended
    to the batch tuple.

    Batch layout (with taxonomy):
        (continuous, species_ids, genus_ids, family_ids, weights, mask, has_cover,
         *targets, mlm_mask, mlm_targets)

    Batch layout (no taxonomy):
        (continuous, species_ids, weights, mask, has_cover,
         *targets, mlm_mask, mlm_targets)
    """

    def __init__(self, base_collate_fn, n_species: int, mask_prob: float = 0.15,
                 has_taxonomy: bool = True):
        self.base_collate_fn = base_collate_fn
        self.n_species = n_species
        self.mask_prob = mask_prob
        self.has_taxonomy = has_taxonomy

    def __call__(self, samples):
        batch = self.base_collate_fn(samples)

        # Extract species_ids and mask from batch tuple
        # Layout: (continuous, species_ids, [genus, family,] weights, mask, has_cover, *targets)
        species_ids = batch[1]  # always at index 1
        if self.has_taxonomy:
            valid_mask = batch[5]  # continuous, sp, g, f, w, mask
        else:
            valid_mask = batch[3]  # continuous, sp, w, mask

        masked_ids, mlm_mask, mlm_targets = mask_species_batch(
            species_ids, valid_mask.bool(), self.n_species, self.mask_prob
        )

        # Replace species_ids with masked version, append mlm tensors
        batch_list = list(batch)
        batch_list[1] = masked_ids
        batch_list.append(mlm_mask)
        batch_list.append(mlm_targets)

        return tuple(batch_list)
