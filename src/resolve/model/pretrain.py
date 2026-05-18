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
from torch.utils.data import default_collate


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
    """Apply MLM masking to a pre-padded rank_pool/transformer batch.

    Consumes batches produced by `torch.utils.data.default_collate` over a
    `TensorDataset` built from the pre-padded tuple returned by
    `Trainer._build_tensors` (rank_pool/transformer branch). Replaces
    `species_ids` with a masked copy and appends `(mlm_mask, mlm_targets)`.

    Batch tuple positions (from `_build_tensors`):
        [0]              continuous
        [1]              species_ids
        [2:2+2*has_tax]  genus_ids, family_ids   (only if taxonomy present)
        next             weights
        next             mask
        next             has_cover
        next             categorical_ids         (only if has_categoricals)
        rest             *targets

    Output layout: `(*batch, mlm_mask, mlm_targets)`.
    """

    def __init__(self, n_species: int, mask_prob: float = 0.15,
                 has_taxonomy: bool = True, has_categoricals: bool = False):
        self.n_species = n_species
        self.mask_prob = mask_prob
        self.has_taxonomy = has_taxonomy
        self.has_categoricals = has_categoricals
        # mask sits after [continuous, species_ids, (g, f,) weights]
        self._mask_index = 2 + (2 if has_taxonomy else 0) + 1

    def __call__(self, samples):
        batch = default_collate(samples)

        species_ids = batch[1]
        valid_mask = batch[self._mask_index]

        masked_ids, mlm_mask, mlm_targets = mask_species_batch(
            species_ids, valid_mask.bool(), self.n_species, self.mask_prob
        )

        batch_list = list(batch)
        batch_list[1] = masked_ids
        batch_list.append(mlm_mask)
        batch_list.append(mlm_targets)
        return tuple(batch_list)
