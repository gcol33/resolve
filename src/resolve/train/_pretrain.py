"""Pretraining mixin for Trainer.

Implements masked species pretraining (MLM) for transformer-based encoders.
The encoder learns species representations via BERT-style masked prediction
before supervised fine-tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from resolve.train._loaders import RankPoolBatchDataset, _rank_pool_collate_fn

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class PretrainMixin:
    """Mixin providing masked species pretraining for Trainer."""

    def pretrain(self: Trainer) -> None:
        """Run masked species pretraining (v6) on the transformer encoder.

        Trains the encoder with BERT-style masked language modelling over species
        tokens. Uses a separate MaskedSpeciesHead (discarded after pretraining).

        Must be called BEFORE fit(). Requires species_encoding="transformer" and
        pretrain_epochs > 0.

        The method:
          1. Prepares data (same pipeline as rank_pool)
          2. Builds model if not already built
          3. Runs MLM pretraining loop
          4. Discards the MaskedSpeciesHead, keeps encoder weights
        """
        if self.species_encoding != "transformer":
            raise ValueError("pretrain() requires species_encoding='transformer'")
        if self.pretrain_epochs < 1:
            raise ValueError("pretrain() requires pretrain_epochs >= 1")

        from resolve.model.pretrain import MaskedSpeciesHead, MaskedSpeciesCollateWrapper

        print("\n=== Masked Species Pretraining (v6) ===")
        print(f"  Epochs: {self.pretrain_epochs}")
        print(f"  Mask prob: {self.pretrain_mask_prob}")
        print(f"  LR: {self.pretrain_lr}")
        print(f"  All data: {self.pretrain_all_data}")

        if self.pretrain_all_data:
            # Fit encoder on full Trainer dataset (no train/test split)
            # No label leakage: pretraining is unsupervised (MLM)
            from dataclasses import replace
            from resolve.encode.rank_pool import RankPoolEncoder

            self._rank_pool_encoder = RankPoolEncoder(
                weighting=self.species_normalization,
                min_species_frequency=self.min_species_frequency,
            )
            self._rank_pool_encoder.fit(self.dataset)

            # Update schema with vocab sizes
            self._schema = replace(
                self._schema,
                n_species_vocab=self._rank_pool_encoder.n_species,
                n_genera_vocab=self._rank_pool_encoder.n_genera,
                n_families_vocab=self._rank_pool_encoder.n_families,
            )
            self._pretrain_fitted_encoder = True

            print(f"  Encoder fitted on full dataset: {len(self.dataset.plot_ids):,} plots")
            print(f"  Vocab: {self._rank_pool_encoder.n_species:,} species, "
                  f"{self._rank_pool_encoder.n_genera:,} genera, "
                  f"{self._rank_pool_encoder.n_families:,} families")

            # Build model with updated schema
            if self.model is None:
                self.model = self._build_model()
            self.model.to(self._device)

            # Build tensors from full dataset
            train_tensors = self._build_tensors(self.dataset, fit_scalers=True)
        else:
            # Standard: fit encoder on train split only
            train_ds, _ = self._prepare_data(fit_encoder=True)

            # Build model if not done yet
            if self.model is None:
                self.model = self._build_model()
            self.model.to(self._device)

            # Build tensors from train split
            train_tensors = self._build_tensors(train_ds, fit_scalers=True)
        has_taxonomy = self._schema.has_taxonomy

        # Create masking collate wrapper
        mlm_collate = MaskedSpeciesCollateWrapper(
            base_collate_fn=_rank_pool_collate_fn,
            n_species=self._schema.n_species_vocab,
            mask_prob=self.pretrain_mask_prob,
            has_taxonomy=has_taxonomy,
        )

        pretrain_loader = DataLoader(
            RankPoolBatchDataset(train_tensors),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=mlm_collate,
            drop_last=True,
        )

        # Build MLM head (will be discarded after pretraining)
        encoder = self.model.encoder
        mlm_head = MaskedSpeciesHead(encoder.d_model, self._schema.n_species_vocab)
        mlm_head.to(self._device)

        # Optimizer for encoder + MLM head only (not task heads)
        pretrain_params = list(encoder.parameters()) + list(mlm_head.parameters())
        optimizer = AdamW(pretrain_params, lr=self.pretrain_lr, weight_decay=self.weight_decay)

        # AMP scaler
        grad_scaler = GradScaler() if self.use_amp else None

        from resolve.csrc.fused_linear_ce import fused_linear_cross_entropy

        for epoch in range(1, self.pretrain_epochs + 1):
            encoder.train()
            mlm_head.train()
            total_loss = 0.0
            n_batches = 0

            for batch in pretrain_loader:
                # Unpack batch: (continuous, masked_sp, [g, f,] w, mask, has_cover, *targets, mlm_mask, mlm_targets)
                idx = 0
                continuous = batch[idx].to(self._device, non_blocking=True); idx += 1
                species_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                if has_taxonomy:
                    pool_genus_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                    pool_family_ids = batch[idx].to(self._device, non_blocking=True); idx += 1
                else:
                    pool_genus_ids = None
                    pool_family_ids = None
                pool_weights = batch[idx].to(self._device, non_blocking=True); idx += 1
                pool_mask = batch[idx].to(self._device, non_blocking=True); idx += 1
                pool_has_cover = batch[idx].to(self._device, non_blocking=True); idx += 1

                # Skip targets (we don't need them for pretraining)
                n_targets = len(self.model.target_configs)
                idx += n_targets

                mlm_mask = batch[idx].to(self._device, non_blocking=True); idx += 1
                mlm_targets = batch[idx].to(self._device, non_blocking=True); idx += 1

                # Forward through encoder to get token-level representations
                # We need the pre-pooling token embeddings, not the pooled output
                # Re-run embedding + transformer without pooling
                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast(device_type="cuda"):
                        token_embs = self._get_pretrain_tokens(
                            encoder, continuous, species_ids, pool_genus_ids,
                            pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                            mlm_mask,
                        )
                        # Fused linear + cross-entropy: avoids materializing (N, V) logits
                        masked_embs = token_embs[mlm_mask]  # (N_masked, d_model)
                        targets = mlm_targets[mlm_mask]  # (N_masked,)
                        loss = fused_linear_cross_entropy(
                            masked_embs, mlm_head.proj.weight, targets,
                            ignore_index=0, label_smoothing=self.label_smoothing,
                        )

                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    token_embs = self._get_pretrain_tokens(
                        encoder, continuous, species_ids, pool_genus_ids,
                        pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                        mlm_mask,
                    )
                    masked_embs = token_embs[mlm_mask]
                    targets = mlm_targets[mlm_mask]
                    loss = fused_linear_cross_entropy(
                        masked_embs, mlm_head.proj.weight, targets,
                        ignore_index=0, label_smoothing=self.label_smoothing,
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if self.verbose >= 1:
                print(f"  Pretrain epoch {epoch}/{self.pretrain_epochs}: MLM loss = {avg_loss:.4f}")

        # Discard MLM head, keep encoder weights
        del mlm_head
        print("  Pretraining complete. MLM head discarded, encoder weights retained.")

    @staticmethod
    def _get_pretrain_tokens(
        encoder, continuous, species_ids, genus_ids, family_ids,
        weights, mask, has_cover, mlm_mask,
    ) -> torch.Tensor:
        """Get token-level embeddings from PlotEncoderTransformer (pre-pooling).

        Runs the embedding + self-attention layers but skips pooling and MLP,
        returning per-token representations for MLM prediction.
        """
        batch_size = continuous.shape[0]

        if has_cover is None:
            has_cover = torch.ones(batch_size, device=continuous.device)

        if mask is None:
            mask = species_ids != 0

        # Additive token embeddings
        tokens = encoder.species_embedding(species_ids)
        if encoder.has_taxonomy and genus_ids is not None and family_ids is not None:
            tokens = tokens + encoder.genus_embedding(genus_ids) + encoder.family_embedding(family_ids)
        if weights is not None:
            tokens = tokens + encoder.weight_proj(weights.unsqueeze(-1))

        # Apply mask embedding at MLM positions
        tokens = tokens.clone()
        tokens[mlm_mask] = encoder.mask_embedding

        # Self-attention
        padding_mask = ~mask
        if encoder.transformer_encoder is not None:
            tokens = encoder.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        return tokens
