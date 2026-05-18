"""Checkpointing mixin for Trainer.

Handles saving and loading training checkpoints for resumable training,
including model state, optimizer state, scalers, and encoder state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from resolve.encode.species import SpeciesEncoder

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class CheckpointMixin:
    """Mixin providing checkpoint save/load methods for Trainer."""

    def _checkpoint_path(self: Trainer) -> Optional[Path]:
        """Get path to checkpoint file."""
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "checkpoint.pt"

    def _progress_path(self: Trainer) -> Optional[Path]:
        """Get path to progress JSON file (human-readable)."""
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "progress.json"

    def save_checkpoint(
        self: Trainer,
        epoch: int,
        best_epoch: int,
        best_metric: float,
        epochs_without_improvement: int,
        history: dict,
        *,
        completed: bool = False,
    ) -> None:
        """Save training checkpoint for resume.

        ``completed=True`` marks the checkpoint as a final post-training save,
        so that a subsequent ``fit()`` call recognizes the job as done and
        fast-returns instead of re-entering the training loop.
        """
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            # Training state
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "completed": completed,
            # Model state
            "model_state_dict": self.model.state_dict(),
            "best_state": self._best_state,
            # Optimizer state
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "grad_scaler_state_dict": self._grad_scaler.state_dict() if self._grad_scaler else None,
            # Data state
            "scalers": self._scalers,
            "target_scalers": {
                k: (v[0].cpu(), v[1].cpu()) for k, v in self._target_scalers.items()
            },
            # Species encoder state
            "species_encoder": {
                "vocab": self._species_encoder._vocab if self._species_encoder else None,
                "species_vocab": self._species_encoder._species_vocab if self._species_encoder else set(),
            },
            # Categorical vocab state
            "categorical_vocabs": self._categorical_vocabs,
            # EMA state (exponential moving average of model weights)
            "ema_state": (
                {k: v.cpu().clone() for k, v in self._ema_state.items()}
                if self._ema_state is not None
                else None
            ),
            # Config (for validation on resume)
            "config": {
                "hash_dim": self.hash_dim,
                "top_k": self.top_k,
                "hidden_dims": self.hidden_dims,
                "max_epochs": self.max_epochs,
                "batch_size": self.batch_size,
                "species_encoding": self.species_encoding,
                "species_selection": self.species_selection,
                "species_representation": self.species_representation,
                "genus_emb_dim": self.genus_emb_dim,
                "family_emb_dim": self.family_emb_dim,
                "n_attention_layers": self.n_attention_layers,
                "n_heads": self.n_heads,
                "transformer_ff_dim": self.transformer_ff_dim,
                "transformer_pooling": self.transformer_pooling,
                "transformer_dropout": self.transformer_dropout,
            },
        }

        # Save checkpoint
        torch.save(checkpoint, self._checkpoint_path())

        # Save human-readable progress
        progress = {
            "epoch": epoch,
            "max_epochs": self.max_epochs,
            "best_epoch": best_epoch,
            "best_metric": float(best_metric),
            "epochs_without_improvement": epochs_without_improvement,
            "patience": self.patience,
            "progress_pct": round(100 * epoch / self.max_epochs, 1),
        }
        # Add latest metrics from history
        if history.get("test_loss"):
            progress["latest_test_loss"] = float(history["test_loss"][-1])
        with open(self._progress_path(), "w") as f:
            json.dump(progress, f, indent=2)

        print(f"  [Checkpoint saved: epoch {epoch}, best={best_metric:.2%}]")

    def load_checkpoint(self: Trainer) -> Optional[dict]:
        """Load checkpoint if exists and resume=True."""
        if not self.resume or self.checkpoint_dir is None:
            return None

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            return None

        print(f"Loading checkpoint from {checkpoint_path}")
        # Note: weights_only=False is required for sklearn scalers and encoder state.
        # Only load checkpoint files from trusted sources.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Validate config matches - architecture parameters must match exactly
        saved_config = checkpoint.get("config", {})
        config_mismatches = []

        # Critical architecture parameters
        if saved_config.get("hash_dim") != self.hash_dim:
            config_mismatches.append(
                f"hash_dim: checkpoint={saved_config.get('hash_dim')}, current={self.hash_dim}"
            )
        if saved_config.get("hidden_dims") != self.hidden_dims:
            config_mismatches.append(
                f"hidden_dims: checkpoint={saved_config.get('hidden_dims')}, current={self.hidden_dims}"
            )
        if saved_config.get("top_k") != self.top_k:
            config_mismatches.append(
                f"top_k: checkpoint={saved_config.get('top_k')}, current={self.top_k}"
            )
        if saved_config.get("species_encoding") != self.species_encoding:
            config_mismatches.append(
                f"species_encoding: checkpoint={saved_config.get('species_encoding')}, current={self.species_encoding}"
            )
        if saved_config.get("species_selection") != self.species_selection:
            config_mismatches.append(
                f"species_selection: checkpoint={saved_config.get('species_selection')}, current={self.species_selection}"
            )

        if config_mismatches:
            print("  Warning: Cannot resume - configuration mismatch:")
            for mismatch in config_mismatches:
                print(f"    - {mismatch}")
            print("  Starting fresh training run.")
            return None

        return checkpoint

    def _restore_scalers_from_checkpoint(self: Trainer, checkpoint: dict) -> None:
        """Restore scalers and species encoder from checkpoint (before building tensors)."""
        # Restore scalers
        if checkpoint.get("scalers"):
            self._scalers = checkpoint["scalers"]
        if checkpoint.get("target_scalers"):
            self._target_scalers = {
                k: (v[0].to(self._device), v[1].to(self._device))
                for k, v in checkpoint["target_scalers"].items()
            }

        # Restore categorical vocabs
        if checkpoint.get("categorical_vocabs"):
            self._categorical_vocabs = checkpoint["categorical_vocabs"]

        # Restore species encoder state (hash mode only; rank_pool/embed use different encoders)
        if checkpoint.get("species_encoder") and self.species_encoding == "hash":
            enc_state = checkpoint["species_encoder"]
            # Create encoder if not exists
            if self._species_encoder is None:
                self._species_encoder = SpeciesEncoder(
                    hash_dim=self.hash_dim,
                    top_k=self.top_k,
                    aggregation=self.species_aggregation,
                    normalization=self.species_normalization,
                    track_unknown_count=self.track_unknown_count,
                    selection=self.species_selection,
                    representation=self.species_representation,
                )
            if enc_state.get("vocab"):
                self._species_encoder._vocab = enc_state["vocab"]
            if enc_state.get("species_vocab"):
                self._species_encoder._species_vocab = enc_state["species_vocab"]
            self._species_encoder._fitted = True

    def _restore_from_checkpoint(
        self: Trainer,
        checkpoint: dict,
    ) -> tuple[int, int, float, int, dict]:
        """Restore training state from checkpoint (model, optimizer, etc.)."""
        # Restore model
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("best_state"):
            self._best_state = checkpoint["best_state"]

        # Restore optimizer (scheduler state is restored separately in fit())
        if checkpoint.get("optimizer_state_dict") and self._optimizer:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("grad_scaler_state_dict") and self._grad_scaler:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

        # Restore EMA state if available
        if checkpoint.get("ema_state") is not None:
            self._ema_state = {
                k: v.to(self._device) for k, v in checkpoint["ema_state"].items()
            }

        # Note: Scalers already restored by _restore_scalers_from_checkpoint (called earlier)

        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_metric = checkpoint["best_metric"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        history = checkpoint["history"]

        print(f"  Resumed from epoch {epoch} (best={best_metric:.2%} at epoch {best_epoch})")

        return epoch, best_epoch, best_metric, epochs_without_improvement, history
