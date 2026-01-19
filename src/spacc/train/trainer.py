"""Trainer: training orchestration for SpaccModel."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from spacc.data.dataset import SpaccDataset
from spacc.encode.species import SpeciesEncoder
from spacc.model.spacc import SpaccModel
from spacc.train.loss import MultiTaskLoss
from spacc.train.metrics import compute_metrics


@dataclass
class TrainResult:
    """Results from training."""

    best_epoch: int
    final_metrics: dict[str, dict[str, float]]
    history: dict[str, list[float]] = field(default_factory=dict)


class Trainer:
    """
    Trains SpaccModel with phased loss schedule.

    Handles:
        - Data preprocessing (encoding, scaling)
        - Training loop with early stopping
        - Checkpointing
        - Evaluation
    """

    def __init__(
        self,
        model: SpaccModel,
        dataset: SpaccDataset,
        batch_size: int = 4096,
        max_epochs: int = 500,
        patience: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        phase_boundaries: tuple[int, int] = (100, 300),
        device: str = "auto",
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.phase_boundaries = phase_boundaries

        # Device selection
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Components to be initialized in fit()
        self._species_encoder: Optional[SpeciesEncoder] = None
        self._scalers: dict[str, StandardScaler] = {}
        self._target_scalers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        self._optimizer: Optional[AdamW] = None
        self._scheduler: Optional[OneCycleLR] = None
        self._loss_fn: Optional[MultiTaskLoss] = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _prepare_data(self) -> tuple[SpaccDataset, SpaccDataset]:
        """Split and encode data."""
        train_ds, test_ds = self.dataset.split(test_size=0.2)

        # Fit species encoder on training data
        self._species_encoder = SpeciesEncoder(
            hash_dim=self.model.hash_dim,
            top_k=self.model.top_k,
        )
        self._species_encoder.fit(train_ds)

        return train_ds, test_ds

    def _build_tensors(
        self,
        dataset: SpaccDataset,
        fit_scalers: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Convert dataset to tensors."""
        # Encode species
        encoded = self._species_encoder.transform(dataset)

        # Get continuous features
        coords = dataset.get_coordinates()
        covariates = dataset.get_covariates()

        # Combine: coords + covariates + hash_embedding
        if covariates is not None:
            continuous = np.hstack([coords, covariates, encoded.hash_embedding])
        else:
            continuous = np.hstack([coords, encoded.hash_embedding])

        # Scale continuous features
        if fit_scalers:
            self._scalers["continuous"] = StandardScaler()
            continuous = self._scalers["continuous"].fit_transform(continuous)
        else:
            continuous = self._scalers["continuous"].transform(continuous)

        continuous = continuous.astype(np.float32)

        # Get targets
        targets = {}
        for name, cfg in self.model.target_configs.items():
            target_vals = dataset.get_target(name)
            mask = dataset.get_target_mask(name)

            if cfg.task == "regression":
                if fit_scalers:
                    self._scalers[f"target_{name}"] = StandardScaler()
                    target_scaled = self._scalers[f"target_{name}"].fit_transform(
                        target_vals[mask].reshape(-1, 1)
                    )
                    # Store scaler params for loss computation
                    scaler = self._scalers[f"target_{name}"]
                    self._target_scalers[name] = (
                        torch.tensor(scaler.mean_[0], dtype=torch.float32, device=self._device),
                        torch.tensor(scaler.scale_[0], dtype=torch.float32, device=self._device),
                    )
                else:
                    target_scaled = self._scalers[f"target_{name}"].transform(
                        target_vals.reshape(-1, 1)
                    )
                targets[name] = target_scaled.flatten().astype(np.float32)
            else:
                targets[name] = target_vals.astype(np.int64)

        # Build tensor dataset
        tensors = [torch.from_numpy(continuous)]

        if encoded.genus_ids is not None:
            tensors.append(torch.from_numpy(encoded.genus_ids))
        if encoded.family_ids is not None:
            tensors.append(torch.from_numpy(encoded.family_ids))

        for name in self.model.target_configs.keys():
            tensors.append(torch.from_numpy(targets[name]))

        return tuple(tensors)

    def _create_loaders(
        self,
        train_tensors: tuple[torch.Tensor, ...],
        test_tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Create data loaders."""
        train_ds = TensorDataset(*train_tensors)
        test_ds = TensorDataset(*test_tensors)

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self._device.type == "cuda",
        )
        self._test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self._device.type == "cuda",
        )

    def fit(self) -> TrainResult:
        """
        Train the model.

        Returns:
            TrainResult with metrics and history
        """
        # Prepare data
        train_ds, test_ds = self._prepare_data()
        train_tensors = self._build_tensors(train_ds, fit_scalers=True)
        test_tensors = self._build_tensors(test_ds, fit_scalers=False)
        self._create_loaders(train_tensors, test_tensors)

        # Move model to device
        self.model.to(self._device)

        # Setup optimizer and scheduler
        self._optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = self.max_epochs * len(self._train_loader)
        self._scheduler = OneCycleLR(
            self._optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        # Setup loss
        self._loss_fn = MultiTaskLoss(
            self.model.target_configs,
            phase_boundaries=self.phase_boundaries,
        )

        # Training loop
        best_metric = -float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = {"train_loss": [], "test_loss": []}

        target_names = list(self.model.target_configs.keys())
        has_taxonomy = self.model.schema.has_taxonomy

        for epoch in range(self.max_epochs):
            # Train
            train_loss = self._train_epoch(epoch, target_names, has_taxonomy)
            history["train_loss"].append(train_loss)

            # Evaluate
            test_loss, metrics = self._eval_epoch(epoch, target_names, has_taxonomy)
            history["test_loss"].append(test_loss)

            # Track best by first regression target's band_25 or classification accuracy
            first_target = target_names[0]
            cfg = self.model.target_configs[first_target]
            if cfg.task == "regression":
                current_metric = metrics[first_target]["band_25"]
            else:
                current_metric = metrics[first_target]["accuracy"]

            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            # Log progress
            phase = self._loss_fn.phased_loss.get_phase(epoch)
            metric_str = " | ".join(
                f"{name}: {metrics[name].get('band_25', metrics[name].get('accuracy', 0)):.2%}"
                for name in target_names
            )
            print(
                f"Epoch {epoch:3d} [P{phase}] | "
                f"train={train_loss:.4f} test={test_loss:.4f} | {metric_str}"
            )

            # Early stopping
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        self.model.load_state_dict(self._best_state)

        # Final evaluation
        _, final_metrics = self._eval_epoch(best_epoch, target_names, has_taxonomy)

        return TrainResult(
            best_epoch=best_epoch,
            final_metrics=final_metrics,
            history=history,
        )

    def _train_epoch(
        self,
        epoch: int,
        target_names: list[str],
        has_taxonomy: bool,
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in self._train_loader:
            # Unpack batch
            idx = 0
            continuous = batch[idx].to(self._device)
            idx += 1

            if has_taxonomy:
                genus_ids = batch[idx].to(self._device)
                idx += 1
                family_ids = batch[idx].to(self._device)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx].to(self._device)
                idx += 1

            # Forward
            self._optimizer.zero_grad()
            predictions = self.model(continuous, genus_ids, family_ids)

            # Reshape predictions for loss
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            # Compute loss
            loss, _ = self._loss_fn(
                predictions, targets, epoch, self._target_scalers
            )

            # Backward
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

            total_loss += loss.item() * continuous.size(0)

        return total_loss / len(self._train_loader.dataset)

    @torch.no_grad()
    def _eval_epoch(
        self,
        epoch: int,
        target_names: list[str],
        has_taxonomy: bool,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0

        all_preds = {name: [] for name in target_names}
        all_targets = {name: [] for name in target_names}

        for batch in self._test_loader:
            idx = 0
            continuous = batch[idx].to(self._device)
            idx += 1

            if has_taxonomy:
                genus_ids = batch[idx].to(self._device)
                idx += 1
                family_ids = batch[idx].to(self._device)
                idx += 1
            else:
                genus_ids = None
                family_ids = None

            targets = {}
            for name in target_names:
                targets[name] = batch[idx].to(self._device)
                idx += 1

            predictions = self.model(continuous, genus_ids, family_ids)

            # Reshape for loss
            targets_for_loss = {}
            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets_for_loss[name] = targets[name].unsqueeze(-1)
                else:
                    targets_for_loss[name] = targets[name]

            loss, _ = self._loss_fn(
                predictions, targets_for_loss, epoch, self._target_scalers
            )
            total_loss += loss.item() * continuous.size(0)

            # Collect predictions
            for name in target_names:
                cfg = self.model.target_configs[name]
                pred = predictions[name]

                if cfg.task == "regression":
                    # Inverse scale
                    scaler = self._scalers[f"target_{name}"]
                    pred_np = pred.cpu().numpy()
                    pred_np = scaler.inverse_transform(pred_np).flatten()
                    target_np = scaler.inverse_transform(
                        targets[name].cpu().numpy().reshape(-1, 1)
                    ).flatten()
                else:
                    pred_np = pred.argmax(dim=-1).cpu().numpy()
                    target_np = targets[name].cpu().numpy()

                all_preds[name].append(pred_np)
                all_targets[name].append(target_np)

        avg_loss = total_loss / len(self._test_loader.dataset)

        # Compute metrics
        metrics = {}
        for name in target_names:
            cfg = self.model.target_configs[name]
            pred = np.concatenate(all_preds[name])
            target = np.concatenate(all_targets[name])
            metrics[name] = compute_metrics(pred, target, cfg.task, cfg.transform)

        return avg_loss, metrics

    def save(self, path: str | Path) -> None:
        """Save model, encoder, and scalers."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "schema": self.model.schema,
            "target_configs": self.model.target_configs,
            "hash_dim": self.model.hash_dim,
            "top_k": self.model.top_k,
            "hidden_dims": self.model.hidden_dims,
            "genus_emb_dim": self.model.genus_emb_dim,
            "family_emb_dim": self.model.family_emb_dim,
            "dropout": self.model.dropout,
            "scalers": self._scalers,
            "vocab": self._species_encoder.vocab if self._species_encoder else None,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> tuple[SpaccModel, SpeciesEncoder, dict]:
        """
        Load model from checkpoint.

        Returns:
            (model, species_encoder, scalers)
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        model = SpaccModel(
            schema=state["schema"],
            targets=state["target_configs"],
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
            hidden_dims=state.get("hidden_dims"),
            genus_emb_dim=state.get("genus_emb_dim", 8),
            family_emb_dim=state.get("family_emb_dim", 8),
            dropout=state.get("dropout", 0.3),
        )
        model.load_state_dict(state["model_state_dict"])

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        model.to(dev)

        encoder = SpeciesEncoder(
            hash_dim=state["hash_dim"],
            top_k=state["top_k"],
        )
        if state["vocab"] is not None:
            encoder._vocab = state["vocab"]
            encoder._fitted = True

        return model, encoder, state["scalers"]
