"""Profiling mixin for Trainer.

Provides detailed timing breakdown of training batches to identify
performance bottlenecks (data loading, forward pass, backward pass,
optimizer step).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from resolve.train._loaders import CUDAPrefetcher
from resolve.train._types import ProfileResult, Timer
from resolve.train.loss import MultiTaskLoss

if TYPE_CHECKING:
    from resolve.train.trainer import Trainer

__all__: list[str] = []


class ProfilingMixin:
    """Mixin providing training profiling methods for Trainer."""

    def profile(
        self: Trainer,
        n_batches: int = 50,
        warmup_batches: int = 5,
        save_trace: bool = False,
        trace_dir: str | Path | None = None,
    ) -> ProfileResult:
        """Profile training performance to identify bottlenecks.

        Runs a small number of training batches with detailed timing,
        breaking down time spent in forward pass, backward pass,
        optimizer step, and data loading.

        Args:
            n_batches: Number of batches to profile (after warmup).
            warmup_batches: Number of warmup batches to run first (not timed).
            save_trace: If True, save detailed Chrome trace for analysis.
            trace_dir: Directory to save trace files. Default: ./profiles/

        Returns:
            ProfileResult with timing breakdown.

        Example:
            >>> trainer = Trainer(dataset)
            >>> result = trainer.profile(n_batches=100)
            >>> print(result)
        """
        # Ensure model and data are ready
        if self.model is None or self._train_loader is None:
            # Do data prep without full training
            print("Preparing data for profiling...")
            checkpoint = self.load_checkpoint()

            t_prep_start = time.time()
            data_cache = self._load_cache()

            if data_cache is not None:
                train_tensors, test_tensors = self._restore_from_cache(data_cache)
            else:
                if checkpoint is not None:
                    self._restore_scalers_from_checkpoint(checkpoint)
                train_ds, test_ds = self._prepare_data(fit_encoder=(checkpoint is None))

                self._ensure_model()

                train_tensors = self._build_tensors(train_ds, fit_scalers=(checkpoint is None))
                test_tensors = self._build_tensors(test_ds, fit_scalers=False)

            self._create_loaders(train_tensors, test_tensors)

            self._ensure_model()
            self.model.to(self._device)
            print(f"  Data prepared in {time.time() - t_prep_start:.1f}s")

        # Setup optimizer if not already
        if self._optimizer is None:
            self._optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Setup loss if not already
        if self._loss_fn is None:
            self._loss_fn = MultiTaskLoss(
                self.model.target_configs,
                phases=self.phases,
                phase_boundaries=self.phase_boundaries,
                label_smoothing=self.label_smoothing,
                class_weights=self.class_weights,
            )

        # Setup AMP
        if self.use_amp and self._grad_scaler is None:
            self._grad_scaler = GradScaler()

        # Track GPU memory
        gpu_memory_peak = 0.0
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

        self.model.train()
        timer = Timer()
        target_names = list(self.model.target_configs.keys())
        has_taxonomy = self.model.schema.has_taxonomy

        # Determine if data is already on GPU (from GPUTensorLoader or CUDAPrefetcher)
        use_gpu_loader = getattr(self, "_using_gpu_loader", False)
        use_prefetch = self.prefetch_data and self._device.type == "cuda" and not use_gpu_loader
        data_on_device = use_gpu_loader or use_prefetch

        # Choose loader
        if use_prefetch:
            loader = CUDAPrefetcher(self._train_loader, self._device)
        else:
            loader = self._train_loader

        total_samples = 0
        batch_count = 0

        # Warmup (not timed)
        print(f"Warming up ({warmup_batches} batches)...")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= warmup_batches:
                break

            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
             categorical_ids, targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)

            self._profile_step(
                continuous, genus_ids, family_ids, species_ids, species_vector,
                pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                categorical_ids, targets,
            )

        # Profile batches
        print(f"Profiling ({n_batches} batches)...")
        # Re-initialize loader for profiling
        if use_prefetch:
            loader = CUDAPrefetcher(self._train_loader, self._device)
        else:
            loader = self._train_loader

        timer.start("total")
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break

            # Data loading timing
            timer.start("data")
            (continuous, genus_ids, family_ids, species_ids, species_vector,
             pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
             categorical_ids, targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

            for name in target_names:
                cfg = self.model.target_configs[name]
                if cfg.task == "regression":
                    targets[name] = targets[name].unsqueeze(-1)
            timer.stop("data")

            # Forward pass
            self._optimizer.zero_grad(set_to_none=True)
            timer.start("forward")
            if self.use_amp:
                with autocast(device_type="cuda"):
                    predictions = self.model(
                        continuous, genus_ids, family_ids, species_ids, species_vector,
                        pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                        pool_weights=pool_weights, pool_mask=pool_mask,
                        pool_has_cover=pool_has_cover,
                        categorical_ids=categorical_ids,
                    )
                    loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            else:
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                    categorical_ids=categorical_ids,
                )
                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            timer.stop("forward")

            # Backward pass
            timer.start("backward")
            if self.use_amp:
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            timer.stop("backward")

            # Optimizer step
            timer.start("optimizer")
            if self.use_amp:
                self._grad_scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optimizer.step()
            timer.stop("optimizer")

            total_samples += continuous.size(0)
            batch_count += 1

        timer.stop("total")

        # Get GPU memory stats
        if self._device.type == "cuda":
            gpu_memory_peak = torch.cuda.max_memory_allocated(self._device) / (1024 * 1024)  # MB

        # Build result
        total_time = timer.get("total")
        result = ProfileResult(
            total_time_ms=total_time,
            forward_time_ms=timer.get("forward"),
            backward_time_ms=timer.get("backward"),
            optimizer_time_ms=timer.get("optimizer"),
            data_time_ms=timer.get("data"),
            n_batches=batch_count,
            avg_batch_time_ms=total_time / batch_count if batch_count > 0 else 0,
            samples_per_second=total_samples / (total_time / 1000) if total_time > 0 else 0,
            gpu_memory_peak_mb=gpu_memory_peak,
        )

        # Optional: save torch.profiler trace
        if save_trace:
            result = self._save_profile_trace(
                result, trace_dir, target_names, has_taxonomy,
                use_prefetch, data_on_device,
            )

        return result

    def _profile_step(
        self: Trainer,
        continuous: torch.Tensor,
        genus_ids: torch.Tensor | None,
        family_ids: torch.Tensor | None,
        species_ids: torch.Tensor | None,
        species_vector: torch.Tensor | None,
        pool_genus_ids: torch.Tensor | None,
        pool_family_ids: torch.Tensor | None,
        pool_weights: torch.Tensor | None,
        pool_mask: torch.Tensor | None,
        pool_has_cover: torch.Tensor | None,
        categorical_ids: torch.Tensor | None,
        targets: dict[str, torch.Tensor],
    ) -> None:
        """Run a single forward+backward+optimizer step (used by warmup and trace)."""
        self._optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            with autocast(device_type="cuda"):
                predictions = self.model(
                    continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                    pool_weights=pool_weights, pool_mask=pool_mask,
                    pool_has_cover=pool_has_cover,
                    categorical_ids=categorical_ids,
                )
                loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.step(self._optimizer)
            self._grad_scaler.update()
        else:
            predictions = self.model(
                continuous, genus_ids, family_ids, species_ids, species_vector,
                pool_genus_ids=pool_genus_ids, pool_family_ids=pool_family_ids,
                pool_weights=pool_weights, pool_mask=pool_mask,
                pool_has_cover=pool_has_cover,
                categorical_ids=categorical_ids,
            )
            loss, _ = self._loss_fn(predictions, targets, 0, self._target_scalers)
            loss.backward()
            self._optimizer.step()

    def _save_profile_trace(
        self: Trainer,
        result: ProfileResult,
        trace_dir: str | Path | None,
        target_names: list[str],
        has_taxonomy: bool,
        use_prefetch: bool,
        data_on_device: bool,
    ) -> ProfileResult:
        """Save a torch.profiler Chrome trace and return updated ProfileResult."""
        trace_path = Path(trace_dir) if trace_dir else Path("./profiles")
        trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path / f"profile_{time.strftime('%Y%m%d_%H%M%S')}.json"

        try:
            from torch.profiler import profile as torch_profile, ProfilerActivity

            print(f"Saving detailed trace to {trace_file}...")
            with torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                # Run a few batches for trace
                loader = CUDAPrefetcher(self._train_loader, self._device) if use_prefetch else self._train_loader
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= 10:
                        break

                    (continuous, genus_ids, family_ids, species_ids, species_vector,
                     pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                     categorical_ids, targets) = self._unpack_batch(batch, target_names, has_taxonomy, data_on_device)

                    for name in target_names:
                        cfg = self.model.target_configs[name]
                        if cfg.task == "regression":
                            targets[name] = targets[name].unsqueeze(-1)

                    self._profile_step(
                        continuous, genus_ids, family_ids, species_ids, species_vector,
                        pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                        categorical_ids, targets,
                    )

            prof.export_chrome_trace(str(trace_file))
            return ProfileResult(
                total_time_ms=result.total_time_ms,
                forward_time_ms=result.forward_time_ms,
                backward_time_ms=result.backward_time_ms,
                optimizer_time_ms=result.optimizer_time_ms,
                data_time_ms=result.data_time_ms,
                n_batches=result.n_batches,
                avg_batch_time_ms=result.avg_batch_time_ms,
                samples_per_second=result.samples_per_second,
                gpu_memory_peak_mb=result.gpu_memory_peak_mb,
                detailed_trace_path=str(trace_file),
            )
        except ImportError:
            print("  Warning: torch.profiler not available, skipping trace")
            return result
