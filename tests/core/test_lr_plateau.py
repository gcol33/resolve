"""The reduce-on-plateau learning-rate schedule through the nanobind bindings.

``LRSchedulerType.ReduceOnPlateau`` is PyTorch's ``ReduceLROnPlateau`` in mode
``min`` on the held-out fold's selection loss. These cases replay a run's own
held-out losses through ``torch.optim.lr_scheduler.ReduceLROnPlateau`` and
require the rates the engine trained at to be the ones PyTorch sets, and pin
the refusal without a held-out fold, the checkpointed knobs and resume.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


def _f32(value: float) -> float:
    """``value`` as the engine's float32 config field holds it."""
    return torch.tensor(value, dtype=torch.float32).item()


def _plateau_config(*, max_epochs: int, threshold: float, **extra) -> "rc.TrainConfig":
    config = make_train_config(max_epochs=max_epochs, lr=5e-2)
    config.loss_config = rc.LossConfigMode.MAE
    config.lr_scheduler = rc.LRSchedulerType.ReduceOnPlateau
    config.lr_plateau_factor = 0.5
    config.lr_plateau_patience = 1
    config.lr_plateau_threshold = threshold
    config.lr_plateau_cooldown = 1
    config.lr_min = 1e-3
    for name, value in extra.items():
        setattr(config, name, value)
    return config


def _trainer(dataset, config, seed: int = 3) -> "rc.Trainer":
    torch.manual_seed(seed)
    trainer = rc.Trainer(rc.ResolveModel(dataset.schema, make_model_config()), config)
    trainer.prepare_data(dataset, 0.25, 42)
    return trainer


def _torch_rates(config: "rc.TrainConfig", losses: list[float]) -> list[float]:
    """The rate each epoch trains at when ``losses`` drive ReduceLROnPlateau."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=_f32(config.lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=_f32(config.lr_plateau_factor),
        patience=config.lr_plateau_patience,
        threshold=_f32(config.lr_plateau_threshold),
        threshold_mode="rel",
        cooldown=config.lr_plateau_cooldown,
        min_lr=_f32(config.lr_min),
    )
    rates = []
    for loss in losses:
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step(loss)
    return rates


@pytest.mark.parametrize("threshold", [1e-4, 0.999])
def test_the_rates_are_torch_reduce_lr_on_plateau_on_the_held_out_loss(hash_dataset, threshold):
    config = _plateau_config(max_epochs=20, threshold=threshold)
    result = _trainer(hash_dataset, config).fit()

    assert len(result.lr_history) == len(result.test_loss_history)
    expected = _torch_rates(config, result.test_loss_history)
    assert result.lr_history == pytest.approx(expected, rel=1e-6)
    if threshold == 0.999:
        # No epoch can beat its best by 99.9 %, so the rate decays on a fixed
        # rhythm down to the floor.
        assert result.lr_history[0] == pytest.approx(5e-2, rel=1e-6)
        assert result.lr_history[3] == pytest.approx(2.5e-2, rel=1e-6)
        assert result.lr_history[-1] == pytest.approx(1e-3, rel=1e-6)


def test_a_plateau_without_a_held_out_fold_is_refused(hash_dataset):
    config = _plateau_config(max_epochs=3, threshold=0.0)
    config.fixed_epochs = 3
    trainer = rc.Trainer(rc.ResolveModel(hash_dataset.schema, make_model_config()), config)
    with pytest.raises(ValueError, match="no held-out fold"):
        trainer.prepare_data(hash_dataset, 0.0, 42)


def test_an_out_of_range_factor_is_refused(hash_dataset):
    config = _plateau_config(max_epochs=3, threshold=0.0, lr_plateau_factor=1.5)
    trainer = rc.Trainer(rc.ResolveModel(hash_dataset.schema, make_model_config()), config)
    with pytest.raises(ValueError, match="lr_plateau_factor"):
        trainer.prepare_data(hash_dataset, 0.25, 42)


def test_the_plateau_knobs_round_trip_through_the_checkpoint(tmp_path, hash_dataset):
    config = _plateau_config(max_epochs=3, threshold=1e-4, lr_plateau_patience=5,
                             lr_plateau_cooldown=2)
    trainer = _trainer(hash_dataset, config)
    trainer.fit()
    path = str(tmp_path / "plateau.pt")
    trainer.save(path)

    loaded = rc.Trainer.load_train_config(path)
    assert loaded.lr_scheduler == rc.LRSchedulerType.ReduceOnPlateau
    assert loaded.lr_plateau_factor == pytest.approx(0.5)
    assert loaded.lr_plateau_patience == 5
    assert loaded.lr_plateau_threshold == pytest.approx(1e-4)
    assert loaded.lr_plateau_cooldown == 2
    assert loaded.lr_min == pytest.approx(1e-3)


def test_a_plateau_run_resumes_with_its_schedule(tmp_path, hash_dataset):
    config = _plateau_config(max_epochs=10, threshold=0.999)
    config.patience = 50
    config.checkpoint_dir = str(tmp_path / "ckpt")
    config.checkpoint_every = 2

    full = _trainer(hash_dataset, config, seed=1).fit()
    resumed = _trainer(hash_dataset, config, seed=2).resume(
        str(tmp_path / "ckpt" / "checkpoint_4.pt"))

    assert resumed.resumed_from_epoch == 4
    assert resumed.lr_history == full.lr_history
    assert resumed.test_loss_history == full.test_loss_history
    assert resumed.best_epoch == full.best_epoch
