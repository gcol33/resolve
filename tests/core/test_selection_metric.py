"""Selecting a run's epochs by the held-out accuracy through the nanobind bindings.

``TrainConfig.selection_metric = SelectionMetric.Accuracy`` makes early
stopping, ``best_epoch`` and a ``ReduceOnPlateau`` schedule follow the held-out
accuracy of the schema's one classification target, maximized. These cases
replay each run's held-out accuracies (``TrainResult.selection_history``)
through ``torch.optim.lr_scheduler.ReduceLROnPlateau(mode="max")`` and require
the rates the engine trained at to be PyTorch's, the kept epoch to be the
first highest accuracy, and pin the refusals, the checkpointed setting and
resume.
"""

from __future__ import annotations

import math

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


def _f32(value: float) -> float:
    """``value`` as the engine's float32 config field holds it."""
    return torch.tensor(value, dtype=torch.float32).item()


def _accuracy_config(*, max_epochs: int, threshold: float, **extra) -> "rc.TrainConfig":
    config = make_train_config(max_epochs=max_epochs, lr=5e-2)
    config.loss_config = rc.LossConfigMode.MAE
    config.selection_metric = rc.SelectionMetric.Accuracy
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


def _torch_rates(config: "rc.TrainConfig", accuracies: list[float]) -> list[float]:
    """The rate each epoch trains at when ``accuracies`` drive ReduceLROnPlateau."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=_f32(config.lr))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=_f32(config.lr_plateau_factor),
        patience=config.lr_plateau_patience,
        threshold=_f32(config.lr_plateau_threshold),
        threshold_mode="rel",
        cooldown=config.lr_plateau_cooldown,
        min_lr=_f32(config.lr_min),
    )
    rates = []
    for accuracy in accuracies:
        rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step(accuracy)
    return rates


def test_selection_metric_defaults_to_the_loss():
    assert rc.TrainConfig().selection_metric == rc.SelectionMetric.Loss


@pytest.mark.parametrize("threshold", [1e-4, 0.5])
def test_the_rates_are_torch_reduce_lr_on_plateau_in_mode_max_on_the_accuracy(
    hash_dataset, threshold
):
    config = _accuracy_config(max_epochs=20, threshold=threshold, patience=100)
    result = _trainer(hash_dataset, config).fit()

    accuracies = result.selection_history
    assert len(accuracies) == len(result.lr_history) == 20
    assert all(0.0 <= a <= 1.0 for a in accuracies)
    assert accuracies != result.test_loss_history
    expected = _torch_rates(config, accuracies)
    assert result.lr_history == pytest.approx(expected, rel=1e-6)
    # Some epochs fail to beat the best accuracy by the threshold, so the rate
    # decays at least once.
    assert result.lr_history[-1] < result.lr_history[0]


def test_the_kept_epoch_is_the_first_highest_accuracy(hash_dataset):
    config = _accuracy_config(max_epochs=25, threshold=0.0, patience=100)
    config.lr_scheduler = rc.LRSchedulerType.None_
    result = _trainer(hash_dataset, config).fit()

    accuracies = result.selection_history
    best = max(accuracies)
    assert result.best_epoch == accuracies.index(best)
    assert result.final_metrics["hab"]["accuracy"] == pytest.approx(best, abs=0.0)

    by_loss = make_train_config(max_epochs=25, lr=5e-2)
    by_loss.loss_config = rc.LossConfigMode.MAE
    by_loss.patience = 100
    loss_result = _trainer(hash_dataset, by_loss).fit()
    assert loss_result.selection_history == loss_result.test_loss_history
    assert loss_result.test_loss_history == result.test_loss_history
    losses = loss_result.test_loss_history
    assert loss_result.best_epoch == losses.index(min(losses))


def test_early_stopping_follows_the_accuracy(hash_dataset):
    config = _accuracy_config(max_epochs=80, threshold=1e-4, patience=4)
    config.lr_plateau_patience = 2
    result = _trainer(hash_dataset, config).fit()

    accuracies = result.selection_history
    best = -math.inf
    since_best = 0
    stop = len(accuracies)
    for epoch, accuracy in enumerate(accuracies):
        if accuracy > best:
            best = accuracy
            since_best = 0
        else:
            since_best += 1
            if since_best >= config.patience:
                stop = epoch + 1
                break
    assert stop < config.max_epochs
    assert len(accuracies) == stop
    assert result.lr_history == pytest.approx(_torch_rates(config, accuracies), rel=1e-6)


def test_accuracy_without_a_held_out_fold_is_refused(hash_dataset):
    config = _accuracy_config(max_epochs=3, threshold=0.0)
    config.lr_scheduler = rc.LRSchedulerType.None_
    config.fixed_epochs = 3
    trainer = rc.Trainer(rc.ResolveModel(hash_dataset.schema, make_model_config()), config)
    with pytest.raises(ValueError, match="no held-out fold"):
        trainer.prepare_data(hash_dataset, 0.0, 42)


def test_accuracy_without_one_classification_target_is_refused(plot_csvs):
    from conftest import make_dataset_config, make_roles

    regression_only = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        [rc.TargetSpec.regression("y")],
        make_dataset_config(rc.SpeciesEncodingMode.Hash),
    )
    config = _accuracy_config(max_epochs=3, threshold=0.0)
    trainer = rc.Trainer(
        rc.ResolveModel(regression_only.schema, make_model_config()), config
    )
    with pytest.raises(ValueError, match="0 classification targets"):
        trainer.prepare_data(regression_only, 0.25, 42)


def test_the_selection_metric_round_trips_through_the_checkpoint(tmp_path, hash_dataset):
    config = _accuracy_config(max_epochs=3, threshold=1e-4)
    trainer = _trainer(hash_dataset, config)
    trainer.fit()
    path = str(tmp_path / "accuracy.pt")
    trainer.save(path)

    loaded = rc.Trainer.load_train_config(path)
    assert loaded.selection_metric == rc.SelectionMetric.Accuracy


def test_an_accuracy_run_resumes_selecting_by_accuracy(tmp_path, hash_dataset):
    config = _accuracy_config(max_epochs=10, threshold=1e-4, patience=50)
    config.checkpoint_dir = str(tmp_path / "ckpt")
    config.checkpoint_every = 2

    full = _trainer(hash_dataset, config, seed=1).fit()
    checkpoint = str(tmp_path / "ckpt" / "checkpoint_4.pt")
    resumed = _trainer(hash_dataset, config, seed=2).resume(checkpoint)

    assert resumed.resumed_from_epoch == 4
    assert resumed.selection_history == full.selection_history
    assert resumed.lr_history == full.lr_history
    assert resumed.test_loss_history == full.test_loss_history
    assert resumed.best_epoch == full.best_epoch

    by_loss = _accuracy_config(max_epochs=10, threshold=1e-4, patience=50)
    by_loss.selection_metric = rc.SelectionMetric.Loss
    with pytest.raises(ValueError, match="selection metric"):
        _trainer(hash_dataset, by_loss, seed=2).resume(checkpoint)
