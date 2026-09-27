"""TrainConfig.fixed_epochs through the nanobind bindings.

The knob runs a fixed number of epochs of the ``max_epochs`` schedule, keeps
the final weights, and is the one mode that fits with nothing held out. These
cases pin the binding surface: the config field and its checkpoint round trip,
a refit on every plot, and the refusals the engine raises.
"""

from __future__ import annotations

import math

import pytest

import resolve_core as rc

from conftest import make_model_config, make_train_config


def _fit(dataset, cfg, test_size):
    model = rc.ResolveModel(dataset.schema, make_model_config())
    trainer = rc.Trainer(model, cfg)
    trainer.prepare_data(dataset, test_size, 42)
    return trainer, trainer.fit()


def test_the_default_stops_early():
    assert rc.TrainConfig().fixed_epochs == 0


def test_a_fixed_duration_refits_on_every_plot(hash_dataset):
    cfg = make_train_config(max_epochs=8)
    cfg.fixed_epochs = 3
    trainer, result = _fit(hash_dataset, cfg, 0.0)
    assert len(trainer.train_plot_ids()) == hash_dataset.n_plots
    assert len(trainer.test_plot_ids()) == 0
    assert len(result.train_loss_history) == 3
    assert len(result.test_loss_history) == 0
    assert result.best_epoch == 2
    assert all(math.isfinite(loss) for loss in result.train_loss_history)


def test_the_knob_survives_a_checkpoint(hash_dataset, tmp_path):
    cfg = make_train_config(max_epochs=8)
    cfg.fixed_epochs = 3
    trainer, _ = _fit(hash_dataset, cfg, 0.0)
    path = tmp_path / "fixed.pt"
    trainer.save(str(path))
    assert rc.Trainer.load_train_config(str(path)).fixed_epochs == 3


def test_early_stopping_without_a_held_out_fold_is_refused(hash_dataset):
    with pytest.raises(ValueError, match="fixed_epochs"):
        _fit(hash_dataset, make_train_config(max_epochs=8), 0.0)


def test_a_duration_past_the_schedule_is_refused(hash_dataset):
    cfg = make_train_config(max_epochs=4)
    cfg.fixed_epochs = 5
    with pytest.raises(ValueError, match="max_epochs"):
        _fit(hash_dataset, cfg, 0.25)
