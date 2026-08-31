"""``Trainer`` lifecycle through the nanobind bindings.

Coverage kind: mostly structure / plumbing (shapes, key sets, definedness) plus
a few behavioural invariants that a broken binding would violate: the held-out
fold accessors must partition the plots, cross-validation must leave the
trainer's own split intact, ``load_state`` must reproduce the saved model's
predictions, and the persisted training config must come back equal to what was
set. Whether the fitter *learns* is asserted separately in ``test_recovery.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import resolve_core as rc

from conftest import (
    make_dataset_config,
    make_model_config,
    make_roles,
    make_train_config,
)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def test_fit_returns_a_populated_result(fitted_trainer):
    result = fitted_trainer.result
    assert result.best_epoch >= 0
    assert len(result.train_loss_history) > 0
    assert len(result.test_loss_history) == len(result.train_loss_history)
    assert all(np.isfinite(v) for v in result.train_loss_history)
    assert set(result.final_metrics) == {"y", "hab"}
    assert "r2" in result.final_metrics["y"]
    assert "accuracy" in result.final_metrics["hab"]
    assert result.train_time_seconds >= 0.0


def test_final_metrics_are_finite(fitted_trainer):
    for target, metrics in fitted_trainer.result.final_metrics.items():
        for name, value in metrics.items():
            assert np.isfinite(value), f"{target}.{name} is not finite"


def test_baseline_metrics_are_reported(fitted_trainer):
    """``fit`` compares the model against a naive baseline per target."""
    baselines = fitted_trainer.result.baselines
    assert set(baselines) == {"y", "hab"}


def test_hash_dim_mismatch_is_rejected_at_prepare_data(hash_dataset):
    """DatasetConfig.hash_dim and ModelConfig.hash_dim are independent knobs.

    A mismatch used to surface as an opaque matmul shape error deep in fit();
    prepare_data names the two values instead.
    """
    config = make_model_config(hash_dim=8)  # dataset was built with 16
    model = rc.ResolveModel(hash_dataset.schema, config)
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    with pytest.raises(Exception, match="hash_dim"):
        trainer.prepare_data(hash_dataset, 0.25, 42)


# ---------------------------------------------------------------------------
# Held-out fold accessors
# ---------------------------------------------------------------------------

def test_split_accessors_partition_the_plots(fitted_trainer):
    trainer, dataset = fitted_trainer.trainer, fitted_trainer.dataset
    train_ids = trainer.train_plot_ids()
    test_ids = trainer.test_plot_ids()

    assert len(train_ids) + len(test_ids) == dataset.n_plots
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(dataset.plot_ids)
    assert len(test_ids) == round(fitted_trainer.test_size * dataset.n_plots)


def test_index_accessors_match_the_plot_id_accessors(fitted_trainer):
    plot_ids = list(fitted_trainer.dataset.plot_ids)
    test_idx = fitted_trainer.trainer.test_indices()
    assert test_idx.dtype == torch.int64
    assert [plot_ids[i] for i in test_idx.tolist()] == list(
        fitted_trainer.trainer.test_plot_ids()
    )


# ---------------------------------------------------------------------------
# Test-fold evaluators
# ---------------------------------------------------------------------------

def test_compute_residuals_shapes(fitted_trainer):
    residuals = fitted_trainer.trainer.compute_residuals("y")
    n_test = len(fitted_trainer.trainer.test_plot_ids())

    assert residuals.target_name == "y"
    assert len(residuals.predictions) == n_test
    assert len(residuals.actuals) == n_test
    assert len(residuals.residuals) == n_test
    assert np.isfinite(residuals.mean_residual)
    assert np.isfinite(residuals.std_residual)
    assert residuals.q05 <= residuals.q50 <= residuals.q95


def test_compute_classification_predictions_shapes(fitted_trainer):
    preds = fitted_trainer.trainer.compute_classification_predictions("hab")
    n_test = len(fitted_trainer.trainer.test_plot_ids())

    assert preds.target_name == "hab"
    assert len(preds.predicted_classes) == n_test
    assert len(preds.actuals) == n_test
    assert np.asarray(preds.probabilities).shape == (n_test, 3)
    assert preds.class_names == ["0", "1", "2"]

    probabilities = np.asarray(preds.probabilities)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4)
    assert set(np.asarray(preds.predicted_classes).tolist()) <= {0, 1, 2}


def test_classification_actuals_match_the_dataset_targets(fitted_trainer):
    """The evaluator's actuals must be the held-out rows, in fold order."""
    trainer, dataset = fitted_trainer.trainer, fitted_trainer.dataset
    preds = trainer.compute_classification_predictions("hab")
    expected = dataset.targets["hab"].index_select(0, trainer.test_indices())
    np.testing.assert_array_equal(
        np.asarray(preds.actuals), expected.numpy()
    )


def test_compute_calibration_bins_a_classification_target(fitted_trainer):
    calibration = fitted_trainer.trainer.compute_calibration("hab")
    assert calibration.target_name == "hab"
    assert len(calibration.bins) > 0
    assert np.isfinite(calibration.expected_calibration_error)
    assert np.isfinite(calibration.max_calibration_error)
    total = 0
    for bin_ in calibration.bins:
        assert bin_.bin_start <= bin_.bin_end
        assert bin_.count >= 0
        if bin_.count > 0:
            assert np.isfinite(bin_.mean_predicted_prob)
            assert 0.0 <= bin_.actual_frequency <= 1.0
        total += bin_.count
    assert total == len(fitted_trainer.trainer.test_plot_ids())


def test_calibration_is_empty_for_a_regression_target(fitted_trainer):
    """Calibration is a classification notion; regression returns no bins."""
    assert fitted_trainer.trainer.compute_calibration("y").bins == []


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cross_validate_returns_per_fold_results(hash_dataset):
    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=2))
    trainer.prepare_data(hash_dataset, 0.25, 42)

    cv = trainer.cross_validate(3, 42)
    assert cv.n_folds == 3
    assert len(cv.fold_results) == 3
    assert set(cv.mean_metrics) == {"y", "hab"}
    assert set(cv.std_metrics) == {"y", "hab"}
    for metrics in cv.mean_metrics.values():
        for value in metrics.values():
            assert np.isfinite(value)


def test_cross_validate_restores_the_trainers_own_split(hash_dataset):
    """CV must not leave the trainer sitting on its last fold.

    The test-fold evaluators read the trainer's split, so a CV run that left the
    final fold in place would silently score against the wrong rows afterwards.
    """
    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=2))
    trainer.prepare_data(hash_dataset, 0.25, 42)

    before = list(trainer.test_plot_ids())
    trainer.cross_validate(3, 42)
    assert list(trainer.test_plot_ids()) == before


# ---------------------------------------------------------------------------
# Checkpoint config / metadata readers
# ---------------------------------------------------------------------------

def test_train_config_round_trips_through_the_checkpoint(tmp_path, fitted_trainer):
    path = str(tmp_path / "cfg.pt")
    fitted_trainer.trainer.save(path)

    loaded = rc.Trainer.load_train_config(path)
    original = fitted_trainer.trainer.config
    assert loaded.batch_size == original.batch_size
    assert loaded.max_epochs == original.max_epochs
    assert loaded.patience == original.patience
    assert loaded.lr == pytest.approx(original.lr)
    assert loaded.batch_size_floor == original.batch_size_floor
    assert loaded.vram_fraction == pytest.approx(original.vram_fraction)


def test_nca_hyperparameters_are_writable_and_round_trip(tmp_path, hash_dataset):
    """The NCA term's temperature / neighbour count / weight are TrainConfig fields.

    They used to be hardcoded constants no caller could reach, so a run could
    select ``LossConfigMode.NCA`` and had no say in how the term behaved.
    """
    config = make_train_config(max_epochs=1)
    config.loss_config = rc.LossConfigMode.NCA
    config.nca_temperature = 0.45
    config.nca_neighbors = 7
    config.nca_weight = 0.6

    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, config)
    trainer.prepare_data(hash_dataset, 0.25, 42)

    path = str(tmp_path / "nca.pt")
    trainer.save(path)

    loaded = rc.Trainer.load_train_config(path)
    assert loaded.loss_config == rc.LossConfigMode.NCA
    assert loaded.nca_temperature == pytest.approx(0.45)
    assert loaded.nca_neighbors == 7
    assert loaded.nca_weight == pytest.approx(0.6)

    # A checkpoint written before these keys existed reads back as the defaults.
    defaults = rc.TrainConfig()
    assert defaults.nca_temperature == pytest.approx(0.1)
    assert defaults.nca_neighbors == 32
    assert defaults.nca_weight == pytest.approx(0.1)


def test_fit_writes_run_metadata_into_its_checkpoint(tmp_path, hash_dataset):
    """``fit`` records its own run in ``checkpoint_dir/checkpoint.pt``.

    ``load_run_metadata`` reads it back; the two used to be write-only, so a
    checkpoint carried no record of the run that produced it.
    """
    config = make_train_config(max_epochs=3)
    config.checkpoint_dir = str(tmp_path / "ckpt")

    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, config)
    trainer.prepare_data(hash_dataset, 0.25, 42)
    result = trainer.fit()

    meta = rc.Trainer.load_run_metadata(str(tmp_path / "ckpt" / "checkpoint.pt"))
    assert meta.resolve_version == rc.__version__
    assert meta.n_plots_train == len(trainer.train_plot_ids())
    assert meta.n_plots_test == len(trainer.test_plot_ids())
    assert meta.best_epoch == result.best_epoch
    assert meta.total_epochs == len(result.train_loss_history)
    assert set(meta.final_metrics) == {"y", "hab"}
    assert meta.completed_at != ""


def test_load_state_restores_weights_in_place(tmp_path, hash_dataset, fitted_trainer):
    """``load_state`` is the first-class way to score a saved checkpoint.

    A freshly constructed trainer predicts from random weights; after
    ``load_state`` it must reproduce the saved model's predictions exactly.
    """
    path = str(tmp_path / "state.pt")
    fitted_trainer.trainer.save(path)
    reference = fitted_trainer.trainer.compute_residuals("y")

    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    fresh = rc.Trainer(model, make_train_config(max_epochs=1))
    fresh.prepare_data(hash_dataset, 0.25, 42)
    fresh.load_state(path, "cpu", 1.0)

    restored = fresh.compute_residuals("y")
    np.testing.assert_allclose(
        np.asarray(restored.predictions),
        np.asarray(reference.predictions),
        rtol=1e-5,
        atol=1e-6,
    )


def test_load_state_rejects_a_mismatched_architecture(
    tmp_path, hash_dataset, fitted_trainer
):
    """A checkpoint whose parameters do not line up must fail loudly.

    A silent skip would leave the mismatched layers at random init and predict
    nonsense from a model that looks loaded.
    """
    path = str(tmp_path / "mismatch.pt")
    fitted_trainer.trainer.save(path)

    wider = make_model_config()
    wider.hidden_dims = [128, 64, 32]
    model = rc.ResolveModel(hash_dataset.schema, wider)
    other = rc.Trainer(model, make_train_config(max_epochs=1))
    other.prepare_data(hash_dataset, 0.25, 42)
    with pytest.raises(Exception):
        other.load_state(path, "cpu", 1.0)


# ---------------------------------------------------------------------------
# A fit needs a target
# ---------------------------------------------------------------------------
#
# A target-less dataset is legal to build -- that is an inference set, and
# ``Predictor.predict`` scores one -- so the guard sits at ``prepare_data``.
# Without it the empty target map reached the training loop and surfaced as
# ``element 0 of tensors does not require grad and does not have a grad_fn``,
# an autograd message that names nothing about targets.

def test_prepare_data_rejects_a_dataset_with_no_targets(plot_csvs):
    targetless = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        [],
        make_dataset_config(rc.SpeciesEncodingMode.Hash),
    )
    assert targetless.targets == {}

    model = rc.ResolveModel(targetless.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    with pytest.raises(Exception, match="no targets"):
        trainer.prepare_data(targetless, 0.25, 42)


def test_prepare_data_accepts_a_dataset_that_has_targets(hash_dataset):
    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    trainer.prepare_data(hash_dataset, 0.25, 42)
    assert trainer.test_indices().numel() > 0
