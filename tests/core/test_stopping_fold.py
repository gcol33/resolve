"""An explicit stopping fold through the nanobind bindings.

``Trainer.prepare_data(dataset, stopping_indices, seed)`` holds out the plots
the caller names instead of a random ``test_size`` fraction, for spatial or
database cross-validation and for protocols that select the epoch on a named
fold. These cases pin the binding surface: the overload and the plot-ID lookup,
the equivalence with the random split, the refusals, and the run metadata that
records the split.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


def _trainer(dataset, max_epochs=4):
    torch.manual_seed(5)
    model = rc.ResolveModel(dataset.schema, make_model_config())
    return rc.Trainer(model, make_train_config(max_epochs=max_epochs))


def test_the_fold_is_held_out_as_given(hash_dataset):
    ids = [hash_dataset.plot_ids[i] for i in (7, 2, 11, 0)]
    fold = hash_dataset.plot_indices(ids)
    assert fold == [7, 2, 11, 0]

    trainer = _trainer(hash_dataset)
    trainer.prepare_data(hash_dataset, fold, 3)
    assert trainer.test_indices().tolist() == fold
    assert trainer.test_plot_ids() == ids
    assert len(trainer.train_plot_ids()) == hash_dataset.n_plots - len(fold)
    assert set(trainer.train_plot_ids()).isdisjoint(ids)

    meta = trainer.run_metadata
    assert meta.hold_out == rc.HoldOutMode.Explicit
    assert meta.test_size == 0.0
    assert meta.split_seed == 3
    assert meta.n_plots_test == len(fold)


def test_the_keyword_form_selects_the_explicit_fold(hash_dataset):
    trainer = _trainer(hash_dataset)
    trainer.prepare_data(hash_dataset, stopping_indices=[4, 5, 6], seed=1)
    assert trainer.test_indices().tolist() == [4, 5, 6]


def test_the_random_fold_named_explicitly_reproduces_the_random_run(hash_dataset):
    random_trainer = _trainer(hash_dataset)
    random_trainer.prepare_data(hash_dataset, 0.25, 9)
    fold = random_trainer.test_indices().tolist()
    random_result = random_trainer.fit()

    explicit_trainer = _trainer(hash_dataset)
    explicit_trainer.prepare_data(hash_dataset, fold, 9)
    assert torch.equal(explicit_trainer.train_indices(), random_trainer.train_indices())
    explicit_result = explicit_trainer.fit()

    assert explicit_result.train_loss_history == random_result.train_loss_history
    assert explicit_result.test_loss_history == random_result.test_loss_history
    assert explicit_result.best_epoch == random_result.best_epoch
    assert explicit_result.final_metrics == random_result.final_metrics
    assert random_trainer.run_metadata.hold_out == rc.HoldOutMode.Random
    assert random_trainer.run_metadata.test_size == pytest.approx(0.25)


@pytest.mark.parametrize(
    "fold, message",
    [
        ([], "empty"),
        ([1, -1], "outside"),
        ([1, 10_000], "outside"),
        ([3, 1, 3], "more than once"),
    ],
)
def test_an_unusable_fold_is_refused(hash_dataset, fold, message):
    trainer = _trainer(hash_dataset)
    with pytest.raises(ValueError, match=message):
        trainer.prepare_data(hash_dataset, stopping_indices=fold, seed=1)


def test_a_fold_of_every_plot_is_refused(hash_dataset):
    trainer = _trainer(hash_dataset)
    with pytest.raises(ValueError, match="none to fit"):
        trainer.prepare_data(hash_dataset, list(range(hash_dataset.n_plots)), 1)


def test_an_unknown_plot_id_is_refused(hash_dataset):
    with pytest.raises(ValueError, match="no_such_plot"):
        hash_dataset.plot_indices(["no_such_plot"])


def test_the_checkpoint_records_the_split(tmp_path, hash_dataset):
    trainer = _trainer(hash_dataset, max_epochs=2)
    fold = [0, 3, 6, 9, 12]
    trainer.prepare_data(hash_dataset, fold, 17)
    result = trainer.fit()

    path = str(tmp_path / "model.pt")
    trainer.save(path, trainer.run_metadata)
    meta = rc.Trainer.load_run_metadata(path)
    assert meta.hold_out == rc.HoldOutMode.Explicit
    assert meta.split_seed == 17
    assert meta.n_plots_test == len(fold)
    assert meta.best_epoch == result.best_epoch
    assert (tmp_path / "model.json").read_text().count('"hold_out": "explicit"') == 1
