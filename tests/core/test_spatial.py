"""Spatial block cross-validation through the bindings.

The splitter itself (`resolve::SpatialBlockSplitter`) is not bound to Python;
its algorithm — every plot in exactly one test fold, co-located plots never
split across folds, determinism under a seed, balanced mode, parameter
rejection — is pinned by Catch2 in ``src/core/tests/test_spatial_cv.cpp``.

What is reachable from Python, and therefore what is tested here, is
``SpatialBlockConfig`` plus ``Trainer.cross_validate_spatial``. Coverage kind:
structure, with three behavioural assertions: the too-few-blocks guard, and the
two halves of what a seed covers -- runs seeded in both the split and the weight
initialisation agree exactly, while runs seeded only in the split do not.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import resolve_core as rc
from resolve_core import SpatialBlockConfig

from conftest import make_model_config, make_train_config


def _trainer(dataset, *, max_epochs: int = 2, weight_seed: int | None = None) -> "rc.Trainer":
    # `weight_seed` seeds the process-global torch RNG that ResolveModel's
    # weight initialisation draws from. The seed passed to prepare_data and to
    # cross_validate_spatial governs the SPLIT only; nothing on the library path
    # seeds initialisation, so two otherwise identical runs start from different
    # weights unless the caller seeds first, the way the CLI's --seed does before
    # it constructs the model (issue #115).
    if weight_seed is not None:
        torch.manual_seed(weight_seed)
    model = rc.ResolveModel(dataset.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=max_epochs))
    trainer.prepare_data(dataset, 0.25, 42)
    return trainer


def _block_config(size: float, *, balance: bool = False) -> SpatialBlockConfig:
    config = SpatialBlockConfig()
    config.lat_size = size
    config.lon_size = size
    config.balance = balance
    return config


def test_spatial_block_config_fields_round_trip():
    config = SpatialBlockConfig()
    config.lat_size = 0.5
    config.lon_size = 2.0
    config.balance = True
    assert config.lat_size == pytest.approx(0.5)
    assert config.lon_size == pytest.approx(2.0)
    assert config.balance is True


def test_spatial_cv_returns_the_requested_folds(hash_dataset):
    cv = _trainer(hash_dataset).cross_validate_spatial(_block_config(1.0), 3, 42)

    assert cv.n_folds == 3
    assert len(cv.fold_results) == 3
    assert set(cv.mean_metrics) == {"y", "hab"}
    for metrics in cv.mean_metrics.values():
        for value in metrics.values():
            assert np.isfinite(value)


def test_spatial_cv_is_deterministic_under_a_seed(hash_dataset):
    """Two runs seeded identically -- split AND weight init -- agree exactly.

    The weight seed is the load-bearing half. Without it this compared two runs
    that started from different weights and only usually landed on the same
    best_epoch, so it failed at random (issue #115). Comparing the fold metrics
    rather than the argmin over them makes a real divergence visible instead of
    hiding behind a tie-break.
    """
    first = _trainer(hash_dataset, weight_seed=7).cross_validate_spatial(
        _block_config(1.0), 3, 7
    )
    second = _trainer(hash_dataset, weight_seed=7).cross_validate_spatial(
        _block_config(1.0), 3, 7
    )

    for a, b in zip(first.fold_results, second.fold_results):
        assert a.best_epoch == b.best_epoch
    for target, metrics in first.mean_metrics.items():
        for name, value in metrics.items():
            assert value == pytest.approx(second.mean_metrics[target][name], rel=0, abs=0)


def test_an_unseeded_run_is_not_reproducible(hash_dataset):
    """The other half of the contract, stated so it is not rediscovered as a flake.

    Model weight initialisation draws from the process-global torch RNG, which
    neither prepare_data's seed nor cross_validate_spatial's covers -- they seed
    the split. A caller that wants a reproducible fit seeds the global RNG
    before constructing the model, which is what the CLI's --seed does.
    """
    torch.manual_seed(0)
    runs = [
        _trainer(hash_dataset).cross_validate_spatial(_block_config(1.0), 3, 7)
        for _ in range(4)
    ]
    maes = [r.mean_metrics["y"]["mae"] for r in runs]
    assert len(set(maes)) > 1, (
        "unseeded runs came out identical; if initialisation became seeded on "
        "the library path, this test and the docs it mirrors are stale"
    )


def test_balanced_mode_runs(hash_dataset):
    cv = _trainer(hash_dataset).cross_validate_spatial(
        _block_config(1.0, balance=True), 3, 42
    )
    assert cv.n_folds == 3


def test_fewer_blocks_than_folds_is_rejected(hash_dataset):
    """A block size that swallows the whole extent would leave folds empty.

    An empty fold divides by zero in the baseline metrics, so the splitter
    refuses the split rather than reporting a degenerate score.
    """
    with pytest.raises(Exception):
        _trainer(hash_dataset).cross_validate_spatial(_block_config(1000.0), 3, 42)


def test_spatial_cv_restores_the_trainers_own_split(hash_dataset):
    trainer = _trainer(hash_dataset)
    before = list(trainer.test_plot_ids())
    trainer.cross_validate_spatial(_block_config(1.0), 3, 42)
    assert list(trainer.test_plot_ids()) == before


def test_spatial_cv_requires_coordinates(plot_csvs):
    """Without lon/lat there is no grid to block on."""
    from conftest import make_dataset_config, make_roles, make_targets

    dataset = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(coordinates=False),
        make_targets(),
        make_dataset_config(),
    )
    assert not dataset.schema.has_coordinates
    with pytest.raises(Exception):
        _trainer(dataset).cross_validate_spatial(_block_config(1.0), 3, 42)
