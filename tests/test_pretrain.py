"""Smoke + regression tests for masked species pretraining.

Catches the silent breakage where `_pretrain.py` passed a pre-padded tensor
tuple into the (now-deleted) `RankPoolBatchDataset`, which expected ragged
`_RankPoolPreparedData`. Anyone setting `pretrain_epochs > 0` previously
crashed at the first batch.
"""

from __future__ import annotations

import inspect

from resolve.train import _pretrain as pretrain_mod
from resolve.train.trainer import Trainer


def _make_pretrain_trainer(dataset, **overrides) -> Trainer:
    kwargs = dict(
        species_encoding="transformer",
        species_embed_dim=16,
        n_attention_layers=1,
        n_heads=2,
        transformer_ff_dim=32,
        hidden_dims=[16],
        max_epochs=1,
        patience=1,
        batch_size=32,
        verbose=0,
        pretrain_epochs=1,
        pretrain_mask_prob=0.3,
        pretrain_lr=1e-4,
    )
    kwargs.update(overrides)
    return Trainer(dataset, **kwargs)


class TestPretrainRuns:
    def test_pretrain_then_fit(self, dataset):
        trainer = _make_pretrain_trainer(dataset)
        trainer.pretrain()
        result = trainer.fit()
        assert result.best_epoch >= 0

    def test_pretrain_all_data(self, dataset):
        trainer = _make_pretrain_trainer(dataset, pretrain_all_data=True)
        trainer.pretrain()
        result = trainer.fit()
        assert result.best_epoch >= 0


class TestPretrainModuleHygiene:
    """Confirms the dead ragged-data path is fully removed from _pretrain.py."""

    def test_no_ragged_dataset_imports(self):
        src = inspect.getsource(pretrain_mod)
        assert "_RankPoolPreparedData" not in src
        assert "RankPoolBatchDataset" not in src
        assert "_rank_pool_collate_fn" not in src
