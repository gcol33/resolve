"""ModelConfig.freeze_composition through the nanobind bindings.

The knob keeps a species encoder's species, genus and family tables at their
initialisation. These cases pin the binding surface: the config field, the
tables ``ResolveModel.composition_parameters()`` returns, a fit that leaves them
unchanged, and the refusal on a model that has no such table.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


def test_the_default_learns_the_composition():
    assert rc.ModelConfig().freeze_composition is False


@pytest.mark.parametrize("freeze", [True, False])
def test_a_fit_moves_the_tables_only_when_they_are_not_frozen(pool_dataset, freeze):
    torch.manual_seed(3)
    cfg = make_model_config(rc.SpeciesEncodingMode.RankPool)
    cfg.freeze_composition = freeze
    model = rc.ResolveModel(pool_dataset.schema, cfg)
    tables = model.composition_parameters()
    assert tables and all(t.requires_grad is not freeze for t in tables)
    before = [t.detach().clone() for t in tables]

    trainer = rc.Trainer(model, make_train_config(max_epochs=3, batch_size=32))
    trainer.prepare_data(pool_dataset, test_size=0.25, seed=1)
    trainer.fit()

    unchanged = all(torch.equal(t, b) for t, b in zip(model.composition_parameters(), before))
    assert unchanged is freeze


def test_a_model_without_a_table_refuses_the_knob(pool_dataset):
    cfg = make_model_config(rc.SpeciesEncodingMode.RankPool)
    cfg.encoder_architecture = rc.EncoderArchitecture.TabNet
    cfg.freeze_composition = True
    with pytest.raises(Exception, match="freeze_composition"):
        rc.ResolveModel(pool_dataset.schema, cfg)
