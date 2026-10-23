"""The tabular encoder architectures through the bindings.

Coverage kind: structure / plumbing. The architectures are pinned by the
Catch2 suite (``test_attention_encoders.cpp``, ``test_tabular_adapter.cpp``,
``test_data_buffers.cpp``); what is asserted here is that the Python surface
reaches them: the named auxiliary terms on ``forward_with_aux``, the TabM
aggregation enum, SAINT's reference set and its chunk-independent predictions.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import (
    make_dataset_config,
    make_model_config,
    make_plot_csvs,
    make_roles,
    make_targets,
    make_train_config,
    trainer_continuous,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    csvs = make_plot_csvs(tmp_path_factory.mktemp("architectures"), n_plots=60, n_species=10)
    return rc.ResolveDataset.from_csv(
        csvs.header, csvs.species, make_roles(), make_targets(),
        make_dataset_config(rc.SpeciesEncodingMode.Hash))


def _tabnet_config():
    cfg = make_model_config(rc.SpeciesEncodingMode.Hash)
    cfg.encoder_architecture = rc.EncoderArchitecture.TabNet
    cfg.tabnet.n_steps = 2
    cfg.tabnet.n_d = 8
    cfg.tabnet.n_a = 8
    cfg.tabnet.virtual_batch_size = 0
    return cfg


def _saint_config(rows=8):
    cfg = make_model_config(rc.SpeciesEncodingMode.Hash)
    cfg.encoder_architecture = rc.EncoderArchitecture.SAINT
    cfg.saint.d_model = 16
    cfg.saint.n_heads = 2
    cfg.saint.n_layers = 1
    cfg.saint.row_attention_rows = rows
    return cfg


def test_each_model_names_the_auxiliary_terms_it_adds(dataset):
    plain = rc.ResolveModel(dataset.schema, make_model_config(rc.SpeciesEncodingMode.Hash))
    assert plain.aux_loss_names == []

    moe_cfg = make_model_config(rc.SpeciesEncodingMode.Hash)
    moe_cfg.moe_routing = rc.MoERoutingType.Soft
    assert rc.ResolveModel(dataset.schema, moe_cfg).aux_loss_names == ["moe"]

    assert rc.ResolveModel(dataset.schema, _tabnet_config()).aux_loss_names == ["tabnet_sparsity"]


def test_forward_with_aux_returns_the_weighted_terms(dataset):
    cfg = _tabnet_config()
    cfg.tabnet.sparsity_coefficient = 0.25
    torch.manual_seed(0)
    model = rc.ResolveModel(dataset.schema, cfg)
    continuous = trainer_continuous(dataset, cfg)
    genus, family = dataset.genus_ids, dataset.family_ids

    result = model.forward_with_aux(continuous, genus, family)
    assert set(result.aux_losses) == {"tabnet_sparsity"}
    term = result.aux_losses["tabnet_sparsity"]
    assert term.dim() == 0
    assert float(term) > 0.0

    plain = rc.ResolveModel(dataset.schema, make_model_config(rc.SpeciesEncodingMode.Hash))
    assert plain.forward_with_aux(continuous, genus, family).aux_losses == {}


def test_tabm_aggregation_is_an_enum():
    cfg = rc.ModelConfig()
    assert cfg.tabm.aggregation == rc.TabMAggregation.Mean
    cfg.tabm.aggregation = rc.TabMAggregation.Median
    assert cfg.tabm.aggregation == rc.TabMAggregation.Median


def test_saint_predictions_do_not_depend_on_the_chunk(dataset, tmp_path):
    torch.manual_seed(0)
    model = rc.ResolveModel(dataset.schema, _saint_config(rows=8))
    assert model.row_reference_rows == 7
    trainer = rc.Trainer(model, make_train_config(max_epochs=1, batch_size=16))
    trainer.prepare_data(dataset, test_size=0.25, seed=1)
    trainer.fit()

    path = str(tmp_path / "saint.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path, device="cpu")
    one_shot = predictor.predict_dataset(dataset, False, -1)
    for batch_size in (1, 7, 25):
        chunked = predictor.predict_dataset(dataset, False, batch_size)
        for target in one_shot.predictions:
            assert torch.allclose(
                chunked.predictions[target].float(),
                one_shot.predictions[target].float(),
                rtol=1e-5,
                atol=1e-6,
            ), f"{target} differs at batch_size={batch_size}"
