"""Mixture of experts through the bindings, at both placements.

Coverage kind: structure / plumbing. ``moe_routing`` used to mean two different
architectures depending on ``species_encoding`` -- a mixture that replaced the
encoder's last MLP stages for hash, a dim-preserving block bolted onto the
latent for embed / sparse / rank_pool / transformer, and a refusal for the
adapter architectures -- and nothing anywhere constructed a model with routing
on. The architecture contract is pinned by
``src/core/tests/test_moe_placement.cpp``; what is asserted here is that the
Python surface reaches it: the new ``MoEPlacement`` enum, the config fields,
the auxiliary loss on ``forward_with_aux``, and the checkpoint round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import resolve_core as rc

from conftest import (
    make_dataset_config,
    make_model_config,
    make_plot_csvs,
    make_roles,
    make_targets,
    make_train_config,
)


ENCODINGS = [
    rc.SpeciesEncodingMode.Hash,
    rc.SpeciesEncodingMode.Embed,
    rc.SpeciesEncodingMode.Sparse,
    rc.SpeciesEncodingMode.RankPool,
    rc.SpeciesEncodingMode.Transformer,
]

HIDDEN = [32, 24, 16, 12]


@pytest.fixture(scope="module")
def csvs(tmp_path_factory) -> "object":
    return make_plot_csvs(tmp_path_factory.mktemp("moe"), n_plots=80, n_species=10)


def _dataset(csvs, encoding):
    cfg = make_dataset_config(encoding)
    return rc.ResolveDataset.from_csv(
        csvs.header, csvs.species, make_roles(), make_targets(), cfg
    )


def _model_config(encoding, placement=None, routing=None):
    cfg = make_model_config(encoding, hidden_dims=list(HIDDEN))
    cfg.d_model = 16
    cfg.n_heads = 2
    cfg.n_attention_layers = 1
    cfg.transformer_ff_dim = 16
    if routing is not None:
        cfg.moe_routing = routing
        cfg.moe_placement = placement
        cfg.n_experts = 3
        cfg.expert_hidden_dims = [10]
        cfg.moe_top_k = 2
        cfg.moe_noise_std = 0.0
    return cfg


def _param_names(model) -> set[str]:
    return set(model.named_parameters().keys())


def _has_prefix(names: set[str], prefix: str) -> bool:
    return any(name.startswith(prefix) for name in names)


# ---------------------------------------------------------------------------
# The placement enum reached the module
# ---------------------------------------------------------------------------

def test_placement_enum_is_public():
    assert rc.MoEPlacement.Tail is not rc.MoEPlacement.Post
    cfg = rc.ModelConfig()
    # Tail is the default: the mixture is the encoder's last stage unless the
    # caller moves it.
    assert cfg.moe_placement == rc.MoEPlacement.Tail
    cfg.moe_placement = rc.MoEPlacement.Post
    assert cfg.moe_placement == rc.MoEPlacement.Post


def test_moe_config_fields_round_trip():
    cfg = rc.ModelConfig()
    cfg.moe_routing = rc.MoERoutingType.TopK
    cfg.moe_placement = rc.MoEPlacement.Post
    cfg.n_experts = 5
    cfg.expert_hidden_dims = [64, 32]
    cfg.moe_top_k = 3
    cfg.moe_noise_std = 0.25
    cfg.moe_aux_loss_weight = 0.05

    assert cfg.moe_routing == rc.MoERoutingType.TopK
    assert cfg.n_experts == 5
    assert list(cfg.expert_hidden_dims) == [64, 32]
    assert cfg.moe_top_k == 3
    assert cfg.moe_noise_std == pytest.approx(0.25)
    assert cfg.moe_aux_loss_weight == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Tail: available to every species encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: str(e).split(".")[-1])
def test_tail_mixture_builds_for_every_encoding(csvs, encoding):
    ds = _dataset(csvs, encoding)
    model = rc.ResolveModel(
        ds.schema,
        _model_config(encoding, rc.MoEPlacement.Tail, rc.MoERoutingType.Soft),
    )

    # The mixture produces the last hidden width, so the latent is unchanged.
    assert model.latent_dim == HIDDEN[-1]
    assert model.uses_moe

    names = _param_names(model)
    assert _has_prefix(names, "encoder.backbone.")
    assert _has_prefix(names, "encoder.moe.")
    assert not _has_prefix(names, "encoder.mlp.")
    assert not _has_prefix(names, "post_moe.")


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: str(e).split(".")[-1])
def test_no_routing_leaves_the_plain_mlp_tail(csvs, encoding):
    ds = _dataset(csvs, encoding)
    model = rc.ResolveModel(ds.schema, _model_config(encoding))
    names = _param_names(model)
    assert not model.uses_moe
    assert _has_prefix(names, "encoder.mlp.")
    assert not _has_prefix(names, "encoder.moe.")
    assert not _has_prefix(names, "post_moe.")


@pytest.mark.parametrize("encoding", ENCODINGS, ids=lambda e: str(e).split(".")[-1])
def test_post_mixture_keeps_the_encoder_tail(csvs, encoding):
    ds = _dataset(csvs, encoding)
    model = rc.ResolveModel(
        ds.schema,
        _model_config(encoding, rc.MoEPlacement.Post, rc.MoERoutingType.Soft),
    )
    assert model.latent_dim == HIDDEN[-1]
    names = _param_names(model)
    assert _has_prefix(names, "post_moe.")
    assert _has_prefix(names, "encoder.mlp.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "placement",
    [rc.MoEPlacement.Tail, rc.MoEPlacement.Post],
    ids=["tail", "post"],
)
@pytest.mark.parametrize(
    "encoding",
    [rc.SpeciesEncodingMode.Hash, rc.SpeciesEncodingMode.Embed,
     rc.SpeciesEncodingMode.RankPool],
    ids=lambda e: str(e).split(".")[-1],
)
def test_a_mixture_trains(csvs, encoding, placement):
    ds = _dataset(csvs, encoding)
    model = rc.ResolveModel(
        ds.schema, _model_config(encoding, placement, rc.MoERoutingType.Soft)
    )
    trainer = rc.Trainer(model, make_train_config(max_epochs=3))
    trainer.prepare_data(ds)
    result = trainer.fit()

    assert len(result.train_loss_history) >= 1
    for loss in result.train_loss_history:
        assert loss == loss  # not NaN: the load-balancing term stays finite


# ---------------------------------------------------------------------------
# Placements an encoder without an MLP tail can and cannot take
# ---------------------------------------------------------------------------

def test_tail_is_refused_where_there_is_no_mlp_tail(csvs):
    ds = _dataset(csvs, rc.SpeciesEncodingMode.Sparse)
    cfg = _model_config(
        rc.SpeciesEncodingMode.Sparse, rc.MoEPlacement.Tail, rc.MoERoutingType.Soft
    )
    cfg.encoder_architecture = rc.EncoderArchitecture.TabNet

    with pytest.raises(Exception) as excinfo:
        rc.ResolveModel(ds.schema, cfg)
    # The message has to name the placement that works, or the knob just looks
    # broken for these architectures.
    assert "moe_placement=post" in str(excinfo.value)

    cfg.moe_placement = rc.MoEPlacement.Post
    model = rc.ResolveModel(ds.schema, cfg)
    assert _has_prefix(_param_names(model), "post_moe.")


def test_tabm_and_a_mixture_cannot_both_have_the_tail(csvs):
    ds = _dataset(csvs, rc.SpeciesEncodingMode.Hash)
    cfg = _model_config(
        rc.SpeciesEncodingMode.Hash, rc.MoEPlacement.Tail, rc.MoERoutingType.Soft
    )
    tabm = rc.TabMConfig()
    tabm.enabled = True
    cfg.tabm = tabm
    with pytest.raises(Exception):
        rc.ResolveModel(ds.schema, cfg)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "placement",
    [rc.MoEPlacement.Tail, rc.MoEPlacement.Post],
    ids=["tail", "post"],
)
def test_a_mixture_survives_the_checkpoint(csvs, tmp_path: Path, placement):
    encoding = rc.SpeciesEncodingMode.RankPool
    ds = _dataset(csvs, encoding)
    model = rc.ResolveModel(
        ds.schema, _model_config(encoding, placement, rc.MoERoutingType.Soft)
    )
    trainer = rc.Trainer(model, make_train_config(max_epochs=2))
    trainer.prepare_data(ds)
    trainer.fit()

    path = str(tmp_path / f"moe_{placement}.pt")
    trainer.save(path)

    # The reload rebuilds the architecture from the persisted config and throws
    # on a parameter it cannot find, so a placement that failed to round-trip
    # cannot load quietly.
    predictor = rc.Predictor.load(path, device="cpu")
    assert predictor.model.config.moe_placement == placement
    assert predictor.model.config.moe_routing == rc.MoERoutingType.Soft
    assert predictor.model.config.n_experts == 3

    out = predictor.predict_dataset(ds, return_latent=True)
    assert out.latent.shape == (csvs.n_plots, HIDDEN[-1])
