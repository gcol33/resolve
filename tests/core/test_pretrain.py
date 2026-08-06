"""Self-supervised pretraining surface (SCARF, JEPA, VAE).

Coverage kind: structure / plumbing plus config validation. The pretrainers run
for a couple of epochs and are checked for a populated loss history and for
having actually moved the encoder's weights -- a pretrainer wired to a detached
copy would report losses while leaving the model untouched. Whether the
pretext task *helps* downstream is not asserted; that is a research question,
not a contract.

The masking semantics (BERT-style 80/10/10, JEPA context masking, SCARF's
species-side view) are pinned by Catch2 in ``src/core/tests/test_mlm_pretrain.cpp``
and ``test_new_modules.cpp``.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, trainer_continuous


@pytest.fixture
def pretrain_inputs(hash_dataset):
    """The tensors the pretrainers take: continuous block plus taxonomy ids."""
    config = make_model_config()
    return (
        rc.ResolveModel(hash_dataset.schema, config),
        trainer_continuous(hash_dataset, config),
        hash_dataset.genus_ids,
        hash_dataset.family_ids,
    )


def _pretrain_config(epochs: int = 2) -> "rc.PretrainConfig":
    config = rc.PretrainConfig()
    config.pretrain_epochs = epochs
    config.batch_size = 16
    config.mask_ratio = 0.3
    config.pretrain_lr = 1e-3
    return config


def _flat_weights(model) -> torch.Tensor:
    named = model.named_parameters()
    return torch.cat([named[key].detach().reshape(-1) for key in sorted(named)])


# ---------------------------------------------------------------------------
# SCARF / JEPA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "cls"),
    [("scarf", rc.SCARFPretrainer), ("jepa", rc.JEPAPretrainer)],
)
def test_pretrainer_runs_and_reports_a_loss_history(pretrain_inputs, name, cls):
    model, continuous, genus_ids, family_ids = pretrain_inputs

    result = cls(model, _pretrain_config(2)).pretrain(continuous, genus_ids, family_ids)

    assert result.epochs_completed == 2
    assert len(result.loss_history) == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in result.loss_history)
    assert result.total_time_seconds >= 0.0


@pytest.mark.parametrize(
    ("name", "cls"),
    [("scarf", rc.SCARFPretrainer), ("jepa", rc.JEPAPretrainer)],
)
def test_pretraining_updates_the_encoder_weights(pretrain_inputs, name, cls):
    """The pretext task must train the model it was handed, not a copy."""
    model, continuous, genus_ids, family_ids = pretrain_inputs

    before = _flat_weights(model).clone()
    cls(model, _pretrain_config(2)).pretrain(continuous, genus_ids, family_ids)
    after = _flat_weights(model)

    assert not torch.allclose(before, after)


def test_omitted_taxonomy_is_read_as_absent_not_as_a_tensor(pretrain_inputs):
    """The optional args must be genuinely omittable.

    They used to be unpacked as tensors unconditionally, so leaving one out
    reinterpreted the ``None`` singleton as a tensor handle. The model here is
    built with taxonomy, so omitting both must raise the encoder's own clear
    error rather than crash.
    """
    model, continuous, _, _ = pretrain_inputs
    with pytest.raises(RuntimeError, match="taxonomy"):
        rc.SCARFPretrainer(model, _pretrain_config(1)).pretrain(continuous)


def test_pretraining_without_taxonomy(plot_csvs):
    """A model built without taxonomy pretrains from the continuous block alone."""
    from conftest import make_dataset_config, make_roles, make_targets

    config = make_dataset_config()
    config.use_taxonomy = False
    dataset = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(taxonomy=False),
        make_targets(),
        config,
    )
    model_config = make_model_config()
    model = rc.ResolveModel(dataset.schema, model_config)

    result = rc.SCARFPretrainer(model, _pretrain_config(1)).pretrain(
        trainer_continuous(dataset, model_config)
    )
    assert result.epochs_completed == 1


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def test_vae_pretrainer_runs_over_a_species_matrix(hash_dataset):
    n_species = hash_dataset.schema.n_species_vocab

    config = rc.VAEConfig()
    config.pretrain_epochs = 2
    config.batch_size = 16
    config.latent_dim = 4
    config.encoder_dims = [16]
    config.decoder_dims = [16]

    pretrainer = rc.VAEPretrainer(n_species, config)
    vectors = torch.rand(hash_dataset.n_plots, n_species)
    result = pretrainer.pretrain(vectors)

    assert result.epochs_completed == 2
    assert len(result.loss_history) == 2
    assert len(result.recon_loss_history) == 2
    assert len(result.kl_loss_history) == 2
    assert pretrainer.get_latent_dim() == 4
    assert pretrainer.get_projection_weights().shape[1] == n_species


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("mask_ratio", 0.0),
        ("mask_ratio", 1.5),
        ("corruption_rate", -0.1),
        ("corruption_rate", 1.5),
    ],
)
def test_invalid_pretrain_config_is_rejected_at_construction(
    pretrain_inputs, field, value
):
    """The pretraining path has no CLI parse site, so the config is the guard.

    ``batch_size = 0`` divides by zero computing step counts and a mask ratio
    outside (0, 1) makes the Block strategy's randint bound non-positive.
    """
    model, _, _, _ = pretrain_inputs
    config = _pretrain_config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        rc.SCARFPretrainer(model, config)


def test_mask_strategy_enum_is_exposed():
    assert set(rc.MaskStrategy.__members__) == {"Random", "Block", "Structured"}
    config = _pretrain_config()
    config.mask_strategy = rc.MaskStrategy.Block
    assert config.mask_strategy == rc.MaskStrategy.Block
