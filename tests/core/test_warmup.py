"""Learning-rate warmup and the linear schedule through the nanobind bindings.

``TrainConfig.warmup_steps`` / ``warmup_epochs`` ramp the rate linearly over
the run's first optimizer steps, composing with every schedule;
``LRSchedulerType.LinearDecay`` lowers it per step from the end of warmup to
``lr_min``. The pretraining configs carry the same warmup together with the
out-of-memory floor and the VRAM cap of the shared pretraining loop.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


def _fit(dataset, config):
    torch.manual_seed(3)
    trainer = rc.Trainer(rc.ResolveModel(dataset.schema, make_model_config()), config)
    trainer.prepare_data(dataset, 0.25, 42)
    return trainer, trainer.fit()


def test_the_defaults_train_without_warmup():
    config = rc.TrainConfig()
    assert config.warmup_steps == 0
    assert config.warmup_epochs == 0
    for cls in (rc.PretrainConfig, rc.MLMPretrainConfig, rc.VAEConfig):
        pretrain = cls()
        assert pretrain.warmup_steps == 0
        assert pretrain.batch_size_floor == 1024
        assert pretrain.vram_fraction == pytest.approx(1.0)


def test_a_warmup_ramps_the_recorded_rate(hash_dataset):
    # 90 fitting plots in batches of 16: six steps an epoch, so a twelve-step
    # warmup closes with epoch 1.
    config = make_train_config(max_epochs=4, batch_size=16, lr=1e-2)
    config.warmup_steps = 12
    _, result = _fit(hash_dataset, config)
    assert result.lr_history == pytest.approx([5e-3, 1e-2, 1e-2, 1e-2], rel=1e-6)

    config.warmup_steps = 0
    _, cold = _fit(hash_dataset, config)
    assert cold.lr_history == pytest.approx([1e-2] * 4, rel=1e-6)
    assert cold.train_loss_history != result.train_loss_history


def test_a_linear_schedule_ends_at_lr_min_and_round_trips(tmp_path, hash_dataset):
    config = make_train_config(max_epochs=4, batch_size=16, lr=1e-2)
    config.lr_scheduler = rc.LRSchedulerType.LinearDecay
    config.lr_min = 1e-4
    config.warmup_epochs = 1
    trainer, result = _fit(hash_dataset, config)
    assert result.lr_history[0] == pytest.approx(1e-2, rel=1e-6)
    assert result.lr_history[-1] == pytest.approx(1e-4, rel=1e-5)

    path = str(tmp_path / "linear.pt")
    trainer.save(path)
    loaded = rc.Trainer.load_train_config(path)
    assert loaded.lr_scheduler == rc.LRSchedulerType.LinearDecay
    assert loaded.warmup_epochs == 1
    assert loaded.warmup_steps == 0


def test_an_unusable_warmup_is_refused(hash_dataset):
    config = make_train_config(max_epochs=2)
    config.warmup_steps = 3
    config.warmup_epochs = 1
    trainer = rc.Trainer(rc.ResolveModel(hash_dataset.schema, make_model_config()), config)
    with pytest.raises(ValueError, match="give one of them"):
        trainer.prepare_data(hash_dataset, 0.25, 42)


def test_masked_species_pretraining_takes_a_warmup(pool_dataset):
    def run(warmup_steps):
        config = make_model_config(rc.SpeciesEncodingMode.Transformer)
        config.d_model = 16
        config.n_heads = 2
        config.n_attention_layers = 1
        torch.manual_seed(7)
        model = rc.ResolveModel(pool_dataset.schema, config)
        mlm = rc.MLMPretrainConfig()
        mlm.pretrain_epochs = 2
        mlm.batch_size = 16
        mlm.mask_prob = 0.5
        mlm.warmup_steps = warmup_steps
        return rc.MaskedSpeciesPretrainer(model, mlm).pretrain(
            species_ids=pool_dataset.species_ids,
            pool_genus_ids=pool_dataset.pool_genus_ids,
            pool_family_ids=pool_dataset.pool_family_ids,
            pool_weights=pool_dataset.pool_weights,
            pool_mask=pool_dataset.pool_mask,
        )

    cold = run(0)
    assert cold.effective_batch_size == 16
    assert run(1).loss_history == cold.loss_history
    assert run(10).loss_history != cold.loss_history
