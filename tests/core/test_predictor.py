"""``Predictor``: checkpoint round-trip and inference through the bindings.

Coverage kind: one genuine numerical invariant (chunked prediction is
bit-equivalent to the one-shot forward, and a reloaded checkpoint reproduces
the trainer's own predictions) plus shape/definedness checks on the embedding
accessors.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config


@pytest.fixture
def saved_model(tmp_path, fitted_trainer) -> str:
    path = str(tmp_path / "model.pt")
    fitted_trainer.trainer.save(path)
    return path


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_predictor_reproduces_the_trainers_predictions(saved_model, fitted_trainer):
    """train -> save -> load -> predict must return the trainer's own numbers."""
    trainer, dataset = fitted_trainer.trainer, fitted_trainer.dataset
    reference = trainer.compute_residuals("y")

    predictor = rc.Predictor.load(saved_model, device="cpu")
    predictions = predictor.predict_dataset(dataset, False, 64)

    test_idx = trainer.test_indices().tolist()
    reloaded = predictions.predictions["y"].numpy()[test_idx]
    np.testing.assert_allclose(
        reloaded, np.asarray(reference.predictions), rtol=1e-5, atol=1e-5
    )


def test_predict_dataset_returns_every_target(saved_model, fitted_trainer):
    predictor = rc.Predictor.load(saved_model, device="cpu")
    out = predictor.predict_dataset(fitted_trainer.dataset, False, 64)

    assert set(out.predictions) == {"y", "hab"}
    assert len(out.plot_ids) == fitted_trainer.dataset.n_plots
    assert list(out.plot_ids) == list(fitted_trainer.dataset.plot_ids)
    assert out.predictions["y"].shape == (fitted_trainer.dataset.n_plots,)
    assert torch.isfinite(out.predictions["y"]).all()
    assert out.predictions["hab"].dtype == torch.int64
    assert int(out.predictions["hab"].max()) < 3


def test_chunked_prediction_matches_the_one_shot_forward(saved_model, fitted_trainer):
    """``batch_size`` only slices the input; the numbers must not move."""
    predictor = rc.Predictor.load(saved_model, device="cpu")
    dataset = fitted_trainer.dataset

    one_shot = predictor.predict_dataset(dataset, False, -1)
    for batch_size in (1, 7, 4096):
        chunked = predictor.predict_dataset(dataset, False, batch_size)
        for target in one_shot.predictions:
            assert torch.allclose(
                chunked.predictions[target].float(),
                one_shot.predictions[target].float(),
                rtol=1e-5,
                atol=1e-6,
            ), f"{target} differs at batch_size={batch_size}"


def test_invalid_batch_size_is_rejected(saved_model, fitted_trainer):
    predictor = rc.Predictor.load(saved_model, device="cpu")
    for bad in (0, -2):
        with pytest.raises(Exception):
            predictor.predict_dataset(fitted_trainer.dataset, False, bad)


def test_latent_is_returned_on_request(saved_model, fitted_trainer):
    predictor = rc.Predictor.load(saved_model, device="cpu")

    without = predictor.predict_dataset(fitted_trainer.dataset, False, 64)
    assert without.latent is None or without.latent.numel() == 0

    with_latent = predictor.predict_dataset(fitted_trainer.dataset, True, 64)
    assert with_latent.latent is not None
    assert with_latent.latent.shape == (
        fitted_trainer.dataset.n_plots,
        predictor.model.latent_dim,
    )
    assert torch.isfinite(with_latent.latent).all()


# ---------------------------------------------------------------------------
# Embedding accessors
# ---------------------------------------------------------------------------

def test_taxonomy_embedding_accessors(saved_model, fitted_trainer):
    predictor = rc.Predictor.load(saved_model, device="cpu")
    schema = fitted_trainer.dataset.schema

    genus = predictor.get_genus_embeddings()
    family = predictor.get_family_embeddings()
    assert genus.ndim == 2
    assert genus.shape[0] == schema.n_genera
    assert family.shape[0] == schema.n_families
    assert torch.isfinite(genus).all()
    assert torch.isfinite(family).all()


def test_predictor_carries_the_model_schema(saved_model, fitted_trainer):
    predictor = rc.Predictor.load(saved_model, device="cpu")
    schema = predictor.model.schema
    assert {t.name for t in schema.targets} == {"y", "hab"}
    assert schema.n_species_vocab == fitted_trainer.dataset.schema.n_species_vocab
    assert predictor.device == "cpu"


# ---------------------------------------------------------------------------
# Pool encoders survive the checkpoint
# ---------------------------------------------------------------------------

def test_rank_pool_checkpoint_round_trip(tmp_path, pool_dataset):
    """A rank-pool checkpoint must reload with its pool hyperparameters intact.

    ``cover_dropout`` and the schema's pool weighting / species cap are what let
    ``predict`` recompute the same pool weights the model trained on.
    """
    config = make_model_config(rc.SpeciesEncodingMode.RankPool)
    config.cover_dropout = 0.15
    model = rc.ResolveModel(pool_dataset.schema, config)
    trainer = rc.Trainer(model, make_train_config(max_epochs=3))
    trainer.prepare_data(pool_dataset, 0.25, 42)
    trainer.fit()

    path = str(tmp_path / "pool.pt")
    trainer.save(path)

    predictor = rc.Predictor.load(path, device="cpu")
    reloaded = predictor.model.config
    assert reloaded.species_encoding == rc.SpeciesEncodingMode.RankPool
    assert reloaded.cover_dropout == pytest.approx(0.15)
    assert predictor.model.schema.pool_weighting == rc.PoolWeighting.Log1p.value
    assert predictor.model.schema.pool_species_cap == pool_dataset.schema.pool_species_cap

    out = predictor.predict_dataset(pool_dataset, False, 64)
    assert torch.isfinite(out.predictions["y"]).all()


def test_transformer_checkpoint_round_trip(tmp_path, plot_csvs):
    """Every transformer hyperparameter must round-trip, or the reload resizes."""
    from conftest import make_dataset_config, make_roles, make_targets

    dataset = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(rc.SpeciesEncodingMode.Transformer),
    )

    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Transformer
    config.d_model = 32
    config.n_heads = 4
    config.n_attention_layers = 1
    config.transformer_ff_dim = 64
    config.transformer_pooling = "attention"
    config.transformer_dropout = 0.05
    config.hidden_dims = [32]

    model = rc.ResolveModel(dataset.schema, config)
    trainer = rc.Trainer(model, make_train_config(max_epochs=2))
    trainer.prepare_data(dataset, 0.25, 42)
    trainer.fit()

    path = str(tmp_path / "transformer.pt")
    trainer.save(path)

    reloaded = rc.Predictor.load(path, device="cpu").model.config
    assert reloaded.d_model == 32
    assert reloaded.n_heads == 4
    assert reloaded.n_attention_layers == 1
    assert reloaded.transformer_ff_dim == 64
    assert reloaded.transformer_pooling == "attention"
    assert reloaded.transformer_dropout == pytest.approx(0.05)
