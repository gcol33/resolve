"""Model inputs through the bindings.

Every method that takes model inputs reads them through one argument reader
built from the engine's own field lists: positionally in the model's field
order or by keyword, with an unknown keyword refused. The pool encoders and
categorical columns reach every such method, including the ones that used to
take a fixed subset of the inputs (forward_single, get_gate_probs,
Predictor.get_embeddings).
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import (
    make_dataset_config,
    make_model_config,
    make_roles,
    make_train_config,
    trainer_continuous,
)

POOL_FIELDS = (
    "species_ids",
    "pool_genus_ids",
    "pool_family_ids",
    "pool_weights",
    "pool_mask",
    "pool_has_cover",
)
CONTINUOUS_PARTS = (
    "coordinates",
    "covariates",
    "unknown_fraction",
    "unknown_count",
    "hash_embedding",
)


def _present(dataset, names):
    return {
        name: getattr(dataset, name)
        for name in names
        if getattr(dataset, name) is not None
    }


@pytest.fixture
def hash_model(hash_dataset):
    config = make_model_config()
    model = rc.ResolveModel(hash_dataset.schema, config)
    model.eval()
    return model, trainer_continuous(hash_dataset, config)


def test_positional_and_keyword_inputs_agree(hash_dataset, hash_model):
    model, continuous = hash_model
    by_keyword = model.forward(
        continuous=continuous,
        genus_ids=hash_dataset.genus_ids,
        family_ids=hash_dataset.family_ids,
    )
    by_position = model(continuous, hash_dataset.genus_ids, hash_dataset.family_ids)
    assert set(by_keyword) == set(by_position) == {"y", "hab"}
    for target in by_keyword:
        assert torch.allclose(by_keyword[target], by_position[target])


def test_an_unknown_keyword_is_refused(hash_dataset, hash_model):
    model, continuous = hash_model
    with pytest.raises(TypeError, match="genus_id"):
        model.forward(continuous, genus_id=hash_dataset.genus_ids)


def test_an_argument_given_twice_is_refused(hash_model):
    model, continuous = hash_model
    with pytest.raises(TypeError, match="continuous"):
        model.get_latent(continuous, continuous=continuous)


def test_too_many_positional_arguments_are_refused(hash_model):
    model, continuous = hash_model
    with pytest.raises(TypeError, match="positional"):
        model.forward(continuous, *([None] * 11))


def test_continuous_is_required(hash_dataset, hash_model):
    model, _ = hash_model
    with pytest.raises(ValueError, match="continuous"):
        model.forward(genus_ids=hash_dataset.genus_ids)


def test_a_non_tensor_input_is_refused(hash_model):
    model, continuous = hash_model
    with pytest.raises(ValueError, match="genus_ids"):
        model.forward(continuous, genus_ids=[1, 2, 3])


def test_forward_single_reads_the_pool_inputs(pool_dataset):
    config = make_model_config(rc.SpeciesEncodingMode.RankPool)
    model = rc.ResolveModel(pool_dataset.schema, config)
    model.eval()
    inputs = _present(pool_dataset, POOL_FIELDS)
    continuous = trainer_continuous(pool_dataset, config)

    single = model.forward_single("y", continuous, **inputs)
    full = model.forward(continuous, **inputs)["y"]
    assert torch.allclose(single, full)


def test_predictor_tensor_paths_match_the_dataset_path(tmp_path, pool_dataset):
    model = rc.ResolveModel(
        pool_dataset.schema, make_model_config(rc.SpeciesEncodingMode.RankPool)
    )
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    trainer.prepare_data(pool_dataset, 0.25, 42)
    path = str(tmp_path / "pool.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path, device="cpu")

    kwargs = {
        **_present(pool_dataset, CONTINUOUS_PARTS),
        **_present(pool_dataset, POOL_FIELDS),
    }
    reference = predictor.predict_dataset(pool_dataset, True, -1)

    by_tensors = predictor.predict(**kwargs, return_latent=True, batch_size=7)
    for target in reference.predictions:
        assert torch.allclose(
            by_tensors.predictions[target].float(),
            reference.predictions[target].float(),
            rtol=1e-5,
            atol=1e-6,
        )
    assert torch.allclose(by_tensors.latent, reference.latent, rtol=1e-5, atol=1e-6)

    embeddings = predictor.get_embeddings(**kwargs)
    assert embeddings.shape == (pool_dataset.n_plots, predictor.model.latent_dim)
    assert torch.allclose(embeddings, reference.latent, rtol=1e-5, atol=1e-6)

    with pytest.raises(TypeError):
        predictor.predict(pool_dataset.coordinates)


def test_embeddings_read_the_categorical_columns(tmp_path, categorical_csvs):
    header, species = categorical_csvs
    roles = make_roles(
        taxonomy=False, coordinates=False, covariates=False, categoricals=["soil"]
    )
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        roles,
        [rc.TargetSpec.regression("y")],
        make_dataset_config(rc.SpeciesEncodingMode.Embed, use_taxonomy=False),
    )
    model = rc.ResolveModel(
        dataset.schema, make_model_config(rc.SpeciesEncodingMode.Embed)
    )
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    trainer.prepare_data(dataset, 0.25, 42)
    path = str(tmp_path / "embed.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path, device="cpu")

    kwargs = {
        **_present(dataset, CONTINUOUS_PARTS),
        **_present(dataset, ("species_ids", "categorical_ids")),
    }
    embeddings = predictor.get_embeddings(**kwargs)
    reference = predictor.predict_dataset(dataset, True, -1).latent
    assert torch.allclose(embeddings, reference, rtol=1e-5, atol=1e-6)

    kwargs["categorical_ids"] = torch.zeros_like(dataset.categorical_ids)
    assert not torch.allclose(predictor.get_embeddings(**kwargs), reference)
