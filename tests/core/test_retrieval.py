"""Retrieval-augmented prediction through ``Predictor.set_retrieval``.

Coverage kind: behavioural. The blend is recomputed here from the model's own
latents and predictions with numpy (cosine top-k, softmax of similarity over
temperature, renormalised over the labelled neighbours) and compared with the
engine's, for a regression target in original units and a classification
target as class probabilities; k = 1, alpha = 1 on the reference plots returns
their own labels.
"""

from __future__ import annotations

import numpy as np
import pytest

import resolve_core as rc


@pytest.fixture
def predictor(tmp_path, fitted_trainer) -> "rc.Predictor":
    path = str(tmp_path / "model.pt")
    fitted_trainer.trainer.save(path)
    return rc.Predictor.load(path)


def _config(k: int, alpha: float, temperature: float = 1.0) -> "rc.RetrievalConfig":
    config = rc.RetrievalConfig()
    config.k_neighbors = k
    config.alpha = alpha
    config.temperature = temperature
    return config


def test_k1_alpha1_on_the_reference_returns_its_labels(predictor, hash_dataset):
    predictor.set_retrieval(hash_dataset, _config(1, 1.0), batch_size=17)
    assert predictor.retrieval.k_neighbors == 1
    assert predictor.n_retrieval_reference == hash_dataset.n_plots

    out = predictor.predict_dataset(hash_dataset, batch_size=50)
    targets = hash_dataset.targets
    np.testing.assert_allclose(
        out.predictions["y"].numpy(), targets["y"].numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(
        out.predictions["hab"].numpy(), targets["hab"].numpy().astype(np.int64)
    )


def test_the_blend_matches_an_independent_computation(predictor, hash_dataset):
    plain = predictor.predict_dataset(hash_dataset, return_latent=True)
    k, alpha, temperature = 5, 0.4, 0.5
    predictor.set_retrieval(hash_dataset, _config(k, alpha, temperature))
    blended = predictor.predict_dataset(hash_dataset, return_latent=True)

    latent = plain.latent.numpy().astype(np.float64)
    unit = latent / np.linalg.norm(latent, axis=1, keepdims=True)
    similarity = unit @ unit.T
    neighbours = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    nearest = np.take_along_axis(similarity, neighbours, axis=1) / temperature
    weights = np.exp(nearest - nearest.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)

    y = hash_dataset.targets["y"].numpy().astype(np.float64)
    expected_y = (1 - alpha) * plain.predictions["y"].numpy() + alpha * (
        weights * y[neighbours]
    ).sum(1)
    np.testing.assert_allclose(
        blended.predictions["y"].numpy(), expected_y, rtol=1e-4, atol=1e-4
    )

    hab = hash_dataset.targets["hab"].numpy().astype(np.int64)
    distribution = np.zeros_like(plain.probabilities["hab"].numpy(), dtype=np.float64)
    for row in range(distribution.shape[0]):
        np.add.at(distribution[row], hab[neighbours[row]], weights[row])
    expected_probs = (1 - alpha) * plain.probabilities[
        "hab"
    ].numpy() + alpha * distribution
    np.testing.assert_allclose(
        blended.probabilities["hab"].numpy(), expected_probs, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_array_equal(
        blended.predictions["hab"].numpy(), expected_probs.argmax(1)
    )
    # The latent is the model's own, not blended.
    np.testing.assert_allclose(blended.latent.numpy(), plain.latent.numpy())


def test_clear_retrieval_restores_the_model(predictor, hash_dataset):
    plain = predictor.predict_dataset(hash_dataset)
    assert predictor.retrieval is None
    predictor.set_retrieval(hash_dataset, _config(3, 0.5))
    predictor.clear_retrieval()
    assert predictor.retrieval is None
    assert predictor.n_retrieval_reference == 0
    again = predictor.predict_dataset(hash_dataset)
    np.testing.assert_array_equal(
        again.predictions["y"].numpy(), plain.predictions["y"].numpy()
    )


def test_retrieval_refuses_an_unlabelled_reference_and_bad_settings(
    predictor, plot_csvs
):
    from conftest import make_dataset_config, make_roles

    unlabelled = rc.ResolveDataset.from_csv_with_vocabs(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        [],
        predictor.external_vocabs,
        make_dataset_config(rc.SpeciesEncodingMode.Hash),
    )
    with pytest.raises(ValueError, match="no labels for target"):
        predictor.set_retrieval(unlabelled)
    assert predictor.retrieval is None
    with pytest.raises(ValueError, match="k_neighbors"):
        predictor.set_retrieval(unlabelled, _config(0, 0.3))
    with pytest.raises(ValueError, match="alpha"):
        _config(5, 1.5).validate()


def test_target_specs_from_schema_load_a_labelled_reference(predictor, plot_csvs):
    from conftest import make_dataset_config, make_roles

    specs = rc.target_specs_from_schema(predictor.schema)
    assert [s.target_name for s in specs] == [t.name for t in predictor.schema.targets]
    assert [s.column_name for s in specs] == [t.name for t in predictor.schema.targets]
    reference = rc.ResolveDataset.from_csv_with_vocabs(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        specs,
        predictor.external_vocabs,
        make_dataset_config(rc.SpeciesEncodingMode.Hash),
    )
    predictor.set_retrieval(reference, _config(1, 1.0))
    out = predictor.predict_dataset(reference)
    np.testing.assert_allclose(
        out.predictions["y"].numpy(),
        reference.targets["y"].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
