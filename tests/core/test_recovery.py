"""End-to-end recovery: does a model fitted through the bindings learn?

Coverage kind: **parameter recovery**, not smoke. Each case simulates data from
a known generating function, fits to convergence on a training split, and
asserts the estimate on a *held-out* split is close to the truth. A binding
that returns correctly-shaped garbage — a mis-threaded pool tensor, a scaler
applied to the wrong axis, an encoder that never sees the species — passes a
shape test and fails these.

The same generating functions are used by ``src/core/tests/test_recovery.cpp``,
so a divergence between the C++ and Python results localizes to the binding
layer rather than to the engine. Thresholds are the C++ ones.

Runtime on a CPU runner is a few minutes; the fits are deliberately small
(600 plots, 12 species, <= 400 epochs with early stopping).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import resolve_core as rc

from conftest import make_train_config, write_csv


# ---------------------------------------------------------------------------
# Generating functions
# ---------------------------------------------------------------------------

def _species_signal_csvs(tmp_path, n_plots: int = 600, n_species: int = 12):
    """y is the sum of per-species contributions over each plot's species set.

    Deterministic species assignment: with 600 plots over 12 species every
    embedding row is visited hundreds of times, so what is being measured is
    whether the pooling reaches the head, not whether a rare row got enough
    gradient.
    """
    values = [math.sin(s * 1.7) * 2.0 for s in range(n_species)]
    header_rows, species_rows = [], []
    for i in range(n_plots):
        s0 = i % n_species
        s1 = (i // n_species) % n_species
        s2 = (i // (n_species * n_species) + i) % n_species
        y = values[s0] + values[s1] + values[s2]
        header_rows.append([f"P{i}", f"{y:.6f}"])
        for s in (s0, s1, s2):
            species_rows.append([f"P{i}", f"sp{s}", "1.0"])

    header = write_csv(tmp_path / "sig_header.csv", ["plot_id", "y"], header_rows)
    species = write_csv(
        tmp_path / "sig_species.csv", ["plot_id", "sp", "cover"], species_rows
    )
    return str(header), str(species)


def _covariate_signal_csvs(tmp_path, n_plots: int = 600):
    """y = 3*cov1 - 2*cov2 + 1, a linear signal carried entirely by covariates."""
    header_rows, species_rows = [], []
    for i in range(n_plots):
        cov1 = (i % 37) / 37.0
        cov2 = (i % 23) / 23.0
        y = 3.0 * cov1 - 2.0 * cov2 + 1.0
        header_rows.append(
            [f"P{i}", f"{cov1:.6f}", f"{cov2:.6f}", f"{y:.6f}"]
        )
        species_rows.append([f"P{i}", f"sp{i % 5}", "1.0"])

    header = write_csv(
        tmp_path / "cov_header.csv", ["plot_id", "cov1", "cov2", "y"], header_rows
    )
    species = write_csv(
        tmp_path / "cov_species.csv", ["plot_id", "sp", "cover"], species_rows
    )
    return str(header), str(species)


def _separable_class_csvs(tmp_path, n_plots: int = 600):
    """Three classes separated by an indicator species, plus a matching covariate."""
    header_rows, species_rows = [], []
    for i in range(n_plots):
        label = i % 3
        header_rows.append([f"P{i}", label, f"{label * 5.0:.3f}"])
        for k in range(3):
            species_rows.append([f"P{i}", f"cls{label}_sp{k}", "1.0"])

    header = write_csv(
        tmp_path / "cls_header.csv", ["plot_id", "label", "signal"], header_rows
    )
    species = write_csv(
        tmp_path / "cls_species.csv", ["plot_id", "sp", "cover"], species_rows
    )
    return str(header), str(species)


def _minimal_roles(*, covariates: list[str] | None = None) -> "rc.RoleMapping":
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    if covariates:
        roles.covariates = list(covariates)
    return roles


def _dataset_config(encoding, **overrides) -> "rc.DatasetConfig":
    config = rc.DatasetConfig()
    config.species_encoding = encoding
    config.use_taxonomy = False
    config.track_unknown_fraction = False
    config.track_unknown_count = False
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _recovery_train_config(max_epochs: int = 400) -> "rc.TrainConfig":
    config = make_train_config(max_epochs=max_epochs, batch_size=64, lr=1e-2)
    config.patience = 40
    return config


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _fit(dataset, model_config, *, max_epochs: int = 400, seed: int = 7):
    model = rc.ResolveModel(dataset.schema, model_config)
    trainer = rc.Trainer(model, _recovery_train_config(max_epochs))
    trainer.prepare_data(dataset, 0.25, seed)
    trainer.fit()
    return trainer


# ---------------------------------------------------------------------------
# Regression from covariates
# ---------------------------------------------------------------------------

def test_regression_recovers_a_linear_covariate_signal(tmp_path):
    header, species = _covariate_signal_csvs(tmp_path)
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        _minimal_roles(covariates=["cov1", "cov2"]),
        [rc.TargetSpec.regression("y")],
        _dataset_config(rc.SpeciesEncodingMode.Hash, hash_dim=16),
    )

    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Hash
    config.hash_dim = 16
    config.hidden_dims = [64, 32]

    trainer = _fit(dataset, config, seed=11)
    residuals = trainer.compute_residuals("y")

    assert len(residuals.predictions) > 10
    assert _pearson(residuals.predictions, residuals.actuals) > 0.9


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classification_separates_a_separable_signal(tmp_path):
    header, species = _separable_class_csvs(tmp_path)
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        _minimal_roles(covariates=["signal"]),
        [rc.TargetSpec.classification("label", 3)],
        _dataset_config(rc.SpeciesEncodingMode.Hash, hash_dim=16),
    )

    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Hash
    config.hash_dim = 16
    config.hidden_dims = [64, 32]

    trainer = _fit(dataset, config, max_epochs=200, seed=3)
    preds = trainer.compute_classification_predictions("label")

    accuracy = float(
        (np.asarray(preds.predicted_classes) == np.asarray(preds.actuals)).mean()
    )
    assert accuracy > 0.9


# ---------------------------------------------------------------------------
# Species-composition encoders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("encoding", "build_model", "threshold"),
    [
        pytest.param(
            rc.SpeciesEncodingMode.RankPool,
            lambda: _model_config_rank_pool(),
            0.8,
            id="rank_pool",
        ),
        pytest.param(
            rc.SpeciesEncodingMode.Transformer,
            lambda: _model_config_transformer(),
            0.7,
            id="transformer",
        ),
        pytest.param(
            rc.SpeciesEncodingMode.Embed,
            lambda: _model_config_embed(),
            0.7,
            id="embed",
        ),
    ],
)
def test_encoder_recovers_a_species_driven_target(
    tmp_path, encoding, build_model, threshold
):
    """The target is a pure function of composition, so a working encoder finds it.

    A broken pooling path — weights dropped, mask ignored, species tensor never
    reaching the encoder — leaves correlation near zero on the held-out fold
    while every shape assertion still passes.
    """
    header, species = _species_signal_csvs(tmp_path)
    config = _dataset_config(encoding)
    if encoding == rc.SpeciesEncodingMode.Embed:
        config.top_k_species = 3
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        _minimal_roles(),
        [rc.TargetSpec.regression("y")],
        config,
    )

    trainer = _fit(dataset, build_model())
    residuals = trainer.compute_residuals("y")

    assert len(residuals.predictions) > 10
    assert _pearson(residuals.predictions, residuals.actuals) > threshold


def _model_config_rank_pool() -> "rc.ModelConfig":
    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.RankPool
    config.species_embed_dim = 16
    config.hidden_dims = [64, 32]
    return config


def _model_config_transformer() -> "rc.ModelConfig":
    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Transformer
    config.d_model = 32
    config.n_heads = 4
    config.n_attention_layers = 1
    config.transformer_ff_dim = 64
    config.transformer_pooling = "attention"
    config.hidden_dims = [32]
    return config


def _model_config_embed() -> "rc.ModelConfig":
    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Embed
    config.species_embed_dim = 16
    config.top_k_species = 3
    config.hidden_dims = [64, 32]
    return config


# ---------------------------------------------------------------------------
# The recovered model survives the checkpoint
# ---------------------------------------------------------------------------

def test_a_recovered_model_still_recovers_after_a_round_trip(tmp_path):
    """Saving and reloading must not lose what the model learned.

    Shape-only checkpoint tests pass even when a layer reloads at random init;
    re-measuring recovery through the reloaded Predictor does not.
    """
    header, species = _species_signal_csvs(tmp_path)
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        _minimal_roles(),
        [rc.TargetSpec.regression("y")],
        _dataset_config(rc.SpeciesEncodingMode.RankPool),
    )

    trainer = _fit(dataset, _model_config_rank_pool())
    path = str(tmp_path / "recovered.pt")
    trainer.save(path)

    predictor = rc.Predictor.load(path, device="cpu")
    out = predictor.predict_dataset(dataset, False, 256)

    test_idx = trainer.test_indices().tolist()
    predicted = out.predictions["y"].numpy()[test_idx]
    actual = dataset.targets["y"].numpy()[test_idx]
    assert _pearson(predicted, actual) > 0.8
