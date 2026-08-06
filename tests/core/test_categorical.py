"""Categorical covariates and class-mapped classification targets.

Coverage kind: behavioural for the encoding contract (which string gets which
code, what happens to NA), structure for the rest. The contract these pin is
the one downstream code depends on: codes are ``1..K`` in sorted order over the
non-NA values, ``0`` is reserved for unknown/NA, and the vocab survives a
checkpoint round-trip so inference on a raw CSV encodes the same way training
did.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import make_dataset_config, make_train_config


def _roles() -> "rc.RoleMapping":
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    roles.covariates = ["altitude"]
    roles.categoricals = ["resurvey", "soil"]
    return roles


def _targets() -> list["rc.TargetSpec"]:
    return [
        rc.TargetSpec.regression("y"),
        rc.TargetSpec.classification("eunis", 9),
    ]


@pytest.fixture
def categorical_dataset(categorical_csvs) -> "rc.ResolveDataset":
    header, species = categorical_csvs
    return rc.ResolveDataset.from_csv(
        header, species, _roles(), _targets(), make_dataset_config(hash_dim=8)
    )


# ---------------------------------------------------------------------------
# Vocabulary contract
# ---------------------------------------------------------------------------

def test_codes_are_sorted_and_start_at_one(categorical_dataset):
    vocab = categorical_dataset.categorical_vocab
    assert vocab.column_names == ["resurvey", "soil"]
    # Sorted unique non-NA values numbered from 1; 0 stays reserved.
    assert vocab.column_map("resurvey") == {"N": 1, "Y": 2}
    assert vocab.column_map("soil") == {"clay": 1, "sand": 2, "silt": 3}


def test_vocab_size_counts_the_reserved_unknown_slot(categorical_dataset):
    vocab = categorical_dataset.categorical_vocab
    assert vocab.vocab_sizes == [3, 4]
    assert vocab.vocab_size("resurvey") == 3
    assert vocab.vocab_size("soil") == 4


def test_unknown_and_na_encode_to_zero(categorical_dataset):
    vocab = categorical_dataset.categorical_vocab
    assert vocab.encode("soil", "sand") == 2
    assert vocab.encode("soil", "not_a_soil") == 0

    # Row 1 has an empty resurvey cell, row 7 the string "NA"; both are unknown.
    codes = categorical_dataset.categorical_ids[:, 0]
    assert int(codes[1]) == 0
    assert int(codes[7]) == 0
    assert int(codes[0]) == 2  # "Y"


def test_categorical_ids_shape_and_dtype(categorical_dataset):
    ids = categorical_dataset.categorical_ids
    assert ids.shape == (categorical_dataset.n_plots, 2)
    assert ids.dtype == torch.int64
    assert int(ids.min()) >= 0
    assert int(ids[:, 0].max()) < 3
    assert int(ids[:, 1].max()) < 4


def test_has_column_predicate(categorical_dataset):
    vocab = categorical_dataset.categorical_vocab
    assert vocab.has_column("soil")
    assert not vocab.has_column("bedrock")


# ---------------------------------------------------------------------------
# Schema wiring
# ---------------------------------------------------------------------------

def test_schema_carries_the_categorical_layout(categorical_dataset):
    schema = categorical_dataset.schema
    assert schema.has_categoricals()
    assert schema.n_categoricals() == 2
    assert schema.categorical_names == ["resurvey", "soil"]
    assert schema.categorical_vocab_sizes == [3, 4]
    assert schema.categorical_embed_dim > 0


def test_numeric_covariate_stays_out_of_the_categorical_block(categorical_dataset):
    assert categorical_dataset.schema.covariate_names == ["altitude"]
    assert categorical_dataset.covariates.shape == (categorical_dataset.n_plots, 1)


# ---------------------------------------------------------------------------
# Letter-coded classification target
# ---------------------------------------------------------------------------

def test_letter_target_factorized_alphabetically(categorical_dataset):
    eunis = next(t for t in categorical_dataset.schema.targets if t.name == "eunis")
    assert eunis.class_names == ["M", "N", "P", "Q", "R", "S", "T", "U", "V"]
    assert eunis.num_classes == 9

    codes = categorical_dataset.targets["eunis"]
    assert codes.dtype == torch.int64
    assert int(codes.min()) == 0
    assert int(codes.max()) == 8


def test_explicit_class_mapping_is_honoured(categorical_csvs):
    header, species = categorical_csvs
    mapping = {letter: i for i, letter in enumerate("VUTSRQPNM")}
    dataset = rc.ResolveDataset.from_csv(
        header,
        species,
        _roles(),
        [
            rc.TargetSpec.regression("y"),
            rc.TargetSpec.classification_with_mapping("eunis", mapping),
        ],
        make_dataset_config(hash_dim=8),
    )
    eunis = next(t for t in dataset.schema.targets if t.name == "eunis")
    assert eunis.num_classes == 9
    # "M" is last in the explicit mapping, unlike the alphabetical default.
    assert eunis.class_names.index("M") == 8


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

def test_schema_source_reuse_keeps_the_same_codes(categorical_csvs, categorical_dataset):
    header, species = categorical_csvs
    rebuilt = rc.ResolveDataset.from_csv_with_schema(
        header, species, _roles(), _targets(), categorical_dataset,
        make_dataset_config(hash_dim=8),
    )
    assert torch.equal(rebuilt.categorical_ids, categorical_dataset.categorical_ids)
    assert rebuilt.categorical_vocab.column_map("soil") == \
        categorical_dataset.categorical_vocab.column_map("soil")


def test_vocab_survives_a_checkpoint_round_trip(tmp_path, categorical_dataset):
    """The Predictor must recover the training vocab to score a raw CSV.

    Without it, inference re-factorizes the strings against whatever values
    happen to appear in the new file and looks up the wrong embedding rows.
    """
    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Hash
    config.hash_dim = 8
    config.hidden_dims = [16]

    model = rc.ResolveModel(categorical_dataset.schema, config)
    trainer = rc.Trainer(model, make_train_config(max_epochs=2, batch_size=16))
    trainer.prepare_data(categorical_dataset, 0.25, 42)
    trainer.fit()

    assert trainer.categorical_vocab.column_names == ["resurvey", "soil"]

    path = str(tmp_path / "cat_model.pt")
    trainer.save(path)

    predictor = rc.Predictor.load(path, device="cpu")
    recovered = predictor.categorical_vocab
    assert recovered.column_names == ["resurvey", "soil"]
    assert recovered.vocab_sizes == [3, 4]
    assert recovered.column_map("soil") == {"clay": 1, "sand": 2, "silt": 3}

    predictions = predictor.predict_dataset(categorical_dataset, False, 32)
    assert len(predictions.plot_ids) == categorical_dataset.n_plots
    assert torch.isfinite(predictions.predictions["y"]).all()
