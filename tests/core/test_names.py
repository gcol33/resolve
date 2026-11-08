"""Name resolution through the Python bindings.

Coverage kind: behaviour. A species name reaches a model's vocabulary as
written, through the release's name table, or as the nearest known spelling;
these tests pin the routes a caller sees, the table file and its cache, the
report a load and a suite prediction return, and that resolved names predict
exactly as the tokens written directly. The routes' edge cases (ties, the
genus bucket, the widening past the first matches) are covered by
``src/core/tests/test_name_resolution.cpp``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import resolve_core as rc
from test_suite import _config, _roles, scoring, suite_dir  # noqa: F401

VOCAB = ["<UNK>", "Quercus robur", "Anemone altaica", "Carex curvicollis"]


@pytest.fixture
def table():
    return rc.NameTable.from_columns(
        ["Anemone nemorosa", "Carex viridula"],
        ["Anemone altaica", "Carex curvicollis"],
        ["eva", "eva"],
    )


def test_each_route_and_what_it_reports(table):
    r = rc.NameResolver(VOCAB, table, 1)
    exact = r.resolve("Quercus robur")
    assert (exact.token, exact.route) == ("Quercus robur", rc.NameRoute.Exact)
    alias = r.resolve("Anemone nemorosa")
    assert (alias.token, alias.route, alias.source) == (
        "Anemone altaica",
        rc.NameRoute.Alias,
        "eva",
    )
    typo = r.resolve("Carex viridul")
    assert (typo.token, typo.route, typo.via, typo.source, typo.distance) == (
        "Carex curvicollis",
        rc.NameRoute.Spelling,
        "Carex viridula",
        "eva",
        1,
    )
    assert r.resolve("Poa annua").route == rc.NameRoute.Unresolved
    assert (
        rc.NameResolver(VOCAB, table, 0).resolve("Carex viridul").route
        == rc.NameRoute.Unresolved
    )
    with pytest.raises(Exception, match="max_edit_distance"):
        rc.NameResolver(VOCAB, table, 9)


def test_resolve_names_returns_one_row_per_name(table):
    out = rc.resolve_names(
        ["Quercus robru", "Anemone nemorosa", "Poa annua"], VOCAB, table=table
    )
    assert list(out.columns) == ["name", "token", "route", "source", "via", "distance"]
    assert out["route"].tolist() == ["spelling", "alias", "unresolved"]
    assert out["token"].tolist() == ["Quercus robur", "Anemone altaica", "Poa annua"]
    # Without a table only the vocabulary and spellings are read.
    assert rc.resolve_names(["Anemone nemorosa"], VOCAB, table=None)[
        "route"
    ].tolist() == ["unresolved"]


def test_the_table_file_is_read_once_per_process(tmp_path: Path):
    path = tmp_path / "names.json"
    path.write_text(
        json.dumps(
            {
                "source": "test",
                "aliases": {"Anemone nemorosa": ["Anemone altaica", "eva"]},
            }
        ),
        encoding="utf-8",
    )
    first = rc.NameTable.open(str(path))
    assert len(first) == 1 and "Anemone nemorosa" in first
    assert first.find("Anemone nemorosa") == ("Anemone altaica", "eva")
    assert first.description == "test"
    assert first.sha256 == rc.sha256_file(str(path))
    assert rc.NameTable.open(str(path)).sha256 == first.sha256
    names = rc.SpeciesNames(first, 2)
    assert names.max_edit_distance == 2 and len(names.table) == 1
    assert rc.SpeciesNames().table is None


def _written(species):
    """The scoring species with an entered name for sp1 and a misspelling of sp_late."""
    out = species.copy()
    out["taxon"] = out["taxon"].replace(
        {"sp1": "Species one entered", "sp_late": "sp_latte"}
    )
    return out


def test_a_suite_reads_resolved_names_as_the_tokens(suite_dir, scoring):  # noqa: F811
    header, species, _, _ = scoring
    suite = rc.SuitePredictor.load(str(suite_dir))
    table = rc.NameTable.from_columns(["Species one entered"], ["sp1"], ["eva"])

    expected = suite.predict(species, header)
    resolved = suite.predict(_written(species), header, names=table)
    plain = suite.predict(_written(species), header)
    for name in ("hab", "y", "aspect"):
        assert torch.equal(resolved.target(name).value, expected.target(name).value)
        assert len(plain.target(name).species_names) == 0

    hab = resolved.target("hab").species_names.to_pandas()
    assert hab["name"].tolist() == ["Species one entered", "sp_latte", "sp_unseen"]
    assert hab["route"].tolist() == ["alias", "unresolved", "unresolved"]
    y = resolved.target("y").species_names.to_pandas()
    assert y["token"].tolist() == ["sp1", "sp_late", "sp_unseen"]
    assert y["route"].tolist() == ["alias", "spelling", "unresolved"]
    assert y["n_records"].tolist()[1] == int((species["taxon"] == "sp_late").sum())

    # Each target's vocabulary is the one its names resolve against.
    by_target = rc.resolve_names(["sp_latte"], suite, target="hab")
    assert by_target["route"].tolist() == ["unresolved"]
    assert rc.resolve_names(["sp_latte"], suite, target="y")["token"].tolist() == [
        "sp_late"
    ]
    with pytest.raises(ValueError, match="target"):
        rc.resolve_names(["sp_latte"], suite)


def test_a_dataset_load_reports_its_names(suite_dir, scoring):  # noqa: F811
    _, species, h_path, _ = scoring
    p = rc.Predictor.load(str(suite_dir / "y" / "seed_3" / "model_final.pt"))
    s_path = h_path.parent / "written_species.csv"
    _written(species).to_csv(s_path, index=False)
    names = rc.SpeciesNames(
        rc.NameTable.from_columns(["Species one entered"], ["sp1"], ["eva"])
    )
    ds = rc.ResolveDataset.from_csv_with_vocabs(
        str(h_path), str(s_path), _roles(), [], p.external_vocabs, _config(), names
    )
    report = ds.species_names.to_pandas()
    assert report["route"].tolist() == ["alias", "spelling", "unresolved"]
    assert rc.resolve_names(["sp_latte"], p)["token"].tolist() == ["sp_late"]
