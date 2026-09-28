"""Model suites through the Python bindings.

A small suite is written to disk -- a class vote, a mean, and a circular mean
read from a sine / cosine pair, each over members of different seeds -- then
loaded and scored from pandas DataFrames and from CSV paths. Every combined
prediction is checked against the same rule applied to the members scored one
at a time through ``Predictor``, and the manifest is checked as the contract it
is: a changed file, a wrong input column or an unknown key stops the load.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config, write_csv


# ---------------------------------------------------------------------------
# A suite on disk
# ---------------------------------------------------------------------------

def _plots(n: int, *, with_targets: bool, extra: bool = False):
    header, species = [], []
    for i in range(n):
        picks = [f"sp{(i + k) % 6}" for k in range(3)]
        if i >= 30:
            picks.append("sp_late")  # only in the later plots
        bearing = (41.0 * i) % 360.0
        row = [f"P{i}", f"{46.0 + 0.05 * i:.4f}", f"{9.0 + 0.03 * i:.4f}", 100.0 + 5 * i]
        if with_targets:
            r = math.radians(bearing)
            row += ["ABC"[i % 3], 1.0 + 0.2 * i, f"{math.sin(r):.6f}", f"{math.cos(r):.6f}"]
        header.append(row)
        for rank, sp in enumerate(picks):
            species.append([f"P{i}", sp, 0 if rank == 2 else 1.0 + rank,
                            f"g{sp[-1]}", "fam"])
    if extra:
        header.append(["NEW", "46.5", "9.5", 150.0])
        species += [["NEW", "sp1", 3.0, "g1", "fam"], ["NEW", "sp_unseen", 1.0, "gz", "fam"]]
    cols = ["plot", "lat", "lon", "elev"]
    if with_targets:
        cols += ["hab", "y", "asp_sin", "asp_cos"]
    return cols, header, species


def _roles():
    roles = rc.RoleMapping()
    roles.plot_id = "plot"
    roles.species_id = "taxon"
    roles.abundance = "cover"
    roles.genus = "genus"
    roles.family = "family"
    roles.latitude = "lat"
    roles.longitude = "lon"
    roles.covariates = ["elev"]
    return roles


def _config():
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.RankPool
    cfg.pool_weighting = rc.PoolWeighting.Log1p
    cfg.zero_abundance_as = 1.0
    return cfg


def _save_member(dataset, seed: int, path: Path) -> None:
    torch.manual_seed(seed)
    model = rc.ResolveModel(dataset.schema,
                            make_model_config(rc.SpeciesEncodingMode.RankPool, hidden_dims=[16, 8]))
    trainer = rc.Trainer(model, make_train_config(max_epochs=1, batch_size=16))
    trainer.prepare_data(dataset, 0.25, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(path))


@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    cols, header, species = _plots(40, with_targets=True)
    full_h = write_csv(tmp_path / "train_header.csv", cols, header)
    full_s = write_csv(tmp_path / "train_species.csv", ["plot", "taxon", "cover", "genus", "family"],
                       species)
    early_h = write_csv(tmp_path / "early_header.csv", cols, header[:30])
    early_s = write_csv(tmp_path / "early_species.csv",
                        ["plot", "taxon", "cover", "genus", "family"],
                        [r for r in species if int(r[0][1:]) < 30])

    hab_ds = rc.ResolveDataset.from_csv(str(early_h), str(early_s), _roles(),
                                        [rc.TargetSpec.classification("hab", 0)], _config())
    y_ds = rc.ResolveDataset.from_csv(str(full_h), str(full_s), _roles(),
                                      [rc.TargetSpec.regression("y")], _config())
    asp_ds = rc.ResolveDataset.from_csv(str(full_h), str(full_s), _roles(),
                                        [rc.TargetSpec.regression("asp_sin"),
                                         rc.TargetSpec.regression("asp_cos")], _config())

    root = tmp_path / "suite"
    manifest = {
        "format": "resolve-suite",
        "format_version": 1,
        "name": "pytest_suite",
        "description": "three targets, two vocabularies",
        "licence": "CC-BY-4.0",
        "engine_version": rc.__version__,
        "taxonomy": "synthetic",
        "training_scope": "40 synthetic plots",
        "inputs": {
            "plot_id": "plot", "species": "taxon", "abundance": "cover",
            "genus": "genus", "family": "family", "latitude": "lat", "longitude": "lon",
            "covariates": ["elev"], "abundance_units": "percent cover",
            "zero_abundance_as": 1.0,
        },
        "targets": [],
    }
    for name, combine, outputs, ds, seeds, extra in (
        ("hab", "vote", ["hab"], hab_ds, [0, 1, 2], {"task": "classification"}),
        ("y", "mean", ["y"], y_ds, [3, 4],
         {"task": "regression", "units": "m", "status": "experimental",
          "limit": "synthetic plots only"}),
        ("aspect", "circular_mean", ["asp_sin", "asp_cos"], asp_ds, [5, 6],
         {"task": "regression", "units": "degrees", "period": 360}),
    ):
        members = []
        for seed in seeds:
            rel = f"{name}/seed_{seed}/model_final.pt"
            _save_member(ds, seed, root / rel)
            members.append({"seed": seed, "file": rel})
        target = {"name": name, "combine": combine, "outputs": outputs,
                  "status": "released", "members": members,
                  "training": {"fixed_epochs": 13},
                  "validation": {"canonical": {"metric": 0.5}}}
        target.update(extra)
        manifest["targets"].append(target)

    m = rc.SuiteManifest.from_json(json.dumps(manifest))
    m.seal(str(root))
    m.write(str(root))
    return root


@pytest.fixture
def scoring(tmp_path: Path):
    cols, header, species = _plots(40, with_targets=False, extra=True)
    h = pd.DataFrame(header, columns=cols)
    s = pd.DataFrame(species, columns=["plot", "taxon", "cover", "genus", "family"])
    h_path = tmp_path / "score_header.csv"
    s_path = tmp_path / "score_species.csv"
    h.to_csv(h_path, index=False)
    s.to_csv(s_path, index=False)
    return h, s, h_path, s_path


def _member_predictions(path: Path, h_path: Path, s_path: Path):
    p = rc.Predictor.load(str(path))
    cfg = p.dataset_config
    cfg.zero_abundance_as = 1.0
    ds = rc.ResolveDataset.from_csv_with_vocabs(str(h_path), str(s_path), _roles(), [],
                                                p.external_vocabs, cfg)
    return p.predict_dataset(ds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_a_suite_combines_members_by_each_targets_rule(suite_dir, scoring):
    h, s, h_path, s_path = scoring
    suite = rc.SuitePredictor.load(str(suite_dir))
    assert suite.target_names == ["hab", "y", "aspect"]
    assert suite.n_encodings == 2

    out = suite.predict(s, header=h, keep_members=True)
    assert out.plot_ids == list(h["plot"])
    ids = out.plot_ids

    manifest = rc.SuiteManifest.read(str(suite_dir))
    stacks = {"hab": [], "hab_p": [], "y": [], "aspect": []}
    for target in manifest.targets:
        for member in target.members:
            p = _member_predictions(suite_dir / member.file, h_path, s_path)
            order = [p.plot_ids.index(i) for i in ids]
            if target.name == "hab":
                stacks["hab"].append(p.predictions["hab"][order])
                stacks["hab_p"].append(p.probabilities["hab"][order])
            elif target.name == "y":
                stacks["y"].append(p.predictions["y"][order])
            else:
                stacks["aspect"].append(rc.bearing_from_components(
                    p.predictions["asp_sin"][order], p.predictions["asp_cos"][order], 360.0))

    hab = out.target("hab")
    vote = rc.combine_vote(torch.stack(stacks["hab"]), 3, torch.stack(stacks["hab_p"]).mean(0))
    assert torch.equal(hab.value, vote.value)
    assert torch.allclose(hab.agreement, vote.agreement)
    assert torch.allclose(hab.probabilities, torch.stack(stacks["hab_p"]).mean(0), atol=1e-6)
    assert hab.class_names == ["A", "B", "C"]
    assert hab.combine == rc.SuiteCombine.Vote
    assert hab.dispersion is None

    y = out.target("y")
    mean = rc.combine_mean(torch.stack(stacks["y"]))
    assert torch.allclose(y.value, mean.value, atol=1e-5)
    assert torch.allclose(y.dispersion, mean.dispersion, atol=1e-5)
    assert y.status == rc.SuiteTargetStatus.Experimental
    assert y.limit == "synthetic plots only"

    asp = out.target("aspect")
    circ = rc.combine_circular(torch.stack(stacks["aspect"]), 360.0)
    assert torch.allclose(asp.value, circ.value, atol=1e-4)
    assert asp.members.shape == (2, len(ids))

    # sp_late is missing from the vote target's vocabulary only; sp_unseen from all.
    late = ids.index("P35")
    assert (hab.recognition.n_recognised[late].item()
            == hab.recognition.n_species[late].item() - 1)
    assert y.recognition.count_share[late].item() == 1.0
    new = ids.index("NEW")
    assert y.recognition.count_share[new].item() == pytest.approx(0.5)
    assert y.recognition.abundance_share[new].item() == pytest.approx(0.75)


def test_csv_paths_score_like_dataframes_and_flatten_to_pandas(suite_dir, scoring):
    h, s, h_path, s_path = scoring
    suite = rc.SuitePredictor.load(str(suite_dir))
    frames = suite.predict(s, header=h)
    paths = suite.predict(s_path, header=h_path)
    for a, b in zip(frames.targets, paths.targets):
        assert torch.allclose(a.value.double(), b.value.double())

    table = paths.to_pandas(probabilities=True)
    assert len(table) == len(h)
    for col in ("plot_id", "hab", "hab_code", "hab_agreement", "hab_prob_A", "y", "y_sd",
                "aspect", "aspect_circular_sd", "y_n_species", "y_recognised_share",
                "y_recognised_abundance_share"):
        assert col in table.columns
    assert set(table["hab"]) <= {"A", "B", "C"}
    assert np.allclose(table[["hab_prob_A", "hab_prob_B", "hab_prob_C"]].sum(axis=1), 1.0,
                       atol=1e-5)


def test_renamed_columns_and_a_target_subset(suite_dir, scoring):
    h, s, _, _ = scoring
    suite = rc.SuitePredictor.load(str(suite_dir), targets=["y"])
    assert suite.target_names == ["y"]
    assert suite.n_encodings == 1
    base = suite.predict(s, header=h)
    renamed = suite.predict(s.rename(columns={"taxon": "name"}),
                            header=h.rename(columns={"elev": "altitude"}),
                            columns={"species_id": "name", "covariates": ["altitude"]})
    assert torch.allclose(base.target("y").value, renamed.target("y").value)
    with pytest.raises(ValueError, match="unknown key"):
        suite.predict(s, header=h, columns={"species": "name"})
    with pytest.raises(Exception, match="header table is required"):
        suite.predict(s)


def test_the_manifest_is_a_contract(suite_dir):
    manifest = rc.SuiteManifest.read(str(suite_dir))
    assert manifest.inputs.zero_abundance_as == 1.0
    assert manifest.target("aspect").period == 360.0
    assert manifest.target("hab").training == {"fixed_epochs": 13}
    assert manifest.target("hab").validation == {"canonical": {"metric": 0.5}}
    assert json.loads(manifest.to_json())["targets"][1]["status"] == "experimental"
    assert manifest.verify(str(suite_dir)) == []

    member = suite_dir / manifest.targets[1].members[0].file
    assert rc.sha256_file(str(member)) == manifest.targets[1].members[0].sha256
    data = bytearray(member.read_bytes())
    data[200] ^= 0xFF
    member.write_bytes(bytes(data))
    assert len(manifest.verify(str(suite_dir))) == 1
    with pytest.raises(Exception, match="SHA-256"):
        rc.SuitePredictor.load(str(suite_dir))
    rc.SuitePredictor.load(str(suite_dir), targets=["hab"])  # untouched targets still load


def test_a_manifest_the_members_do_not_match_is_refused(suite_dir):
    manifest = rc.SuiteManifest.read(str(suite_dir))
    inputs = manifest.inputs
    inputs.covariates = ["slope"]
    manifest.inputs = inputs
    manifest.write(str(suite_dir))
    with pytest.raises(Exception, match="does not list in that order"):
        rc.SuitePredictor.load(str(suite_dir))

    text = json.loads(rc.SuiteManifest.read(str(suite_dir)).to_json())
    text["licnce"] = "x"
    with pytest.raises(Exception, match="licnce is not a member"):
        rc.SuiteManifest.from_json(json.dumps(text))


def test_combination_rules_and_recognition():
    vote = rc.combine_vote(torch.tensor([[0, 1, 2], [1, 1, 2], [2, 0, 2]]), 3)
    assert vote.value.tolist() == [0, 1, 2]  # first column is a three-way tie
    assert vote.agreement.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.0, 0.0, 1.0]])
    broken = rc.combine_vote(torch.tensor([[0, 1, 2], [1, 1, 2], [2, 0, 2]]), 3, probs)
    assert broken.value.tolist() == [2, 1, 2]  # the tie goes to the most probable class

    circ = rc.combine_circular(torch.tensor([[350.0], [10.0]]), 360.0)
    assert min(circ.value.item(), 360.0 - circ.value.item()) < 1e-3

    vocab = rc.SpeciesVocab()
    records = []
    for pid, sp, a in (("p1", "a", 2.0), ("p1", "b", 2.0), ("p2", "a", 1.0)):
        r = rc.SpeciesRecord()
        r.plot_id, r.species_id, r.abundance = pid, sp, a
        records.append(r)
    vocab = rc.SpeciesVocab.from_records(records[:1], 1)
    rec = rc.compute_species_recognition(records, ["p1", "p2", "p3"], vocab)
    assert rec.count_share[0].item() == pytest.approx(0.5)
    assert rec.abundance_share[1].item() == pytest.approx(1.0)
    assert math.isnan(rec.count_share[2].item())


def test_zero_abundance_as_reaches_the_schema(tmp_path):
    header = write_csv(tmp_path / "h.csv", ["plot", "y"], [["a", 1], ["b", 2], ["c", 3], ["d", 4]])
    species = write_csv(tmp_path / "s.csv", ["plot", "taxon", "cover"],
                        [["a", "x", 0], ["b", "x", 1], ["c", "z", 2], ["d", "x", 3]])
    roles = rc.RoleMapping()
    roles.plot_id, roles.species_id, roles.abundance = "plot", "taxon", "cover"
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.RankPool
    cfg.use_taxonomy = False
    cfg.zero_abundance_as = 1.0
    ds = rc.ResolveDataset.from_csv(str(header), str(species), roles,
                                    [rc.TargetSpec.regression("y")], cfg)
    assert ds.schema.zero_abundance_as == 1.0
    weights = ds.pool_weights
    assert torch.allclose(weights[0], weights[1])  # a read as 1, like b
