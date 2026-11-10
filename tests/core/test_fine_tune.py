"""Fine-tuning from a trained model through the Python bindings.

A parent trained on one target of a composition gradient is opened as a
``ParentModel`` (a checkpoint, or a member of a suite), new heads are trained
on other targets of the same plots with ``Trainer.init_from``, and the result
is released as a derived suite with ``write_derived_suite``. The cases pin the
binding surface: the parent's vocabularies and settings, the copied encoder,
the frozen encoder, the parent record in the checkpoint, the refusals, the
recovery of a known rule, and the derived manifest SuitePredictor scores.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import resolve_core as rc

from conftest import make_train_config, write_csv

N_SPECIES = 24


def _gradient(i: int) -> float:
    return math.fmod(0.6180339887498949 * (i + 1), 1.0)


def _write_plots(tmp_path: Path, first: int, last: int, name: str) -> tuple[str, str]:
    header, species = [], []
    for i in range(first, last):
        t = _gradient(i)
        header.append(
            [
                f"p{i}",
                46.0 + math.fmod(0.37 * i, 1.0),
                10.0 + math.fmod(0.53 * i, 1.0),
                t,
                1.0 + 3.0 * t,
                f"Z{min(2, int(3 * t))}",
            ]
        )
        for j in range(N_SPECIES):
            d = abs(t - j / 23.0)
            if d < 0.22:
                species.append(
                    [
                        f"p{i}",
                        f"sp_{j}",
                        1.0 + 10.0 * (1.0 - d / 0.22),
                        f"gen_{j // 4}",
                        f"fam_{j // 8}",
                    ]
                )
    h = write_csv(
        tmp_path / f"{name}_header.csv",
        ["plot_id", "lat", "lon", "g", "soil", "zone"],
        header,
    )
    s = write_csv(
        tmp_path / f"{name}_species.csv",
        ["plot_id", "sp", "cover", "genus", "family"],
        species,
    )
    return str(h), str(s)


def _roles() -> rc.RoleMapping:
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    roles.genus = "genus"
    roles.family = "family"
    roles.latitude = "lat"
    roles.longitude = "lon"
    return roles


def _dataset_config() -> rc.DatasetConfig:
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.Embed
    cfg.top_k = 2
    cfg.top_k_species = 12
    return cfg


def _parent_model_config() -> rc.ModelConfig:
    cfg = rc.ModelConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.Embed
    cfg.species_embed_dim = 8
    cfg.genus_emb_dim = 4
    cfg.family_emb_dim = 4
    cfg.hidden_dims = [32, 16]
    return cfg


def _train_parent(
    tmp_path: Path, path: Path, *, epochs: int = 2, seed: int = 7
) -> None:
    h, s = _write_plots(tmp_path, 0, 300, f"parent{seed}")
    ds = rc.ResolveDataset.from_csv(
        h, s, _roles(), [rc.TargetSpec.regression("g")], _dataset_config()
    )
    torch.manual_seed(seed)
    model = rc.ResolveModel(
        ds.schema, rc.model_config_for_dataset(ds, _parent_model_config())
    )
    trainer = rc.Trainer(model, make_train_config(max_epochs=epochs, batch_size=32))
    trainer.prepare_data(ds, 0.2, 1)
    trainer.fit()
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(path), trainer.run_metadata)


def _child(
    parent: rc.ParentModel,
    tmp_path: Path,
    targets,
    *,
    freeze=False,
    epochs: int = 3,
    first: int = 300,
    last: int = 380,
    name: str = "child",
):
    h, s = _write_plots(tmp_path, first, last, name)
    ds = rc.ResolveDataset.from_csv_with_vocabs(
        h, s, _roles(), targets, parent.vocabs(), parent.dataset_config()
    )
    mc = parent.model_config
    mc.freeze_encoder = freeze
    model = rc.ResolveModel(ds.schema, rc.model_config_for_dataset(ds, mc))
    trainer = rc.Trainer(model, make_train_config(max_epochs=epochs, batch_size=32))
    return ds, trainer, (h, s)


@pytest.fixture
def parent_path(tmp_path: Path) -> Path:
    path = tmp_path / "parent.pt"
    _train_parent(tmp_path, path)
    return path


def test_init_from_copies_the_encoder(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    assert parent.record.sha256 == rc.sha256_file(str(parent_path))
    assert parent.record.suite == ""
    ds, trainer, _ = _child(parent, tmp_path, [rc.TargetSpec.regression("soil")])
    trainer.init_from(parent)

    reference = rc.Predictor.load(str(parent_path)).model
    ref = dict(reference.named_parameters())
    for name, value in trainer.model.named_parameters().items():
        if name.startswith("head_"):
            continue
        assert torch.equal(value.detach(), ref[name].detach()), name
    assert trainer.parent.reuse == rc.InitReuse.Encoder
    assert trainer.parent.n_tensors > 0


def test_a_frozen_encoder_stays_fixed(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    ds, trainer, _ = _child(
        parent, tmp_path, [rc.TargetSpec.regression("soil")], freeze=True
    )
    trainer.init_from(parent)
    trainer.prepare_data(ds, 0.25, 3)
    before = {
        n: v.detach().clone() for n, v in trainer.model.named_parameters().items()
    }
    trainer.fit()
    after = dict(trainer.model.named_parameters())
    moved_heads = [
        n
        for n in before
        if n.startswith("head_") and not torch.equal(before[n], after[n].detach())
    ]
    assert moved_heads
    for name in before:
        if not name.startswith("head_"):
            assert torch.equal(before[name], after[name].detach()), name
    assert torch.equal(trainer.scalers.continuous_mean, parent.scalers.continuous_mean)


def test_the_checkpoint_records_its_parent(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    ds, trainer, _ = _child(
        parent, tmp_path, [rc.TargetSpec.regression("soil")], freeze=True
    )
    trainer.init_from(parent)
    trainer.prepare_data(ds, 0.25, 3)
    trainer.fit()
    trainer.save(str(tmp_path / "child.pt"), trainer.run_metadata)
    meta = rc.Trainer.load_run_metadata(str(tmp_path / "child.pt"))
    assert meta.parent is not None
    assert meta.parent.sha256 == rc.sha256_file(str(parent_path))
    assert meta.parent == trainer.parent


def test_encoder_heads_carries_a_shared_head(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    ds, trainer, _ = _child(
        parent,
        tmp_path,
        [rc.TargetSpec.regression("g"), rc.TargetSpec.regression("soil")],
    )
    trainer.init_from(parent, rc.InitReuse.EncoderHeads)
    assert trainer.parent.heads == ["g"]


def test_a_model_that_reads_the_plots_otherwise_is_refused(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    h, s = _write_plots(tmp_path, 300, 380, "own")
    own = rc.ResolveDataset.from_csv(
        h, s, _roles(), [rc.TargetSpec.regression("soil")], _dataset_config()
    )
    model = rc.ResolveModel(
        own.schema, rc.model_config_for_dataset(own, parent.model_config)
    )
    trainer = rc.Trainer(model, make_train_config())
    with pytest.raises(ValueError, match="species vocabularies"):
        trainer.init_from(parent)

    ds, _, _ = _child(parent, tmp_path, [rc.TargetSpec.regression("soil")])
    mc = parent.model_config
    mc.hidden_dims = [24, 16]
    wide = rc.Trainer(
        rc.ResolveModel(ds.schema, rc.model_config_for_dataset(ds, mc)),
        make_train_config(),
    )
    with pytest.raises(ValueError, match="hidden_dims"):
        wide.init_from(parent)
    assert "hidden_dims" in " ".join(rc.fine_tune_mismatches(parent, mc, ds.schema))
    assert "freeze_encoder" in rc.FINE_TUNE_MODEL_FIELDS


def test_a_frozen_encoder_recovers_a_rule_of_the_gradient(tmp_path):
    path = tmp_path / "parent.pt"
    _train_parent(tmp_path, path, epochs=150)
    parent = rc.ParentModel.open(str(path))
    targets = [
        rc.TargetSpec.regression("soil"),
        rc.TargetSpec.classification("zone", 3),
    ]
    ds, trainer, _ = _child(
        parent, tmp_path, targets, freeze=True, epochs=200, first=300, last=400
    )
    torch.manual_seed(11)
    trainer.init_from(parent)
    trainer.prepare_data(ds, 0.25, 5)
    trainer.fit()
    trainer.save(str(tmp_path / "child.pt"), trainer.run_metadata)

    h, s = _write_plots(tmp_path, 400, 500, "held_out")
    test = rc.ResolveDataset.from_csv_with_vocabs(
        h, s, _roles(), targets, parent.vocabs(), parent.dataset_config()
    )
    out = rc.Predictor.load(str(tmp_path / "child.pt")).predict_dataset(test)
    soil = out.predictions["soil"].numpy()
    truth = test.targets["soil"].numpy()
    r2 = 1.0 - np.sum((soil - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
    assert r2 > 0.9
    zone = out.predictions["zone"].numpy()
    assert np.mean(zone == test.targets["zone"].numpy()) > 0.85


def _parent_suite(tmp_path: Path) -> Path:
    suite = tmp_path / "parent_suite"
    for seed in (1, 2):
        _train_parent(tmp_path, suite / "g" / f"seed_{seed}.pt", seed=seed)
    m = rc.SuiteManifest()
    m.name = "gradient_parent"
    m.licence = "CC-BY-4.0"
    m.engine_version = rc.__version__ if hasattr(rc, "__version__") else "test"
    m.taxonomy = "synthetic"
    m.training_scope = "300 synthetic plots"
    inputs = rc.SuiteInputContract()
    inputs.plot_id, inputs.species, inputs.abundance = "plot_id", "sp", "cover"
    inputs.genus, inputs.family = "genus", "family"
    inputs.latitude, inputs.longitude = "lat", "lon"
    m.inputs = inputs
    t = rc.SuiteTarget()
    t.name = "g"
    t.task = rc.TaskType.Regression
    t.combine = rc.SuiteCombine.Mean
    t.outputs = ["g"]
    t.members = [rc.SuiteMember(1, "g/seed_1.pt"), rc.SuiteMember(2, "g/seed_2.pt")]
    m.targets = [t]
    m.seal(str(suite))
    m.write(str(suite))
    return suite


def test_a_derived_suite_is_written_and_scored(tmp_path):
    suite = _parent_suite(tmp_path)
    members = []
    for seed in (1, 2):
        cfg = rc.InitFromConfig()
        cfg.suite_target = "g"
        cfg.suite_seed = seed
        parent = rc.ParentModel.open(str(suite), cfg)
        assert parent.record.suite == "gradient_parent"
        assert parent.record.suite_seed == seed
        ds, trainer, paths = _child(
            parent,
            tmp_path,
            [rc.TargetSpec.regression("soil"), rc.TargetSpec.classification("zone", 3)],
            freeze=True,
        )
        trainer.init_from(parent)
        trainer.prepare_data(ds, 0.25, seed)
        trainer.fit()
        out = tmp_path / f"tuned_{seed}.pt"
        trainer.save(str(out), trainer.run_metadata)
        members.append(rc.DerivedSuiteMember(str(out), seed))

    spec = rc.DerivedSuiteSpec()
    spec.parent = str(suite)
    spec.name = "gradient_soil"
    spec.licence = "CC-BY-4.0"
    spec.training_scope = "80 synthetic plots"
    soil = rc.DerivedSuiteTarget()
    soil.name = "soil"
    soil.units = "pH"
    soil.members = members
    zone = rc.DerivedSuiteTarget()
    zone.name = "zone"
    zone.members = members
    spec.targets = [soil, zone]

    derived = tmp_path / "derived"
    manifest = rc.write_derived_suite(str(derived), spec)
    assert manifest.format_version == 5
    assert manifest.targets[0].derived_from.target == "g"
    assert manifest.targets[1].classes == ["Z0", "Z1", "Z2"]
    parent_manifest = rc.SuiteManifest.read(str(suite))
    assert [m.parent_sha256 for m in manifest.targets[0].members] == [
        m.sha256 for m in parent_manifest.targets[0].members
    ]

    scored = rc.SuitePredictor.load(str(derived)).predict_csv(paths[1], paths[0])
    assert scored.target("soil").value.shape[0] == 80


def test_data_frames_load_in_the_parent_namespace(parent_path, tmp_path):
    parent = rc.ParentModel.open(str(parent_path))
    h, s = _write_plots(tmp_path, 300, 380, "frames")
    import pandas as pd

    header, species = pd.read_csv(h), pd.read_csv(s)
    targets = [rc.TargetSpec.regression("soil")]
    from_frames = rc.ResolveDataset.from_pandas(
        header,
        species,
        _roles(),
        targets,
        config=parent.dataset_config(),
        vocabs=parent.vocabs(),
    )
    from_csv = rc.ResolveDataset.from_csv_with_vocabs(
        h, s, _roles(), targets, parent.vocabs(), parent.dataset_config()
    )
    assert torch.equal(from_frames.species_ids, from_csv.species_ids)
    with pytest.raises(ValueError, match="schema_source or vocabs"):
        rc.ResolveDataset.from_pandas(
            header,
            species,
            _roles(),
            targets,
            schema_source=from_csv,
            vocabs=parent.vocabs(),
        )
