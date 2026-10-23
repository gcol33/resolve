"""Per-target missing-label masking through the nanobind bindings (#119).

``TargetSpec.missing`` decides what a fit does with a plot whose label for that
target is missing: ``Drop`` (the default) leaves the plot out, ``Mask`` keeps it
with a sentinel (class code -1, regression NaN) so it trains and is scored on
its other targets. These cases pin the binding surface: the enum and its
default, the sentinels in the dataset tensors, the policy on the schema and in
the checkpoint, the per-target ``n_labelled`` metric, the ``test_rows`` that
place a masked target's held-out rows in the test fold, and the loss histories
split by target.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config, write_csv


def _csvs(tmp_path: Path, n: int = 160) -> tuple[str, str]:
    header_rows, species_rows = [], []
    for i in range(n):
        c1 = math.sin(i * 0.13)
        c2 = math.cos(i * 0.07)
        y = "" if i % 3 == 0 else f"{2.0 * c1 - 1.5 * c2 + 3.0:.5f}"
        cls = "" if i % 4 == 1 else ("pos" if c1 + c2 > 0 else "neg")
        if i % 12 == 0:
            cls = ""  # with i % 3 == 0 this plot is labelled for nothing
        header_rows.append([f"P{i}", f"{c1:.5f}", f"{c2:.5f}", y, cls])
        species_rows.append([f"P{i}", f"sp{i % 6}", "1.0"])
    header = write_csv(tmp_path / "mask_header.csv", ["plot_id", "c1", "c2", "y", "cls"],
                       header_rows)
    species = write_csv(tmp_path / "mask_species.csv", ["plot_id", "sp", "cover"],
                        species_rows)
    return str(header), str(species)


def _roles() -> "rc.RoleMapping":
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    roles.covariates = ["c1", "c2"]
    return roles


def _targets(policy: "rc.TargetMissingPolicy") -> list:
    y = rc.TargetSpec.regression("y")
    y.missing = policy
    cls = rc.TargetSpec()
    cls.column_name = "cls"
    cls.target_name = "cls"
    cls.task = rc.TaskType.Classification
    cls.missing = policy
    return [y, cls]


def _load(tmp_path: Path, policy: "rc.TargetMissingPolicy") -> "rc.ResolveDataset":
    header, species = _csvs(tmp_path)
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.Hash
    cfg.hash_dim = 8
    cfg.track_unknown_fraction = False
    return rc.ResolveDataset.from_csv(header, species, _roles(), _targets(policy), cfg)


def test_drop_is_the_default_policy():
    assert rc.TargetSpec().missing == rc.TargetMissingPolicy.Drop
    assert rc.TargetConfig().missing == rc.TargetMissingPolicy.Drop
    assert rc.TargetSpec.regression("y").missing == rc.TargetMissingPolicy.Drop


def test_mask_keeps_partly_labelled_plots_with_sentinels(tmp_path):
    drop = _load(tmp_path, rc.TargetMissingPolicy.Drop)
    mask = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    n = 160
    labelled_y = {i for i in range(n) if i % 3 != 0}
    labelled_cls = {i for i in range(n) if i % 4 != 1 and i % 12 != 0}
    assert drop.n_plots == len(labelled_y & labelled_cls)
    assert mask.n_plots == len(labelled_y | labelled_cls)

    ids = list(mask.plot_ids)
    y = mask.targets["y"]
    cls = mask.targets["cls"]
    assert math.isnan(y[ids.index("P3")].item())
    assert cls[ids.index("P1")].item() == -1
    assert "P0" not in ids  # labelled for no target
    assert int(torch.isfinite(y).sum()) == len(labelled_y)
    assert int((cls >= 0).sum()) == len(labelled_cls)
    assert [t.missing for t in mask.schema.targets] == [rc.TargetMissingPolicy.Mask] * 2


def test_masked_fit_reports_labelled_counts_and_round_trips(tmp_path):
    ds = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    torch.manual_seed(0)
    model = rc.ResolveModel(ds.schema, make_model_config(rc.SpeciesEncodingMode.Hash, hash_dim=8))
    trainer = rc.Trainer(model, make_train_config(max_epochs=4, batch_size=16))
    trainer.prepare_data(ds, test_size=0.25, seed=1)
    result = trainer.fit()

    test_idx = trainer.test_indices()
    y_test = ds.targets["y"][test_idx]
    cls_test = ds.targets["cls"][test_idx]
    assert result.final_metrics["y"]["n_labelled"] == int(torch.isfinite(y_test).sum())
    assert result.final_metrics["cls"]["n_labelled"] == int((cls_test >= 0).sum())
    assert math.isfinite(result.final_metrics["y"]["mae"])

    classified = trainer.compute_classification_predictions("cls")
    assert torch.equal(classified.actuals, cls_test[classified.test_rows])
    assert (classified.actuals >= 0).all()

    residuals = trainer.compute_residuals("y")
    assert len(residuals.test_rows) == len(residuals.actuals) == int(torch.isfinite(y_test).sum())

    path = str(tmp_path / "masked.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path)
    assert [t.missing for t in predictor.schema.targets] == [rc.TargetMissingPolicy.Mask] * 2
    predictions = predictor.predict_dataset(ds).predictions
    # Every plot is predicted, labelled or not.
    assert predictions["y"].shape[0] == ds.n_plots
    assert torch.isfinite(predictions["y"]).all()


def _decomposes(by_target, total, names, aux=None):
    """Per epoch, the targets' terms (plus the auxiliary terms) add up to the total."""
    assert sorted(by_target) == sorted(names)
    for name in names:
        assert len(by_target[name]) == len(total)
    for epoch, value in enumerate(total):
        terms = sum(by_target[name][epoch] for name in names)
        if aux is not None:
            terms += sum(history[epoch] for history in aux.values())
        assert terms == pytest.approx(value, rel=1e-5, abs=1e-6), epoch


def _fit_masked(ds, *, test_size=0.25, fixed_epochs=0, moe=False, tabnet=False,
                batch_size=16):
    torch.manual_seed(0)
    model_cfg = make_model_config(rc.SpeciesEncodingMode.Hash, hash_dim=8)
    if moe:
        model_cfg.moe_routing = rc.MoERoutingType.Soft
    if tabnet:
        model_cfg.encoder_architecture = rc.EncoderArchitecture.TabNet
        model_cfg.tabnet.n_steps = 2
        model_cfg.tabnet.n_d = 8
        model_cfg.tabnet.n_a = 8
        model_cfg.tabnet.sparsity_coefficient = 0.5
    train_cfg = make_train_config(max_epochs=4, batch_size=batch_size)
    train_cfg.fixed_epochs = fixed_epochs
    trainer = rc.Trainer(rc.ResolveModel(ds.schema, model_cfg), train_cfg)
    trainer.prepare_data(ds, test_size=test_size, seed=1)
    return trainer.fit()


def test_the_loss_history_splits_by_target(tmp_path):
    ds = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    result = _fit_masked(ds, batch_size=4)

    assert isinstance(result.train_loss_history_by_target, dict)
    assert isinstance(result.test_loss_history_by_target, dict)
    assert len(result.train_loss_history) == 4
    _decomposes(result.train_loss_history_by_target, result.train_loss_history, ["y", "cls"])
    _decomposes(result.test_loss_history_by_target, result.test_loss_history, ["y", "cls"])
    assert result.train_aux_loss_history == {}


def test_the_moe_term_is_recorded_beside_the_targets(tmp_path):
    ds = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    result = _fit_masked(ds, moe=True)
    assert list(result.train_aux_loss_history) == ["moe"]
    assert len(result.train_aux_loss_history["moe"]) == len(result.train_loss_history)
    _decomposes(result.train_loss_history_by_target, result.train_loss_history, ["y", "cls"],
                aux=result.train_aux_loss_history)
    _decomposes(result.test_loss_history_by_target, result.test_loss_history, ["y", "cls"])


def test_the_tabnet_sparsity_term_is_recorded_beside_the_targets(tmp_path):
    ds = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    result = _fit_masked(ds, tabnet=True)
    assert list(result.train_aux_loss_history) == ["tabnet_sparsity"]
    assert all(term > 0 for term in result.train_aux_loss_history["tabnet_sparsity"])
    _decomposes(result.train_loss_history_by_target, result.train_loss_history, ["y", "cls"],
                aux=result.train_aux_loss_history)
    _decomposes(result.test_loss_history_by_target, result.test_loss_history, ["y", "cls"])


def test_nothing_held_out_leaves_the_test_split_empty(tmp_path):
    ds = _load(tmp_path, rc.TargetMissingPolicy.Mask)
    result = _fit_masked(ds, test_size=0.0, fixed_epochs=3)
    assert result.test_loss_history == []
    assert result.test_loss_history_by_target == {}
    _decomposes(result.train_loss_history_by_target, result.train_loss_history, ["y", "cls"])


def test_a_single_target_history_is_the_total(tmp_path):
    header, species = _csvs(tmp_path)
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.Hash
    cfg.hash_dim = 8
    cfg.track_unknown_fraction = False
    ds = rc.ResolveDataset.from_csv(header, species, _roles(),
                                    [rc.TargetSpec.regression("y")], cfg)
    result = _fit_masked(ds)
    assert result.train_loss_history_by_target == {"y": result.train_loss_history}
    assert result.test_loss_history_by_target == {"y": result.test_loss_history}
