"""Missing covariates and coordinates through the nanobind bindings.

The engine keeps a blank covariate or coordinate cell as NaN and
``DatasetConfig.missing_values`` decides how the model reads it. These cases pin
the binding surface: the policy on the config, the schema and the checkpoint,
the NaN in the dataset tensors, the flag columns in the continuous block, and a
fit plus reload that stays finite through the missing cells.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import resolve_core as rc

from conftest import make_model_config, make_train_config, write_csv


def _csvs(tmp_path: Path) -> tuple[str, str]:
    header_rows, species_rows = [], []
    for i in range(80):
        lat = "" if i % 9 == 4 else f"{45.0 + 0.1 * i:.3f}"
        lon = "" if i % 9 == 4 else f"{10.0 + 0.05 * i:.3f}"
        elev = "" if i % 5 == 1 else ("0" if i % 5 == 3 else f"{200 + 10 * (i % 7)}")
        y = 1.0 if i % 5 == 1 else 0.01 * float(elev)
        header_rows.append([f"P{i}", lat, lon, elev, f"{y:.4f}"])
        species_rows.append([f"P{i}", f"sp{i % 6}", "2.0", f"g{i % 3}", f"f{i % 2}"])
        species_rows.append([f"P{i}", f"sp{(i + 1) % 6}", "1.0", f"g{(i + 1) % 3}", "f0"])
    header = write_csv(tmp_path / "missing_header.csv", ["plot_id", "lat", "lon", "elev", "y"],
                       header_rows)
    species = write_csv(tmp_path / "missing_species.csv",
                        ["plot_id", "sp", "cover", "genus", "family"], species_rows)
    return str(header), str(species)


def _roles() -> "rc.RoleMapping":
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    roles.genus = "genus"
    roles.family = "family"
    roles.latitude = "lat"
    roles.longitude = "lon"
    roles.covariates = ["elev"]
    return roles


def _load(tmp_path: Path, policy: "rc.MissingValuePolicy") -> "rc.ResolveDataset":
    header, species = _csvs(tmp_path)
    cfg = rc.DatasetConfig()
    cfg.species_encoding = rc.SpeciesEncodingMode.Hash
    cfg.hash_dim = 16
    cfg.track_unknown_fraction = False
    cfg.missing_values = policy
    return rc.ResolveDataset.from_csv(header, species, _roles(),
                                      [rc.TargetSpec.regression("y")], cfg)


def test_the_default_policy_flags_missing_values():
    assert rc.DatasetConfig().missing_values == rc.MissingValuePolicy.Indicate


def test_a_missing_cell_is_nan_and_a_recorded_zero_is_not(tmp_path):
    ds = _load(tmp_path, rc.MissingValuePolicy.Indicate)
    ids = list(ds.plot_ids)
    covariates = ds.covariates
    assert math.isnan(covariates[ids.index("P1"), 0].item())
    assert covariates[ids.index("P3"), 0].item() == 0.0
    assert torch.isnan(ds.coordinates[ids.index("P4")]).all()
    assert ds.schema.missing_values == rc.MissingValuePolicy.Indicate
    assert ds.schema.missing_flag_width() == 2


def test_the_block_carries_one_flag_per_covariate_and_one_for_the_coordinates(tmp_path):
    indicated = _load(tmp_path, rc.MissingValuePolicy.Indicate)
    zero = _load(tmp_path, rc.MissingValuePolicy.Zero)
    flagged = indicated.continuous_block(True)
    plain = zero.continuous_block(True)
    # coords (2) | coordinate flag | elev | elev flag | hash (16)
    assert flagged.shape[1] == plain.shape[1] + 2 == 2 + 1 + 1 + 1 + 16
    ids = list(indicated.plot_ids)
    assert flagged[ids.index("P4"), 2].item() == 1.0
    assert flagged[ids.index("P1"), 4].item() == 1.0
    assert flagged[ids.index("P3"), 4].item() == 0.0
    assert not torch.isnan(plain).any()


@pytest.mark.parametrize("policy", ["Indicate", "Zero"])
def test_fit_and_reload_stay_finite_through_missing_values(tmp_path, policy):
    ds = _load(tmp_path, getattr(rc.MissingValuePolicy, policy))
    torch.manual_seed(0)
    model = rc.ResolveModel(ds.schema, make_model_config(rc.SpeciesEncodingMode.Hash))
    trainer = rc.Trainer(model, make_train_config(max_epochs=3, batch_size=16))
    trainer.prepare_data(ds, test_size=0.25, seed=1)
    trainer.fit()
    fill = trainer.scalers.continuous_fill
    assert fill is not None and torch.isfinite(fill).all()

    path = str(tmp_path / f"missing_{policy}.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path)
    assert predictor.schema.missing_values == getattr(rc.MissingValuePolicy, policy)
    assert torch.allclose(predictor.scalers.continuous_fill, fill)
    predictions = predictor.predict_dataset(ds).predictions["y"]
    assert torch.isfinite(predictions).all()
