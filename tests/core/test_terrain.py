"""Terrain correction, terrain sources and terrain fields through the Python bindings.

The grids are float32 tiles written as GLO-30 lays its own (pixel is point, a
tile's first row on its northern edge) over a surface linear in latitude and
longitude, whose descriptors have closed forms. The correction is fitted on
plots whose elevation model is exact where the coordinate is precise and
wrong by hundreds of metres for a share that grows with the recorded
uncertainty; it must learn to trust the model there and the prediction
elsewhere.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import resolve_core as rc


def _plane(lat, lon):
    return 100.0 + 1000.0 * (lat - 46.0) + 500.0 * (lon - 10.0)


def _write_plane_grids(directory, squares, cells=30):
    d = 1.0 / cells
    tiles = []
    for south, west in squares:
        lat = south + 1.0 - np.arange(cells) * d
        lon = west + np.arange(cells) * d
        grid = torch.tensor(_plane(lat[:, None], lon[None, :]), dtype=torch.float32)
        name = f"T{south}_{west}"
        rc.write_npy_grid(str(directory / f"{name}.npy"), grid)
        tiles.append({"name": name, "file": f"{name}.npy", "rows": cells, "cols": cells,
                      "lat_first": south + 1.0, "lon_first": float(west), "dlat": d, "dlon": d})
    (directory / "tiles.json").write_text(json.dumps({"format": "resolve-terrain-grids", "version": 1,
                                                      "tiles": tiles}))
    return rc.TerrainGrids(str(directory))


def test_npy_grids_round_trip_with_numpy(tmp_path):
    grid = np.arange(12, dtype=np.float32).reshape(3, 4) * 1.5
    np.save(tmp_path / "numpy.npy", grid)
    assert torch.equal(rc.read_npy_grid(str(tmp_path / "numpy.npy")), torch.from_numpy(grid))
    rc.write_npy_grid(str(tmp_path / "engine.npy"), torch.from_numpy(grid))
    assert np.array_equal(np.load(tmp_path / "engine.npy"), grid)


def test_descriptors_of_a_plane_have_their_closed_forms(tmp_path):
    grids = _write_plane_grids(tmp_path, [(46, 9), (45, 9), (46, 10), (45, 10)])
    lat = torch.tensor([46.31, 45.77], dtype=torch.float64)
    lon = torch.tensor([9.42, 9.95], dtype=torch.float64)
    d = rc.terrain_descriptors(grids, lat, lon)
    expected = _plane(lat.numpy(), lon.numpy())
    np.testing.assert_allclose(d.elevation.numpy(), expected, atol=1e-3)
    np.testing.assert_allclose(d.elevation_1km.numpy(), expected, atol=1e-3)
    gy = 1000.0 / (6371000.0 * math.pi / 180.0)
    gx = 500.0 / (6371000.0 * math.pi / 180.0 * np.cos(np.radians(lat.numpy())))
    np.testing.assert_allclose(d.slope_1km.numpy(), np.degrees(np.arctan(np.hypot(gx, gy))), atol=1e-4)
    assert math.isnan(grids.elevation(torch.tensor([50.5], dtype=torch.float64),
                                      torch.tensor([9.5], dtype=torch.float64))[0].item())


def _corpus(n, seed):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(200, 1700, n)
    log_u = rng.uniform(0, 3.5, n)
    wrong = rng.random(n) < 0.05 + 0.15 * log_u
    dem = truth + rng.normal(0, 2, n) + np.where(wrong, rng.normal(0, 300, n), 0)
    spread = rng.uniform(5, 45, n)
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    return dict(truth=t(truth), prediction=t(truth + rng.normal(0, spread)), dem=t(dem),
                dem_1km=t(dem + rng.normal(0, 15, n)), uncertainty_m=t(10 ** log_u), spread=t(spread))


@pytest.mark.parametrize("seed", [1, 2])
def test_terrain_correction_learns_which_source_to_trust(tmp_path, seed):
    train, test = _corpus(6000, 100 + seed), _corpus(2000, 200 + seed)
    cfg = rc.TerrainCorrectionConfig()
    cfg.epochs, cfg.batch_size, cfg.hidden_dim, cfg.seed = 80, 256, 64, seed
    inputs = lambda c: {k: v for k, v in c.items() if k != "truth"}  # noqa: E731
    model = rc.fit_terrain_correction("altitude", rc.TerrainDescriptor.Elevation, truth=train["truth"],
                                      config=cfg, **inputs(train))
    r = model.apply(**inputs(test))
    mae = lambda x: (x - test["truth"]).abs().mean().item()  # noqa: E731
    assert mae(r.corrected) < 0.5 * min(mae(test["prediction"]), mae(test["dem"]))
    precise = test["uncertainty_m"] < 3
    assert r.weights[precise, 1].mean().item() > 0.6
    assert model.report["n_train"] == 6000
    assert rc.TerrainCorrection.weight_names == ["resolve", "dem", "dem_1km"]

    model.save(str(tmp_path / "a.terrain"))
    back = rc.TerrainCorrection.load(str(tmp_path / "a.terrain"))
    assert back.target == "altitude" and back.descriptor == rc.TerrainDescriptor.Elevation
    torch.testing.assert_close(back.apply(**inputs(test)).corrected, r.corrected, rtol=1e-6, atol=1e-4)


def test_coded_and_neural_fields_read_like_their_grids(tmp_path):
    grids = _write_plane_grids(tmp_path, [(46, 9)], cells=60)
    codec = rc.TerrainCodecConfig()
    codec.max_error = 0.5
    metrics = rc.encode_terrain_grids(grids, codec, str(tmp_path / "coded.bin"))
    assert metrics["T46_9"]["max_abs"] <= 0.5 + 1e-3
    coded = rc.open_terrain_field(str(tmp_path / "coded.bin"))
    assert isinstance(coded, rc.TerrainCodedField)
    assert (coded.grid(0) - grids.grid(0)).abs().max().item() <= 0.5 + 1e-3
    tight = rc.TerrainCodedField(str(tmp_path / "coded.bin"), cache_blocks=1)
    probe_lat = torch.tensor([46.9, 46.1, 46.5, 46.05], dtype=torch.float64)
    probe_lon = torch.tensor([9.05, 9.95, 9.5, 9.1], dtype=torch.float64)
    torch.testing.assert_close(tight.elevation(probe_lat, probe_lon), coded.elevation(probe_lat, probe_lon))

    field_cfg = rc.TerrainFieldConfig()
    field_cfg.levels, field_cfg.log2_table_size, field_cfg.hidden_dim = 6, 10, 16
    field_cfg.coarsest_resolution, field_cfg.base_cells, field_cfg.steps, field_cfg.batch_size = 8, 8, 600, 4096
    tile = grids.tiles[0]
    m = rc.fit_terrain_tile(rc.read_npy_grid(str(tmp_path / "T46_9.npy")), tile, field_cfg, str(tmp_path / "t.tile"))
    assert m["rmse"] < 5
    rc.assemble_terrain_field([str(tmp_path / "t.tile")], str(tmp_path / "field.bin"))
    neural = rc.open_terrain_field(str(tmp_path / "field.bin"))
    assert isinstance(neural, rc.TerrainField)

    lat = torch.tensor([46.2, 46.55, 46.8], dtype=torch.float64)
    lon = torch.tensor([9.2, 9.4, 9.7], dtype=torch.float64)
    g = rc.terrain_descriptors(grids, lat, lon)
    assert (rc.terrain_descriptors(coded, lat, lon).elevation - g.elevation).abs().max().item() <= 0.5 + 1e-3
    assert (rc.terrain_descriptors(neural, lat, lon).elevation - g.elevation).abs().mean().item() < 5
    whole = neural.grid(0)
    assert whole.shape == (60, 60) and whole.dtype == torch.float32
    assert (whole - grids.grid(0)).abs().mean().item() < 5


def test_terrain_options_are_read_from_names_dicts_and_objects():
    from resolve_core._suite import _terrain, _terrain_columns

    assert _terrain(None) is None and _terrain(False) is None
    assert _terrain(True).source == rc.TerrainSourceKind.Field
    assert _terrain("columns").source == rc.TerrainSourceKind.Columns
    opts = _terrain({"source": "grids", "grids": "somewhere", "device": "cpu"})
    assert opts.source == rc.TerrainSourceKind.Grids and opts.grids == "somewhere"
    with pytest.raises(ValueError):
        _terrain({"source": "satellite"})
    with pytest.raises(ValueError):
        _terrain({"grid": "x"})
    cols = _terrain_columns({"elevation": "dem", "uncertainty": "unc"})
    assert cols.elevation == "dem" and cols.uncertainty == "unc" and cols.slope is None
    with pytest.raises(ValueError):
        _terrain_columns({"altitude": "x"})
