"""Gap fitting through the Python bindings.

Each case simulates complete plots and gaps from a known relation between the
raw prediction, the plots' neighbours and the truth, and checks that
``resolve_core.fit_gaps`` recovers it: a calibration slope, a turn of
bearings, spatial neighbours chosen where the gaps are near and not for a far
region, fold layouts of squares and of survey-year periods for gaps far in km
and in years, and an estimated error that meets the gaps' error.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

import resolve_core as rc


def _gapped(truth: np.ndarray, gaps: np.ndarray) -> torch.Tensor:
    recorded = truth.astype(float).copy()
    recorded[gaps] = np.nan
    return torch.from_numpy(recorded)


def _no_species(n: int) -> torch.Tensor:
    return torch.zeros(n, 1)


def _random_species(rng, n: int, n_species: int = 60, per_plot: int = 12) -> torch.Tensor:
    picks = np.argsort(rng.random((n, n_species)), axis=1)[:, :per_plot]
    out = np.zeros((n, n_species), np.float32)
    np.put_along_axis(out, picks, 1.0, axis=1)
    return torch.from_numpy(out).to_sparse_csr()


def _only(*methods) -> rc.GapFittingConfig:
    cfg = rc.GapFittingConfig()
    cfg.candidates = list(methods)
    return cfg


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("slope", [0.0, 0.4, 1.0])
def test_offset_slope_recovers_the_local_relation(seed, slope):
    rng = np.random.default_rng(seed)
    pred = rng.uniform(1950, 2020, 3000)
    truth = 1990 + slope * (pred - pred.mean()) + 8.0 + rng.normal(0, 3, pred.size)
    gaps = np.arange(3000) >= 2400
    fit = rc.fit_gaps(rc.GapTarget("year"), torch.from_numpy(pred), _gapped(truth, gaps),
                      _no_species(3000), config=_only(rc.GapMethod.OffsetSlope))
    group = fit.report["groups"]["unlocated"]
    assert group["method"] == "offset_slope"
    assert abs(group["coefficients"]["resolve"] - slope) < 0.02
    assert fit.method[:2400] == ["recorded"] * 2400
    assert set(fit.method[2400:]) == {"offset_slope"}
    assert np.allclose(fit.fitted.numpy()[:2400], truth[:2400])


@pytest.mark.parametrize("seed", range(5))
def test_a_turn_of_bearings_is_recovered(seed):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 360, 2000)
    pred = (truth - 40 + rng.normal(0, 5, truth.size)) % 360
    gaps = np.arange(2000) >= 1500
    fit = rc.fit_gaps(rc.GapTarget("aspect", rc.GapScale.Circular, 360.0), torch.from_numpy(pred),
                      _gapped(truth, gaps), _no_species(2000),
                      config=_only(rc.GapMethod.OffsetSlope))
    turn = fit.report["groups"]["unlocated"]["coefficients"]["resolve"]
    assert abs(turn["angle_deg"] - 40) < 1
    assert abs(turn["modulus"] - 1) < 0.05
    d = np.abs(fit.fitted.numpy()[gaps] - truth[gaps]) % 360
    assert np.minimum(d, 360 - d).mean() < 6


def _wavy(rng, n):
    lat = rng.uniform(45.0, 50.0, n)
    lon = rng.uniform(14.0, 14.5, n)
    truth = 1985 + 30 * np.sin(2 * np.pi * lat / 0.5) + rng.normal(0, 1, n)
    return lat, lon, truth, truth + rng.normal(0, 6, n)


def _mae(fit, truth, gaps):
    return float(np.abs(fit.fitted.numpy()[gaps] - truth[gaps]).mean())


@pytest.mark.parametrize("seed", range(3))
def test_spatial_neighbours_only_where_the_gaps_are_near(seed):
    rng = np.random.default_rng(seed)
    lat, lon, truth, pred = _wavy(rng, 3000)
    coords = torch.from_numpy(np.column_stack([lat, lon]))
    target = rc.GapTarget("altitude")

    gaps = rng.random(3000) < 0.2
    fit = rc.fit_gaps(target, torch.from_numpy(pred), _gapped(truth, gaps), _no_species(3000),
                      coordinates=coords)
    assert fit.report["groups"]["located"]["method"] in ("knn_spatial", "stack_species_spatial")
    best = min(_mae(rc.fit_gaps(target, torch.from_numpy(pred), _gapped(truth, gaps), _no_species(3000),
                                coordinates=coords, config=_only(m)), truth, gaps)
               for m in (rc.GapMethod.KnnSpatial, rc.GapMethod.StackSpeciesSpatial))
    assert _mae(fit, truth, gaps) <= 1.1 * best

    keep = (lat < 48.5) | (lat > 49.0)
    t, p, c = truth[keep], pred[keep], coords[torch.from_numpy(keep)]
    gaps = lat[keep] > 49.0
    n = int(keep.sum())
    fit = rc.fit_gaps(target, torch.from_numpy(p), _gapped(t, gaps), _no_species(n), coordinates=c)
    assert fit.report["groups"]["located"]["method"] != "knn_spatial"
    plot_by_plot = torch.arange(fit.report["n_complete"]) % 10
    plain = rc.fit_gaps(target, torch.from_numpy(p), _gapped(t, gaps), _no_species(n), coordinates=c,
                        config=_only(rc.GapMethod.StackSpeciesSpatial), folds=plot_by_plot)
    calibrated = rc.fit_gaps(target, torch.from_numpy(p), _gapped(t, gaps), _no_species(n),
                             coordinates=c, config=_only(rc.GapMethod.OffsetSlope))
    assert _mae(fit, t, gaps) < 0.8 * _mae(plain, t, gaps)
    assert _mae(fit, t, gaps) <= 1.1 * _mae(calibrated, t, gaps)


def test_far_gaps_are_matched_by_a_buffer_in_km_and_in_years():
    rng = np.random.default_rng(4)
    n_cal, n_gap = 800, 200
    lat = np.concatenate([rng.uniform(45.0, 47.0, n_cal), 47.0 + rng.uniform(0.9, 1.35, n_gap)])
    lon = rng.uniform(14.0, 15.0, n_cal + n_gap)
    year = np.concatenate([rng.uniform(1960, 2020, n_cal), rng.uniform(1930, 1940, n_gap)])
    gaps = np.arange(n_cal + n_gap) >= n_cal
    coords = torch.from_numpy(np.column_stack([lat, lon]))
    found = rc.gap_folds(rc.GapTarget("altitude"), _gapped(np.zeros(n_cal + n_gap), gaps),
                         _random_species(rng, n_cal + n_gap), coordinates=coords, years=torch.from_numpy(year))
    layout = found["layout"]
    assert found["folds"].shape == (n_cal,)
    assert found["complete"].numpy().tolist() == list(range(n_cal))
    assert np.bincount(found["folds"].numpy()).max() <= 0.5 * n_cal
    assert layout["n_folds"] >= 2
    # The gaps lie beyond the complete plots in space and in time: a buffer
    # drawn from their distances on both axes reaches them.
    assert layout["buffer_km"] > 90 and layout["buffer_years"] > 15
    assert set(layout["mismatch"]) == {"similarity", "km", "years"}
    assert layout["mismatch"]["km"] < 0.2 and layout["mismatch"]["years"] < 0.2


def test_injected_folds_reproduce_the_matched_fit_and_the_report_is_plain_data():
    rng = np.random.default_rng(1)
    n = 400
    truth = rng.normal(size=n)
    gaps = rng.random(n) < 0.3
    species = _random_species(rng, n)
    raw = torch.from_numpy(truth + 0.1)
    matched = rc.fit_gaps(rc.GapTarget("slope"), raw, _gapped(truth, gaps), species)
    folds = rc.gap_folds(rc.GapTarget("slope"), _gapped(truth, gaps), species)["folds"]
    injected = rc.fit_gaps(rc.GapTarget("slope"), raw, _gapped(truth, gaps), species, folds=folds)
    assert torch.equal(matched.fitted, injected.fitted)
    assert injected.report["folds"]["injected"]
    report = matched.report
    assert report["n_gaps"] == int(gaps.sum())
    assert set(report["axes"]) == {"similarity", "km", "years"}
    assert report["axes"]["similarity"]["matched"]
    group = report["groups"]["unlocated"]
    assert isinstance(group["cv_error"]["resolve"], float)
    assert isinstance(group["estimated_error"], float)
    assert group["n_estimated"] == report["n_complete"]
    with pytest.raises(ValueError):
        rc.fit_gaps(rc.GapTarget("slope"), raw, _gapped(truth, gaps), species, folds=torch.zeros(3, dtype=torch.int64))


@pytest.mark.parametrize("seed", range(3))
def test_the_estimated_error_meets_the_gaps_error(seed):
    rng = np.random.default_rng(seed)
    n = 3000
    truth = 1985 + rng.normal(0, 15, n)
    pred = truth + rng.normal(0, 6, n)
    gaps = rng.random(n) < 0.2
    coords = torch.from_numpy(np.column_stack([rng.uniform(45, 48, n), rng.uniform(14, 17, n)]))
    fit = rc.fit_gaps(rc.GapTarget("year"), torch.from_numpy(pred), _gapped(truth, gaps),
                      _random_species(rng, n), coordinates=coords)
    group = fit.report["groups"]["located"]
    assert abs(group["estimated_error"] - _mae(fit, truth, gaps)) < 0.5


def test_the_config_validates_and_lists_its_candidates():
    cfg = rc.GapFittingConfig()
    assert cfg.k_grid == [1, 3, 5, 10, 20, 50]
    assert cfg.spatial and cfg.matching_seed == 20261106
    assert cfg.n_folds == 10 and cfg.max_fold_share == 0.5 and cfg.matching_sample == 1000
    assert len(cfg.candidates) == 7
    cfg.k_grid = [0]
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = rc.GapFittingConfig()
    cfg.n_folds = 1
    with pytest.raises(ValueError):
        cfg.validate()


def test_gap_tables_read_every_named_species_from_dataframes():
    header = pd.DataFrame({"plot": ["a", "b", "c"], "Year": ["1990", "", "2001"],
                           "alt": ["300", "NA", "500"]})
    species = pd.DataFrame({"plot": ["a", "a", "b", "c"],
                            "taxon": ["Poa annua", "Bellis perennis", "Poa annua", "Abies alba"]})
    cols = rc.GapColumns()
    cols.plot_id, cols.species_id, cols.survey_year = "plot", "taxon", "Year"
    cols.recorded = [("altitude", ["alt"])]
    tables = rc.read_gap_tables_columns(
        list(species.columns), [species[c].astype(str).tolist() for c in species.columns],
        list(header.columns), [header[c].astype(str).tolist() for c in header.columns],
        cols, ["c", "a", "b"])
    presence = tables.species.to_dense()
    assert presence.shape == (3, 3)
    assert presence.sum(1).tolist() == [1.0, 2.0, 1.0]
    assert tables.recorded["altitude"].tolist()[0] == 500.0
    assert math.isnan(tables.years.tolist()[2])


def test_nearest_by_great_circle_is_the_brute_force():
    rng = np.random.default_rng(7)
    points = np.column_stack([rng.uniform(40, 60, 300), rng.uniform(-5, 25, 300)])
    points[5] = np.nan
    idx, km = rc.nearest_by_great_circle(torch.from_numpy(points), torch.from_numpy(points), 4, True)
    lat, lon = np.radians(points[:, 0]), np.radians(points[:, 1])
    d = 2 * 6371.0 * np.arcsin(np.sqrt(
        np.sin((lat[None, :] - lat[:, None]) / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[None, :] - lon[:, None]) / 2) ** 2))
    d[np.arange(300), np.arange(300)] = np.inf
    d = np.where(np.isnan(d), np.inf, d)
    expected = np.argsort(d, axis=1)[:, :4]
    ok = np.isfinite(points).all(1)
    assert (idx.numpy()[ok] == expected[ok]).all()
    assert (idx.numpy()[5] == -1).all()
    assert np.allclose(km.numpy()[ok], np.take_along_axis(d, expected, 1)[ok])
