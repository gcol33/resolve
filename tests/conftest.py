"""Shared test fixtures and synthetic-data helpers.

Centralizes `make_synthetic_data` and the `dataset` fixture so individual test
modules don't import across each other (which breaks on hosts where `tests/`
is not on sys.path as a package).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from resolve.data.dataset import ResolveDataset, TargetConfig
from resolve.data.roles import RoleMapping


def make_synthetic_data(
    n_plots: int = 200,
    n_species: int = 50,
    n_genera: int = 15,
    n_families: int = 5,
    species_per_plot: int = 10,
    seed: int = 42,
) -> ResolveDataset:
    """Generate a synthetic vegetation plot dataset."""
    rng = np.random.default_rng(seed)

    plot_ids = [f"P{i:04d}" for i in range(n_plots)]
    lat = rng.uniform(45, 55, n_plots)
    lon = rng.uniform(5, 15, n_plots)
    area = np.exp(rng.normal(2.0 + 0.1 * (lat - 50), 0.5))
    habitat = rng.integers(0, 3, n_plots)

    header = pl.DataFrame({
        "plot_id": plot_ids,
        "lat": lat,
        "lon": lon,
        "area": area,
        "habitat": habitat,
    })

    species_names = [f"sp_{i}" for i in range(n_species)]
    genus_names = [f"genus_{i % n_genera}" for i in range(n_species)]
    family_names = [f"family_{i % n_families}" for i in range(n_species)]
    sp_to_genus = dict(zip(species_names, genus_names))
    sp_to_family = dict(zip(species_names, family_names))

    rows = []
    for pid in plot_ids:
        n_sp = rng.integers(3, species_per_plot + 1)
        chosen = rng.choice(species_names, size=n_sp, replace=False)
        abundances = rng.exponential(5.0, size=n_sp)
        for sp, abd in zip(chosen, abundances):
            rows.append({
                "plot_id": pid,
                "species": sp,
                "abundance": float(abd),
                "genus": sp_to_genus[sp],
                "family": sp_to_family[sp],
            })

    species_df = pl.DataFrame(rows)

    roles = RoleMapping(
        plot_id="plot_id",
        species_id="species",
        species_plot_id="plot_id",
        coords_lat="lat",
        coords_lon="lon",
        abundance="abundance",
        taxonomy_genus="genus",
        taxonomy_family="family",
    )

    targets = {
        "area": TargetConfig(
            column="area",
            task="regression",
            transform="log1p",
        ),
        "habitat": TargetConfig(
            column="habitat",
            task="classification",
            num_classes=3,
        ),
    }

    return ResolveDataset(
        header=header,
        species=species_df,
        roles=roles,
        targets=targets,
    )


@pytest.fixture
def dataset() -> ResolveDataset:
    return make_synthetic_data()
