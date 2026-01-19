"""Test fixtures for RESOLVE."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def sample_header_df():
    """Create sample header dataframe."""
    np.random.seed(42)
    n_plots = 100

    return pd.DataFrame({
        "PlotID": [f"P{i:04d}" for i in range(n_plots)],
        "Latitude": np.random.uniform(45, 55, n_plots),
        "Longitude": np.random.uniform(5, 15, n_plots),
        "Area": np.random.exponential(100, n_plots),
        "Elevation": np.random.uniform(100, 2000, n_plots),
        "Habitat": np.random.choice(["forest", "grassland", "wetland"], n_plots),
        "Temperature": np.random.uniform(5, 15, n_plots),
    })


@pytest.fixture
def sample_species_df():
    """Create sample species dataframe."""
    np.random.seed(42)
    n_plots = 100
    species_per_plot = 20

    genera = ["Quercus", "Fagus", "Pinus", "Abies", "Betula", "Acer", "Fraxinus", "Tilia"]
    families = ["Fagaceae", "Pinaceae", "Betulaceae", "Sapindaceae", "Oleaceae", "Malvaceae"]

    rows = []
    for i in range(n_plots):
        n_species = np.random.randint(5, species_per_plot)
        for j in range(n_species):
            genus = np.random.choice(genera)
            family = {
                "Quercus": "Fagaceae",
                "Fagus": "Fagaceae",
                "Pinus": "Pinaceae",
                "Abies": "Pinaceae",
                "Betula": "Betulaceae",
                "Acer": "Sapindaceae",
                "Fraxinus": "Oleaceae",
                "Tilia": "Malvaceae",
            }[genus]
            rows.append({
                "PlotID": f"P{i:04d}",
                "Species": f"{genus}_sp{j}",
                "Genus": genus,
                "Family": family,
                "Cover": np.random.uniform(1, 50),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_roles():
    """Create sample role mapping."""
    return {
        "plot_id": "PlotID",
        "species_id": "Species",
        "species_plot_id": "PlotID",
        "coords_lat": "Latitude",
        "coords_lon": "Longitude",
        "abundance": "Cover",
        "taxonomy_genus": "Genus",
        "taxonomy_family": "Family",
        "covariates": ["Temperature"],
    }


@pytest.fixture
def sample_targets():
    """Create sample target configs."""
    return {
        "area": {
            "column": "Area",
            "task": "regression",
            "transform": "log1p",
        },
        "elevation": {
            "column": "Elevation",
            "task": "regression",
        },
    }


@pytest.fixture
def sample_csv_files(sample_header_df, sample_species_df):
    """Write sample data to temporary CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        header_path = Path(tmpdir) / "header.csv"
        species_path = Path(tmpdir) / "species.csv"

        sample_header_df.to_csv(header_path, index=False)
        sample_species_df.to_csv(species_path, index=False)

        yield header_path, species_path
