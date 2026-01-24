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


# === Edge Case Fixtures ===

@pytest.fixture
def empty_header_df():
    """Create empty header dataframe (0 plots)."""
    return pd.DataFrame({
        "PlotID": [],
        "Latitude": [],
        "Longitude": [],
        "Area": [],
        "Elevation": [],
        "Temperature": [],
    })


@pytest.fixture
def empty_species_df():
    """Create empty species dataframe (0 observations)."""
    return pd.DataFrame({
        "PlotID": [],
        "Species": [],
        "Genus": [],
        "Family": [],
        "Cover": [],
    })


@pytest.fixture
def header_with_null_targets():
    """Create header with all-null target values."""
    np.random.seed(42)
    n_plots = 20

    return pd.DataFrame({
        "PlotID": [f"P{i:04d}" for i in range(n_plots)],
        "Latitude": np.random.uniform(45, 55, n_plots),
        "Longitude": np.random.uniform(5, 15, n_plots),
        "Area": [np.nan] * n_plots,  # All null
        "Elevation": np.random.uniform(100, 2000, n_plots),
        "Temperature": np.random.uniform(5, 15, n_plots),
    })


@pytest.fixture
def header_with_partial_null_targets():
    """Create header with partially null target values."""
    np.random.seed(42)
    n_plots = 20

    area = np.random.exponential(100, n_plots)
    area[:10] = np.nan  # First 10 are null

    return pd.DataFrame({
        "PlotID": [f"P{i:04d}" for i in range(n_plots)],
        "Latitude": np.random.uniform(45, 55, n_plots),
        "Longitude": np.random.uniform(5, 15, n_plots),
        "Area": area,
        "Elevation": np.random.uniform(100, 2000, n_plots),
        "Temperature": np.random.uniform(5, 15, n_plots),
    })


@pytest.fixture
def species_with_single_species_per_plot():
    """Create species data with only 1 species per plot."""
    np.random.seed(42)
    n_plots = 20

    rows = []
    for i in range(n_plots):
        rows.append({
            "PlotID": f"P{i:04d}",
            "Species": "Quercus_robur",
            "Genus": "Quercus",
            "Family": "Fagaceae",
            "Cover": np.random.uniform(1, 50),
        })

    return pd.DataFrame(rows)


@pytest.fixture
def species_with_zero_abundance():
    """Create species data with some zero abundance values."""
    np.random.seed(42)
    n_plots = 20

    rows = []
    for i in range(n_plots):
        for j in range(5):
            cover = 0.0 if j == 0 else np.random.uniform(1, 50)  # First species has 0 cover
            rows.append({
                "PlotID": f"P{i:04d}",
                "Species": f"Species_{j}",
                "Genus": "Quercus",
                "Family": "Fagaceae",
                "Cover": cover,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def header_with_nan_coordinates():
    """Create header with NaN coordinate values."""
    np.random.seed(42)
    n_plots = 20

    lat = np.random.uniform(45, 55, n_plots)
    lon = np.random.uniform(5, 15, n_plots)
    lat[:5] = np.nan  # First 5 have NaN coordinates
    lon[:5] = np.nan

    return pd.DataFrame({
        "PlotID": [f"P{i:04d}" for i in range(n_plots)],
        "Latitude": lat,
        "Longitude": lon,
        "Area": np.random.exponential(100, n_plots),
        "Elevation": np.random.uniform(100, 2000, n_plots),
        "Temperature": np.random.uniform(5, 15, n_plots),
    })


@pytest.fixture
def small_species_df():
    """Create small species dataframe for 20 plots."""
    np.random.seed(42)
    n_plots = 20

    genera = ["Quercus", "Fagus", "Pinus", "Abies"]
    families = ["Fagaceae", "Pinaceae"]

    rows = []
    for i in range(n_plots):
        n_species = np.random.randint(3, 8)
        for j in range(n_species):
            genus = np.random.choice(genera)
            family = "Fagaceae" if genus in ["Quercus", "Fagus"] else "Pinaceae"
            rows.append({
                "PlotID": f"P{i:04d}",
                "Species": f"{genus}_sp{j}",
                "Genus": genus,
                "Family": family,
                "Cover": np.random.uniform(1, 50),
            })

    return pd.DataFrame(rows)
