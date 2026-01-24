"""
pytest configuration and fixtures for RESOLVE tests.
"""

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture(scope="session")
def seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    return 42


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Sample Data Generation ---

def make_sample_data(n_plots=50, n_species=100, seed=42):
    """
    Create synthetic data/obs data for testing.

    Returns:
        data_df: Plot-level data (one row per plot)
        obs_df: Observation data (many rows per plot)
    """
    np.random.seed(seed)
    genera = ['Quercus', 'Pinus', 'Acer', 'Betula', 'Fagus', 'Abies', 'Picea', 'Larix']
    families = ['Fagaceae', 'Pinaceae', 'Sapindaceae', 'Betulaceae']

    obs_records = []
    for i in range(n_plots):
        plot_id = f'plot_{i:03d}'
        for _ in range(np.random.randint(3, 12)):
            genus = np.random.choice(genera)
            obs_records.append({
                'plot_id': plot_id,
                'species_id': f'species_{np.random.randint(0, n_species):03d}',
                'genus': genus,
                'family': families[genera.index(genus) % len(families)],
                'abundance': np.random.exponential(10)
            })

    data_df = pd.DataFrame({
        'plot_id': [f'plot_{i:03d}' for i in range(n_plots)],
        'latitude': np.random.uniform(40, 60, n_plots),
        'longitude': np.random.uniform(-10, 30, n_plots),
        'elevation': np.random.uniform(0, 2000, n_plots),
        'ph': np.random.uniform(4, 8, n_plots),
        'nitrogen': np.random.exponential(0.2, n_plots),
        'carbon': np.random.exponential(3, n_plots),
    })

    return data_df, pd.DataFrame(obs_records)


@pytest.fixture
def sample_data():
    """Create sample data using data/obs pattern."""
    return make_sample_data(n_plots=50, n_species=100, seed=42)


@pytest.fixture
def small_sample_data():
    """Create small sample data for quick tests."""
    return make_sample_data(n_plots=10, n_species=20, seed=42)


@pytest.fixture
def large_sample_data():
    """Create larger sample data for performance testing."""
    return make_sample_data(n_plots=200, n_species=200, seed=42)
