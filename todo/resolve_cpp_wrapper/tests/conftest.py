"""Pytest fixtures for RESOLVE tests."""

import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_species_csv(temp_dir):
    """Create a sample species CSV file."""
    csv_content = """plot_id,species,cover,lon,lat,genus,family,area,habitat
p1,sp1,0.5,10.0,50.0,Quercus,Fagaceae,100,2
p1,sp2,0.3,10.0,50.0,Fagus,Fagaceae,100,2
p1,sp3,0.2,10.0,50.0,Pinus,Pinaceae,100,2
p2,sp1,0.8,11.0,51.0,Quercus,Fagaceae,200,5
p2,sp4,0.2,11.0,51.0,Abies,Pinaceae,200,5
p3,sp2,0.6,12.0,52.0,Fagus,Fagaceae,150,3
p3,sp3,0.4,12.0,52.0,Pinus,Pinaceae,150,3
"""
    csv_path = temp_dir / "species.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def sample_header_csv(temp_dir):
    """Create a sample header CSV file."""
    csv_content = """plot_id,lon,lat,elevation,area,habitat
p1,10.0,50.0,500,100,2
p2,11.0,51.0,600,200,5
p3,12.0,52.0,550,150,3
"""
    csv_path = temp_dir / "header.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def simple_schema():
    """Create a simple schema for testing."""
    try:
        from resolve_core import ResolveSchema, TargetConfig, TaskType, TransformType

        schema = ResolveSchema()
        schema.n_plots = 100
        schema.n_species = 50
        schema.has_coordinates = True
        schema.has_taxonomy = False
        schema.track_unknown_fraction = True

        target = TargetConfig()
        target.name = "area"
        target.task = TaskType.Regression
        target.transform = TransformType.None_
        target.num_classes = 0
        target.weight = 1.0

        schema.targets = [target]
        return schema
    except ImportError:
        pytest.skip("resolve_core not installed")


@pytest.fixture
def multi_task_schema():
    """Create a multi-task schema for testing."""
    try:
        from resolve_core import ResolveSchema, TargetConfig, TaskType, TransformType

        schema = ResolveSchema()
        schema.n_plots = 100
        schema.n_species = 50
        schema.has_coordinates = True
        schema.has_taxonomy = True
        schema.n_genera = 20
        schema.n_families = 10
        schema.n_genera_vocab = 25
        schema.n_families_vocab = 15
        schema.track_unknown_fraction = True

        # Regression target
        area_target = TargetConfig()
        area_target.name = "area"
        area_target.task = TaskType.Regression
        area_target.transform = TransformType.Log1p
        area_target.num_classes = 0
        area_target.weight = 1.0

        # Classification target
        habitat_target = TargetConfig()
        habitat_target.name = "habitat"
        habitat_target.task = TaskType.Classification
        habitat_target.transform = TransformType.None_
        habitat_target.num_classes = 9
        habitat_target.weight = 1.0

        schema.targets = [area_target, habitat_target]
        return schema
    except ImportError:
        pytest.skip("resolve_core not installed")
