"""Tests for RESOLVE dataset loading."""

import pytest
import torch


class TestResolveDataset:
    """Test the ResolveDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve_core or skip."""
        try:
            from resolve_core import (
                ResolveDataset, RoleMapping, TargetSpec, DatasetConfig,
                SpeciesEncodingMode, TaskType, TransformType
            )
            self.ResolveDataset = ResolveDataset
            self.RoleMapping = RoleMapping
            self.TargetSpec = TargetSpec
            self.DatasetConfig = DatasetConfig
            self.SpeciesEncodingMode = SpeciesEncodingMode
            self.TaskType = TaskType
            self.TransformType = TransformType
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_load_from_species_csv(self, sample_species_csv):
        """Test loading dataset from species CSV."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"
        roles.abundance = "cover"
        roles.longitude = "lon"
        roles.latitude = "lat"
        roles.genus = "genus"
        roles.family = "family"

        targets = [self.TargetSpec.regression("area")]

        config = self.DatasetConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 16

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, targets, config
        )

        assert dataset.n_plots == 3
        assert len(dataset.plot_ids) == 3
        assert dataset.coordinates.shape == (3, 2)
        assert dataset.hash_embedding.shape[1] == 16

    def test_load_with_taxonomy(self, sample_species_csv):
        """Test loading dataset with taxonomy encoding."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"
        roles.genus = "genus"
        roles.family = "family"

        config = self.DatasetConfig()
        config.top_k = 2
        config.use_taxonomy = True

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, [], config
        )

        assert dataset.genus_ids.defined()
        assert dataset.family_ids.defined()
        assert dataset.genus_ids.shape[1] == 2  # top_k

    def test_load_embed_mode(self, sample_species_csv):
        """Test loading dataset in embed mode."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"
        roles.abundance = "cover"

        config = self.DatasetConfig()
        config.species_encoding = self.SpeciesEncodingMode.Embed
        config.top_k_species = 3

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, [], config
        )

        assert dataset.species_ids.defined()
        assert dataset.species_ids.shape[1] == 3

    def test_load_sparse_mode(self, sample_species_csv):
        """Test loading dataset in sparse mode."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"
        roles.abundance = "cover"

        config = self.DatasetConfig()
        config.species_encoding = self.SpeciesEncodingMode.Sparse

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, [], config
        )

        assert dataset.species_vector.defined()
        # Vocab includes unique species
        assert dataset.species_vector.shape[1] >= 4

    def test_load_classification_target(self, sample_species_csv):
        """Test loading classification target."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"

        targets = [self.TargetSpec.classification("habitat", 10)]

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, targets
        )

        assert "habitat" in dataset.targets
        assert dataset.targets["habitat"].dtype == torch.int64

    def test_schema_populated(self, sample_species_csv):
        """Test that schema is correctly populated."""
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"
        roles.longitude = "lon"
        roles.latitude = "lat"
        roles.genus = "genus"
        roles.family = "family"

        targets = [
            self.TargetSpec.regression("area"),
            self.TargetSpec.classification("habitat", 9)
        ]

        dataset = self.ResolveDataset.from_species_csv(
            str(sample_species_csv), roles, targets
        )

        schema = dataset.schema

        assert schema.n_plots == 3
        assert schema.has_coordinates == True
        assert schema.has_taxonomy == True
        assert len(schema.targets) == 2


class TestRoleMapping:
    """Test RoleMapping helper methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve_core import RoleMapping
            self.RoleMapping = RoleMapping
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_has_coordinates(self):
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"

        assert not roles.has_coordinates()

        roles.longitude = "lon"
        roles.latitude = "lat"

        assert roles.has_coordinates()

    def test_has_taxonomy(self):
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"

        assert not roles.has_taxonomy()

        roles.genus = "genus"
        assert roles.has_taxonomy()

    def test_has_abundance(self):
        roles = self.RoleMapping()
        roles.plot_id = "plot_id"
        roles.species_id = "species"

        assert not roles.has_abundance()

        roles.abundance = "cover"
        assert roles.has_abundance()


class TestTargetSpec:
    """Test TargetSpec convenience constructors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve_core import TargetSpec, TaskType, TransformType
            self.TargetSpec = TargetSpec
            self.TaskType = TaskType
            self.TransformType = TransformType
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_regression_target(self):
        spec = self.TargetSpec.regression("area", self.TransformType.Log1p)

        assert spec.column_name == "area"
        assert spec.target_name == "area"
        assert spec.task == self.TaskType.Regression
        assert spec.transform == self.TransformType.Log1p

    def test_classification_target(self):
        spec = self.TargetSpec.classification("habitat", 9)

        assert spec.column_name == "habitat"
        assert spec.target_name == "habitat"
        assert spec.task == self.TaskType.Classification
        assert spec.num_classes == 9
