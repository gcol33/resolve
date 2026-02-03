"""Tests for RESOLVE model."""

import pytest
import torch


class TestResolveModel:
    """Test the ResolveModel class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve_core or skip."""
        try:
            from resolve_core import (
                ResolveModel, ResolveSchema, ModelConfig,
                TargetConfig, TaskType, TransformType, SpeciesEncodingMode
            )
            self.ResolveModel = ResolveModel
            self.ResolveSchema = ResolveSchema
            self.ModelConfig = ModelConfig
            self.TargetConfig = TargetConfig
            self.TaskType = TaskType
            self.TransformType = TransformType
            self.SpeciesEncodingMode = SpeciesEncodingMode
        except ImportError:
            pytest.skip("resolve_core not installed")

    def _create_schema(self, has_taxonomy=True):
        """Helper to create a test schema."""
        schema = self.ResolveSchema()
        schema.n_plots = 100
        schema.n_species = 50
        schema.n_species_vocab = 100
        schema.has_coordinates = True
        schema.has_taxonomy = has_taxonomy
        schema.n_genera = 20 if has_taxonomy else 0
        schema.n_families = 10 if has_taxonomy else 0
        schema.n_genera_vocab = 25 if has_taxonomy else 0
        schema.n_families_vocab = 15 if has_taxonomy else 0
        schema.track_unknown_fraction = True

        target = self.TargetConfig()
        target.name = "area"
        target.task = self.TaskType.Regression
        target.transform = self.TransformType.None_
        target.num_classes = 0
        target.weight = 1.0
        schema.targets = [target]

        return schema

    def test_model_creation_hash_mode(self):
        schema = self._create_schema()

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 32
        config.hidden_dims = [64, 32]

        model = self.ResolveModel(schema, config)

        assert model.species_encoding == self.SpeciesEncodingMode.Hash
        assert model.latent_dim == 32  # Last hidden dim

    def test_model_forward_hash_mode(self):
        schema = self._create_schema()

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 32
        config.n_taxonomy_slots = 3
        config.hidden_dims = [64, 32]

        model = self.ResolveModel(schema, config)

        # n_continuous = 2 (coords) + 1 (unknown_fraction) + 32 (hash) = 35
        continuous = torch.randn(8, 35)
        genus_ids = torch.randint(0, 25, (8, 3))
        family_ids = torch.randint(0, 15, (8, 3))

        outputs = model.forward(continuous, genus_ids, family_ids)

        assert "area" in outputs
        assert outputs["area"].shape == (8, 1)

    def test_model_forward_embed_mode(self):
        schema = self._create_schema()

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Embed
        config.species_embed_dim = 16
        config.top_k_species = 5
        config.n_taxonomy_slots = 3
        config.hidden_dims = [64, 32]

        model = self.ResolveModel(schema, config)

        # n_continuous = 2 (coords) + 1 (unknown_fraction) = 3
        continuous = torch.randn(8, 3)
        species_ids = torch.randint(1, 100, (8, 5))
        genus_ids = torch.randint(1, 25, (8, 3))
        family_ids = torch.randint(1, 15, (8, 3))

        outputs = model.forward(continuous, genus_ids, family_ids, species_ids)

        assert "area" in outputs
        assert outputs["area"].shape == (8, 1)

    def test_model_get_latent(self):
        schema = self._create_schema(has_taxonomy=False)

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 32
        config.hidden_dims = [64, 32]

        model = self.ResolveModel(schema, config)

        # n_continuous = 2 (coords) + 1 (unknown) + 32 (hash) = 35
        continuous = torch.randn(8, 35)

        latent = model.get_latent(continuous)

        assert latent.shape == (8, 32)

    def test_model_multi_task(self):
        schema = self._create_schema()

        # Add classification target
        clf_target = self.TargetConfig()
        clf_target.name = "habitat"
        clf_target.task = self.TaskType.Classification
        clf_target.num_classes = 9
        clf_target.weight = 1.0
        schema.targets.append(clf_target)

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 32
        config.n_taxonomy_slots = 3
        config.hidden_dims = [64, 32]

        model = self.ResolveModel(schema, config)

        continuous = torch.randn(8, 35)
        genus_ids = torch.randint(0, 25, (8, 3))
        family_ids = torch.randint(0, 15, (8, 3))

        outputs = model.forward(continuous, genus_ids, family_ids)

        assert "area" in outputs
        assert "habitat" in outputs
        assert outputs["area"].shape == (8, 1)
        assert outputs["habitat"].shape == (8, 9)

    def test_model_train_eval_mode(self):
        schema = self._create_schema(has_taxonomy=False)

        config = self.ModelConfig()
        config.hidden_dims = [32, 16]

        model = self.ResolveModel(schema, config)

        # Test train mode
        model.train(True)
        # Test eval mode
        model.eval()

        # Should not raise
        continuous = torch.randn(4, 35)
        _ = model.forward(continuous)
