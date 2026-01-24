"""Tests for ResolveModel Python bindings."""

import pytest
import torch
import numpy as np

# Try to import C++ bindings
try:
    from resolve import ResolveModel, ModelConfig, TargetConfig, TaskType, SpeciesEncodingMode
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    pytest.skip("C++ bindings not available", allow_module_level=True)


@pytest.fixture
def regression_target_config():
    """Create a regression target configuration."""
    config = TargetConfig()
    config.name = "ph"
    config.task = TaskType.Regression
    config.n_classes = 1
    return config


@pytest.fixture
def classification_target_config():
    """Create a classification target configuration."""
    config = TargetConfig()
    config.name = "soil_type"
    config.task = TaskType.Classification
    config.n_classes = 5
    return config


@pytest.fixture
def model_config():
    """Create a model configuration."""
    config = ModelConfig()
    config.encoder_dim = 64
    config.hidden_dim = 128
    config.n_encoder_layers = 2
    config.dropout = 0.1
    config.hash_dim = 16
    config.genus_vocab_size = 50
    config.family_vocab_size = 25
    config.n_continuous = 10
    config.top_k = 3
    config.mode = SpeciesEncodingMode.Hash
    return config


class TestResolveModelConstruction:
    """Test ResolveModel construction."""

    def test_construction(self, model_config, regression_target_config):
        """Test basic model construction."""
        model = ResolveModel(model_config, [regression_target_config])
        assert model is not None

    def test_construction_multiple_targets(self, model_config, regression_target_config, classification_target_config):
        """Test model construction with multiple targets."""
        model = ResolveModel(model_config, [regression_target_config, classification_target_config])
        assert model is not None


class TestResolveModelForward:
    """Test ResolveModel forward pass."""

    def test_forward_regression(self, model_config, regression_target_config):
        """Test forward pass for regression."""
        model = ResolveModel(model_config, [regression_target_config])
        model.eval()

        batch_size = 4
        n_continuous = model_config.hash_dim + model_config.n_continuous
        n_taxonomy = 2 * model_config.top_k

        continuous = torch.randn(batch_size, n_continuous)
        genus_ids = torch.randint(0, model_config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, model_config.family_vocab_size, (batch_size, n_taxonomy))

        outputs = model.forward(continuous, genus_ids, family_ids)

        assert "ph" in outputs
        assert outputs["ph"].shape == (batch_size, 1)

    def test_forward_classification(self, model_config, classification_target_config):
        """Test forward pass for classification."""
        model = ResolveModel(model_config, [classification_target_config])
        model.eval()

        batch_size = 4
        n_continuous = model_config.hash_dim + model_config.n_continuous
        n_taxonomy = 2 * model_config.top_k

        continuous = torch.randn(batch_size, n_continuous)
        genus_ids = torch.randint(0, model_config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, model_config.family_vocab_size, (batch_size, n_taxonomy))

        outputs = model.forward(continuous, genus_ids, family_ids)

        assert "soil_type" in outputs
        assert outputs["soil_type"].shape == (batch_size, 5)  # 5 classes


class TestResolveModelModes:
    """Test ResolveModel with different encoding modes."""

    def test_embed_mode(self, regression_target_config):
        """Test model with embed mode."""
        config = ModelConfig()
        config.encoder_dim = 64
        config.hidden_dim = 128
        config.n_encoder_layers = 2
        config.dropout = 0.1
        config.species_vocab_size = 100
        config.genus_vocab_size = 50
        config.family_vocab_size = 25
        config.n_continuous = 10
        config.top_k = 5
        config.mode = SpeciesEncodingMode.Embed

        model = ResolveModel(config, [regression_target_config])
        model.eval()

        batch_size = 4
        n_taxonomy = 2 * config.top_k

        continuous = torch.randn(batch_size, config.n_continuous)
        genus_ids = torch.randint(0, config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, config.family_vocab_size, (batch_size, n_taxonomy))
        species_ids = torch.randint(0, config.species_vocab_size, (batch_size, config.top_k))

        outputs = model.forward(continuous, genus_ids, family_ids, species_ids=species_ids)

        assert "ph" in outputs
        assert outputs["ph"].shape == (batch_size, 1)

    def test_sparse_mode(self, regression_target_config):
        """Test model with sparse mode."""
        config = ModelConfig()
        config.encoder_dim = 64
        config.hidden_dim = 128
        config.n_encoder_layers = 2
        config.dropout = 0.1
        config.n_species_vector = 200
        config.genus_vocab_size = 50
        config.family_vocab_size = 25
        config.n_continuous = 10
        config.top_k = 3
        config.mode = SpeciesEncodingMode.Sparse

        model = ResolveModel(config, [regression_target_config])
        model.eval()

        batch_size = 4
        n_taxonomy = 2 * config.top_k

        continuous = torch.randn(batch_size, config.n_continuous)
        genus_ids = torch.randint(0, config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, config.family_vocab_size, (batch_size, n_taxonomy))
        species_vector = torch.rand(batch_size, config.n_species_vector)

        outputs = model.forward(continuous, genus_ids, family_ids, species_vector=species_vector)

        assert "ph" in outputs
        assert outputs["ph"].shape == (batch_size, 1)


class TestResolveModelLatent:
    """Test ResolveModel latent representation extraction."""

    def test_get_latent(self, model_config, regression_target_config):
        """Test latent representation extraction."""
        model = ResolveModel(model_config, [regression_target_config])
        model.eval()

        batch_size = 4
        n_continuous = model_config.hash_dim + model_config.n_continuous
        n_taxonomy = 2 * model_config.top_k

        continuous = torch.randn(batch_size, n_continuous)
        genus_ids = torch.randint(0, model_config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, model_config.family_vocab_size, (batch_size, n_taxonomy))

        latent = model.get_latent(continuous, genus_ids, family_ids)

        assert latent.shape == (batch_size, model_config.encoder_dim)


class TestResolveModelTraining:
    """Test ResolveModel training functionality."""

    def test_train_eval_modes(self, model_config, regression_target_config):
        """Test switching between train and eval modes."""
        model = ResolveModel(model_config, [regression_target_config])

        model.train()
        assert model.training

        model.eval()
        assert not model.training

    def test_gradient_computation(self, model_config, regression_target_config):
        """Test that gradients can be computed."""
        model = ResolveModel(model_config, [regression_target_config])
        model.train()

        batch_size = 4
        n_continuous = model_config.hash_dim + model_config.n_continuous
        n_taxonomy = 2 * model_config.top_k

        continuous = torch.randn(batch_size, n_continuous)
        genus_ids = torch.randint(0, model_config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, model_config.family_vocab_size, (batch_size, n_taxonomy))
        targets = torch.randn(batch_size)

        outputs = model.forward(continuous, genus_ids, family_ids)
        loss = torch.nn.functional.mse_loss(outputs["ph"].squeeze(), targets)

        loss.backward()

        # Check that some gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad


class TestResolveModelSerialization:
    """Test ResolveModel save/load functionality."""

    def test_save_load_roundtrip(self, model_config, regression_target_config, tmp_path):
        """Test that save/load preserves model state."""
        model1 = ResolveModel(model_config, [regression_target_config])
        model2 = ResolveModel(model_config, [regression_target_config])

        # Create test input
        batch_size = 2
        n_continuous = model_config.hash_dim + model_config.n_continuous
        n_taxonomy = 2 * model_config.top_k

        continuous = torch.randn(batch_size, n_continuous)
        genus_ids = torch.randint(0, model_config.genus_vocab_size, (batch_size, n_taxonomy))
        family_ids = torch.randint(0, model_config.family_vocab_size, (batch_size, n_taxonomy))

        # Get output from model1
        model1.eval()
        out1 = model1.forward(continuous, genus_ids, family_ids)

        # Save model1
        save_path = tmp_path / "model.pt"
        model1.save(str(save_path))

        # Load into model2
        model2.load(str(save_path))

        # Get output from model2
        model2.eval()
        out2 = model2.forward(continuous, genus_ids, family_ids)

        # Outputs should match
        torch.testing.assert_close(out1["ph"], out2["ph"])
