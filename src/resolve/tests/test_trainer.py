"""Tests for RESOLVE trainer."""

import pytest
import torch


class TestTrainer:
    """Test the Trainer class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve_core or skip."""
        try:
            from resolve_core import (
                Trainer, ResolveModel, ResolveSchema, ModelConfig, TrainConfig,
                TargetConfig, TaskType, TransformType, SpeciesEncodingMode,
                LRSchedulerType
            )
            self.Trainer = Trainer
            self.ResolveModel = ResolveModel
            self.ResolveSchema = ResolveSchema
            self.ModelConfig = ModelConfig
            self.TrainConfig = TrainConfig
            self.TargetConfig = TargetConfig
            self.TaskType = TaskType
            self.TransformType = TransformType
            self.SpeciesEncodingMode = SpeciesEncodingMode
            self.LRSchedulerType = LRSchedulerType
        except ImportError:
            pytest.skip("resolve_core not installed")

    def _create_simple_setup(self):
        """Create a simple model and data for testing."""
        schema = self.ResolveSchema()
        schema.n_plots = 100
        schema.n_species = 20
        schema.has_coordinates = True
        schema.has_taxonomy = False
        schema.track_unknown_fraction = False

        target = self.TargetConfig()
        target.name = "area"
        target.task = self.TaskType.Regression
        target.transform = self.TransformType.None_
        target.num_classes = 0
        target.weight = 1.0
        schema.targets = [target]

        config = self.ModelConfig()
        config.species_encoding = self.SpeciesEncodingMode.Hash
        config.hash_dim = 16
        config.hidden_dims = [32, 16]

        model = self.ResolveModel(schema, config)

        # Create synthetic data
        n_samples = 50
        # continuous = coords(2) + hash(16) = 18
        coordinates = torch.randn(n_samples, 2)
        hash_embedding = torch.randn(n_samples, 16)
        targets = {"area": torch.randn(n_samples) * 100 + 500}

        return model, coordinates, hash_embedding, targets

    def test_trainer_creation(self):
        model, _, _, _ = self._create_simple_setup()

        train_config = self.TrainConfig()
        train_config.batch_size = 16
        train_config.max_epochs = 10

        trainer = self.Trainer(model, train_config)

        assert trainer.config.batch_size == 16
        assert trainer.config.max_epochs == 10

    def test_trainer_prepare_data(self):
        model, coordinates, hash_embedding, targets = self._create_simple_setup()

        train_config = self.TrainConfig()
        trainer = self.Trainer(model, train_config)

        # Prepare data with raw tensor API
        trainer.prepare_data_raw(
            coordinates,
            torch.Tensor(),  # no covariates
            hash_embedding,
            torch.Tensor(),  # no species_ids
            torch.Tensor(),  # no species_vector
            torch.Tensor(),  # no genus_ids
            torch.Tensor(),  # no family_ids
            torch.Tensor(),  # no unknown_fraction
            torch.Tensor(),  # no unknown_count
            targets,
            test_size=0.2,
            seed=42
        )

        # Should not raise - data is prepared

    def test_trainer_fit_short(self):
        """Test that training runs (very short run)."""
        model, coordinates, hash_embedding, targets = self._create_simple_setup()

        train_config = self.TrainConfig()
        train_config.batch_size = 16
        train_config.max_epochs = 2
        train_config.patience = 5

        trainer = self.Trainer(model, train_config)

        trainer.prepare_data_raw(
            coordinates,
            torch.Tensor(),
            hash_embedding,
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            targets,
            test_size=0.2
        )

        result = trainer.fit()

        assert result.best_epoch >= 0
        assert len(result.train_loss_history) == 2
        assert len(result.test_loss_history) == 2
        assert "area" in result.final_metrics

    def test_trainer_with_lr_scheduler(self):
        """Test training with LR scheduler."""
        model, coordinates, hash_embedding, targets = self._create_simple_setup()

        train_config = self.TrainConfig()
        train_config.batch_size = 16
        train_config.max_epochs = 3
        train_config.lr = 0.01
        train_config.lr_scheduler = self.LRSchedulerType.CosineAnnealing
        train_config.lr_min = 0.0001

        trainer = self.Trainer(model, train_config)

        trainer.prepare_data_raw(
            coordinates,
            torch.Tensor(),
            hash_embedding,
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            targets,
            test_size=0.2
        )

        result = trainer.fit()

        assert result.best_epoch >= 0


class TestTrainConfig:
    """Test TrainConfig settings."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve_core import TrainConfig, LRSchedulerType, LossConfigMode
            self.TrainConfig = TrainConfig
            self.LRSchedulerType = LRSchedulerType
            self.LossConfigMode = LossConfigMode
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_default_values(self):
        config = self.TrainConfig()

        assert config.batch_size == 4096
        assert config.max_epochs == 500
        assert config.patience == 50
        assert config.lr == 0.001
        assert config.lr_scheduler == self.LRSchedulerType.None_

    def test_lr_scheduler_options(self):
        config = self.TrainConfig()

        config.lr_scheduler = self.LRSchedulerType.StepLR
        config.lr_step_size = 50
        config.lr_gamma = 0.5

        assert config.lr_scheduler == self.LRSchedulerType.StepLR
        assert config.lr_step_size == 50
        assert config.lr_gamma == 0.5

    def test_loss_config_mode(self):
        config = self.TrainConfig()

        config.loss_config = self.LossConfigMode.MAE

        assert config.loss_config == self.LossConfigMode.MAE


class TestClassWeights:
    """Test class weights for imbalanced classification."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve_core import TargetConfig, TaskType
            self.TargetConfig = TargetConfig
            self.TaskType = TaskType
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_class_weights_assignment(self):
        config = self.TargetConfig()
        config.name = "habitat"
        config.task = self.TaskType.Classification
        config.num_classes = 5
        config.class_weights = [1.0, 2.0, 1.5, 3.0, 1.0]

        assert len(config.class_weights) == 5
        assert config.class_weights[1] == 2.0
