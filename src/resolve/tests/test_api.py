"""Tests for RESOLVE high-level Python API."""

import pytest
import tempfile
from pathlib import Path


class TestResolveDataset:
    """Test the high-level ResolveDataset class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve or skip."""
        try:
            from resolve import ResolveDataset, RoleMapping, TargetConfig
            self.ResolveDataset = ResolveDataset
            self.RoleMapping = RoleMapping
            self.TargetConfig = TargetConfig
        except ImportError:
            pytest.skip("resolve not installed")

    def test_from_csv(self, sample_header_csv, sample_species_csv):
        """Test loading dataset from CSV files."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
                "abundance": "cover",
                "longitude": "lon",
                "latitude": "lat",
                "genus": "genus",
                "family": "family",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        assert dataset.n_plots == 3
        assert len(dataset.plot_ids) == 3

    def test_from_csv_with_config(self, sample_header_csv, sample_species_csv):
        """Test loading dataset with custom config."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression", "transform": "log1p"},
            },
            species_encoding="hash",
            hash_dim=64,
            top_k=3,
        )

        assert dataset.n_plots == 3
        # Hash embedding should have 64 dimensions
        assert dataset.hash_embedding.shape[1] == 64

    def test_role_mapping_from_dict(self):
        """Test RoleMapping.from_dict() helper."""
        roles = self.RoleMapping.from_dict({
            "plot_id": "my_plot_id",
            "species_id": "my_species",
            "abundance": "cover",
        })

        assert roles.plot_id == "my_plot_id"
        assert roles.species_id == "my_species"
        assert roles.abundance == "cover"

    def test_target_config_from_dict(self):
        """Test TargetConfig.from_dict() helper."""
        config = self.TargetConfig.from_dict({
            "column": "area",
            "task": "regression",
            "transform": "log1p",
            "weight": 2.0,
        })

        assert config["column"] == "area"
        assert config["task"] == "regression"
        assert config["transform"] == "log1p"
        assert config["weight"] == 2.0


class TestTrainer:
    """Test the high-level Trainer class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve or skip."""
        try:
            from resolve import ResolveDataset, Trainer, TrainerConfig
            self.ResolveDataset = ResolveDataset
            self.Trainer = Trainer
            self.TrainerConfig = TrainerConfig
        except ImportError:
            pytest.skip("resolve not installed")

    def test_trainer_creation(self, sample_header_csv, sample_species_csv):
        """Test creating a Trainer from dataset."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
        )

        assert trainer._dataset is not None

    def test_trainer_fit_short(self, sample_header_csv, sample_species_csv):
        """Test that training runs (very short run)."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            batch_size=16,
        )

        results = trainer.fit()

        assert results.best_epoch >= 0
        assert len(results.train_loss_history) == 2
        assert "area" in results.final_metrics

    def test_trainer_save_load(self, sample_header_csv, sample_species_csv, temp_dir):
        """Test saving and loading a trained model."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            batch_size=16,
        )

        trainer.fit()

        # Save model
        model_path = temp_dir / "test_model.pt"
        trainer.save(str(model_path))

        assert model_path.exists()

    def test_trainer_create_predictor(self, sample_header_csv, sample_species_csv):
        """Test creating a predictor from trained model."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            batch_size=16,
        )

        trainer.fit()

        # Create predictor
        predictor = trainer.create_predictor()

        assert predictor is not None


class TestTrainerConfig:
    """Test the TrainerConfig dataclass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve import TrainerConfig
            self.TrainerConfig = TrainerConfig
        except ImportError:
            pytest.skip("resolve not installed")

    def test_default_values(self):
        """Test default configuration values."""
        config = self.TrainerConfig()

        assert config.hash_dim == 32
        assert config.hidden_dims == [2048, 1024, 512, 256, 128, 64]
        assert config.max_epochs == 500
        assert config.patience == 50
        assert config.batch_size == 4096
        assert config.lr == 0.001

    def test_custom_values(self):
        """Test custom configuration values."""
        config = self.TrainerConfig(
            hash_dim=64,
            hidden_dims=[256, 128],
            max_epochs=100,
            lr=0.01,
        )

        assert config.hash_dim == 64
        assert config.hidden_dims == [256, 128]
        assert config.max_epochs == 100
        assert config.lr == 0.01


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from resolve import ResolveDataset, Trainer
            self.ResolveDataset = ResolveDataset
            self.Trainer = Trainer
        except ImportError:
            pytest.skip("resolve not installed")

    def test_full_pipeline(self, sample_header_csv, sample_species_csv, temp_dir):
        """Test full train -> save -> load -> predict pipeline."""
        # 1. Load dataset
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
                "longitude": "lon",
                "latitude": "lat",
            },
            targets={
                "area": {"column": "area", "task": "regression"},
            },
        )

        # 2. Train model
        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=3,
            batch_size=16,
        )

        results = trainer.fit()
        assert results.best_epoch >= 0

        # 3. Save model
        model_path = temp_dir / "model.pt"
        trainer.save(str(model_path))
        assert model_path.exists()

        # 4. Create predictor and make predictions
        predictor = trainer.create_predictor()
        predictions = predictor.predict(dataset)

        assert "area" in predictions.predictions
        assert len(predictions.predictions["area"]) == 3  # 3 plots

    def test_multi_task_training(self, sample_header_csv, sample_species_csv):
        """Test training with multiple targets."""
        dataset = self.ResolveDataset.from_csv(
            header=str(sample_header_csv),
            species=str(sample_species_csv),
            roles={
                "plot_id": "plot_id",
                "species_id": "species",
            },
            targets={
                "area": {"column": "area", "task": "regression", "transform": "log1p"},
                "habitat": {"column": "habitat", "task": "classification", "num_classes": 10},
            },
        )

        trainer = self.Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            batch_size=16,
        )

        results = trainer.fit()

        # Both targets should have metrics
        assert "area" in results.final_metrics
        assert "habitat" in results.final_metrics
