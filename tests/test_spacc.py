"""Tests for spacc package."""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from spacc.data.dataset import SpaccDataset
from spacc.data.roles import RoleMapping, TargetConfig
from spacc.encode.species import SpeciesEncoder
from spacc.encode.vocab import TaxonomyVocab
from spacc.model.encoder import PlotEncoder
from spacc.model.head import TaskHead
from spacc.model.spacc import SpaccModel
from spacc.train.trainer import Trainer
from spacc.inference.predictor import Predictor


class TestRoleMapping:
    """Tests for RoleMapping."""

    def test_from_dict(self, sample_roles):
        roles = RoleMapping.from_dict(sample_roles)
        assert roles.plot_id == "PlotID"
        assert roles.has_abundance
        assert roles.has_taxonomy

    def test_validate_missing_required(self):
        roles = RoleMapping(
            plot_id="",
            species_id="Species",
            species_plot_id="PlotID",
            coords_lat="Lat",
            coords_lon="Lon",
        )
        with pytest.raises(ValueError, match="Missing required roles"):
            roles.validate()


class TestTargetConfig:
    """Tests for TargetConfig."""

    def test_regression(self):
        cfg = TargetConfig(column="Area", task="regression", transform="log1p")
        assert cfg.task == "regression"
        assert cfg.transform == "log1p"

    def test_classification_requires_num_classes(self):
        with pytest.raises(ValueError, match="num_classes required"):
            TargetConfig(column="Habitat", task="classification")

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="task must be"):
            TargetConfig(column="X", task="invalid")


class TestSpaccDataset:
    """Tests for SpaccDataset."""

    def test_from_csv(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        assert dataset.n_plots == 100
        assert len(dataset.targets) == 2

    def test_schema(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        schema = dataset.schema
        assert schema.n_plots == 100
        assert schema.has_taxonomy
        assert schema.has_abundance
        assert "Temperature" in schema.covariate_names

    def test_split(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        train, test = dataset.split(test_size=0.2)
        assert train.n_plots == 80
        assert test.n_plots == 20

    def test_get_coordinates(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        coords = dataset.get_coordinates()
        assert coords.shape == (100, 2)
        assert coords.dtype == np.float32

    def test_get_target(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        area = dataset.get_target("area")
        assert len(area) == 100
        # Should be log-transformed
        assert area.min() >= 0


class TestTaxonomyVocab:
    """Tests for TaxonomyVocab."""

    def test_from_species_data(self, sample_species_df):
        vocab = TaxonomyVocab.from_species_data(
            sample_species_df, "Genus", "Family"
        )
        assert vocab.n_genera > 1
        assert vocab.n_families > 1

    def test_encode(self, sample_species_df):
        vocab = TaxonomyVocab.from_species_data(
            sample_species_df, "Genus", "Family"
        )
        assert vocab.encode_genus("Quercus") > 0
        assert vocab.encode_genus("Unknown") == 0
        assert vocab.encode_genus(None) == 0

    def test_save_load(self, sample_species_df):
        vocab = TaxonomyVocab.from_species_data(
            sample_species_df, "Genus", "Family"
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            vocab.save(f.name)
            loaded = TaxonomyVocab.load(f.name)

        assert loaded.n_genera == vocab.n_genera
        assert loaded.n_families == vocab.n_families


class TestSpeciesEncoder:
    """Tests for SpeciesEncoder."""

    def test_fit_transform(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        encoder = SpeciesEncoder(hash_dim=32, top_k=3)
        encoder.fit(dataset)

        encoded = encoder.transform(dataset)
        assert encoded.hash_embedding.shape == (100, 32)
        assert encoded.genus_ids.shape == (100, 3)
        assert encoded.family_ids.shape == (100, 3)

    def test_without_taxonomy(self, sample_header_df, sample_species_df):
        # Remove taxonomy columns
        species_df = sample_species_df.drop(columns=["Genus", "Family"])

        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = Path(tmpdir) / "header.csv"
            species_path = Path(tmpdir) / "species.csv"

            sample_header_df.to_csv(header_path, index=False)
            species_df.to_csv(species_path, index=False)

            roles = {
                "plot_id": "PlotID",
                "species_id": "Species",
                "species_plot_id": "PlotID",
                "coords_lat": "Latitude",
                "coords_lon": "Longitude",
                "abundance": "Cover",
            }
            targets = {
                "area": {"column": "Area", "task": "regression"},
            }

            dataset = SpaccDataset.from_csv(
                header=header_path,
                species=species_path,
                roles=roles,
                targets=targets,
            )

            encoder = SpeciesEncoder(hash_dim=32, top_k=3)
            encoder.fit(dataset)
            encoded = encoder.transform(dataset)

            assert encoded.hash_embedding.shape == (100, 32)
            assert encoded.genus_ids is None
            assert encoded.family_ids is None


class TestPlotEncoder:
    """Tests for PlotEncoder."""

    def test_forward_with_taxonomy(self):
        encoder = PlotEncoder(
            n_continuous=35,  # 2 + 1 + 32
            n_genera=10,
            n_families=5,
            genus_emb_dim=8,
            family_emb_dim=8,
            top_k=3,
        )

        batch_size = 16
        continuous = torch.randn(batch_size, 35)
        genus_ids = torch.randint(0, 10, (batch_size, 3))
        family_ids = torch.randint(0, 5, (batch_size, 3))

        latent = encoder(continuous, genus_ids, family_ids)
        assert latent.shape == (batch_size, 64)

    def test_forward_without_taxonomy(self):
        encoder = PlotEncoder(
            n_continuous=35,
            n_genera=0,
            n_families=0,
        )

        batch_size = 16
        continuous = torch.randn(batch_size, 35)

        latent = encoder(continuous)
        assert latent.shape == (batch_size, 64)


class TestTaskHead:
    """Tests for TaskHead."""

    def test_regression(self):
        head = TaskHead(latent_dim=64, task="regression", transform="log1p")
        latent = torch.randn(16, 64)
        out = head(latent)
        assert out.shape == (16, 1)

    def test_classification(self):
        head = TaskHead(latent_dim=64, task="classification", num_classes=5)
        latent = torch.randn(16, 64)
        out = head(latent)
        assert out.shape == (16, 5)

    def test_predict_regression(self):
        head = TaskHead(latent_dim=64, task="regression", transform="log1p")
        latent = torch.randn(16, 64)
        pred = head.predict(latent)
        assert pred.shape == (16,)

    def test_predict_classification(self):
        head = TaskHead(latent_dim=64, task="classification", num_classes=5)
        latent = torch.randn(16, 64)
        pred = head.predict(latent)
        assert pred.shape == (16,)
        assert pred.max() < 5


class TestSpaccModel:
    """Tests for SpaccModel."""

    def test_forward(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=32,
        )

        batch_size = 16
        n_continuous = 2 + 1 + 32  # lat, lon, temp, hash
        continuous = torch.randn(batch_size, n_continuous)
        # Use valid indices based on schema
        n_genera = dataset.schema.n_genera + 1
        n_families = dataset.schema.n_families + 1
        genus_ids = torch.randint(0, n_genera, (batch_size, 3))
        family_ids = torch.randint(0, n_families, (batch_size, 3))

        outputs = model(continuous, genus_ids, family_ids)
        assert "area" in outputs
        assert "elevation" in outputs
        assert outputs["area"].shape == (batch_size, 1)
        assert outputs["elevation"].shape == (batch_size, 1)


class TestTrainer:
    """Tests for Trainer."""

    def test_fit_cpu(self, sample_csv_files, sample_roles, sample_targets):
        """Test training on CPU with small dataset."""
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=16,
            hidden_dims=[32, 16],
        )

        trainer = Trainer(
            model=model,
            dataset=dataset,
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert "area" in result.final_metrics
        assert "elevation" in result.final_metrics

    def test_save_load(self, sample_csv_files, sample_roles, sample_targets):
        """Test model save and load."""
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=16,
            hidden_dims=[32, 16],
        )

        trainer = Trainer(
            model=model,
            dataset=dataset,
            batch_size=32,
            max_epochs=2,
            device="cpu",
        )
        trainer.fit()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save(f.name)

            loaded_model, loaded_encoder, loaded_scalers = Trainer.load(f.name)
            assert loaded_model is not None
            assert loaded_encoder is not None


class TestPredictor:
    """Tests for Predictor."""

    def test_predict(self, sample_csv_files, sample_roles, sample_targets):
        """Test prediction on new data."""
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=16,
            hidden_dims=[32, 16],
        )

        trainer = Trainer(
            model=model,
            dataset=dataset,
            batch_size=32,
            max_epochs=2,
            device="cpu",
        )
        trainer.fit()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save(f.name)

            predictor = Predictor.load(f.name, device="cpu")
            predictions = predictor.predict(dataset)

            assert "area" in predictions.predictions
            assert "elevation" in predictions.predictions
            assert len(predictions["area"]) == 100

    def test_get_embeddings(self, sample_csv_files, sample_roles, sample_targets):
        """Test getting latent embeddings."""
        header_path, species_path = sample_csv_files
        dataset = SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=16,
            hidden_dims=[32, 16],
        )

        trainer = Trainer(
            model=model,
            dataset=dataset,
            batch_size=32,
            max_epochs=2,
            device="cpu",
        )
        trainer.fit()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save(f.name)

            predictor = Predictor.load(f.name, device="cpu")
            embeddings = predictor.get_embeddings(dataset)

            assert embeddings.shape[0] == 100
            assert embeddings.shape[1] == 16  # last hidden dim
