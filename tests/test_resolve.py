"""Tests for RESOLVE package."""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from resolve.data.dataset import ResolveDataset
from resolve.data.roles import RoleMapping, TargetConfig
from resolve.encode.species import SpeciesEncoder
from resolve.encode.vocab import TaxonomyVocab
from resolve.model.encoder import PlotEncoder
from resolve.model.head import TaskHead
from resolve.model.resolve import ResolveModel
from resolve.train.trainer import Trainer
from resolve.inference.predictor import Predictor


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


class TestResolveDataset:
    """Tests for ResolveDataset."""

    def test_from_csv(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )
        assert dataset.n_plots == 100
        assert len(dataset.targets) == 2

    def test_schema(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
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
        dataset = ResolveDataset.from_csv(
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
        dataset = ResolveDataset.from_csv(
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
        dataset = ResolveDataset.from_csv(
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
        dataset = ResolveDataset.from_csv(
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

    def test_selection_modes(self, sample_csv_files, sample_roles, sample_targets):
        """Test that different selection modes produce different taxonomy IDs."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        # Test all three selection modes
        encoder_top = SpeciesEncoder(hash_dim=32, top_k=4, selection="top")
        encoder_bottom = SpeciesEncoder(hash_dim=32, top_k=4, selection="bottom")
        encoder_top_bottom = SpeciesEncoder(hash_dim=32, top_k=4, selection="top_bottom")

        encoder_top.fit(dataset)
        encoder_bottom.fit(dataset)
        encoder_top_bottom.fit(dataset)

        enc_top = encoder_top.transform(dataset)
        enc_bottom = encoder_bottom.transform(dataset)
        enc_top_bottom = encoder_top_bottom.transform(dataset)

        # top and bottom have K items each
        assert enc_top.genus_ids.shape == (100, 4)
        assert enc_bottom.genus_ids.shape == (100, 4)
        # top_bottom has 2K items (K top + K bottom)
        assert enc_top_bottom.genus_ids.shape == (100, 8)

        # Top and bottom should produce different orderings
        # (At least some plots should differ)
        assert not np.array_equal(enc_top.genus_ids, enc_bottom.genus_ids)

    def test_selection_invalid(self):
        """Test that invalid selection mode raises error."""
        with pytest.raises(ValueError, match="selection must be"):
            SpeciesEncoder(hash_dim=32, top_k=3, selection="invalid")


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


class TestResolveModel:
    """Tests for ResolveModel."""

    def test_forward(self, sample_csv_files, sample_roles, sample_targets):
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        model = ResolveModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=32,
        )

        batch_size = 16
        n_continuous = 2 + 1 + 32 + 1  # lat, lon, temp, hash, unknown_fraction
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
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert "area" in result.final_metrics
        assert "elevation" in result.final_metrics

    def test_resume_missing_continuous_scaler(self, sample_csv_files, sample_roles, sample_targets):
        """Test resuming from checkpoint with missing continuous scaler key."""
        import warnings

        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            trainer.fit()

            # Load checkpoint and remove continuous scaler key to simulate old checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            del checkpoint["scalers"]["continuous"]
            torch.save(checkpoint, checkpoint_path)

            # Create new trainer and try to resume - should warn but succeed
            new_trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=4,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = new_trainer.fit()
                # Check that warning was issued
                assert any("continuous" in str(warning.message) for warning in w)

            assert result.best_epoch >= 0

    def test_resume_scaler_dimension_mismatch(self, sample_csv_files, sample_roles, sample_targets):
        """Test resuming from checkpoint with different feature dimensions."""
        import warnings
        from sklearn.preprocessing import StandardScaler

        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            trainer.fit()

            # Load checkpoint and replace continuous scaler with one that has different dimensions
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            # Create a scaler with different number of features
            wrong_scaler = StandardScaler()
            wrong_scaler.fit(np.random.randn(10, 5))  # Wrong dimension
            checkpoint["scalers"]["continuous"] = wrong_scaler
            torch.save(checkpoint, checkpoint_path)

            # Create new trainer and try to resume - should warn but succeed
            new_trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=4,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = new_trainer.fit()
                # Check that dimension mismatch warning was issued
                assert any("dimension mismatch" in str(warning.message).lower() for warning in w)

            assert result.best_epoch >= 0

    def test_resume_missing_target_scaler(self, sample_csv_files, sample_roles, sample_targets):
        """Test resuming from checkpoint with missing target scaler key."""
        import warnings

        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            trainer.fit()

            # Load checkpoint and remove target scaler key
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            del checkpoint["scalers"]["target_area"]
            torch.save(checkpoint, checkpoint_path)

            # Create new trainer and try to resume - should warn but succeed
            new_trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=4,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = new_trainer.fit()
                # Check that warning was issued
                assert any("target_area" in str(warning.message) for warning in w)

            assert result.best_epoch >= 0

    def test_checkpoint_has_all_scalers(self, sample_csv_files, sample_roles, sample_targets):
        """Test that checkpoints always contain all required scaler keys."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            trainer.fit()

            # Load checkpoint and verify structure
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Verify scalers dict has required keys
            assert "scalers" in checkpoint
            assert "continuous" in checkpoint["scalers"], "Checkpoint missing 'continuous' scaler"
            for target_name in dataset.targets.keys():
                target_cfg = dataset.targets[target_name]
                if target_cfg.task == "regression":
                    assert f"target_{target_name}" in checkpoint["scalers"], \
                        f"Checkpoint missing 'target_{target_name}' scaler"

    def test_resume_no_warnings_normal_operation(self, sample_csv_files, sample_roles, sample_targets):
        """Test that normal resume operation produces no warnings."""
        import warnings

        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"

            # First training run
            trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=1,
            )
            trainer.fit()

            # Resume training - should produce NO scaler warnings
            new_trainer = Trainer(
                dataset,
                hash_dim=16,
                hidden_dims=[32, 16],
                batch_size=32,
                max_epochs=4,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                resume=True,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = new_trainer.fit()
                # No scaler-related warnings should occur in normal operation
                scaler_warnings = [
                    warning for warning in w
                    if "scaler" in str(warning.message).lower()
                ]
                assert len(scaler_warnings) == 0, \
                    f"Unexpected scaler warnings: {[str(w.message) for w in scaler_warnings]}"

            assert result.best_epoch >= 0

    def test_fit_embed_mode(self, sample_csv_files, sample_roles, sample_targets):
        """Test training with embed mode (learnable species embeddings)."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            hidden_dims=[32, 16],
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert "area" in result.final_metrics
        assert "elevation" in result.final_metrics
        # Verify model uses embed mode
        assert trainer.model.species_encoding == "embed"

    def test_fit_top_bottom_selection(self, sample_csv_files, sample_roles, sample_targets):
        """Test training with top_bottom species selection mode."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            hash_dim=16,
            top_k=6,  # Will be split: 3 top + 3 bottom
            hidden_dims=[32, 16],
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
            species_selection="top_bottom",
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert trainer.species_selection == "top_bottom"

    def test_fit_all_species_mode(self, sample_csv_files, sample_roles, sample_targets):
        """Test training with all species selection (full abundance vector)."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            species_encoding="hash",
            species_selection="all",
            species_embed_dim=16,
            hidden_dims=[32, 16],
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
            min_species_frequency=1,
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert "area" in result.final_metrics
        assert "elevation" in result.final_metrics
        # Verify model uses explicit vector mode
        assert trainer.model.uses_explicit_vector is True
        assert trainer.species_selection == "all"

    def test_fit_presence_absence_mode(self, sample_csv_files, sample_roles, sample_targets):
        """Test training with presence/absence species selection (binary vector)."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            species_encoding="hash",
            species_selection="all",
            species_representation="presence_absence",
            species_embed_dim=16,
            hidden_dims=[32, 16],
            batch_size=32,
            max_epochs=3,
            patience=2,
            device="cpu",
            min_species_frequency=1,
        )

        result = trainer.fit()
        assert result.best_epoch >= 0
        assert "area" in result.final_metrics
        # Verify model uses explicit vector mode
        assert trainer.model.uses_explicit_vector is True
        assert trainer.species_selection == "all"
        assert trainer.species_representation == "presence_absence"

    def test_save_load(self, sample_csv_files, sample_roles, sample_targets):
        """Test model save and load."""
        header_path, species_path = sample_csv_files
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
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
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
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
        dataset = ResolveDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=sample_roles,
            targets=sample_targets,
        )

        trainer = Trainer(
            dataset,
            hash_dim=16,
            hidden_dims=[32, 16],
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
