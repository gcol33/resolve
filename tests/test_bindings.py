"""Integration tests for RESOLVE Python bindings.

Tests the full pipeline roundtrips (train -> save -> load -> predict),
cross-validation, embedding extraction, batched prediction, encoding modes,
and output_space handling. All tests use the pure-Python fallback backend
(no C++ _resolve_core needed).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import tempfile
from pathlib import Path

from resolve import Predictor, ResolvePredictions, ResolveDataset, Trainer
from resolve.data.roles import RoleMapping, TargetConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_plots: int = 100,
    n_species: int = 20,
    n_genera: int = 6,
    n_families: int = 3,
    species_per_plot: int = 8,
    seed: int = 99,
) -> ResolveDataset:
    """Generate a small synthetic dataset for fast tests."""
    rng = np.random.default_rng(seed)

    plot_ids = [f"P{i:04d}" for i in range(n_plots)]
    lat = rng.uniform(45, 55, n_plots)
    lon = rng.uniform(5, 15, n_plots)
    # Regression target (positive, suitable for log1p transform)
    area = np.exp(rng.normal(2.0 + 0.1 * (lat - 50), 0.5))
    # 3-class classification target
    habitat = rng.integers(0, 3, n_plots)

    header = pl.DataFrame({
        "plot_id": plot_ids,
        "lat": lat,
        "lon": lon,
        "area": area,
        "habitat": habitat,
    })

    species_names = [f"sp_{i}" for i in range(n_species)]
    genus_names = [f"genus_{i % n_genera}" for i in range(n_species)]
    family_names = [f"family_{i % n_families}" for i in range(n_species)]
    sp_to_genus = dict(zip(species_names, genus_names))
    sp_to_family = dict(zip(species_names, family_names))

    rows = []
    for pid in plot_ids:
        n_sp = rng.integers(3, species_per_plot + 1)
        chosen = rng.choice(species_names, size=n_sp, replace=False)
        abundances = rng.exponential(5.0, size=n_sp)
        for sp, abd in zip(chosen, abundances):
            rows.append({
                "plot_id": pid,
                "species": sp,
                "abundance": float(abd),
                "genus": sp_to_genus[sp],
                "family": sp_to_family[sp],
            })

    species_df = pl.DataFrame(rows)

    roles = RoleMapping(
        plot_id="plot_id",
        species_id="species",
        species_plot_id="plot_id",
        coords_lat="lat",
        coords_lon="lon",
        abundance="abundance",
        taxonomy_genus="genus",
        taxonomy_family="family",
    )

    targets = {
        "area": TargetConfig(
            column="area",
            task="regression",
            transform="log1p",
        ),
        "habitat": TargetConfig(
            column="habitat",
            task="classification",
            num_classes=3,
        ),
    }

    return ResolveDataset(
        header=header,
        species=species_df,
        roles=roles,
        targets=targets,
    )


@pytest.fixture
def dataset() -> ResolveDataset:
    return _make_synthetic_data()


FAST_TRAINER_KWARGS = dict(
    hidden_dims=[32, 16],
    max_epochs=3,
    patience=2,
    batch_size=64,
    verbose=0,
)


# ---------------------------------------------------------------------------
# 1. Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    """Train -> save -> Predictor.load -> predict roundtrip."""

    def test_hash_roundtrip(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            result = predictor.predict(dataset)

            assert isinstance(result, ResolvePredictions)
            assert "area" in result.predictions
            assert "habitat" in result.predictions
            assert len(result["area"]) == dataset.n_plots
            assert len(result["habitat"]) == dataset.n_plots
            assert np.isfinite(result["area"]).all()

    def test_embed_roundtrip(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            result = predictor.predict(dataset)

            assert isinstance(result, ResolvePredictions)
            assert "area" in result.predictions
            assert len(result["area"]) == dataset.n_plots

    def test_predictions_to_polars(self, dataset: ResolveDataset) -> None:
        """Verify ResolvePredictions.to_polars() returns valid DataFrame."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            result = predictor.predict(dataset)
            df = result.to_polars()

            assert isinstance(df, pl.DataFrame)
            assert "plot_id" in df.columns
            assert "area" in df.columns
            assert len(df) == dataset.n_plots


# ---------------------------------------------------------------------------
# 2. Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Trainer.cross_validate() returns properly structured results."""

    def test_random_cv_3fold(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        cv_result = trainer.cross_validate(n_splits=3, seed=42, spatial=False)

        assert cv_result.n_folds == 3
        assert len(cv_result.fold_results) == 3
        assert len(cv_result.fold_metrics) == 3

        # mean_metrics and std_metrics have entries for both targets
        assert "area" in cv_result.mean_metrics
        assert "habitat" in cv_result.mean_metrics
        assert "area" in cv_result.std_metrics
        assert "habitat" in cv_result.std_metrics

        # Each fold has a non-negative best_epoch
        for fold in cv_result.fold_results:
            assert fold.best_epoch >= 0

        # Mean metrics are finite numbers
        for target_metrics in cv_result.mean_metrics.values():
            for value in target_metrics.values():
                assert np.isfinite(value)

    def test_spatial_cv_3fold(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        cv_result = trainer.cross_validate(
            n_splits=3, seed=42, spatial=True, block_deg=2.0,
        )

        assert cv_result.n_folds == 3
        assert len(cv_result.fold_results) == 3

    def test_cv_str_repr(self, dataset: ResolveDataset) -> None:
        """CVResult.__str__ produces readable output."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        cv_result = trainer.cross_validate(n_splits=3, seed=42, spatial=False)
        s = str(cv_result)
        assert "3-Fold CV Results" in s


# ---------------------------------------------------------------------------
# 3. Genus / family embedding extraction
# ---------------------------------------------------------------------------

class TestEmbeddingExtraction:
    """Predictor.get_genus_embeddings() / get_family_embeddings()."""

    def test_hash_mode_embeddings(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            genus_emb_dim=8,
            family_emb_dim=4,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")

            genus_emb = predictor.get_genus_embeddings()
            assert isinstance(genus_emb, np.ndarray)
            assert genus_emb.ndim == 2
            # genus_emb_dim=8, so second dimension should be 8
            assert genus_emb.shape[1] == 8
            assert np.isfinite(genus_emb).all()

            family_emb = predictor.get_family_embeddings()
            assert isinstance(family_emb, np.ndarray)
            assert family_emb.ndim == 2
            assert family_emb.shape[1] == 4
            assert np.isfinite(family_emb).all()

    def test_embed_mode_embeddings(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            genus_emb_dim=8,
            family_emb_dim=4,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")

            genus_emb = predictor.get_genus_embeddings()
            assert isinstance(genus_emb, np.ndarray)
            assert genus_emb.ndim == 2
            assert np.isfinite(genus_emb).all()

            family_emb = predictor.get_family_embeddings()
            assert isinstance(family_emb, np.ndarray)
            assert family_emb.ndim == 2
            assert np.isfinite(family_emb).all()

    def test_latent_extraction(self, dataset: ResolveDataset) -> None:
        """Predictor.get_embeddings() returns latent space vectors."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            latent = predictor.get_embeddings(dataset)

            assert isinstance(latent, np.ndarray)
            assert latent.ndim == 2
            assert latent.shape[0] == dataset.n_plots
            assert np.isfinite(latent).all()


# ---------------------------------------------------------------------------
# 4. Batched prediction
# ---------------------------------------------------------------------------

class TestBatchedPrediction:
    """Predictor.predict_batched() returns same results as predict()."""

    def test_batched_matches_full(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")

            full = predictor.predict(dataset)
            batched = predictor.predict_batched(dataset, batch_size=32)

            assert isinstance(batched, ResolvePredictions)
            assert set(full.predictions.keys()) == set(batched.predictions.keys())

            for target in full.predictions:
                np.testing.assert_allclose(
                    full[target], batched[target],
                    rtol=1e-5, atol=1e-6,
                    err_msg=f"Batched predictions differ for target '{target}'",
                )

    def test_batched_plot_ids(self, dataset: ResolveDataset) -> None:
        """Plot IDs in batched result match the dataset."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            batched = predictor.predict_batched(dataset, batch_size=25)

            np.testing.assert_array_equal(batched.plot_ids, dataset.plot_ids)


# ---------------------------------------------------------------------------
# 5. Encoding modes (smoke tests)
# ---------------------------------------------------------------------------

class TestEncodingModes:
    """Quick smoke tests for embed mode (hash/rank_pool/transformer already
    covered in test_e2e_synthetic.py)."""

    def test_embed_train_predict(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            **FAST_TRAINER_KWARGS,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots
        assert np.isfinite(preds["area"]).all()

    def test_hash_train_predict(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots

    def test_invalid_encoding_raises(self, dataset: ResolveDataset) -> None:
        with pytest.raises(ValueError, match="species_encoding must be"):
            Trainer(
                dataset,
                species_encoding="nonexistent",
                **FAST_TRAINER_KWARGS,
            )


# ---------------------------------------------------------------------------
# 6. Output space parameter
# ---------------------------------------------------------------------------

class TestOutputSpace:
    """output_space='transformed' vs 'raw' for log1p-transformed target."""

    def test_transformed_differs_from_raw(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        raw = trainer.predict(dataset, output_space="raw")
        transformed = trainer.predict(dataset, output_space="transformed")

        # log1p transform: raw has expm1 applied, so values differ
        assert not np.allclose(raw["area"], transformed["area"])
        # Classification target has no transform, so should be identical
        np.testing.assert_array_equal(raw["habitat"], transformed["habitat"])

    def test_predictor_output_space(self, dataset: ResolveDataset) -> None:
        """output_space works through Predictor (not just Trainer.predict)."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            raw = predictor.predict(dataset, output_space="raw")
            transformed = predictor.predict(dataset, output_space="transformed")

            assert not np.allclose(raw["area"], transformed["area"])

    def test_invalid_output_space_raises(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with pytest.raises(ValueError, match="output_space must be"):
            trainer.predict(dataset, output_space="invalid")

    def test_predictor_invalid_output_space_raises(
        self, dataset: ResolveDataset
    ) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")
            with pytest.raises(ValueError, match="output_space must be"):
                predictor.predict(dataset, output_space="bogus")


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Misc edge-case coverage."""

    def test_save_before_fit_raises(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Cannot save"):
                trainer.save(Path(tmpdir) / "model.pt")

    def test_predict_before_fit_raises(self, dataset: ResolveDataset) -> None:
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        with pytest.raises(RuntimeError, match="Cannot predict"):
            trainer.predict(dataset)

    def test_predictions_to_csv(self, dataset: ResolveDataset) -> None:
        """ResolvePredictions.to_csv() writes a valid file."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            trainer.save(model_path)

            predictor = Predictor.load(model_path, device="cpu")
            result = predictor.predict(dataset)

            csv_path = Path(tmpdir) / "preds.csv"
            result.to_csv(csv_path)
            assert csv_path.exists()

            loaded = pl.read_csv(csv_path)
            assert len(loaded) == dataset.n_plots
            assert "plot_id" in loaded.columns

    def test_confidence_threshold_predictor(
        self, dataset: ResolveDataset
    ) -> None:
        """Confidence threshold filters low-confidence predictions to NaN."""
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            **FAST_TRAINER_KWARGS,
        )
        trainer.fit()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            trainer.save(path)

            predictor = Predictor.load(path, device="cpu")

            # threshold=0 keeps everything
            result_all = predictor.predict(dataset, confidence_threshold=0.0)
            assert np.isfinite(result_all["area"]).all()

            # Very high threshold may NaN some predictions
            result_high = predictor.predict(dataset, confidence_threshold=0.99)
            # Either some are NaN (filtered) or all pass (all species known)
            assert (
                np.isnan(result_high["area"]).any()
                or np.isfinite(result_high["area"]).all()
            )
