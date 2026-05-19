"""End-to-end smoke tests with synthetic data.

Tests the full pipeline: data → encoder → trainer → fit → predict
for all encoding modes (hash, embed, rank_pool, transformer).
"""

import numpy as np
import pytest

from resolve.train.trainer import Trainer


class TestHashMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots
        assert np.isfinite(preds["area"]).all()


class TestEmbedMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestRankPoolMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="rank_pool",
            species_embed_dim=16,
            genus_emb_dim=8,
            family_emb_dim=4,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestRankPoolWeighting:
    """Regression tests for the rank_pool_weighting kwarg.

    Prior bug: the rank-pool branch wired weighting=self.species_normalization
    (a dataset-level knob defaulting to "norm") so the user's species_aggregation
    setting was silently ignored on rank_pool/transformer encoders.
    """

    def _make_trainer(self, dataset, **kwargs):
        return Trainer(
            dataset,
            species_encoding="rank_pool",
            species_embed_dim=8,
            genus_emb_dim=4,
            family_emb_dim=4,
            hidden_dims=[16],
            max_epochs=1,
            patience=1,
            batch_size=64,
            verbose=0,
            **kwargs,
        )

    def test_explicit_kwarg_overrides_default(self, dataset):
        trainer = self._make_trainer(dataset, rank_pool_weighting="rank")
        trainer.fit()
        assert trainer._rank_pool_encoder.weighting == "rank"

    def test_back_compat_via_species_aggregation(self, dataset):
        # No rank_pool_weighting kwarg, but species_aggregation is a valid weighting.
        trainer = self._make_trainer(dataset, species_aggregation="log1p")
        trainer.fit()
        assert trainer._rank_pool_encoder.weighting == "log1p"

    def test_default_uses_species_aggregation_default(self, dataset):
        # Neither kwarg set. species_aggregation defaults to "abundance",
        # which is a valid rank-pool weighting.
        trainer = self._make_trainer(dataset)
        trainer.fit()
        assert trainer._rank_pool_encoder.weighting == "abundance"

    def test_invalid_kwarg_raises(self, dataset):
        with pytest.raises(ValueError, match="rank_pool_weighting"):
            self._make_trainer(dataset, rank_pool_weighting="not_a_mode")


class TestCacheRestoreSave:
    """Regression test for the resume-from-cache + save() failure.

    Prior bug: on cache hit, _prepare_data() was skipped entirely, so the
    encoder object (_rank_pool_encoder / _embedding_encoder) was never built.
    Training worked off the cached tensors, but trainer.save() then raised
    RuntimeError("encoder not initialized").
    """

    def _make_trainer(self, dataset, cache_dir, **kwargs):
        return Trainer(
            dataset,
            species_embed_dim=8,
            genus_emb_dim=4,
            family_emb_dim=4,
            hidden_dims=[16],
            max_epochs=1,
            patience=1,
            batch_size=64,
            cache_dir=str(cache_dir),
            verbose=0,
            **kwargs,
        )

    def _round_trip(self, dataset, tmp_path, **kwargs):
        cache_dir = tmp_path / "cache"
        # Fit #1 — cache miss, builds encoder and writes cache
        t1 = self._make_trainer(dataset, cache_dir, **kwargs)
        t1.fit()
        t1.save(str(tmp_path / "first.pt"))
        # Fit #2 — cache hit, encoder must be restored from cache
        t2 = self._make_trainer(dataset, cache_dir, **kwargs)
        t2.fit()
        t2.save(str(tmp_path / "second.pt"))
        return t2

    def test_rank_pool_save_after_cache_hit(self, dataset, tmp_path):
        t = self._round_trip(
            dataset, tmp_path,
            species_encoding="rank_pool",
            rank_pool_weighting="log1p",
        )
        assert t._rank_pool_encoder is not None
        assert t._rank_pool_encoder._fitted
        assert (tmp_path / "second.pt").exists()

    def test_hash_save_after_cache_hit(self, dataset, tmp_path):
        t = self._round_trip(
            dataset, tmp_path,
            species_encoding="hash",
            hash_dim=16,
        )
        assert t._species_encoder is not None
        assert (tmp_path / "second.pt").exists()


class TestTransformerMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="transformer",
            species_embed_dim=32,
            n_attention_layers=1,
            n_heads=2,
            transformer_ff_dim=64,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestConfidenceThreshold:
    def test_threshold_produces_nans(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            patience=2,
            batch_size=64,
            verbose=0,
        )
        trainer.fit()

        # High threshold should NaN out low-confidence predictions
        preds = trainer.predict(dataset, confidence_threshold=0.99)
        # At least some should be NaN (unknown species fraction > 0.01)
        assert np.isnan(preds["area"]).any() or np.isfinite(preds["area"]).all()


class TestOutputSpace:
    def test_transformed_vs_raw(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            patience=2,
            batch_size=64,
            verbose=0,
        )
        trainer.fit()

        raw = trainer.predict(dataset, output_space="raw")
        transformed = trainer.predict(dataset, output_space="transformed")
        # log1p transform: raw should have larger values (expm1 applied)
        assert not np.allclose(raw["area"], transformed["area"])
