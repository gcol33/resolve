"""
RESOLVE: Relational Encoding via Structured Observation Learning with Vector Embeddings

Tests for PlotEncoder and metrics equivalence.
"""

import pytest
import numpy as np
import torch

try:
    from resolve import (
        PlotEncoder, EncodingType, DataSource,
        Metrics, PhasedLoss, PhaseConfig, to_records,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ implementation required")


# --- PlotEncoder Tests ---

class TestPlotEncoder:
    """Test PlotEncoder functionality."""

    def test_creation(self):
        encoder = PlotEncoder()
        assert not encoder.is_fitted()

    def test_add_specs(self):
        encoder = PlotEncoder()
        encoder.add_numeric("coords", ["lat", "lon"], source=DataSource.Plot)
        encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")
        encoder.add_embed("genus", ["genus"], dim=8, top_k=3, rank_by="abundance")

        specs = encoder.specs()
        assert len(specs) == 3
        assert specs[0].type == EncodingType.Numeric
        assert specs[1].type == EncodingType.Hash
        assert specs[2].type == EncodingType.Embed

    def test_fit_transform(self, sample_data):
        data_df, obs_df = sample_data
        encoder = PlotEncoder()
        encoder.add_numeric("coords", ["latitude", "longitude"], source=DataSource.Plot)
        encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")

        encoded = encoder.fit_transform(
            to_records(data_df, numeric_cols=['latitude', 'longitude']),
            to_records(obs_df, cat_cols=['species_id'], numeric_cols=['abundance'], is_obs=True),
            data_df['plot_id'].tolist()
        )

        assert encoder.is_fitted()
        assert encoded.continuous_features().shape[0] == len(data_df)

    def test_deterministic(self, sample_data):
        data_df, obs_df = sample_data
        plot_records = to_records(data_df, numeric_cols=['latitude', 'longitude'])
        obs_records = to_records(obs_df, cat_cols=['species_id'], numeric_cols=['abundance'], is_obs=True)
        plot_ids = data_df['plot_id'].tolist()

        def make_encoder():
            enc = PlotEncoder()
            enc.add_numeric("coords", ["latitude", "longitude"], source=DataSource.Plot)
            enc.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")
            return enc

        result1 = make_encoder().fit_transform(plot_records, obs_records, plot_ids)
        result2 = make_encoder().fit_transform(plot_records, obs_records, plot_ids)

        np.testing.assert_allclose(
            result1.continuous_features().numpy(),
            result2.continuous_features().numpy(),
            rtol=1e-5
        )


# --- Metrics Tests ---

class TestMetrics:
    """Test metrics computations."""

    def test_mae(self):
        pred, target = torch.randn(100), torch.randn(100)
        assert np.isclose(Metrics.mae(pred, target), torch.abs(pred - target).mean().item(), rtol=1e-5)

    def test_rmse(self):
        pred, target = torch.randn(100), torch.randn(100)
        expected = torch.sqrt(torch.mean((pred - target) ** 2)).item()
        assert np.isclose(Metrics.rmse(pred, target), expected, rtol=1e-5)

    def test_band_accuracy(self):
        pred = torch.tensor([100.0, 200.0, 300.0, 400.0])
        target = torch.tensor([100.0, 180.0, 400.0, 600.0])
        rel_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
        expected = (rel_error <= 0.25).float().mean().item()
        assert np.isclose(Metrics.band_accuracy(pred, target, 0.25), expected, rtol=1e-5)

    def test_smape(self):
        pred = torch.abs(torch.randn(100)) + 0.1
        target = torch.abs(torch.randn(100)) + 0.1
        denom = (torch.abs(pred) + torch.abs(target)) / 2 + 1e-8
        expected = (torch.abs(pred - target) / denom).mean().item()
        assert np.isclose(Metrics.smape(pred, target), expected, rtol=1e-4)


# --- Loss Tests ---

class TestLoss:
    """Test loss computations."""

    @staticmethod
    def make_phase(mae=0, mse=0, huber=0, smape=0, band=0):
        phase = PhaseConfig()
        phase.mae, phase.mse, phase.huber, phase.smape, phase.band = mae, mse, huber, smape, band
        return phase

    def test_mae_loss(self):
        pred, target = torch.randn(32), torch.randn(32)
        result = PhasedLoss([self.make_phase(mae=1)], [])(pred, target, 0)
        expected = torch.abs(pred - target).mean()
        assert np.isclose(result.item(), expected.item(), rtol=1e-5)

    def test_mse_loss(self):
        pred, target = torch.randn(32), torch.randn(32)
        result = PhasedLoss([self.make_phase(mse=1)], [])(pred, target, 0)
        expected = torch.mean((pred - target) ** 2)
        assert np.isclose(result.item(), expected.item(), rtol=1e-5)
