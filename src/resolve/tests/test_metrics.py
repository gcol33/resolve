"""Tests for RESOLVE metrics."""

import pytest
import torch


class TestMetrics:
    """Test the Metrics class from resolve_core."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Import resolve_core or skip."""
        try:
            from resolve_core import Metrics
            self.Metrics = Metrics
        except ImportError:
            pytest.skip("resolve_core not installed")

    def test_mae(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.5, 2.0, 2.5, 4.5, 5.0])

        mae = self.Metrics.mae(pred, target)

        assert abs(mae - 0.3) < 1e-5

    def test_rmse(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 4.0])

        rmse = self.Metrics.rmse(pred, target)

        assert abs(rmse - 0.8165) < 0.001

    def test_r_squared_perfect(self):
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = self.Metrics.r_squared(vals, vals)

        assert r2 == 1.0

    def test_r_squared_good_fit(self):
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target = torch.tensor([1.1, 2.0, 2.9, 4.1, 5.0])

        r2 = self.Metrics.r_squared(pred, target)

        assert r2 > 0.95

    def test_smape(self):
        pred = torch.tensor([100.0, 200.0, 300.0])
        target = torch.tensor([110.0, 200.0, 280.0])

        smape = self.Metrics.smape(pred, target)

        assert smape > 0.0
        assert smape < 0.1

    def test_band_accuracy(self):
        pred = torch.tensor([100.0, 200.0, 300.0, 400.0])
        target = torch.tensor([100.0, 180.0, 250.0, 500.0])

        acc_25 = self.Metrics.band_accuracy(pred, target, 0.25)
        acc_10 = self.Metrics.band_accuracy(pred, target, 0.10)

        assert acc_25 >= acc_10  # Wider band should have >= accuracy

    def test_accuracy(self):
        # 3 classes, batch of 5
        pred = torch.tensor([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],  # wrong
            [0.1, 0.7, 0.2]
        ])
        target = torch.tensor([0, 1, 2, 1, 1])

        acc = self.Metrics.accuracy(pred, target)

        assert abs(acc - 0.8) < 1e-5

    def test_confusion_matrix(self):
        pred = torch.tensor([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.3, 0.7]
        ])
        target = torch.tensor([0, 1, 1, 0])

        cm = self.Metrics.confusion_matrix(pred, target, 2)

        assert cm.shape == (2, 2)
        assert cm.sum().item() == 4

    def test_classification_metrics(self):
        pred = torch.tensor([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9]
        ])
        target = torch.tensor([0, 1, 2])

        metrics = self.Metrics.classification_metrics(pred, target, 3)

        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert len(metrics.per_class_f1) == 3

    def test_accuracy_at_threshold(self):
        pred = torch.tensor([
            [0.9, 0.1],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.5, 0.5]
        ])
        target = torch.tensor([0, 1, 1, 1])
        confidence = torch.tensor([0.9, 0.6, 0.7, 0.5])

        # All samples
        result = self.Metrics.accuracy_at_threshold(pred, target, confidence, 0.0)
        assert result.n_samples == 4
        assert abs(result.coverage - 1.0) < 1e-5

        # High confidence only
        result = self.Metrics.accuracy_at_threshold(pred, target, confidence, 0.65)
        assert result.n_samples == 2
        assert result.accuracy == 1.0  # Both high-confidence are correct

    def test_accuracy_coverage_curve(self):
        pred = torch.tensor([
            [0.9, 0.1],
            [0.6, 0.4],
            [0.3, 0.7]
        ])
        target = torch.tensor([0, 0, 1])
        confidence = torch.tensor([0.9, 0.6, 0.7])

        thresholds = [0.0, 0.5, 0.8, 1.0]
        curve = self.Metrics.accuracy_coverage_curve(pred, target, confidence, thresholds)

        assert len(curve) == 4
        # Coverage should decrease as threshold increases
        for i in range(len(curve) - 1):
            assert curve[i].coverage >= curve[i + 1].coverage
