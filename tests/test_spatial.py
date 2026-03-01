"""Tests for SpatialBlockSplitter."""

import numpy as np
import pytest

from resolve.data.spatial import SpatialBlockSplitter


class TestSpatialBlockSplitter:
    """Tests for spatial block cross-validation splitting."""

    def test_basic_split(self):
        """All plots appear in exactly one test fold."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(-10, 10, size=(200, 2)).astype(np.float32)
        splitter = SpatialBlockSplitter(block_size=1.0, n_splits=5, seed=42)
        folds = splitter.split(coords)

        assert len(folds) == 5

        # Every index appears in exactly one test fold
        all_test = np.concatenate([test for _, test in folds])
        assert len(all_test) == 200
        assert len(np.unique(all_test)) == 200

    def test_no_train_test_overlap(self):
        """No plot appears in both train and test within a fold."""
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 5, size=(100, 2)).astype(np.float32)
        splitter = SpatialBlockSplitter(block_size=0.5, n_splits=5, seed=7)
        folds = splitter.split(coords)

        for k, (train_idx, test_idx) in enumerate(folds):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Fold {k} has train/test overlap"

    def test_train_test_union_is_complete(self):
        """Train + test indices cover all plots in each fold."""
        n = 150
        rng = np.random.default_rng(2)
        coords = rng.uniform(-5, 5, size=(n, 2)).astype(np.float32)
        splitter = SpatialBlockSplitter(block_size=0.5, n_splits=3, seed=0)
        folds = splitter.split(coords)

        all_indices = set(range(n))
        for k, (train_idx, test_idx) in enumerate(folds):
            union = set(train_idx.tolist()) | set(test_idx.tolist())
            assert union == all_indices, f"Fold {k} missing indices"

    def test_same_block_same_fold(self):
        """Plots in the same spatial block are always in the same fold."""
        # Create clustered coords: 4 tight clusters
        coords = np.array([
            # Cluster A: block (0, 0) with block_size=1.0
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            # Cluster B: block (5, 5)
            [5.1, 5.2],
            [5.3, 5.4],
            # Cluster C: block (-3, -3)
            [-2.9, -2.8],
            [-2.7, -2.6],
            # Cluster D: block (10, 10)
            [10.1, 10.2],
            [10.3, 10.4],
            [10.5, 10.6],
        ], dtype=np.float32)

        splitter = SpatialBlockSplitter(block_size=1.0, n_splits=3, seed=42)
        folds = splitter.split(coords)

        # Within each fold, check that cluster members are always on the same side
        cluster_a = {0, 1, 2}
        cluster_b = {3, 4}
        cluster_c = {5, 6}
        cluster_d = {7, 8, 9}

        for k, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            for cluster_name, cluster in [
                ("A", cluster_a), ("B", cluster_b),
                ("C", cluster_c), ("D", cluster_d),
            ]:
                in_train = cluster & train_set
                in_test = cluster & test_set
                assert not (in_train and in_test), (
                    f"Fold {k}: Cluster {cluster_name} split across "
                    f"train ({in_train}) and test ({in_test})"
                )

    def test_reproducibility(self):
        """Same seed produces identical splits."""
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, 10, size=(80, 2)).astype(np.float32)

        folds_a = SpatialBlockSplitter(block_size=0.5, n_splits=4, seed=99).split(coords)
        folds_b = SpatialBlockSplitter(block_size=0.5, n_splits=4, seed=99).split(coords)

        for k in range(4):
            np.testing.assert_array_equal(folds_a[k][0], folds_b[k][0])
            np.testing.assert_array_equal(folds_a[k][1], folds_b[k][1])

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, 10, size=(100, 2)).astype(np.float32)

        folds_a = SpatialBlockSplitter(block_size=0.5, n_splits=5, seed=1).split(coords)
        folds_b = SpatialBlockSplitter(block_size=0.5, n_splits=5, seed=2).split(coords)

        # At least one fold should differ
        any_different = False
        for k in range(5):
            if not np.array_equal(folds_a[k][1], folds_b[k][1]):
                any_different = True
                break
        assert any_different, "Different seeds produced identical splits"

    def test_invalid_coords_shape(self):
        """Raises on wrong coordinate shape."""
        with pytest.raises(ValueError, match="shape"):
            SpatialBlockSplitter().split(np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="shape"):
            SpatialBlockSplitter().split(np.ones((10, 3)))

    def test_invalid_params(self):
        """Raises on invalid constructor params."""
        with pytest.raises(ValueError, match="block_size"):
            SpatialBlockSplitter(block_size=0)
        with pytest.raises(ValueError, match="block_size"):
            SpatialBlockSplitter(block_size=-1)
        with pytest.raises(ValueError, match="n_splits"):
            SpatialBlockSplitter(n_splits=1)

    def test_negative_coordinates(self):
        """Works correctly with negative coordinates."""
        coords = np.array([
            [-50.1, -70.2],
            [-50.05, -70.15],  # Same block as above at 0.1° resolution
            [30.5, 120.3],
            [30.55, 120.35],   # Same block
        ], dtype=np.float32)

        splitter = SpatialBlockSplitter(block_size=0.1, n_splits=2, seed=0)
        folds = splitter.split(coords)

        # Both southern plots should be in same fold
        for _, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            southern = {0, 1}
            assert not (southern & train_set and southern & test_set)
            northern = {2, 3}
            assert not (northern & train_set and northern & test_set)

    def test_single_block(self):
        """All plots in one block: one fold gets all, rest empty."""
        coords = np.array([
            [0.01, 0.01],
            [0.02, 0.02],
            [0.03, 0.03],
        ], dtype=np.float32)

        splitter = SpatialBlockSplitter(block_size=1.0, n_splits=3, seed=0)
        folds = splitter.split(coords)

        # Only one fold should have test data (single block)
        non_empty_test = [k for k, (_, test) in enumerate(folds) if len(test) > 0]
        assert len(non_empty_test) == 1
        # That fold's test set should contain all plots
        assert len(folds[non_empty_test[0]][1]) == 3
