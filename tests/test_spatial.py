"""Tests for SpatialBlockSplitter."""

import warnings

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


class TestBlockDeg:
    """Tests for the block_deg mode."""

    def test_block_deg_scalar(self):
        """Scalar block_deg produces valid folds identical to old block_size."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(-10, 10, size=(200, 2)).astype(np.float32)

        folds_deg = SpatialBlockSplitter(
            block_deg=1.0, n_splits=5, seed=42,
        ).split(coords)

        # Same result as deprecated block_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            folds_old = SpatialBlockSplitter(
                block_size=1.0, n_splits=5, seed=42,
            ).split(coords)

        for k in range(5):
            np.testing.assert_array_equal(folds_deg[k][0], folds_old[k][0])
            np.testing.assert_array_equal(folds_deg[k][1], folds_old[k][1])

    def test_block_deg_tuple(self):
        """Tuple block_deg creates rectangular blocks."""
        # Wide blocks (0.5 lat, 2.0 lon): points at same lat but different
        # lon should end up in the same block more often than with square blocks
        coords = np.array([
            [0.1, 0.1],
            [0.1, 1.5],  # same lat block, different lon block at 0.5 square
            [0.1, 0.1],
            [5.0, 5.0],
        ], dtype=np.float32)

        splitter = SpatialBlockSplitter(
            block_deg=(0.5, 2.0), n_splits=2, seed=0,
        )
        folds = splitter.split(coords)

        # Points 0 and 1 are in same block (lat_block=0, lon_block=0)
        # because 0.1/0.5=0 and 1.5/2.0=0
        for _, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            group = {0, 1, 2}
            assert not (group & train_set and group & test_set)

    def test_block_deg_negative_raises(self):
        """Negative block_deg raises ValueError."""
        with pytest.raises(ValueError, match="block_deg"):
            SpatialBlockSplitter(block_deg=-1.0)
        with pytest.raises(ValueError, match="block_deg"):
            SpatialBlockSplitter(block_deg=(1.0, -0.5))


class TestBlockKm:
    """Tests for the block_km mode."""

    def test_block_km_scalar(self):
        """Scalar block_km produces valid folds."""
        rng = np.random.default_rng(10)
        coords = rng.uniform(40, 50, size=(100, 2)).astype(np.float32)

        splitter = SpatialBlockSplitter(block_km=100.0, n_splits=5, seed=42)
        folds = splitter.split(coords)

        assert len(folds) == 5
        all_test = np.concatenate([test for _, test in folds])
        assert len(all_test) == 100
        assert len(np.unique(all_test)) == 100

    def test_block_km_tuple(self):
        """Tuple block_km produces valid rectangular blocks."""
        rng = np.random.default_rng(11)
        coords = rng.uniform(40, 50, size=(100, 2)).astype(np.float32)

        splitter = SpatialBlockSplitter(
            block_km=(50.0, 100.0), n_splits=5, seed=42,
        )
        folds = splitter.split(coords)

        assert len(folds) == 5
        all_test = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test)) == 100

    def test_block_km_latitude_correction(self):
        """cos(lat) correction produces wider lon blocks at higher latitudes."""
        # At 60°N, cos(60°) ≈ 0.5, so 111 km in lon ≈ 2° (not 1°)
        # Use float64 coords to avoid float32 boundary precision issues
        coords_60 = np.array([
            [60.0, 10.5],
            [60.0, 11.5],  # 1.0° lon apart, well within 2° block
        ], dtype=np.float64)

        # At equator, 111 km ≈ 1° lon, so points 1.5° apart are in
        # different blocks
        coords_eq = np.array([
            [0.0, 10.0],
            [0.0, 11.5],
        ], dtype=np.float64)

        # 111 km blocks: at 60°N, 111 km ≈ 2° lon, so both points
        # should be in the same block (1.0° < 2°)
        splitter_60 = SpatialBlockSplitter(block_km=111.0, n_splits=2, seed=0)
        folds_60 = splitter_60.split(coords_60)

        # At equator, 111 km ≈ 1° lon, so points are in different blocks
        # (1.5° > 1°)
        splitter_eq = SpatialBlockSplitter(block_km=111.0, n_splits=2, seed=0)
        folds_eq = splitter_eq.split(coords_eq)

        # At 60°N: same block -> both in same fold's test set
        for _, (train_idx, test_idx) in enumerate(folds_60):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            pair = {0, 1}
            assert not (pair & train_set and pair & test_set), (
                "At 60°N, 1.0° lon < 2° block width, should be same block"
            )

        # At equator: different blocks -> can be split
        n_blocks_eq = len(np.unique([
            int(np.floor(c[1] / 1.0))
            for c in coords_eq
        ]))
        assert n_blocks_eq == 2, "At equator, 1.5° lon > 1° block width"

    def test_block_km_negative_raises(self):
        """Negative block_km raises ValueError."""
        with pytest.raises(ValueError, match="block_km"):
            SpatialBlockSplitter(block_km=-50.0)
        with pytest.raises(ValueError, match="block_km"):
            SpatialBlockSplitter(block_km=(100.0, -50.0))


class TestBlockIds:
    """Tests for the block_ids mode."""

    def test_block_ids_basic(self):
        """Pre-assigned block IDs produce correct folds."""
        # 10 plots in 3 blocks
        block_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        splitter = SpatialBlockSplitter(block_ids=block_ids, n_splits=3, seed=0)
        folds = splitter.split()

        assert len(folds) == 3
        all_test = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test)) == 10

        # Plots within same block stay together
        for _, (train_idx, test_idx) in enumerate(folds):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            for group in [{0, 1, 2}, {3, 4, 5}, {6, 7, 8, 9}]:
                assert not (group & train_set and group & test_set)

    def test_block_ids_no_coords(self):
        """block_ids mode works without coords."""
        block_ids = np.array([0, 0, 1, 1, 2, 2])
        splitter = SpatialBlockSplitter(block_ids=block_ids, n_splits=3, seed=0)
        # Should not raise, coords not needed
        folds = splitter.split()
        assert len(folds) == 3

    def test_block_ids_single_block(self):
        """Single block ID: one fold gets all plots."""
        block_ids = np.array([5, 5, 5, 5])
        splitter = SpatialBlockSplitter(block_ids=block_ids, n_splits=3, seed=0)
        folds = splitter.split()

        non_empty = [k for k, (_, test) in enumerate(folds) if len(test) > 0]
        assert len(non_empty) == 1
        assert len(folds[non_empty[0]][1]) == 4

    def test_block_ids_wrong_ndim(self):
        """2-D block_ids raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            SpatialBlockSplitter(block_ids=np.ones((5, 2)))

    def test_block_ids_list_input(self):
        """Plain list is accepted and converted to array."""
        splitter = SpatialBlockSplitter(
            block_ids=[0, 0, 1, 1, 2, 2], n_splits=3, seed=0,
        )
        folds = splitter.split()
        assert len(folds) == 3


class TestMutualExclusivity:
    """Tests for mutual exclusivity of block modes."""

    def test_two_modes_raises(self):
        """Specifying >1 block mode raises ValueError."""
        with pytest.raises(ValueError, match="Only one"):
            SpatialBlockSplitter(block_deg=1.0, block_km=100.0)

        with pytest.raises(ValueError, match="Only one"):
            SpatialBlockSplitter(
                block_deg=1.0, block_ids=np.array([0, 1, 2]),
            )

    def test_block_size_combined_with_new_raises(self):
        """Mixing deprecated block_size with new params raises."""
        with pytest.raises(ValueError, match="Only one"):
            SpatialBlockSplitter(block_size=1.0, block_deg=0.5)

        with pytest.raises(ValueError, match="Only one"):
            SpatialBlockSplitter(block_size=1.0, block_km=100.0)


class TestDeprecation:
    """Tests for backward compatibility with block_size."""

    def test_block_size_emits_warning(self):
        """block_size emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SpatialBlockSplitter(block_size=1.0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "block_size" in str(w[0].message)

    def test_block_size_matches_block_deg(self):
        """block_size=X produces identical results to block_deg=X."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 10, size=(100, 2)).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            folds_old = SpatialBlockSplitter(
                block_size=0.5, n_splits=5, seed=42,
            ).split(coords)

        folds_new = SpatialBlockSplitter(
            block_deg=0.5, n_splits=5, seed=42,
        ).split(coords)

        for k in range(5):
            np.testing.assert_array_equal(folds_old[k][0], folds_new[k][0])
            np.testing.assert_array_equal(folds_old[k][1], folds_new[k][1])


class TestDefaults:
    """Tests for default behavior."""

    def test_default_no_args(self):
        """SpatialBlockSplitter() defaults to block_deg=0.1, no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            splitter = SpatialBlockSplitter()
            # No DeprecationWarning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0

        assert splitter._mode == "block_deg"
        assert splitter._lat_size == 0.1
        assert splitter._lon_size == 0.1

    def test_default_produces_valid_folds(self):
        """Default splitter produces valid folds."""
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 5, size=(50, 2)).astype(np.float32)

        folds = SpatialBlockSplitter().split(coords)
        assert len(folds) == 10
        all_test = np.concatenate([test for _, test in folds])
        assert len(np.unique(all_test)) == 50

    def test_grid_mode_requires_coords(self):
        """Grid modes raise ValueError if coords is None."""
        splitter = SpatialBlockSplitter(block_deg=1.0)
        with pytest.raises(ValueError, match="coords is required"):
            splitter.split()
