"""Spatial block cross-validation splitter for RESOLVE.

Assigns plots to spatial grid blocks based on coordinates, then splits
blocks (not individual plots) into folds. This prevents spatial
autocorrelation leakage between train and test sets.
"""

from __future__ import annotations



import numpy as np

__all__ = ["SpatialBlockSplitter"]


class SpatialBlockSplitter:
    """Grid-based spatial block splitter for cross-validation.

    Assigns each plot to a grid cell based on ``floor(coord / block_deg)``,
    then distributes entire grid cells across folds.
    Plots within the same grid cell always stay together.

    Three mutually exclusive block-specification modes:

    * **block_deg**: block size in degrees (scalar for square, tuple for
      rectangular ``(lat_deg, lon_deg)``).
    * **block_km**: block size in kilometres (scalar for square, tuple for
      ``(ns_km, ew_km)``). Converted to degrees in :meth:`split` using
      mean latitude: ``deg_lat = km / 111``,
      ``deg_lon = km / (111 * cos(mean_lat))``.
    * **block_ids**: pre-computed integer block labels (one per plot).
      Skips grid computation entirely.

    Parameters
    ----------
    n_splits : int
        Number of CV folds. Default 10.
    seed : int
        Random seed for block shuffling. Default 42.
    block_deg : float or tuple[float, float] or None
        Block size in degrees.
    block_km : float or tuple[float, float] or None
        Block size in kilometres.
    block_ids : np.ndarray or None
        Pre-assigned 1-D integer block labels.
    balance : bool
        If True, use greedy bin-packing to equalise plot counts across
        folds (assign each block to the fold with fewest plots so far).
        If False (default), assign blocks round-robin after shuffling.
        Matches Verde's ``BlockKFold(balance=True)`` behaviour.
    """

    def __init__(
        self,
        n_splits: int = 10,
        seed: int = 42,
        *,
        block_deg: float | tuple[float, float] | None = None,
        block_km: float | tuple[float, float] | None = None,
        block_ids: np.ndarray | None = None,
        balance: bool = False,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        # --- resolve mutually exclusive modes --------------------------------
        specified = []
        if block_deg is not None:
            specified.append("block_deg")
        if block_km is not None:
            specified.append("block_km")
        if block_ids is not None:
            specified.append("block_ids")

        if len(specified) > 1:
            raise ValueError(
                f"Only one block mode may be specified, got: "
                f"{', '.join(specified)}"
            )

        # Default when nothing specified
        if not specified:
            block_deg = 0.1

        # --- store mode + values ---------------------------------------------
        self.n_splits = n_splits
        self.seed = seed
        self.balance = balance

        if block_ids is not None:
            block_ids = np.asarray(block_ids)
            if block_ids.ndim != 1:
                raise ValueError(
                    f"block_ids must be 1-D, got {block_ids.ndim}-D"
                )
            self._mode = "block_ids"
            self._block_ids = block_ids
            self._lat_size: float | None = None
            self._lon_size: float | None = None
        elif block_km is not None:
            self._mode = "block_km"
            self._block_ids = None
            if isinstance(block_km, (list, tuple)):
                ns_km, ew_km = float(block_km[0]), float(block_km[1])
            else:
                ns_km = ew_km = float(block_km)
            if ns_km <= 0 or ew_km <= 0:
                raise ValueError(
                    f"block_km values must be > 0, got {block_km}"
                )
            self._ns_km = ns_km
            self._ew_km = ew_km
            # Actual degree sizes computed in split() from mean latitude
            self._lat_size = None
            self._lon_size = None
        else:
            # block_deg (including default and deprecated block_size)
            self._mode = "block_deg"
            self._block_ids = None
            if isinstance(block_deg, (list, tuple)):
                lat_deg, lon_deg = float(block_deg[0]), float(block_deg[1])
            else:
                lat_deg = lon_deg = float(block_deg)
            if lat_deg <= 0 or lon_deg <= 0:
                raise ValueError(
                    f"block_deg values must be > 0, got {block_deg}"
                )
            self._lat_size = lat_deg
            self._lon_size = lon_deg


    def split(
        self, coords: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split plots into spatial CV folds.

        Parameters
        ----------
        coords : np.ndarray or None
            Array of shape ``(n_plots, 2)`` with ``(lat, lon)`` per plot.
            Required for grid modes (*block_deg*, *block_km*). Not required
            for *block_ids* mode.

        Returns
        -------
        list of (train_indices, test_indices)
            One tuple per fold. Indices are integer arrays into the
            original plot order.
        """
        # --- block_ids mode: skip grid computation ---------------------------
        if self._mode == "block_ids":
            block_labels = self._block_ids
            n = len(block_labels)
            return self._assign_folds(block_labels, n)

        # --- grid modes require coords --------------------------------------
        if coords is None:
            raise ValueError(
                "coords is required for grid-based block modes "
                "(block_deg / block_km)"
            )
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (n, 2), got {coords.shape}"
            )

        n = len(coords)

        # Compute degree sizes for block_km mode
        if self._mode == "block_km":
            mean_lat_rad = np.deg2rad(np.mean(coords[:, 0]))
            self._lat_size = self._ns_km / 111.0
            self._lon_size = self._ew_km / (111.0 * np.cos(mean_lat_rad))

        lat_size = self._lat_size
        lon_size = self._lon_size

        # Grid-hash each plot into a block ID
        block_row = np.floor(coords[:, 0] / lat_size).astype(np.int64)
        block_col = np.floor(coords[:, 1] / lon_size).astype(np.int64)

        # Cantor-style pairing to get unique block IDs
        block_labels = block_row * 1_000_000 + block_col

        return self._assign_folds(block_labels, n)

    def _assign_folds(
        self, block_labels: np.ndarray, n: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Assign plots to folds based on block labels."""
        unique_blocks, plot_block_idx = np.unique(
            block_labels, return_inverse=True,
        )
        n_blocks = len(unique_blocks)

        # Count plots per block
        block_sizes = np.bincount(plot_block_idx, minlength=n_blocks)

        # Shuffle blocks
        rng = np.random.default_rng(self.seed)
        block_order = rng.permutation(n_blocks)

        block_to_fold = np.empty(n_blocks, dtype=np.int32)

        if self.balance:
            # Greedy bin-packing: sort blocks largest-first, assign each
            # to the fold with fewest plots so far. Matches Verde's
            # BlockKFold(balance=True) behaviour.
            sorted_order = sorted(
                block_order, key=lambda b: block_sizes[b], reverse=True,
            )
            fold_totals = np.zeros(self.n_splits, dtype=np.int64)
            for block_pos in sorted_order:
                target_fold = int(np.argmin(fold_totals))
                block_to_fold[block_pos] = target_fold
                fold_totals[target_fold] += block_sizes[block_pos]
        else:
            # Round-robin assignment
            for i, block_pos in enumerate(block_order):
                block_to_fold[block_pos] = i % self.n_splits

        # Map plot-level fold assignment
        plot_folds = block_to_fold[plot_block_idx]

        # Build index arrays per fold
        all_indices = np.arange(n)
        folds = []
        for k in range(self.n_splits):
            test_mask = plot_folds == k
            test_idx = all_indices[test_mask]
            train_idx = all_indices[~test_mask]
            folds.append((train_idx, test_idx))

        return folds
