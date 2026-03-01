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

    Assigns each plot to a grid cell based on ``floor(coord / block_size)``,
    then distributes entire grid cells across folds via round-robin.
    Plots within the same grid cell always stay together.

    Parameters
    ----------
    block_size : float
        Grid cell size in coordinate units (degrees for lat/lon). Default 0.1.
    n_splits : int
        Number of CV folds. Default 10.
    seed : int
        Random seed for block shuffling. Default 42.
    """

    def __init__(
        self,
        block_size: float = 0.1,
        n_splits: int = 10,
        seed: int = 42,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.block_size = block_size
        self.n_splits = n_splits
        self.seed = seed

    def split(
        self, coords: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split plots into spatial CV folds.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape ``(n_plots, 2)`` with ``(lat, lon)`` per plot.

        Returns
        -------
        list of (train_indices, test_indices)
            One tuple per fold. Indices are integer arrays into the
            original plot order.
        """
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"coords must have shape (n, 2), got {coords.shape}"
            )

        n = len(coords)

        # Grid-hash each plot into a block ID string
        block_row = np.floor(coords[:, 0] / self.block_size).astype(np.int64)
        block_col = np.floor(coords[:, 1] / self.block_size).astype(np.int64)

        # Cantor-style pairing to get unique block IDs
        # Shift to non-negative first (coords can be negative)
        block_ids = block_row * 1_000_000 + block_col

        # Map each plot to its block
        unique_blocks, plot_block_idx = np.unique(block_ids, return_inverse=True)
        n_blocks = len(unique_blocks)

        # Shuffle blocks
        rng = np.random.default_rng(self.seed)
        block_order = rng.permutation(n_blocks)

        # Assign blocks to folds round-robin
        block_to_fold = np.empty(n_blocks, dtype=np.int32)
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
