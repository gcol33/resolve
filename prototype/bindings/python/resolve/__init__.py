"""
RESOLVE - Relational Encoding via Structured Observation Learning with Vector Embeddings

This package provides Python bindings to the C++ core implementation.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import numpy as np

# Import C++ bindings
try:
    from resolve._core import (
        # Enums
        TaskType,
        TransformType,
        SpeciesEncodingMode,
        SelectionMode,
        NormalizationMode,
        EncodingType,
        DataSource,
        # Configs
        TargetConfig,
        RoleMapping,
        ResolveSchema,
        TrainResult,
        # Vocabularies
        SpeciesVocab,
        TaxonomyVocab,
        # Loss
        PhaseConfig,
        PhasedLoss,
        Metrics,
        # PlotEncoder
        PlotEncoder,
        PlotRecord,
        ObservationRecord,
        # Trainer
        Trainer as _CppTrainer,
        # Version
        __version__ as _cpp_version,
    )
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    _cpp_version = "N/A"

__version__ = "0.1.0"


def is_cpp_available() -> bool:
    """Check if C++ bindings are available."""
    return _CPP_AVAILABLE


def to_records(
    df: "pd.DataFrame",
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    is_obs: bool = False,
) -> List:
    """
    Convert a pandas DataFrame to PlotRecord or ObservationRecord objects.

    Args:
        df: DataFrame with a 'plot_id' column
        numeric_cols: Columns to include as numeric features
        cat_cols: Columns to include as categorical features (obs only)
        is_obs: If True, create ObservationRecord; else PlotRecord

    Returns:
        List of PlotRecord or ObservationRecord objects
    """
    records = []
    for _, row in df.iterrows():
        if is_obs:
            rec = ObservationRecord()
            rec.categorical = {c: str(row[c]) for c in (cat_cols or [])}
        else:
            rec = PlotRecord()
        rec.plot_id = row['plot_id']
        rec.numeric = {c: float(row[c]) for c in (numeric_cols or [])}
        records.append(rec)
    return records


def make_sample_data(n_plots: int = 50, n_species: int = 100, seed: int = 42):
    """
    Create synthetic data/obs data for testing and examples.

    Args:
        n_plots: Number of plots to generate
        n_species: Number of unique species
        seed: Random seed for reproducibility

    Returns:
        Tuple of (data_df, obs_df) where:
        - data_df: Plot-level data (one row per plot) with targets
        - obs_df: Observation data (many rows per plot)
    """
    import pandas as pd
    np.random.seed(seed)

    genera = ['Quercus', 'Pinus', 'Acer', 'Betula', 'Fagus', 'Abies', 'Picea', 'Larix']
    families = ['Fagaceae', 'Pinaceae', 'Sapindaceae', 'Betulaceae']

    obs_records = []
    for i in range(n_plots):
        plot_id = f'plot_{i:03d}'
        for _ in range(np.random.randint(3, 12)):
            genus = np.random.choice(genera)
            obs_records.append({
                'plot_id': plot_id,
                'species_id': f'species_{np.random.randint(0, n_species):03d}',
                'genus': genus,
                'family': families[genera.index(genus) % len(families)],
                'abundance': np.random.exponential(10)
            })

    data_df = pd.DataFrame({
        'plot_id': [f'plot_{i:03d}' for i in range(n_plots)],
        'latitude': np.random.uniform(40, 60, n_plots),
        'longitude': np.random.uniform(-10, 30, n_plots),
        'elevation': np.random.uniform(0, 2000, n_plots),
        'ph': np.random.uniform(4, 8, n_plots),
        'nitrogen': np.random.exponential(0.2, n_plots),
        'carbon': np.random.exponential(3, n_plots),
    })

    return data_df, pd.DataFrame(obs_records)


# Re-export for convenience
__all__ = [
    # Enums
    "TaskType",
    "TransformType",
    "SpeciesEncodingMode",
    "SelectionMode",
    "NormalizationMode",
    "EncodingType",
    "DataSource",
    # Configs
    "TargetConfig",
    "RoleMapping",
    "ResolveSchema",
    "TrainResult",
    # Vocabularies
    "SpeciesVocab",
    "TaxonomyVocab",
    # Loss
    "PhaseConfig",
    "PhasedLoss",
    "Metrics",
    # PlotEncoder
    "PlotEncoder",
    "PlotRecord",
    "ObservationRecord",
    # Training
    "Trainer",
    "ResolveDataset",
    # Utilities
    "to_records",
    "make_sample_data",
    "is_cpp_available",
    "__version__",
]


# ============================================================================
# Pythonic wrapper classes
# ============================================================================

class ResolveDataset:
    """
    Dataset container for ecological plot data.

    This is a thin wrapper that delegates to the C++ implementation
    when available, or falls back to pure Python.
    """

    def __init__(
        self,
        header: "pd.DataFrame",
        species: "pd.DataFrame",
        roles: Dict[str, Any],
        targets: Dict[str, Dict[str, Any]],
        species_normalization: str = "norm",
        track_unknown_fraction: bool = True,
        track_unknown_count: bool = False,
    ):
        """
        Create dataset from pandas DataFrames.

        Args:
            header: Plot-level data (one row per plot)
            species: Species occurrence data (one row per species-plot)
            roles: Column name mappings
            targets: Target configurations
            species_normalization: "raw", "norm", or "log1p"
            track_unknown_fraction: Track unknown species fraction
            track_unknown_count: Track unknown species count
        """
        self._header = header.copy()
        self._species = species.copy()
        self._roles = roles
        self._targets = targets
        self._species_normalization = species_normalization
        self._track_unknown_fraction = track_unknown_fraction
        self._track_unknown_count = track_unknown_count

    @classmethod
    def from_csv(
        cls,
        header_path: str,
        species_path: str,
        roles: Dict[str, Any],
        targets: Dict[str, Dict[str, Any]],
        **kwargs,
    ) -> "ResolveDataset":
        """Load dataset from CSV files."""
        import pandas as pd
        header = pd.read_csv(header_path, low_memory=False)
        species = pd.read_csv(species_path, low_memory=False)
        return cls(header, species, roles, targets, **kwargs)

    @property
    def n_plots(self) -> int:
        return len(self._header)

    @property
    def schema(self) -> ResolveSchema:
        """Derive schema from dataset."""
        # TODO: Implement schema derivation
        pass

    def split(self, test_size: float = 0.2, seed: int = 42):
        """Split into train and test datasets."""
        from sklearn.model_selection import train_test_split

        plot_ids = self._header[self._roles["plot_id"]].values
        train_ids, test_ids = train_test_split(
            plot_ids, test_size=test_size, random_state=seed
        )

        train_ids_set = set(train_ids)
        test_ids_set = set(test_ids)

        train_header = self._header[
            self._header[self._roles["plot_id"]].isin(train_ids_set)
        ]
        test_header = self._header[
            self._header[self._roles["plot_id"]].isin(test_ids_set)
        ]

        train_species = self._species[
            self._species[self._roles["species_plot_id"]].isin(train_ids_set)
        ]
        test_species = self._species[
            self._species[self._roles["species_plot_id"]].isin(test_ids_set)
        ]

        train_ds = ResolveDataset(
            train_header,
            train_species,
            self._roles,
            self._targets,
            self._species_normalization,
            self._track_unknown_fraction,
            self._track_unknown_count,
        )
        test_ds = ResolveDataset(
            test_header,
            test_species,
            self._roles,
            self._targets,
            self._species_normalization,
            self._track_unknown_fraction,
            self._track_unknown_count,
        )

        return train_ds, test_ds


class Trainer:
    """
    Trainer for RESOLVE models.

    Uses C++ implementation when available for performance,
    with fallback to pure Python.
    """

    def __init__(
        self,
        dataset: ResolveDataset,
        species_encoding: str = "hash",
        hash_dim: int = 32,
        top_k: int = 5,
        hidden_dims: Optional[list] = None,
        batch_size: int = 4096,
        max_epochs: int = 500,
        patience: int = 50,
        lr: float = 1e-3,
        loss_config: str = "mae",
        checkpoint_dir: Optional[str] = None,
        device: str = "auto",
        **kwargs,
    ):
        self.dataset = dataset
        self.species_encoding = species_encoding
        self.hash_dim = hash_dim
        self.top_k = top_k
        self.hidden_dims = hidden_dims or [2048, 1024, 512, 256, 128, 64]
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.loss_config = loss_config
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.kwargs = kwargs

        self._fitted = False
        self._model = None

    def fit(self) -> TrainResult:
        """Train the model."""
        if _CPP_AVAILABLE:
            # Use C++ implementation
            # TODO: Convert dataset to C++ format and call _CppTrainer
            pass

        # Fallback to pure Python implementation
        from resolve.train.trainer import Trainer as PyTrainer

        py_trainer = PyTrainer(
            self.dataset,
            species_encoding=self.species_encoding,
            hash_dim=self.hash_dim,
            top_k=self.top_k,
            hidden_dims=self.hidden_dims,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr=self.lr,
            loss_config=self.loss_config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
            **self.kwargs,
        )

        result = py_trainer.fit()
        self._model = py_trainer.model
        self._fitted = True

        return result

    def predict(
        self,
        dataset: ResolveDataset,
        confidence_threshold: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Predict on a dataset."""
        if not self._fitted:
            raise RuntimeError("Trainer must be fit before predict")

        # TODO: Use C++ implementation when available
        pass

    def save(self, path: str) -> None:
        """Save model to file."""
        if not self._fitted:
            raise RuntimeError("Trainer must be fit before save")
        # TODO: Implement
        pass

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "Trainer":
        """Load model from file."""
        # TODO: Implement
        pass
