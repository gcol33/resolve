"""Shared fixtures for the ``resolve_core`` (C++ engine) test suite.

Every module under ``tests/core`` exercises the compiled nanobind extension.
When it is not installed the whole directory is skipped from collection rather
than erroring, so ``pytest`` at the repo root still works in an environment
that only has the pure-Python tooling.

The synthetic-data builders here write real CSV files because ``from_csv`` is
the loader every binding, the CLI, and the R package share; building from disk
keeps these tests on the same path production uses. ``from_pandas`` parity is
asserted separately in ``test_dataset.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

try:
    import resolve_core as rc
except ImportError:  # pragma: no cover - exercised only on installs without the engine
    rc = None
    collect_ignore_glob = ["test_*.py"]


# ---------------------------------------------------------------------------
# CSV builders
# ---------------------------------------------------------------------------

def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    lines = [",".join(header)]
    lines.extend(",".join(str(cell) for cell in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class PlotCsvs:
    """Paths to a header/species CSV pair plus the truth used to build them."""

    header: str
    species: str
    n_plots: int
    n_species: int
    species_value: dict[str, float]


def make_plot_csvs(
    tmp_path: Path,
    n_plots: int = 120,
    n_species: int = 12,
    species_per_plot: int = 3,
    prefix: str = "plots",
) -> PlotCsvs:
    """Header + species CSVs whose targets are a known function of composition.

    ``y`` is the sum of per-species contributions over a plot's species set and
    ``hab`` bins that sum into three classes, so a model that actually learns
    the composition recovers both. Species assignment is deterministic (the
    same construction as ``src/core/tests/test_recovery.cpp``) so every species
    is seen often enough for its embedding row to train.
    """
    values = {f"sp{s}": math.sin(s * 1.7) * 2.0 for s in range(n_species)}
    lo, hi = -1.0, 1.0

    header_rows = []
    species_rows = []
    for i in range(n_plots):
        picks = [
            (i + k * (i // n_species + 1)) % n_species for k in range(species_per_plot)
        ]
        y = sum(values[f"sp{s}"] for s in picks)
        hab = 0 if y < lo else (1 if y < hi else 2)
        header_rows.append(
            [
                f"P{i}",
                f"{45.0 + (i % 10) * 0.75:.5f}",
                f"{5.0 + (i // 10) * 0.75:.5f}",
                f"{y:.6f}",
                hab,
                10.0 + i,
            ]
        )
        for rank, s in enumerate(picks):
            species_rows.append(
                [f"P{i}", f"sp{s}", f"{1.0 + rank:.3f}", f"g{s % 4}", f"f{s % 2}"]
            )

    header = write_csv(
        tmp_path / f"{prefix}_header.csv",
        ["plot_id", "lat", "lon", "y", "hab", "elev"],
        header_rows,
    )
    species = write_csv(
        tmp_path / f"{prefix}_species.csv",
        ["plot_id", "sp", "cover", "genus", "family"],
        species_rows,
    )
    return PlotCsvs(str(header), str(species), n_plots, n_species, values)


def make_categorical_csvs(tmp_path: Path) -> tuple[str, str]:
    """Header with a letter-coded classification target and two string covariates.

    Mirrors the shape the paper data has: an EUNIS-style letter target, a Y/N
    covariate carrying blanks and ``NA``, and an ordinary numeric covariate.
    """
    letters = ["M", "N", "P", "Q", "R", "S", "T", "U", "V"]
    header_rows = []
    species_rows = []
    for i in range(54):
        resurvey = "Y" if i % 3 == 0 else "N"
        if i == 1:
            resurvey = ""
        elif i == 7:
            resurvey = "NA"
        header_rows.append(
            [
                f"P{i}",
                letters[i % len(letters)],
                resurvey,
                ["sand", "clay", "silt"][i % 3],
                float(i),
                f"{i * 0.5:.3f}",
            ]
        )
        for j in range(3):
            species_rows.append([f"P{i}", f"sp{(i + j) % 7}", f"{1.0 + j:.2f}"])

    header = write_csv(
        tmp_path / "cat_header.csv",
        ["plot_id", "eunis", "resurvey", "soil", "altitude", "y"],
        header_rows,
    )
    species = write_csv(
        tmp_path / "cat_species.csv", ["plot_id", "sp", "cover"], species_rows
    )
    return str(header), str(species)


# ---------------------------------------------------------------------------
# Role / target / config builders
# ---------------------------------------------------------------------------

def make_roles(
    *,
    taxonomy: bool = True,
    coordinates: bool = True,
    covariates: bool = True,
    categoricals: list[str] | None = None,
) -> "rc.RoleMapping":
    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    if taxonomy:
        roles.genus = "genus"
        roles.family = "family"
    if coordinates:
        roles.latitude = "lat"
        roles.longitude = "lon"
    if covariates:
        roles.covariates = ["elev"]
    if categoricals:
        roles.categoricals = list(categoricals)
    return roles


def make_targets() -> list["rc.TargetSpec"]:
    return [
        rc.TargetSpec.regression("y"),
        rc.TargetSpec.classification("hab", 3),
    ]


def make_dataset_config(
    encoding: "rc.SpeciesEncodingMode | None" = None,
    *,
    hash_dim: int = 16,
    pool_weighting: "rc.PoolWeighting | None" = None,
    use_taxonomy: bool = True,
) -> "rc.DatasetConfig":
    cfg = rc.DatasetConfig()
    cfg.species_encoding = encoding or rc.SpeciesEncodingMode.Hash
    cfg.hash_dim = hash_dim
    cfg.use_taxonomy = use_taxonomy
    if pool_weighting is not None:
        cfg.pool_weighting = pool_weighting
    return cfg


def make_model_config(
    encoding: "rc.SpeciesEncodingMode | None" = None,
    *,
    hash_dim: int = 16,
    hidden_dims: list[int] | None = None,
) -> "rc.ModelConfig":
    cfg = rc.ModelConfig()
    cfg.species_encoding = encoding or rc.SpeciesEncodingMode.Hash
    cfg.hash_dim = hash_dim
    cfg.hidden_dims = hidden_dims or [32, 16]
    cfg.species_embed_dim = 16
    return cfg


def make_train_config(
    *, max_epochs: int = 5, batch_size: int = 64, lr: float = 1e-2
) -> "rc.TrainConfig":
    cfg = rc.TrainConfig()
    cfg.batch_size = batch_size
    cfg.max_epochs = max_epochs
    cfg.patience = max_epochs
    cfg.lr = lr
    cfg.device = "cpu"
    return cfg


def trainer_continuous(dataset: "rc.ResolveDataset", model_config: "rc.ModelConfig"):
    """Assemble ``continuous`` the way ``Trainer::prepare_data`` does.

    The pretrainers take the encoder's continuous block directly rather than a
    dataset, so a caller has to concatenate coordinates, covariates, the
    unknown-mass features, and (in hash mode) the hash embedding in that order.
    """
    import torch

    parts = []
    for tensor in (dataset.coordinates, dataset.covariates):
        if tensor is not None and tensor.numel() > 0:
            parts.append(tensor)
    for tensor in (dataset.unknown_fraction, dataset.unknown_count):
        if tensor is not None and tensor.numel() > 0:
            parts.append(tensor.reshape(-1, 1).to(torch.float32))
    if model_config.species_encoding == rc.SpeciesEncodingMode.Hash:
        hashed = dataset.hash_embedding
        if hashed is not None and hashed.numel() > 0:
            parts.append(hashed)
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plot_csvs(tmp_path: Path) -> PlotCsvs:
    return make_plot_csvs(tmp_path)


@pytest.fixture
def hash_dataset(plot_csvs: PlotCsvs) -> "rc.ResolveDataset":
    return rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(rc.SpeciesEncodingMode.Hash),
    )


@pytest.fixture
def pool_dataset(plot_csvs: PlotCsvs) -> "rc.ResolveDataset":
    return rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(
            rc.SpeciesEncodingMode.RankPool, pool_weighting=rc.PoolWeighting.Log1p
        ),
    )


@dataclass(frozen=True)
class Fitted:
    """A trainer that has run one short fit, plus the result it returned."""

    trainer: "rc.Trainer"
    result: "rc.TrainResult"
    dataset: "rc.ResolveDataset"
    test_size: float


@pytest.fixture
def fitted_trainer(hash_dataset: "rc.ResolveDataset") -> Fitted:
    model = rc.ResolveModel(hash_dataset.schema, make_model_config())
    trainer = rc.Trainer(model, make_train_config(max_epochs=5))
    trainer.prepare_data(hash_dataset, 0.25, 42)
    result = trainer.fit()
    return Fitted(trainer, result, hash_dataset, 0.25)


@pytest.fixture
def categorical_csvs(tmp_path: Path) -> tuple[str, str]:
    return make_categorical_csvs(tmp_path)
