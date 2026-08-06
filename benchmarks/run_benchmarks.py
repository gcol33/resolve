#!/usr/bin/env python3
"""RESOLVE benchmark suite: compare species encodings and encoder architectures.

Every configuration is cross-validated on the same data with the same folds, so
the numbers in one run are comparable to each other. Results are written to JSON
after every configuration, so a long run that dies part-way keeps what it had.

Usage:
    python benchmarks/run_benchmarks.py --data-size 10k --configs all
    python benchmarks/run_benchmarks.py --data-size 10k --configs encodings
    python benchmarks/run_benchmarks.py --data-size 50k --configs all --epochs 100
    python benchmarks/run_benchmarks.py --synthetic
    python benchmarks/run_benchmarks.py --configs hash_32,embed,rank_pool
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import resolve_core as rc
import torch
from resolve_core import SpatialBlockConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REAL_DATA_FILES = {
    "10k": {
        "header": "header_preprocessed_sample10000.csv",
        "species": "species_preprocessed_sample10000.csv",
    },
    "50k": {
        "header": "header_preprocessed_sample50000.csv",
        "species": "species_preprocessed_sample50000.csv",
    },
}

SYNTHETIC_PLOTS = {"10k": 10_000, "50k": 50_000}

# Column names in the real (ASAAS) sample files.
REAL_COLUMNS = {
    "plot_id": "PlotObservationID",
    "species_id": "WFO_TAXON",
    "abundance": "Cover %",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "genus": "WFO_GENUS",
    "family": "WFO_FAMILY",
    "area": "Relevé area (m²)",
}

# Candidate habitat column names, checked against the header at runtime.
EUNIS_CANDIDATES = [
    "EUNIS_ESy",
    "EUNIS",
    "eunis",
    "habitat",
    "Habitat",
    "EUNIS_habitat",
]


def research_data_dir() -> Path:
    """Directory holding the real sample files.

    Resolved on demand rather than at import, so ``--synthetic`` runs on a
    machine that has never set ``RESEARCH_DATA``.
    """
    root = os.environ.get("RESEARCH_DATA")
    if not root:
        raise FileNotFoundError(
            "RESEARCH_DATA is not set. It must point at the research data root, "
            "e.g. setx RESEARCH_DATA E:\\research. Pass --synthetic to benchmark "
            "on generated data instead."
        )
    return Path(root) / "outputs" / "resolve-2026" / "data"


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """One benchmark run, in the engine's three configuration structs.

    The fields are flat here and expanded into ``DatasetConfig`` /
    ``ModelConfig`` / ``TrainConfig`` by the builders below, so a config entry
    stays readable and the mapping onto the engine lives in one place.
    """

    name: str
    species_encoding: str = "hash"
    encoder_architecture: str = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    max_epochs: int = 50
    patience: int = 10
    batch_size: int = 4096
    lr: float = 1e-3
    n_cv_folds: int = 3
    # Encoding-specific
    hash_dim: int = 32
    pool_weighting: str = "log1p"
    species_embed_dim: int = 32
    d_model: int = 128
    n_attention_layers: int = 0
    n_heads: int = 4
    transformer_ff_dim: int = 256
    transformer_pooling: str = "attention"
    # Mixture of experts
    moe_routing: str = "none"
    n_experts: int = 4
    # AMP (disable for attention-based encoders that overflow in fp16)
    use_amp: bool = True
    # Tag for group selection
    group: str = "encodings"

    @property
    def dataset_key(self) -> tuple:
        """Identity of the ``ResolveDataset`` this config needs.

        The species encoding is chosen when the dataset is built, so configs
        differing only in model or training settings share one loaded dataset.
        """
        return (self.species_encoding, self.hash_dim, self.pool_weighting)


SPECIES_ENCODINGS = {
    "hash": rc.SpeciesEncodingMode.Hash,
    "embed": rc.SpeciesEncodingMode.Embed,
    "sparse": rc.SpeciesEncodingMode.Sparse,
    "rank_pool": rc.SpeciesEncodingMode.RankPool,
    "transformer": rc.SpeciesEncodingMode.Transformer,
}

POOL_WEIGHTINGS = {
    "binary": rc.PoolWeighting.Binary,
    "abundance": rc.PoolWeighting.Abundance,
    "log1p": rc.PoolWeighting.Log1p,
    "norm": rc.PoolWeighting.Norm,
    "rank": rc.PoolWeighting.Rank,
}

ENCODER_ARCHITECTURES = {
    "mlp": rc.EncoderArchitecture.MLP,
    "ft_transformer": rc.EncoderArchitecture.FTTransformer,
    "tabnet": rc.EncoderArchitecture.TabNet,
    "saint": rc.EncoderArchitecture.SAINT,
    "excelformer": rc.EncoderArchitecture.ExcelFormer,
    "gnn": rc.EncoderArchitecture.GNN,
}

MOE_ROUTINGS = {
    "none": rc.MoERoutingType.None_,
    "soft": rc.MoERoutingType.Soft,
    "topk": rc.MoERoutingType.TopK,
}


def build_dataset_config(cfg: BenchmarkConfig) -> rc.DatasetConfig:
    config = rc.DatasetConfig()
    config.species_encoding = SPECIES_ENCODINGS[cfg.species_encoding]
    config.hash_dim = cfg.hash_dim
    config.pool_weighting = POOL_WEIGHTINGS[cfg.pool_weighting]
    return config


def build_model_config(cfg: BenchmarkConfig) -> rc.ModelConfig:
    config = rc.ModelConfig()
    config.species_encoding = SPECIES_ENCODINGS[cfg.species_encoding]
    config.encoder_architecture = ENCODER_ARCHITECTURES[cfg.encoder_architecture]
    # DatasetConfig.hash_dim and ModelConfig.hash_dim are independent knobs and
    # prepare_data rejects a mismatch; one benchmark field feeds both.
    config.hash_dim = cfg.hash_dim
    config.hidden_dims = list(cfg.hidden_dims)
    config.species_embed_dim = cfg.species_embed_dim
    config.d_model = cfg.d_model
    config.n_heads = cfg.n_heads
    config.n_attention_layers = cfg.n_attention_layers
    config.transformer_ff_dim = cfg.transformer_ff_dim
    config.transformer_pooling = cfg.transformer_pooling
    config.moe_routing = MOE_ROUTINGS[cfg.moe_routing]
    config.n_experts = cfg.n_experts
    return config


def build_train_config(cfg: BenchmarkConfig, device: str) -> rc.TrainConfig:
    config = rc.TrainConfig()
    config.batch_size = cfg.batch_size
    config.max_epochs = cfg.max_epochs
    config.patience = cfg.patience
    config.lr = cfg.lr
    config.use_amp = cfg.use_amp and device == "cuda"
    config.device = device
    return config


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

CONFIGS: dict[str, BenchmarkConfig] = {}


def _register(*configs: BenchmarkConfig) -> None:
    for c in configs:
        CONFIGS[c.name] = c


_register(
    # --- Encoding modes ---
    BenchmarkConfig(
        name="hash_32",
        species_encoding="hash",
        hash_dim=32,
        group="encodings",
    ),
    BenchmarkConfig(
        name="hash_64",
        species_encoding="hash",
        hash_dim=64,
        group="encodings",
    ),
    BenchmarkConfig(
        name="embed",
        species_encoding="embed",
        group="encodings",
    ),
    BenchmarkConfig(
        name="sparse",
        species_encoding="sparse",
        group="encodings",
    ),
    BenchmarkConfig(
        name="rank_pool",
        species_encoding="rank_pool",
        pool_weighting="log1p",
        group="encodings",
    ),
    BenchmarkConfig(
        name="transformer_pool",
        species_encoding="transformer",
        n_attention_layers=0,
        lr=3e-4,
        use_amp=False,
        group="encodings",
    ),
    BenchmarkConfig(
        name="transformer_attn2",
        species_encoding="transformer",
        n_attention_layers=2,
        n_heads=4,
        lr=3e-4,
        use_amp=False,
        group="encodings",
    ),
    # --- Mixture of experts (hash-mode routing) ---
    BenchmarkConfig(
        name="hash_moe_soft",
        species_encoding="hash",
        moe_routing="soft",
        n_experts=4,
        group="moe",
    ),
    BenchmarkConfig(
        name="hash_moe_topk",
        species_encoding="hash",
        moe_routing="topk",
        n_experts=4,
        group="moe",
    ),
    # --- Encoder architectures above the species encoder ---
    BenchmarkConfig(
        name="ft_transformer",
        species_encoding="hash",
        encoder_architecture="ft_transformer",
        group="architectures",
    ),
    BenchmarkConfig(
        name="tabnet",
        species_encoding="hash",
        encoder_architecture="tabnet",
        group="architectures",
    ),
)

CONFIG_GROUPS = {
    "all": list(CONFIGS.keys()),
    "encodings": [n for n, c in CONFIGS.items() if c.group == "encodings"],
    "architectures": [n for n, c in CONFIGS.items() if c.group == "architectures"],
    "moe": [n for n, c in CONFIGS.items() if c.group == "moe"],
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    config_name: str
    species_encoding: str
    encoder_architecture: str
    mean_metrics: dict[str, dict[str, float]]
    std_metrics: dict[str, dict[str, float]]
    train_time_s: float
    n_folds: int
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


@dataclass
class DataSource:
    """Frames and role/target declarations, one species encoding away from a dataset.

    ``species`` is either an in-memory frame or a path to a species CSV; the
    engine's ``from_pandas`` takes both, and streams the (large) species table
    from disk when given a path. Species rows whose plot is absent from the
    header are dropped by the loader, so filtering the header is enough.
    """

    header: pd.DataFrame
    species: pd.DataFrame | str
    roles: rc.RoleMapping
    targets: list
    label: str

    @property
    def n_plots(self) -> int:
        return len(self.header)

    @property
    def has_coordinates(self) -> bool:
        return self.roles.has_coordinates()

    def build(self, dataset_config: rc.DatasetConfig) -> rc.ResolveDataset:
        return rc.ResolveDataset.from_pandas(
            self.header,
            self.species,
            self.roles,
            self.targets,
            config=dataset_config,
        )


def _roles(**columns: str | list[str] | None) -> rc.RoleMapping:
    roles = rc.RoleMapping()
    for role, column in columns.items():
        if column is not None:
            setattr(roles, role, column)
    return roles


def generate_synthetic_source(
    n_plots: int = 10_000,
    n_species: int = 500,
    seed: int = 42,
) -> DataSource:
    """Plots whose target is a known function of their species composition.

    Area is driven by a per-species contribution plus noise and habitat bins
    that contribution, so a model that reads the composition can recover both.
    A purely random target would make every encoding score the same and say
    nothing about the encodings.
    """
    rng = np.random.default_rng(seed)

    species_names = [f"Species_{i}" for i in range(n_species)]
    genera = [f"Genus_{i}" for i in range(50)]
    families = [f"Family_{i}" for i in range(20)]
    species_effect = rng.normal(0.0, 1.0, n_species)

    plot_ids: list[int] = []
    species_rows: list[dict] = []
    log_area = np.empty(n_plots)

    for pid in range(n_plots):
        n_spp = int(rng.integers(5, 51))
        spp_ids = rng.choice(n_species, size=n_spp, replace=False)
        covers = rng.uniform(0.1, 80.0, n_spp)
        weights = covers / covers.sum()

        plot_ids.append(pid)
        log_area[pid] = 3.0 + 1.5 * float(species_effect[spp_ids] @ weights) + float(
            rng.normal(0.0, 0.25)
        )

        for sid, cover in zip(spp_ids, covers):
            species_rows.append(
                {
                    "PlotObservationID": pid,
                    "WFO_TAXON": species_names[sid],
                    "Cover %": float(cover),
                    "WFO_GENUS": genera[sid % 50],
                    "WFO_FAMILY": families[sid % 20],
                }
            )

    quantiles = np.quantile(log_area, np.linspace(0, 1, 10)[1:-1])
    habitat = np.searchsorted(quantiles, log_area)

    header = pd.DataFrame(
        {
            "PlotObservationID": plot_ids,
            "Latitude": rng.uniform(35.0, 70.0, n_plots),
            "Longitude": rng.uniform(-10.0, 40.0, n_plots),
            "Releve area (m2)": np.exp(log_area),
            "EUNIS_ESy": habitat,
        }
    )

    return DataSource(
        header=header,
        species=pd.DataFrame(species_rows),
        roles=_roles(
            plot_id="PlotObservationID",
            species_id="WFO_TAXON",
            abundance="Cover %",
            latitude="Latitude",
            longitude="Longitude",
            genus="WFO_GENUS",
            family="WFO_FAMILY",
        ),
        targets=[
            rc.TargetSpec.regression("Releve area (m2)", rc.TransformType.Log1p),
            rc.TargetSpec.classification("EUNIS_ESy", 9),
        ],
        label=f"synthetic({n_plots} plots, {n_species} species)",
    )


def load_real_source(data_size: str) -> DataSource:
    """Load the real ASAAS sample of the requested size."""
    files = REAL_DATA_FILES[data_size]
    data_dir = research_data_dir()
    header_path = data_dir / files["header"]
    species_path = data_dir / files["species"]

    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")
    if not species_path.exists():
        raise FileNotFoundError(f"Species file not found: {species_path}")

    header = pd.read_csv(header_path)
    n_read = len(header)

    area_col = REAL_COLUMNS["area"]
    targets = [rc.TargetSpec.regression(area_col, rc.TransformType.Log1p)]

    eunis_col = next((c for c in EUNIS_CANDIDATES if c in header.columns), None)
    if eunis_col is not None:
        n_classes = int(header[eunis_col].dropna().nunique())
        print(f"  Detected habitat column: '{eunis_col}' ({n_classes} classes)")
        targets.append(rc.TargetSpec.classification(eunis_col, n_classes))

    required = [area_col, REAL_COLUMNS["latitude"], REAL_COLUMNS["longitude"]]
    if eunis_col is not None:
        required.append(eunis_col)
    present = [c for c in required if c in header.columns]
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise KeyError(f"Header {header_path} is missing required columns: {missing}")

    header = header.dropna(subset=present)
    print(f"  Kept {len(header)}/{n_read} plots after dropping missing targets/coords")

    return DataSource(
        header=header,
        species=str(species_path),
        roles=_roles(
            plot_id=REAL_COLUMNS["plot_id"],
            species_id=REAL_COLUMNS["species_id"],
            abundance=REAL_COLUMNS["abundance"],
            latitude=REAL_COLUMNS["latitude"],
            longitude=REAL_COLUMNS["longitude"],
            genus=REAL_COLUMNS["genus"],
            family=REAL_COLUMNS["family"],
        ),
        targets=targets,
        label=f"ASAAS {data_size} sample",
    )


class DatasetCache:
    """Datasets keyed on the settings that change how the species set is encoded.

    Loading is the expensive part of a benchmark sweep, and configurations that
    differ only in model or training settings encode identically.
    """

    def __init__(self, source: DataSource) -> None:
        self._source = source
        self._cache: dict[tuple, rc.ResolveDataset] = {}

    def get(self, cfg: BenchmarkConfig) -> rc.ResolveDataset:
        key = cfg.dataset_key
        if key not in self._cache:
            # flush: the engine writes its own progress straight to the OS
            # handle, so an unflushed Python print lands after it in the log.
            print(f"  Encoding dataset for {key[0]} (hash_dim={key[1]})...", flush=True)
            self._cache[key] = self._source.build(build_dataset_config(cfg))
        return self._cache[key]


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def run_single_benchmark(
    cfg: BenchmarkConfig,
    datasets: DatasetCache,
    device: str,
    spatial_cv: bool,
    seed: int,
) -> BenchmarkResult:
    """Run a single benchmark configuration and return its cross-validated result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {cfg.name}")
    print(f"  encoding={cfg.species_encoding}  arch={cfg.encoder_architecture}  "
          f"hash_dim={cfg.hash_dim}  moe={cfg.moe_routing}")
    print(f"  epochs={cfg.max_epochs}  patience={cfg.patience}  "
          f"batch_size={cfg.batch_size}  lr={cfg.lr}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    try:
        dataset = datasets.get(cfg)
        model = rc.ResolveModel(dataset.schema, build_model_config(cfg))
        trainer = rc.Trainer(model, build_train_config(cfg, device))
        trainer.prepare_data(dataset, 0.2, seed)

        t_start = time.perf_counter()
        if spatial_cv:
            blocks = SpatialBlockConfig()
            cv_result = trainer.cross_validate_spatial(blocks, cfg.n_cv_folds, seed)
        else:
            cv_result = trainer.cross_validate(cfg.n_cv_folds, seed)
        t_elapsed = time.perf_counter() - t_start

        return BenchmarkResult(
            config_name=cfg.name,
            species_encoding=cfg.species_encoding,
            encoder_architecture=cfg.encoder_architecture,
            mean_metrics=cv_result.mean_metrics,
            std_metrics=cv_result.std_metrics,
            train_time_s=t_elapsed,
            n_folds=cv_result.n_folds,
        )

    except Exception as e:
        print(f"  ERROR in '{cfg.name}': {e}")
        print(traceback.format_exc())
        return BenchmarkResult(
            config_name=cfg.name,
            species_encoding=cfg.species_encoding,
            encoder_architecture=cfg.encoder_architecture,
            mean_metrics={},
            std_metrics={},
            train_time_s=0.0,
            n_folds=0,
            status="error",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def _fmt_metric(mean: float, std: float) -> str:
    """Format a metric as 'mean +/- std'."""
    return f"{mean:.4f} +/- {std:.4f}"


def _safe_get(
    result: BenchmarkResult, target: str, metric: str
) -> tuple[float | None, float | None]:
    """Extract mean and std for a target/metric pair, tolerating absent keys."""
    mean = result.mean_metrics.get(target, {}).get(metric)
    std = result.std_metrics.get(target, {}).get(metric)
    return mean, std


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Collect all target/metric pairs across results
    all_targets: dict[str, set[str]] = {}
    for r in results:
        for target, metrics in r.mean_metrics.items():
            if target not in all_targets:
                all_targets[target] = set()
            all_targets[target].update(metrics.keys())

    # Headline metrics, in display order. A nine-class target also carries
    # precision_<c> / recall_<c> / f1_<c> for every class, which would make the
    # table thousands of columns wide; the JSON keeps all of them.
    headline = [
        "mae", "rmse", "r2", "smape",
        "band_10", "band_25", "band_50",
        "accuracy", "macro_f1", "weighted_f1",
    ]

    print("\n")
    print("=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    print("Headline metrics only; every per-class metric is in the JSON.")

    for target, metrics in sorted(all_targets.items()):
        sorted_metrics = [m for m in headline if m in metrics]
        if not sorted_metrics:
            sorted_metrics = sorted(metrics)
        print(f"\n--- Target: {target} ---\n")

        # Build header
        col_widths = {"Config": 20, "Encoding": 12, "Arch": 16, "Time (s)": 10}
        for m in sorted_metrics:
            col_widths[m] = 22

        header_parts = []
        for col, width in col_widths.items():
            header_parts.append(col.ljust(width))
        header_line = "  ".join(header_parts)
        print(header_line)
        print("-" * len(header_line))

        for r in results:
            if r.status != "ok":
                row = [
                    r.config_name.ljust(col_widths["Config"]),
                    r.species_encoding.ljust(col_widths["Encoding"]),
                    r.encoder_architecture.ljust(col_widths["Arch"]),
                    f"[{r.status}]".ljust(col_widths["Time (s)"]),
                ]
                for m in sorted_metrics:
                    row.append("---".ljust(col_widths[m]))
                print("  ".join(row))
                continue

            row = [
                r.config_name.ljust(col_widths["Config"]),
                r.species_encoding.ljust(col_widths["Encoding"]),
                r.encoder_architecture.ljust(col_widths["Arch"]),
                f"{r.train_time_s:.1f}".ljust(col_widths["Time (s)"]),
            ]
            for m in sorted_metrics:
                mean, std = _safe_get(r, target, m)
                if mean is not None and std is not None:
                    row.append(_fmt_metric(mean, std).ljust(col_widths[m]))
                else:
                    row.append("---".ljust(col_widths[m]))
            print("  ".join(row))

    print()


# ---------------------------------------------------------------------------
# Incremental JSON persistence
# ---------------------------------------------------------------------------


def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"results_{ts}.json"


def _save_results(
    results: list[BenchmarkResult],
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save results incrementally to JSON."""
    payload = {
        "metadata": metadata,
        "results": [r.to_dict() for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Results saved to {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RESOLVE Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-size",
        choices=["10k", "50k"],
        default="10k",
        help="Dataset size to use (default: 10k)",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help=(
            "Config group or comma-separated names. "
            "Groups: all, encodings, architectures, moe. "
            "Example: --configs hash_32,embed,rank_pool"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs for all configs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override patience for all configs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size for all configs",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: benchmarks/results_<timestamp>.json)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated data instead of the real sample files",
    )
    parser.add_argument(
        "--synthetic-plots",
        type=int,
        default=None,
        help=(
            "Number of generated plots (default: from --data-size). "
            "For smoke runs of the harness itself; a reported comparison should "
            "use the full size."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--spatial-cv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use spatial block CV (default: True if coordinates available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the split and the CV folds (default: 42)",
    )
    return parser.parse_args(argv)


def load_source(args: argparse.Namespace) -> DataSource:
    """Build the data source the whole sweep runs on."""
    if args.synthetic:
        n = args.synthetic_plots or SYNTHETIC_PLOTS[args.data_size]
        print(f"Generating synthetic dataset ({n:,} plots)...")
        return generate_synthetic_source(n_plots=n, seed=args.seed)

    print(f"Loading real data ({args.data_size})...")
    return load_real_source(args.data_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    output_path = Path(args.output) if args.output else _default_output_path()

    if args.configs in CONFIG_GROUPS:
        config_names = CONFIG_GROUPS[args.configs]
    else:
        config_names = [n.strip() for n in args.configs.split(",")]
        unknown = [n for n in config_names if n not in CONFIGS]
        if unknown:
            print(f"ERROR: Unknown config names: {unknown}")
            print(f"Available: {sorted(CONFIGS.keys())}")
            sys.exit(1)

    selected_configs = [CONFIGS[n] for n in config_names]

    for cfg in selected_configs:
        if args.epochs is not None:
            cfg.max_epochs = args.epochs
        if args.patience is not None:
            cfg.patience = args.patience
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        cfg.n_cv_folds = args.cv_folds

    source = load_source(args)
    datasets = DatasetCache(source)
    print(f"Data: {source.label} — {source.n_plots:,} plots")

    if args.spatial_cv is not None:
        spatial_cv = args.spatial_cv
    else:
        spatial_cv = source.has_coordinates

    if spatial_cv and not source.has_coordinates:
        print("WARNING: --spatial-cv requested but the data has no coordinates. "
              "Falling back to random CV.")
        spatial_cv = False

    cv_label = "spatial block" if spatial_cv else "random"
    print(f"CV strategy: {args.cv_folds}-fold {cv_label}")
    print(f"Device: {device}")
    print(f"Engine: resolve_core {rc.__version__}")
    print(f"Configs to run ({len(selected_configs)}): "
          f"{[c.name for c in selected_configs]}")

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_size": args.data_size,
        "synthetic": args.synthetic,
        "data_label": source.label,
        "n_plots": source.n_plots,
        "device": device,
        "cv_folds": args.cv_folds,
        "spatial_cv": spatial_cv,
        "seed": args.seed,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
        "resolve_core_version": rc.__version__,
    }

    results: list[BenchmarkResult] = []
    total_start = time.perf_counter()

    for i, cfg in enumerate(selected_configs, 1):
        print(f"\n[{i}/{len(selected_configs)}] ", end="")
        result = run_single_benchmark(cfg, datasets, device, spatial_cv, args.seed)
        results.append(result)

        # Save incrementally after each run
        _save_results(results, output_path, metadata)

    total_elapsed = time.perf_counter() - total_start
    metadata["total_time_s"] = total_elapsed

    _save_results(results, output_path, metadata)
    print_results_table(results)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = sum(1 for r in results if r.status == "error")
    print(f"Completed: {n_ok} ok, {n_err} errors")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print(f"Results: {output_path}")

    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
