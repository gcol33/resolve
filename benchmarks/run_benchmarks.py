#!/usr/bin/env python3
"""RESOLVE Benchmark Suite — compare encoding modes and architectures.

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
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resolve import ResolveDataset, Trainer
from resolve.data.roles import RoleMapping, TargetConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("J:/Phd Local/Gilles_paper_resolve/data")

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

ROLE_MAPPING = {
    "plot_id": "PlotObservationID",
    "species_id": "WFO_TAXON",
    "species_plot_id": "PlotObservationID",
    "abundance": "Cover %",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "taxonomy_genus": "WFO_GENUS",
    "taxonomy_family": "WFO_FAMILY",
}

# Candidate EUNIS column names (checked at runtime)
EUNIS_CANDIDATES = [
    "EUNIS_ESy",
    "EUNIS",
    "eunis",
    "habitat",
    "Habitat",
    "EUNIS_habitat",
]

# Architectures that require the C++ backend
CPP_ARCHITECTURES = {"ft_transformer", "tabnet", "saint", "gnn", "excelformer"}


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Single benchmark run configuration."""

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
    n_attention_layers: int = 0
    n_heads: int = 4
    transformer_ff_dim: int = 256
    # MoE
    moe_routing: str = "none"
    n_experts: int = 4
    # AMP (disable for attention-based models that overflow in fp16)
    use_amp: bool = True
    # Tags for filtering
    group: str = "encodings"

    @property
    def requires_cpp(self) -> bool:
        return self.encoder_architecture in CPP_ARCHITECTURES


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
        name="rank_pool",
        species_encoding="rank_pool",
        group="encodings",
    ),
    BenchmarkConfig(
        name="transformer_v4",
        species_encoding="transformer",
        n_attention_layers=0,
        lr=3e-4,
        use_amp=False,
        group="encodings",
    ),
    BenchmarkConfig(
        name="transformer_v5",
        species_encoding="transformer",
        n_attention_layers=2,
        n_heads=4,
        lr=3e-4,
        use_amp=False,
        group="encodings",
    ),
    # --- MoE variants ---
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
    # --- Architecture variants (C++ backend) ---
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
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_dataset(
    n_plots: int = 10_000,
    n_species: int = 500,
    seed: int = 42,
) -> ResolveDataset:
    """Generate a synthetic dataset for benchmarking when real data is unavailable."""
    rng = np.random.default_rng(seed)

    plot_ids = np.arange(n_plots)
    latitudes = rng.uniform(35.0, 70.0, n_plots).astype(np.float64)
    longitudes = rng.uniform(-10.0, 40.0, n_plots).astype(np.float64)
    areas = rng.lognormal(mean=3.0, sigma=1.5, size=n_plots).astype(np.float64)
    habitats = rng.integers(0, 9, n_plots)

    header = pl.DataFrame({
        "PlotObservationID": plot_ids,
        "Latitude": latitudes,
        "Longitude": longitudes,
        "Releve area (m2)": areas,
        "EUNIS_ESy": habitats,
    })

    # Species occurrences: each plot gets 5-50 species
    species_rows = []
    genera = [f"Genus_{i}" for i in range(50)]
    families = [f"Family_{i}" for i in range(20)]
    species_names = [f"Species_{i}" for i in range(n_species)]

    for pid in range(n_plots):
        n_spp = rng.integers(5, 51)
        spp_ids = rng.choice(n_species, size=n_spp, replace=False)
        covers = rng.uniform(0.1, 80.0, n_spp).astype(np.float64)
        for sid, cover in zip(spp_ids, covers):
            species_rows.append({
                "PlotObservationID": pid,
                "WFO_TAXON": species_names[sid],
                "Cover %": cover,
                "WFO_GENUS": genera[sid % 50],
                "WFO_FAMILY": families[sid % 20],
            })

    species = pl.DataFrame(species_rows)

    roles = RoleMapping(
        plot_id="PlotObservationID",
        species_id="WFO_TAXON",
        species_plot_id="PlotObservationID",
        abundance="Cover %",
        coords_lat="Latitude",
        coords_lon="Longitude",
        taxonomy_genus="WFO_GENUS",
        taxonomy_family="WFO_FAMILY",
    )

    targets = {
        "area": TargetConfig(
            column="Releve area (m2)",
            task="regression",
            transform="log1p",
        ),
        "habitat": TargetConfig(
            column="EUNIS_ESy",
            task="classification",
            num_classes=9,
        ),
    }

    return ResolveDataset(header, species, roles, targets)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _detect_eunis_column(header_path: Path) -> Optional[str]:
    """Scan CSV header row for a known EUNIS column name."""
    sample = pl.read_csv(header_path, n_rows=0)
    for candidate in EUNIS_CANDIDATES:
        if candidate in sample.columns:
            return candidate
    return None


def _count_classes(header_path: Path, column: str) -> int:
    """Count distinct non-null values in a column."""
    df = pl.read_csv(header_path, columns=[column], infer_schema_length=10000)
    return df[column].drop_nulls().n_unique()


def load_real_dataset(data_size: str) -> ResolveDataset:
    """Load real ASAAS data from disk."""
    files = REAL_DATA_FILES[data_size]
    header_path = DATA_DIR / files["header"]
    species_path = DATA_DIR / files["species"]

    if not header_path.exists():
        raise FileNotFoundError(f"Header file not found: {header_path}")
    if not species_path.exists():
        raise FileNotFoundError(f"Species file not found: {species_path}")

    # Build targets dict
    targets_dict: dict[str, dict] = {
        "area": {
            "column": "Relevé area (m²)",
            "task": "regression",
            "transform": "log1p",
        },
    }

    # Detect EUNIS column
    eunis_col = _detect_eunis_column(header_path)
    if eunis_col is not None:
        n_classes = _count_classes(header_path, eunis_col)
        print(f"  Detected EUNIS column: '{eunis_col}' ({n_classes} classes)")
        targets_dict["habitat"] = {
            "column": eunis_col,
            "task": "classification",
            "num_classes": n_classes,
        }

    # Filter to plots with non-null targets and coordinates
    print("  Filtering plots with missing targets/coordinates...")
    header = pl.read_csv(header_path)
    plot_id_col = ROLE_MAPPING["plot_id"]

    # Keep plots where all target columns and coords are non-null
    filter_cols = ["Relevé area (m²)", "Latitude", "Longitude"]
    if eunis_col is not None:
        filter_cols.append(eunis_col)
    mask = pl.lit(True)
    for col in filter_cols:
        if col in header.columns:
            mask = mask & header[col].is_not_null()
    header_clean = header.filter(mask)
    valid_ids = set(header_clean[plot_id_col].to_list())

    # Write filtered CSVs to temp files
    import tempfile
    species = pl.read_csv(species_path)
    species_clean = species.filter(pl.col(plot_id_col).is_in(list(valid_ids)))

    tmpdir = Path(tempfile.mkdtemp(prefix="resolve_bench_"))
    h_path = tmpdir / "header.csv"
    s_path = tmpdir / "species.csv"
    header_clean.write_csv(h_path)
    species_clean.write_csv(s_path)
    print(f"  Kept {header_clean.shape[0]}/{header.shape[0]} plots after null filtering")

    dataset = ResolveDataset.from_csv(
        header=str(h_path),
        species=str(s_path),
        roles=ROLE_MAPPING,
        targets=targets_dict,
    )
    return dataset


def load_dataset(data_size: str, synthetic: bool) -> ResolveDataset:
    """Load real data or fall back to synthetic."""
    if synthetic:
        n = 10_000 if data_size == "10k" else 50_000
        print(f"Generating synthetic dataset ({n:,} plots)...")
        return _generate_synthetic_dataset(n_plots=n)

    try:
        print(f"Loading real data ({data_size})...")
        return load_real_dataset(data_size)
    except FileNotFoundError as e:
        print(f"WARNING: Real data not found ({e}). Falling back to synthetic data.")
        n = 10_000 if data_size == "10k" else 50_000
        return _generate_synthetic_dataset(n_plots=n)


# ---------------------------------------------------------------------------
# C++ backend detection
# ---------------------------------------------------------------------------


def _cpp_backend_available() -> bool:
    """Check whether the C++ backend (_resolve_core) is importable."""
    try:
        import _resolve_core  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------


def run_single_benchmark(
    cfg: BenchmarkConfig,
    dataset: ResolveDataset,
    device: str,
    spatial_cv: bool,
) -> BenchmarkResult:
    """Run a single benchmark configuration and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running: {cfg.name}")
    print(f"  encoding={cfg.species_encoding}  arch={cfg.encoder_architecture}  "
          f"hash_dim={cfg.hash_dim}  moe={cfg.moe_routing}")
    print(f"  epochs={cfg.max_epochs}  patience={cfg.patience}  "
          f"batch_size={cfg.batch_size}  lr={cfg.lr}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    # Skip C++ architectures if backend unavailable
    if cfg.requires_cpp and not _cpp_backend_available():
        msg = (
            f"Skipping '{cfg.name}': encoder_architecture='{cfg.encoder_architecture}' "
            f"requires C++ backend (_resolve_core), which is not installed."
        )
        print(f"  {msg}")
        return BenchmarkResult(
            config_name=cfg.name,
            species_encoding=cfg.species_encoding,
            encoder_architecture=cfg.encoder_architecture,
            mean_metrics={},
            std_metrics={},
            train_time_s=0.0,
            n_folds=0,
            status="skipped",
            error=msg,
        )

    try:
        trainer = Trainer(
            dataset=dataset,
            species_encoding=cfg.species_encoding,
            encoder_architecture=cfg.encoder_architecture,
            hash_dim=cfg.hash_dim,
            hidden_dims=cfg.hidden_dims,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            n_attention_layers=cfg.n_attention_layers,
            n_heads=cfg.n_heads,
            transformer_ff_dim=cfg.transformer_ff_dim,
            moe_routing=cfg.moe_routing,
            n_experts=cfg.n_experts,
            use_amp=cfg.use_amp,
            device=device,
            verbose=1,
        )

        t_start = time.perf_counter()
        cv_result = trainer.cross_validate(
            n_splits=cfg.n_cv_folds,
            spatial=spatial_cv,
        )
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
        tb = traceback.format_exc()
        print(f"  ERROR in '{cfg.name}': {e}")
        print(tb)
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
) -> tuple[Optional[float], Optional[float]]:
    """Safely extract mean and std for a target/metric pair."""
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

    # Preferred metric display order
    metric_order = ["mae", "rmse", "band_25", "accuracy", "smape", "r2"]

    def _sort_key(m: str) -> int:
        try:
            return metric_order.index(m)
        except ValueError:
            return len(metric_order)

    print("\n")
    print("=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)

    for target, metrics in sorted(all_targets.items()):
        sorted_metrics = sorted(metrics, key=_sort_key)
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
    print(f"Results saved to {output_path}")


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
        help="Use synthetic data instead of real data",
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
        help="Use spatial CV (default: True if coordinates available)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Resolve output path
    output_path = Path(args.output) if args.output else _default_output_path()

    # Select configs
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

    # Apply CLI overrides
    for cfg in selected_configs:
        if args.epochs is not None:
            cfg.max_epochs = args.epochs
        if args.patience is not None:
            cfg.patience = args.patience
        cfg.n_cv_folds = args.cv_folds

    # Load dataset once
    dataset = load_dataset(args.data_size, args.synthetic)
    print(f"Dataset: {dataset.n_plots:,} plots")

    # Determine spatial CV
    has_coords = dataset.get_coordinates() is not None
    if args.spatial_cv is not None:
        spatial_cv = args.spatial_cv
    else:
        spatial_cv = has_coords

    if spatial_cv and not has_coords:
        print("WARNING: --spatial-cv requested but dataset has no coordinates. "
              "Falling back to random CV.")
        spatial_cv = False

    cv_label = "spatial block" if spatial_cv else "random"
    print(f"CV strategy: {args.cv_folds}-fold {cv_label}")
    print(f"Device: {device}")
    print(f"Configs to run ({len(selected_configs)}): "
          f"{[c.name for c in selected_configs]}")

    # Metadata for JSON output
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_size": args.data_size,
        "synthetic": args.synthetic,
        "n_plots": dataset.n_plots,
        "device": device,
        "cv_folds": args.cv_folds,
        "spatial_cv": spatial_cv,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
    }

    # Run benchmarks
    results: list[BenchmarkResult] = []
    total_start = time.perf_counter()

    for i, cfg in enumerate(selected_configs, 1):
        print(f"\n[{i}/{len(selected_configs)}] ", end="")
        result = run_single_benchmark(cfg, dataset, device, spatial_cv)
        results.append(result)

        # Save incrementally after each run
        _save_results(results, output_path, metadata)

    total_elapsed = time.perf_counter() - total_start
    metadata["total_time_s"] = total_elapsed

    # Final save and display
    _save_results(results, output_path, metadata)
    print_results_table(results)

    # Summary
    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_err = sum(1 for r in results if r.status == "error")
    print(f"Completed: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
