"""RESOLVE command-line interface.

Usage::

    resolve train --species data/species.csv --header data/header.csv \\
        --plot-id plot_id --species-id species_id \\
        --target area:regression --target habitat:classification:9 \\
        --save model.pt

    resolve predict --model model.pt --species data/species.csv \\
        --header data/header.csv --plot-id plot_id --species-id species_id \\
        --output predictions.csv

    resolve serve --model model.pt --port 8000

    resolve version

    resolve ext install <name>
    resolve ext uninstall <name>
    resolve ext list [--installed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared argument builders
# ---------------------------------------------------------------------------

def _add_data_args(parser: argparse.ArgumentParser) -> None:
    """Add data-loading arguments shared by train and predict."""
    data = parser.add_argument_group("data")
    data.add_argument(
        "--species", required=True, type=Path,
        help="Path to species CSV (one row per species-plot occurrence)",
    )
    data.add_argument(
        "--header", type=Path, default=None,
        help="Path to header CSV (one row per plot). Required for multi-table layout.",
    )

    roles = parser.add_argument_group("role mapping")
    roles.add_argument("--plot-id", required=True, help="Column name for plot identifier")
    roles.add_argument("--species-id", required=True, help="Column name for species identifier")
    roles.add_argument(
        "--species-plot-id", default=None,
        help="FK column in species table linking to plot_id (defaults to --plot-id value)",
    )
    roles.add_argument("--abundance", default=None, help="Column for abundance/cover values")
    roles.add_argument("--longitude", default=None, help="Column for longitude coordinate")
    roles.add_argument("--latitude", default=None, help="Column for latitude coordinate")
    roles.add_argument("--genus", default=None, help="Column for genus name")
    roles.add_argument("--family", default=None, help="Column for family name")

    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for computation (default: cpu)",
    )


def _build_roles_dict(args: argparse.Namespace) -> dict[str, str]:
    """Build a roles dictionary from parsed CLI arguments."""
    species_plot_id = args.species_plot_id or args.plot_id
    roles: dict[str, str] = {
        "plot_id": args.plot_id,
        "species_id": args.species_id,
        "species_plot_id": species_plot_id,
    }
    if args.abundance is not None:
        roles["abundance"] = args.abundance
    if args.longitude is not None:
        roles["coords_lon"] = args.longitude
    if args.latitude is not None:
        roles["coords_lat"] = args.latitude
    if args.genus is not None:
        roles["taxonomy_genus"] = args.genus
    if args.family is not None:
        roles["taxonomy_family"] = args.family
    return roles


def _parse_target(raw: str) -> tuple[str, dict]:
    """Parse a --target string into (name, config_dict).

    Accepted formats:
        name:task              e.g. area:regression
        name:task:num_classes  e.g. habitat:classification:9
    """
    parts = raw.split(":")
    if len(parts) == 2:
        name, task = parts
        cfg: dict = {"column": name, "task": task}
    elif len(parts) == 3:
        name, task, nc = parts
        cfg = {"column": name, "task": task, "num_classes": int(nc)}
    else:
        raise argparse.ArgumentTypeError(
            f"--target must be 'name:task' or 'name:task:num_classes', got '{raw}'"
        )
    if task not in ("regression", "classification"):
        raise argparse.ArgumentTypeError(
            f"task must be 'regression' or 'classification', got '{task}'"
        )
    if task == "classification" and "num_classes" not in cfg:
        raise argparse.ArgumentTypeError(
            f"classification target '{name}' requires num_classes (name:classification:N)"
        )
    return name, cfg


def _parse_hidden_dims(raw: str) -> list[int]:
    """Parse comma-separated hidden dimensions."""
    try:
        return [int(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--hidden-dims must be comma-separated integers, got '{raw}'"
        )


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> None:
    from resolve.data.dataset import ResolveDataset
    from resolve.train.trainer import Trainer

    # --- validate target list ---
    if not args.target:
        print("Error: at least one --target is required", file=sys.stderr)
        sys.exit(1)

    targets: dict[str, dict] = {}
    for raw in args.target:
        name, cfg = _parse_target(raw)
        if name in targets:
            print(f"Error: duplicate target '{name}'", file=sys.stderr)
            sys.exit(1)
        targets[name] = cfg

    roles = _build_roles_dict(args)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)

    # --- validate file paths ---
    if not args.species.exists():
        print(f"Error: species file not found: {args.species}", file=sys.stderr)
        sys.exit(1)
    if args.header is not None and not args.header.exists():
        print(f"Error: header file not found: {args.header}", file=sys.stderr)
        sys.exit(1)
    # --- load dataset ---
    if args.header is not None:
        print(f"Loading dataset from {args.header} + {args.species} ...")
        dataset = ResolveDataset.from_csv(
            header=str(args.header),
            species=str(args.species),
            roles=roles,
            targets=targets,
        )
    else:
        print(f"Loading dataset from {args.species} (single-table mode) ...")
        dataset = ResolveDataset.from_species_csv(
            species=str(args.species),
            roles=roles,
            targets=targets,
        )

    # --- build trainer ---
    trainer = Trainer(
        dataset=dataset,
        species_encoding=args.encoding,
        hidden_dims=hidden_dims,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    # --- train or cross-validate ---
    if args.cv is not None:
        print(f"Running {args.cv}-fold cross-validation ...")
        cv_result = trainer.cross_validate(n_splits=args.cv)
        print(f"\nCV complete. Mean metrics across {args.cv} folds:")
        for target_name, metrics in cv_result.mean_metrics.items():
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {target_name}: {metrics_str}")
    else:
        print("Training ...")
        result = trainer.fit()
        print(f"\nTraining complete (best epoch: {result.best_epoch}, "
              f"time: {result.train_time:.1f}s)")
        for target_name, metrics in result.final_metrics.items():
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"  {target_name}: {metrics_str}")

    # --- save ---
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(save_path)
    print(f"Model saved to {save_path}")


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------

def _cmd_predict(args: argparse.Namespace) -> None:
    from resolve.data.dataset import ResolveDataset
    from resolve.inference.predictor import Predictor

    # --- validate files ---
    if not args.model.exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.species.exists():
        print(f"Error: species file not found: {args.species}", file=sys.stderr)
        sys.exit(1)
    if args.header is not None and not args.header.exists():
        print(f"Error: header file not found: {args.header}", file=sys.stderr)
        sys.exit(1)
    # --- load model ---
    print(f"Loading model from {args.model} ...")
    predictor = Predictor.load(str(args.model), device=args.device)

    # --- load dataset ---
    roles = _build_roles_dict(args)
    if args.header is not None:
        print(f"Loading dataset from {args.header} + {args.species} ...")
        dataset = ResolveDataset.from_csv(
            header=str(args.header),
            species=str(args.species),
            roles=roles,
            targets={},
        )
    else:
        print(f"Loading dataset from {args.species} (single-table mode) ...")
        dataset = ResolveDataset.from_species_csv(
            species=str(args.species),
            roles=roles,
            targets={},
        )

    # --- predict ---
    print("Predicting ...")
    preds = predictor.predict(dataset)

    # --- save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path)
    print(f"Predictions saved to {output_path} ({len(preds.plot_ids)} plots)")


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------

def _cmd_version(_args: argparse.Namespace) -> None:
    from resolve import __version__
    print(f"resolve {__version__}")


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------

def _cmd_serve(args: argparse.Namespace) -> None:
    from resolve.serve import serve
    print(f"Starting RESOLVE prediction server on {args.host}:{args.port}...")
    serve(str(args.model), host=args.host, port=args.port, device=args.device)


# ---------------------------------------------------------------------------
# ext commands
# ---------------------------------------------------------------------------

def _ext_install(args: argparse.Namespace) -> None:
    from resolve.ext import install
    install(args.name)


def _ext_uninstall(args: argparse.Namespace) -> None:
    from resolve.ext import uninstall
    try:
        uninstall(args.name)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _ext_list(args: argparse.Namespace) -> None:
    from resolve.ext import list_installed, list_available

    if args.installed:
        extensions = list_installed()
        if not extensions:
            print("No extensions installed.")
            return
        print("Installed extensions:")
        for ext in extensions:
            tag = f" ({ext['source']})" if ext.get("source") else ""
            print(f"  {ext['name']}{tag}")
        return

    # Default: show available + mark installed
    installed_names = {e["name"] for e in list_installed()}

    try:
        available = list_available()
    except ConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        # Still show installed
        if installed_names:
            print("\nInstalled extensions:")
            for name in sorted(installed_names):
                print(f"  {name}")
        sys.exit(1)

    if not available and not installed_names:
        print("No extensions available.")
        return

    print("Available extensions:")
    for ext in available:
        name = ext["name"] if isinstance(ext, dict) else ext
        desc = ext.get("description", "") if isinstance(ext, dict) else ""
        marker = " [installed]" if name in installed_names else ""
        line = f"  {name}{marker}"
        if desc:
            line += f" - {desc}"
        print(line)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="resolve",
        description="RESOLVE: predict sample attributes from compositional data",
    )
    sub = parser.add_subparsers(dest="command")

    # --- train subcommand ---
    train_parser = sub.add_parser(
        "train", help="Train a model on species composition data",
    )
    _add_data_args(train_parser)

    train_parser.add_argument(
        "--target", action="append", default=[],
        help="Target spec: name:task or name:task:num_classes (repeatable)",
    )
    train_parser.add_argument("--encoding", default="hash", help="Species encoding (default: hash)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    train_parser.add_argument("--batch-size", type=int, default=4096, help="Batch size (default: 4096)")
    train_parser.add_argument(
        "--hidden-dims", default="2048,1024,512,256,128,64",
        help="Comma-separated hidden layer dimensions (default: 2048,1024,512,256,128,64)",
    )
    train_parser.add_argument("--save", required=True, type=Path, help="Output checkpoint path")
    train_parser.add_argument("--cv", type=int, default=None, help="Number of CV folds (omit for single train)")
    train_parser.set_defaults(func=_cmd_train)

    # --- predict subcommand ---
    predict_parser = sub.add_parser(
        "predict", help="Generate predictions from a trained model",
    )
    _add_data_args(predict_parser)
    predict_parser.add_argument("--model", required=True, type=Path, help="Trained model checkpoint path")
    predict_parser.add_argument("--output", required=True, type=Path, help="Output CSV path for predictions")
    predict_parser.set_defaults(func=_cmd_predict)

    # --- version subcommand ---
    version_parser = sub.add_parser("version", help="Print version and exit")
    version_parser.set_defaults(func=_cmd_version)

    # --- serve subcommand ---
    serve_parser = sub.add_parser("serve", help="Start prediction REST API server")
    serve_parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    serve_parser.set_defaults(func=_cmd_serve)

    # --- ext subcommand ---
    ext_parser = sub.add_parser("ext", help="Manage extensions")
    ext_sub = ext_parser.add_subparsers(dest="ext_command")

    # ext install
    p_install = ext_sub.add_parser("install", help="Install an extension")
    p_install.add_argument("name", help="Extension name (e.g. gbif)")
    p_install.set_defaults(func=_ext_install)

    # ext uninstall
    p_uninstall = ext_sub.add_parser("uninstall", help="Uninstall an extension")
    p_uninstall.add_argument("name", help="Extension name (e.g. gbif)")
    p_uninstall.set_defaults(func=_ext_uninstall)

    # ext list
    p_list = ext_sub.add_parser("list", help="List extensions")
    p_list.add_argument(
        "--installed", action="store_true",
        help="Show only installed extensions",
    )
    p_list.set_defaults(func=_ext_list)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "ext" and not args.ext_command:
        ext_parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
