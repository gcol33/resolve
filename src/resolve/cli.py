"""RESOLVE command-line interface.

Usage::

    resolve ext install <name>
    resolve ext uninstall <name>
    resolve ext list [--installed]
"""

from __future__ import annotations

import argparse
import sys


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="resolve",
        description="RESOLVE: Species composition prediction with neural networks",
    )
    sub = parser.add_subparsers(dest="command")

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
