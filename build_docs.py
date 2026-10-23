#!/usr/bin/env python3
"""Build the RESOLVE documentation site.

The site is MkDocs Material (``mkdocs.yml``, ``docs/``), built with ``--strict``
so a broken link or a page missing from the nav fails the build. With ``--r``
the R package's pkgdown site is built into ``site/r/`` (``r/build_site.R``),
which is how ``.github/workflows/docs.yml`` publishes both.

Usage:
    python build_docs.py            # mkdocs build --strict into site/
    python build_docs.py --r        # also the pkgdown site into site/r/
    python build_docs.py --serve    # live preview at http://localhost:8000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SITE = REPO / "site"
R_SITE = SITE / "r"


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", action="store_true",
                        help="serve the MkDocs site with live reload instead of building it")
    parser.add_argument("--r", action="store_true",
                        help="also build the R package's pkgdown site into site/r/")
    args = parser.parse_args(argv)

    mkdocs = [sys.executable, "-m", "mkdocs"]
    if args.serve:
        run([*mkdocs, "serve", "--strict"])
        return 0

    # mkdocs empties site/ first, so the pkgdown site is built after it.
    run([*mkdocs, "build", "--strict", "--site-dir", str(SITE)])
    if args.r:
        rscript = shutil.which("Rscript")
        if rscript is None:
            print("--r needs Rscript on PATH, with pkgdown installed", file=sys.stderr)
            return 1
        run([rscript, str(REPO / "r" / "build_site.R"), str(R_SITE)])
    print(f"Site built at {SITE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
