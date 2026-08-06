#!/usr/bin/env python3
"""Keep every declaration of the RESOLVE version in sync with the VERSION file.

The repo-root ``VERSION`` file is the single source of truth. Every file that
spells the version out in its own syntax -- a C++ constant, two package
manifests, a citation record -- is listed in ``SITES``. ``--check`` reports
drift and exits non-zero (the CI gate); ``--set X.Y.Z`` rewrites them all.

Adding a new place that carries the version is one ``Site`` row here, not a new
code path.

Deliberately absent from ``SITES``: everything that reads the version at build
or run time and therefore cannot drift.

  src/core/CMakeLists.txt         reads this VERSION file at configure time
  resolve_core.__version__        re-exported from the compiled engine constant
  r/inst/CITATION                 reads meta$Version from r/DESCRIPTION
  resolve --version (CLI),        read resolve::VERSION, i.e. the types.hpp
  R resolve.version(),            constant this script keeps canonical
  RunMetadata.resolve_version
  pyproject.toml (repo root)      tool configuration only; declares no package

The R package version doubles as the release-tag version: resolve.install_backend()
builds its download URL from packageVersion("resolve"), so a bump here is only
usable once a matching v<VERSION> release carries the backend assets.

Usage
-----
    python tools/version.py                               print the canonical version
    python tools/version.py --check                       exit 1 on drift
    python tools/version.py --check-built                 exit 1 if the compiled
                                                          engine is a stale build
    python tools/version.py --set 0.7.4                   rewrite every site
    python tools/version.py --set 0.7.4 --date 2026-08-06 also stamp CITATION.cff
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "VERSION"
NEWS_FILE = REPO_ROOT / "NEWS.md"

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Site:
    """One file that spells the version out, and how to find it there."""

    path: str
    pattern: str  # exactly one capture group, spanning the version literal
    label: str


SITES: tuple[Site, ...] = (
    Site(
        "src/core/include/resolve/types.hpp",
        r'^inline constexpr const char\* VERSION = "([^"]+)";',
        "C++ engine constant resolve::VERSION",
    ),
    Site(
        "src/core/python/pyproject.toml",
        r'^version = "([^"]+)"',
        "resolve-core wheel (PyPI)",
    ),
    Site(
        "r/DESCRIPTION",
        r"^Version:[ \t]*(\S+)",
        "R package (also the release-tag version)",
    ),
    Site(
        "CITATION.cff",
        r"^version:[ \t]*(\S+)",
        "citation record",
    ),
)

DATE_SITE = Site(
    "CITATION.cff",
    r'^date-released:[ \t]*"?(\d{4}-\d{2}-\d{2})"?',
    "citation release date",
)


def read(path: Path) -> str:
    """Read a file preserving its line endings, so rewrites do not churn them."""
    return path.read_text(encoding="utf-8", newline="")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def locate(site: Site) -> tuple[Path, re.Match[str], str]:
    """Return the file, its single matching occurrence, and the value found."""
    path = REPO_ROOT / site.path
    if not path.is_file():
        raise SystemExit(f"error: {site.path} does not exist")
    text = read(path)
    matches = list(re.finditer(site.pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise SystemExit(
            f"error: {site.path} has {len(matches)} lines matching the version "
            f"pattern (expected exactly 1). Update the pattern in tools/version.py."
        )
    return path, matches[0], matches[0].group(1)


def splice(site: Site, value: str) -> bool:
    """Replace the version at ``site`` with ``value``. True when the file changed."""
    path, match, current = locate(site)
    if current == value:
        return False
    text = read(path)
    start, end = match.span(1)
    write(path, text[:start] + value + text[end:])
    return True


def canonical() -> str:
    if not VERSION_FILE.is_file():
        raise SystemExit("error: VERSION file is missing from the repo root")
    value = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not SEMVER.match(value):
        raise SystemExit(f"error: VERSION holds {value!r}, which is not X.Y.Z")
    return value


def news_heading(version: str, text: str) -> bool:
    return re.search(rf"^## v{re.escape(version)}\b", text, re.MULTILINE) is not None


def cmd_check(version: str) -> int:
    width = max(len(site.path) for site in SITES)
    drifted = []
    print(f"canonical  {'VERSION':<{width}}  {version}")
    for site in SITES:
        _, _, found = locate(site)
        ok = found == version
        if not ok:
            drifted.append((site, found))
        print(f"{'ok       ' if ok else 'DRIFT    '}  {site.path:<{width}}  {found}")

    if not news_heading(version, read(NEWS_FILE)):
        print(f"DRIFT      {'NEWS.md':<{width}}  no '## v{version}' section")
        drifted.append((Site("NEWS.md", "", "changelog section"), "missing"))

    if drifted:
        print()
        print(f"{len(drifted)} file(s) disagree with VERSION ({version}).")
        print("Fix with: python tools/version.py --set " + version)
        return 1
    print()
    print("All version declarations agree.")
    return 0


def cmd_check_built(version: str) -> int:
    """Assert the compiled engine reports the canonical version.

    Catches the one drift ``--check`` cannot see: sources that agree with each
    other but a binary built before the last bump.
    """
    try:
        import resolve_core
    except ImportError as exc:
        raise SystemExit(f"error: resolve_core is not importable: {exc}") from exc

    built = resolve_core.__version__
    print(f"VERSION file: {version}   built engine: {built}")
    if built != version:
        print(f"error: the built engine reports {built}, not {version}. Rebuild it.")
        return 1
    return 0


def cmd_set(new: str, date: str | None) -> int:
    if not SEMVER.match(new):
        raise SystemExit(f"error: {new!r} is not a X.Y.Z version")
    if date is not None and not ISO_DATE.match(date):
        raise SystemExit(f"error: {date!r} is not a YYYY-MM-DD date")

    changed = []
    if canonical_unchecked() != new:
        write(VERSION_FILE, new + "\n")
        changed.append("VERSION")
    for site in SITES:
        if splice(site, new):
            changed.append(site.path)
    if date is not None and splice(DATE_SITE, date):
        changed.append(f"{DATE_SITE.path} (date-released)")

    news = read(NEWS_FILE)
    if not news_heading(new, news):
        write(NEWS_FILE, insert_news_stub(news, new))
        changed.append("NEWS.md (stub section)")

    if not changed:
        print(f"Already at {new}; nothing to do.")
        return 0
    print(f"Set version to {new} in:")
    for path in changed:
        print(f"  {path}")
    print()
    print("The engine constant changed, so rebuild before the new version is")
    print("reported by resolve --version / resolve.version() / resolve_core.")
    return 0


def canonical_unchecked() -> str:
    """The VERSION file's contents, tolerating a missing or malformed file."""
    if not VERSION_FILE.is_file():
        return ""
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def insert_news_stub(news: str, version: str) -> str:
    """Open a section for ``version`` above the newest existing one."""
    stub = f"## v{version} (unreleased)\n\n"
    first = re.search(r"^## ", news, re.MULTILINE)
    if first is None:
        return news.rstrip("\n") + "\n\n" + stub
    return news[: first.start()] + stub + news[first.start() :]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync or check every declaration of the RESOLVE version.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="report drift against the VERSION file and exit 1 if any",
    )
    group.add_argument(
        "--check-built",
        action="store_true",
        help="import resolve_core and check the compiled engine reports VERSION",
    )
    group.add_argument(
        "--set",
        metavar="X.Y.Z",
        help="rewrite VERSION and every declaration to this version",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="with --set, also stamp CITATION.cff date-released",
    )
    args = parser.parse_args(argv)

    if args.set:
        return cmd_set(args.set, args.date)
    if args.date:
        raise SystemExit("error: --date is only meaningful with --set")
    if args.check:
        return cmd_check(canonical())
    if args.check_built:
        return cmd_check_built(canonical())
    print(canonical())
    return 0


if __name__ == "__main__":
    sys.exit(main())
