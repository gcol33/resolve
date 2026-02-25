"""Optional domain-specific extensions for RESOLVE.

Bundled extensions (like ``wfo``) live in this package and are imported
normally.  External extensions are installed as standalone packages named
``resolve_ext_<name>`` and discovered automatically via entry points.

Import bridge
~~~~~~~~~~~~~
A custom ``MetaPathFinder`` redirects ``from resolve.ext.X import Y`` to
the ``resolve_ext_X`` package when *X* is not a bundled submodule.  This
lets users write ``from resolve.ext.gbif import fetch`` regardless of
whether *gbif* is bundled or pip-installed externally.

Management API
~~~~~~~~~~~~~~
- :func:`list_installed` -- bundled + entry-point discovered extensions
- :func:`list_available` -- fetch registry from the extensions monorepo
- :func:`install` / :func:`uninstall` -- pip wrapper for extensions
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import json
import subprocess
import sys
import types
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLED: set[str] = {"wfo"}
"""Names of extensions that ship inside ``resolve.ext`` as submodules."""

_REGISTRY_URL: str = (
    "https://raw.githubusercontent.com/gcol33/resolve-extensions"
    "/main/registry.json"
)
"""URL of the extension registry in the monorepo."""

_REPO_URL: str = "https://github.com/gcol33/resolve-extensions.git"
"""Git URL used by ``pip install git+...``."""

_ENTRY_POINT_GROUP: str = "resolve.ext"
"""Entry-point group that external extension packages register under."""

__all__ = [
    "install",
    "uninstall",
    "list_available",
    "list_installed",
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _discover_installed() -> dict[str, str]:
    """Return ``{name: package_name}`` for externally installed extensions.

    Reads the ``resolve.ext`` entry-point group.  Each entry point maps
    an extension name (e.g. ``gbif``) to its top-level package
    (e.g. ``resolve_ext_gbif``).
    """
    eps = importlib.metadata.entry_points()
    # Python 3.12+ returns a SelectableGroups; older returns dict
    if hasattr(eps, "select"):
        group = eps.select(group=_ENTRY_POINT_GROUP)
    else:
        group = eps.get(_ENTRY_POINT_GROUP, [])

    return {ep.name: ep.value for ep in group}


# ---------------------------------------------------------------------------
# MetaPathFinder — import bridge for external extensions
# ---------------------------------------------------------------------------

class _ExtFinder(importlib.abc.MetaPathFinder):
    """Redirect ``resolve.ext.<name>`` imports to ``resolve_ext_<name>``.

    Only activates for names that are *not* bundled submodules (those are
    found by the normal filesystem-based finders that precede us in
    ``sys.meta_path``).
    """

    def find_module(self, fullname: str, path=None):
        """Python 3.3 legacy hook — delegate to find_spec."""
        spec = self.find_spec(fullname, path)
        return spec.loader if spec is not None else None

    def find_spec(self, fullname: str, path, target=None):
        """Return a ModuleSpec if *fullname* is an external extension."""
        # Only handle resolve.ext.<name> (exactly 3 parts)
        parts = fullname.split(".")
        if len(parts) < 3 or parts[:2] != ["resolve", "ext"]:
            return None

        ext_name = parts[2]

        # Skip bundled extensions — let normal finders handle them
        if ext_name in _BUNDLED:
            return None

        flat_name = f"resolve_ext_{ext_name}"

        # Check if the flat package is importable
        try:
            flat_spec = importlib.util.find_spec(flat_name)
        except (ModuleNotFoundError, ValueError):
            flat_spec = None

        if flat_spec is None:
            return None

        # For the top-level ext submodule (resolve.ext.X), load resolve_ext_X
        if len(parts) == 3:
            return importlib.machinery.ModuleSpec(
                fullname,
                _ExtLoader(flat_name),
                origin=flat_spec.origin,
                is_package=flat_spec.submodule_search_locations is not None,
            )

        # For deeper imports (resolve.ext.X.sub), map to resolve_ext_X.sub
        remainder = ".".join(parts[3:])
        deep_flat = f"{flat_name}.{remainder}"
        try:
            deep_spec = importlib.util.find_spec(deep_flat)
        except (ModuleNotFoundError, ValueError):
            return None
        if deep_spec is None:
            return None

        return importlib.machinery.ModuleSpec(
            fullname,
            _ExtLoader(deep_flat),
            origin=deep_spec.origin,
            is_package=deep_spec.submodule_search_locations is not None,
        )


class _ExtLoader(importlib.abc.Loader):
    """Load a ``resolve_ext_<name>`` package and alias it into
    ``resolve.ext.<name>``."""

    def __init__(self, flat_name: str) -> None:
        self._flat_name = flat_name

    def create_module(self, spec):
        """Import the flat package and return it as the module object."""
        return importlib.import_module(self._flat_name)

    def exec_module(self, module):
        """Register the module under its ``resolve.ext.*`` alias."""
        # The module is already executed by import_module in create_module.
        # We just ensure it's registered in sys.modules under the alias.
        alias = module.__spec__.name if module.__spec__ else None
        # find the resolve.ext.X name from the call stack
        # It's already set by the import machinery, nothing to do.


# Register the finder (once)
if not any(isinstance(f, _ExtFinder) for f in sys.meta_path):
    sys.meta_path.append(_ExtFinder())


# ---------------------------------------------------------------------------
# __getattr__ — attribute-access fallback for ``resolve.ext.X``
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Allow ``resolve.ext.X`` attribute access for external extensions."""
    if name.startswith("_"):
        raise AttributeError(name)

    # Try importing as a submodule (bundled or external via _ExtFinder)
    try:
        return importlib.import_module(f"resolve.ext.{name}")
    except ImportError:
        pass

    raise AttributeError(
        f"module 'resolve.ext' has no attribute {name!r}. "
        f"If this is an external extension, install it with: "
        f"resolve ext install {name}"
    )


# ---------------------------------------------------------------------------
# Management API
# ---------------------------------------------------------------------------

def list_installed() -> list[dict[str, str]]:
    """List all installed extensions (bundled + external).

    Returns:
        List of dicts with keys ``name``, ``source`` (``"bundled"`` or
        ``"entry_point"``), and ``package`` (Python package name).
    """
    result = []

    for name in sorted(_BUNDLED):
        result.append({
            "name": name,
            "source": "bundled",
            "package": f"resolve.ext.{name}",
        })

    for name, package in sorted(_discover_installed().items()):
        if name not in _BUNDLED:
            result.append({
                "name": name,
                "source": "entry_point",
                "package": package,
            })

    return result


def list_available(timeout: float = 10.0) -> list[dict]:
    """Fetch the extension registry from the monorepo.

    Args:
        timeout: HTTP timeout in seconds.

    Returns:
        List of extension dicts from ``registry.json``.

    Raises:
        ConnectionError: If the registry cannot be fetched.
    """
    try:
        with urlopen(_REGISTRY_URL, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise ConnectionError(
            f"Could not fetch extension registry from {_REGISTRY_URL}: {exc}"
        ) from exc

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "extensions" in data:
        return data["extensions"]

    raise ValueError(f"Unexpected registry format: {type(data).__name__}")


def install(name: str) -> None:
    """Install an extension from the monorepo.

    Runs ``pip install git+<repo>@main#subdirectory=<name>``.

    Args:
        name: Extension name (e.g. ``"gbif"``).

    Raises:
        ValueError: If the extension is not in the registry.
        subprocess.CalledProcessError: If pip fails.
    """
    if name in _BUNDLED:
        print(f"'{name}' is a bundled extension, no install needed.")
        return

    # Validate against registry
    try:
        available = list_available()
        known_names = {
            ext["name"] if isinstance(ext, dict) else ext
            for ext in available
        }
        if name not in known_names:
            raise ValueError(
                f"Unknown extension {name!r}. "
                f"Available: {', '.join(sorted(known_names))}"
            )
    except ConnectionError:
        print(
            f"Warning: could not fetch registry. "
            f"Attempting install of '{name}' anyway."
        )

    pip_url = f"git+{_REPO_URL}@main#subdirectory={name}"
    print(f"Installing resolve extension '{name}'...")
    print(f"  pip install {pip_url}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pip_url],
    )
    print(f"Extension '{name}' installed successfully.")


def uninstall(name: str) -> None:
    """Uninstall an external extension.

    Args:
        name: Extension name (e.g. ``"gbif"``).

    Raises:
        ValueError: If trying to uninstall a bundled extension.
        subprocess.CalledProcessError: If pip fails.
    """
    if name in _BUNDLED:
        raise ValueError(
            f"Cannot uninstall '{name}': it is a bundled extension."
        )

    pkg_name = f"resolve-ext-{name}"
    print(f"Uninstalling {pkg_name}...")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg_name],
    )
    print(f"Extension '{name}' uninstalled.")
