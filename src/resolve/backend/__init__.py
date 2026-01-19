"""Backend selection for RESOLVE.

Automatically uses C++ backend if resolve_core is available,
otherwise falls back to pure Python/PyTorch implementation.
"""

import warnings

# Try to import C++ backend
_BACKEND = "python"
_cpp_available = False

try:
    import resolve_core
    _cpp_available = True
    _BACKEND = "cpp"
except ImportError:
    pass


def get_backend() -> str:
    """Return the current backend name ('cpp' or 'python')."""
    return _BACKEND


def use_backend(backend: str) -> None:
    """Set the backend to use.

    Args:
        backend: Either 'cpp' or 'python'

    Raises:
        ValueError: If backend is not recognized
        ImportError: If cpp backend requested but not available
    """
    global _BACKEND

    if backend not in ("cpp", "python"):
        raise ValueError(f"Unknown backend: {backend}. Use 'cpp' or 'python'.")

    if backend == "cpp" and not _cpp_available:
        raise ImportError(
            "C++ backend (resolve_core) not available. "
            "Install with: pip install resolve-core"
        )

    _BACKEND = backend


def cpp_available() -> bool:
    """Check if C++ backend is available."""
    return _cpp_available


# Re-export key classes with backend dispatch
def get_species_encoder():
    """Get SpeciesEncoder class for current backend."""
    if _BACKEND == "cpp" and _cpp_available:
        return resolve_core.SpeciesEncoder
    else:
        from ..encode.species import SpeciesEncoder
        return SpeciesEncoder


def get_metrics():
    """Get Metrics class for current backend."""
    if _BACKEND == "cpp" and _cpp_available:
        return resolve_core.Metrics
    else:
        from ..train.metrics import Metrics
        return Metrics


__all__ = [
    "get_backend",
    "use_backend",
    "cpp_available",
    "get_species_encoder",
    "get_metrics",
]
