"""
RESOLVE - Species composition-based prediction.

High-level Python API for training and inference.

Example:
    from resolve import ResolveDataset, Trainer

    dataset = ResolveDataset.from_csv(
        header="header.csv",
        species="species.csv",
        roles={"plot_id": "plot_id", "species_id": "species"},
        targets={"area": {"column": "area", "task": "regression"}},
    )

    trainer = Trainer(dataset, hash_dim=64, hidden_dims=[256, 128])
    results = trainer.fit()
"""

import os
import sys


def _find_cuda_paths() -> list[str]:
    """
    Find CUDA installation paths on the system.

    Search order:
    1. CUDA_PATH environment variable (standard NVIDIA installer)
    2. CUDA_HOME environment variable (common alternative)
    3. Common Windows installation paths for various CUDA versions
    4. PyTorch's bundled CUDA (if available)

    Returns:
        List of valid CUDA bin directories found
    """
    cuda_dirs = []

    # 1. Check standard environment variables
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        cuda_path = os.environ.get(env_var)
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.isdir(cuda_bin) and cuda_bin not in cuda_dirs:
                cuda_dirs.append(cuda_bin)

    # 2. Check common Windows installation paths
    # NVIDIA installs to Program Files by default
    common_roots = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA GPU Computing Toolkit", "CUDA"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
    ]

    # Check for versioned CUDA directories (v11.x, v12.x, v13.x)
    for root in common_roots:
        if os.path.isdir(root):
            try:
                for entry in os.listdir(root):
                    if entry.startswith("v"):
                        cuda_bin = os.path.join(root, entry, "bin")
                        if os.path.isdir(cuda_bin) and cuda_bin not in cuda_dirs:
                            cuda_dirs.append(cuda_bin)
            except OSError:
                pass

    # 3. Check if PyTorch has bundled CUDA DLLs
    # Modern PyTorch wheels include CUDA runtime libraries
    try:
        import torch
        torch_lib = os.path.dirname(torch.__file__)
        torch_lib_path = os.path.join(torch_lib, "lib")
        if os.path.isdir(torch_lib_path) and torch_lib_path not in cuda_dirs:
            cuda_dirs.append(torch_lib_path)
    except ImportError:
        pass

    return cuda_dirs


def _setup_cuda_dll_paths():
    """Add CUDA DLL directories on Windows for proper library loading."""
    if sys.platform != "win32":
        return

    # Python 3.8+ requires explicit DLL directory registration on Windows
    if not hasattr(os, "add_dll_directory"):
        return

    cuda_paths = _find_cuda_paths()
    for path in cuda_paths:
        try:
            os.add_dll_directory(path)
        except OSError:
            # Directory doesn't exist or can't be added
            pass


# Setup CUDA paths before importing torch
_setup_cuda_dll_paths()

import torch  # noqa: E402
from .dataset import ResolveDataset, RoleMapping, TargetConfig
from .trainer import Trainer
from .config import TrainerConfig

__version__ = "0.1.0"

__all__ = [
    "ResolveDataset",
    "Trainer",
    "RoleMapping",
    "TargetConfig",
    "TrainerConfig",
]
