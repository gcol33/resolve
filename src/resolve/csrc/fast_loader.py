"""
Fast CSV loader using C++ with memory-mapped files.

Compiles C++ code on first import using torch.utils.cpp_extension.
~10x faster than pandas, memory-efficient (no OOM on large files).
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Lazy-load the C++ extension
_fast_csv = None
_msvc_env_initialized = False


def _find_vs_install_via_vswhere():
    """Return the latest VS install path with VC.Tools workload, or None."""
    import subprocess

    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.exists(vswhere):
        return None
    try:
        result = subprocess.run(
            [
                vswhere,
                "-latest",
                "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        path = result.stdout.strip()
        return path or None
    except Exception:
        return None


def _initialize_msvc_environment():
    """Source vcvars64.bat into os.environ so torch's `where cl` and ninja work.

    Idempotent: only runs once per process. Skips if INCLUDE+LIB already set
    (caller is in a Developer Command Prompt or has set them manually).
    """
    global _msvc_env_initialized
    if _msvc_env_initialized:
        return
    if os.environ.get("INCLUDE") and os.environ.get("LIB"):
        _msvc_env_initialized = True
        return

    import platform
    if platform.system() != "Windows":
        _msvc_env_initialized = True
        return

    import subprocess

    # Try vswhere first (most reliable across editions, including pre-release)
    install_path = _find_vs_install_via_vswhere()

    # Fallback: scan known install locations
    if not install_path:
        candidates = [
            r"C:\Program Files\Microsoft Visual Studio\18\Community",
            r"C:\Program Files\Microsoft Visual Studio\18\Professional",
            r"C:\Program Files\Microsoft Visual Studio\18\Enterprise",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "VC", "Auxiliary", "Build", "vcvars64.bat")):
                install_path = c
                break

    if not install_path:
        return  # No MSVC found; let torch produce its own error

    vcvars = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.exists(vcvars):
        return

    # Run vcvars64 in a subshell, then `set` to dump the resulting environment.
    try:
        result = subprocess.run(
            f'"{vcvars}" >NUL 2>&1 && set',
            shell=True, capture_output=True, timeout=30, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return

    # Decode with cp1252 (Windows default) — output may contain non-UTF8 chars
    raw = result.stdout.decode("cp1252", errors="replace")
    for line in raw.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            # Only update keys we actually need; preserve existing PATH semantics
            if key.upper() in {"PATH", "INCLUDE", "LIB", "LIBPATH", "WINDOWSSDKDIR",
                                "VCTOOLSINSTALLDIR", "VSINSTALLDIR", "WINDOWSSDKVERSION"}:
                os.environ[key] = value

    _msvc_env_initialized = True


def _find_msvc_paths():
    """Find MSVC and Windows SDK include and library paths on Windows.

    Returns (include_paths, library_paths). Used as a fallback when the
    process has not initialized the MSVC environment via vcvars64; the
    paths are passed to torch's load() as compile flags.
    """
    import glob

    include_paths = []
    library_paths = []

    install_path = _find_vs_install_via_vswhere()
    msvc_root = None
    if install_path:
        msvc_dirs = glob.glob(os.path.join(install_path, "VC", "Tools", "MSVC", "*"))
        if msvc_dirs:
            msvc_root = sorted(msvc_dirs)[-1]

    if msvc_root is None:
        # Fallback to scanning known VS install paths
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio\18\Community",
            r"C:\Program Files\Microsoft Visual Studio\18\Professional",
            r"C:\Program Files\Microsoft Visual Studio\18\Enterprise",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
        ]
        for vs_path in vs_paths:
            msvc_dirs = glob.glob(os.path.join(vs_path, "VC", "Tools", "MSVC", "*"))
            if msvc_dirs:
                msvc_root = sorted(msvc_dirs)[-1]
                break

    if msvc_root:
        include_paths.append(os.path.join(msvc_root, "include"))
        library_paths.append(os.path.join(msvc_root, "lib", "x64"))

    # Find Windows SDK
    sdk_paths = [
        r"C:\Program Files (x86)\Windows Kits\10",
        r"C:\Program Files\Windows Kits\10",
    ]

    for sdk_base in sdk_paths:
        sdk_include = os.path.join(sdk_base, "Include")
        sdk_lib = os.path.join(sdk_base, "Lib")

        if os.path.exists(sdk_include):
            versions = [d for d in os.listdir(sdk_include) if d.startswith("10.")]
            if versions:
                latest = sorted(versions)[-1]
                for subdir in ["ucrt", "shared", "um", "winrt"]:
                    subpath = os.path.join(sdk_include, latest, subdir)
                    if os.path.exists(subpath):
                        include_paths.append(subpath)
                for subdir in ["ucrt", "um"]:
                    libpath = os.path.join(sdk_lib, latest, subdir, "x64")
                    if os.path.exists(libpath):
                        library_paths.append(libpath)
                break

    return include_paths, library_paths


def _get_extension():
    """Load or compile the C++ extension."""
    global _fast_csv
    if _fast_csv is not None:
        return _fast_csv

    from torch.utils.cpp_extension import load
    import platform
    import hashlib

    # Find the C++ source file
    csrc_dir = Path(__file__).parent
    cpp_file = csrc_dir / "fast_csv.cpp"

    if not cpp_file.exists():
        raise RuntimeError(f"C++ source not found: {cpp_file}")

    # Generate unique module name based on source hash to avoid stale cache
    source_hash = hashlib.md5(cpp_file.read_bytes()).hexdigest()[:8]
    module_name = f"fast_csv_{source_hash}"

    print(f"Compiling fast CSV loader ({module_name})...", end=" ", flush=True)

    # Use appropriate compiler flags
    extra_include_paths = []
    extra_ldflags = []

    if platform.system() == "Windows":
        # Use /std:c++20 to match torch's own header requirements (newer torch
        # versions use C++20 features like default initializers for bit-fields).
        extra_cflags = ["/O2", "/std:c++20"]
        # Source vcvars64.bat into os.environ so torch's `where cl` / ninja can
        # resolve the toolchain. Idempotent and a no-op if INCLUDE+LIB already set.
        _initialize_msvc_environment()
        # If sourcing failed (no VS install found), fall back to passing paths
        # explicitly as compile flags.
        if not os.environ.get("INCLUDE") or not os.environ.get("LIB"):
            include_paths, library_paths = _find_msvc_paths()
            if include_paths:
                extra_include_paths = include_paths
                print(f"\n  Found {len(include_paths)} include paths")
            if library_paths:
                # Add library paths as linker flags (no quotes - ninja handles this)
                extra_ldflags = [f"/LIBPATH:{p}" for p in library_paths]
                print(f"  Found {len(library_paths)} library paths")
    else:
        extra_cflags = ["-O3", "-std=c++17"]

    _fast_csv = load(
        name=module_name,
        sources=[str(cpp_file)],
        verbose=True,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
    )

    print("done")
    return _fast_csv


def load_species_csv(
    path: str,
    plot_id_col: str = "PlotObservationID",
    species_id_col: str = "WFO_TAXON",
    weight_col: str = "Cover %",
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load species CSV directly to PyTorch tensors (legacy API).

    Uses memory-mapped file reading - no OOM even for huge files.

    Args:
        path: Path to species CSV file
        plot_id_col: Column name for plot IDs
        species_id_col: Column name for species IDs (will be hashed)
        weight_col: Column name for abundance weights
        verbose: Print progress

    Returns:
        Dict with:
            - plot_indices: (n_records,) int64 - plot index for each record
            - species_ids: (n_records,) int64 - MurmurHash of species ID
            - weights: (n_records,) float32 - abundance values
            - plot_offsets: (n_plots+1,) int64 - CSR offsets for batch access
    """
    ext = _get_extension()
    return ext.load_species_csv(str(path), plot_id_col, species_id_col, weight_col, verbose)


def load_grouped_csv(
    path: str,
    group_id_col: str,
    numeric_cols: List[str],
    string_cols: List[str],
    hash_string_cols: bool = True,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Load grouped CSV with arbitrary columns (generic API).

    Uses memory-mapped file reading - no OOM even for huge files.
    Suitable for any grouped data: plots in ecology, patients in medicine,
    sessions in web analytics, etc.

    Args:
        path: Path to CSV file
        group_id_col: Column name for group IDs (e.g., plot IDs, patient IDs)
        numeric_cols: List of numeric column names to load as float32 tensors
        string_cols: List of string column names to load (hashed to int64 if hash_string_cols=True)
        hash_string_cols: If True, hash string columns to int64 tensors; if False, return as Python lists
        verbose: Print progress

    Returns:
        Dict with:
            - group_indices: (n_records,) int64 - group index for each record
            - group_offsets: (n_groups+1,) int64 - CSR offsets for batch access
            - numeric columns: float32 tensors
            - string columns: int64 tensors (if hashed) or Python lists
            - "_n_records": total record count
            - "_n_groups": unique group count
    """
    ext = _get_extension()
    return ext.load_grouped_csv(
        str(path), group_id_col, numeric_cols, string_cols, hash_string_cols, verbose
    )


def load_header_csv(
    path: str,
    numeric_cols: List[str],
    plot_id_col: str = "PlotObservationID",
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load header CSV numeric columns to tensors.

    Args:
        path: Path to header CSV file
        numeric_cols: List of numeric column names to load
        plot_id_col: Column name for plot IDs
        verbose: Print progress

    Returns:
        Dict mapping column_name -> float32 tensor
    """
    ext = _get_extension()
    return ext.load_header_csv(str(path), numeric_cols, plot_id_col, verbose)


def load_header_csv_full(
    path: str,
    numeric_cols: List[str],
    string_cols: List[str],
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Load header CSV with both numeric and string columns.

    Uses C++ memory-mapped file reading for maximum speed.

    Args:
        path: Path to header CSV file
        numeric_cols: List of numeric column names to load as float32 tensors
        string_cols: List of string column names to load as Python lists
        verbose: Print progress

    Returns:
        Dict with:
            - numeric columns: torch.Tensor (float32)
            - string columns: List[str]
            - "_n_rows": int (row count)
    """
    ext = _get_extension()
    return ext.load_header_csv_full(str(path), numeric_cols, string_cols, verbose)


def load_dataset_fast(
    header_path: str,
    species_path: str,
    target_cols: List[str],
    covariate_cols: Optional[List[str]] = None,
    coord_cols: Tuple[str, str] = ("Latitude", "Longitude"),
    plot_id_col: str = "PlotObservationID",
    species_id_col: str = "WFO_TAXON",
    weight_col: str = "Cover %",
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load full dataset from header + species CSVs.

    Returns all data needed for training:
    - Species data in COO/CSR format for on-the-fly hash computation
    - Target values
    - Coordinates
    - Covariates

    Args:
        header_path: Path to header CSV (one row per plot)
        species_path: Path to species CSV (many rows per plot)
        target_cols: Target column names to load
        covariate_cols: Optional covariate column names
        coord_cols: Tuple of (lat, lon) column names
        plot_id_col: Plot ID column name
        species_id_col: Species ID column name
        weight_col: Abundance weight column name
        verbose: Print progress

    Returns:
        Dict with all tensors needed for training
    """
    if verbose:
        print("Loading dataset (C++ fast loader)...")

    # Load species data
    species_data = load_species_csv(
        species_path, plot_id_col, species_id_col, weight_col, verbose
    )

    # Determine which header columns to load
    header_cols = list(target_cols)
    if coord_cols:
        header_cols.extend(coord_cols)
    if covariate_cols:
        header_cols.extend(covariate_cols)

    # Load header data
    header_data = load_header_csv(header_path, header_cols, plot_id_col, verbose)

    # Combine results
    result = {
        # Species data (COO format)
        "plot_indices": species_data["plot_indices"],
        "species_ids": species_data["species_ids"],
        "weights": species_data["weights"],
        "plot_offsets": species_data["plot_offsets"],
        "n_plots": len(species_data["plot_offsets"]) - 1,
    }

    # Add targets
    for col in target_cols:
        if col in header_data:
            result[f"target_{col}"] = header_data[col]

    # Add coordinates
    if coord_cols and coord_cols[0] in header_data and coord_cols[1] in header_data:
        result["coordinates"] = torch.stack([
            header_data[coord_cols[0]],
            header_data[coord_cols[1]]
        ], dim=1)

    # Add covariates
    if covariate_cols:
        cov_tensors = [header_data[c] for c in covariate_cols if c in header_data]
        if cov_tensors:
            result["covariates"] = torch.stack(cov_tensors, dim=1)

    return result


# Test function
def _test():
    """Quick test with small data."""
    import tempfile

    # Create test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("PlotObservationID,WFO_TAXON,Cover %\n")
        f.write("p1,species_a,0.5\n")
        f.write("p1,species_b,0.3\n")
        f.write("p2,species_a,0.8\n")
        f.write("p2,species_c,0.2\n")
        test_path = f.name

    try:
        result = load_species_csv(test_path, verbose=True)
        print(f"  plot_indices: {result['plot_indices']}")
        print(f"  species_ids: {result['species_ids']}")
        print(f"  weights: {result['weights']}")
        print(f"  plot_offsets: {result['plot_offsets']}")
        print("Test passed!")
    finally:
        os.unlink(test_path)


if __name__ == "__main__":
    _test()
