"""GPU knobs: the platform-aware ``PYTORCH_CUDA_ALLOC_CONF`` default and the
VRAM cap.

The import-time side effect cannot be observed reliably from a single
pytest process (resolve_core has already been imported by the time the
test starts, so `os.environ["PYTORCH_CUDA_ALLOC_CONF"]` is already set).
We exercise the helper two ways:

1. **Subprocess** — spawn a fresh interpreter that imports resolve_core in a
   clean environment and reads back the env var. This proves the
   import-time side effect actually sets the variable, with the right
   platform-dependent content.

2. **Direct call** — within the current process, call
   ``configure_cuda_allocator(force=True)`` and verify it overwrites the
   variable with the platform-correct string.

Coverage kind: behavioural for the allocator string (the exact content is the
contract that keeps Windows off the unimplemented ``expandable_segments``
path), structure for the VRAM cap, whose effect is only observable on a CUDA
device and is pinned by Catch2 in ``src/core/tests/test_gpu.cpp``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import resolve_core


def _expected_base() -> str:
    base = "garbage_collection_threshold:0.8,max_split_size_mb:256"
    if sys.platform != "win32":
        base = "expandable_segments:True," + base
    return base


def test_import_side_effect_sets_env_var() -> None:
    """A fresh interpreter that imports resolve_core ends up with the env var set."""
    expected = _expected_base()
    script = textwrap.dedent(
        """
        import os
        # Make sure we observe the import-time side effect (no prior value).
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        import resolve_core  # noqa: F401  (side effect: configure_cuda_allocator)
        print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        """
    ).strip()
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip().splitlines()[-1]
    assert actual == expected


def test_existing_env_var_is_preserved_without_force() -> None:
    """A pre-set PYTORCH_CUDA_ALLOC_CONF must NOT be silently overwritten."""
    sentinel = "max_split_size_mb:64"
    script = textwrap.dedent(
        f"""
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = {sentinel!r}
        import resolve_core  # noqa: F401
        print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        """
    ).strip()
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = sentinel
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip().splitlines()[-1]
    assert actual == sentinel


def test_configure_cuda_allocator_force_overwrites() -> None:
    """``configure_cuda_allocator(force=True)`` must set the platform default."""
    original = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        returned = resolve_core.configure_cuda_allocator(force=True)
        assert returned == _expected_base()
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == _expected_base()
    finally:
        if original is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original


def test_platform_specific_expandable_segments_prefix() -> None:
    """expandable_segments must NOT appear on Windows; SHOULD appear elsewhere."""
    original = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    try:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        returned = resolve_core.configure_cuda_allocator(force=True)
        if sys.platform == "win32":
            assert "expandable_segments" not in returned, (
                f"Windows must not request expandable_segments; got {returned!r}"
            )
        else:
            assert returned.startswith("expandable_segments:True,"), (
                f"Non-Windows expected expandable_segments prefix; got {returned!r}"
            )
        assert "garbage_collection_threshold:0.8" in returned
        assert "max_split_size_mb:256" in returned
    finally:
        if original is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = original


def test_batch_size_floor_default_exposed_through_train_config() -> None:
    """TrainConfig.batch_size_floor is bound and defaults to 1024."""
    cfg = resolve_core.TrainConfig()
    assert hasattr(cfg, "batch_size_floor")
    assert cfg.batch_size_floor == 1024
    cfg.batch_size_floor = 256
    assert cfg.batch_size_floor == 256


# ---------------------------------------------------------------------------
# VRAM cap
# ---------------------------------------------------------------------------

def test_vram_fraction_defaults_to_the_whole_device() -> None:
    """A dedicated training job on a solo GPU should not be capped by default."""
    assert resolve_core.TrainConfig().vram_fraction == pytest.approx(1.0)


def test_set_vram_fraction_is_callable_without_cuda() -> None:
    """The helper is a no-op on a CPU-only build rather than an error.

    CI runs the CPU wheel, so this only proves the binding is reachable and
    tolerant. The cap's effect on the caching allocator is asserted in
    ``src/core/tests/test_gpu.cpp``.
    """
    resolve_core.set_vram_fraction(0.8, 0)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_set_vram_fraction_rejects_out_of_range(fraction: float) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("range validation runs on the CUDA path only")
    with pytest.raises(Exception):
        resolve_core.set_vram_fraction(fraction, 0)


def test_process_hardening_helpers_are_exposed() -> None:
    """Windows crash/teardown hardening is armed at import and callable again."""
    resolve_core.install_crash_handler(0)
    resolve_core.set_thread_pools(1, 1)
