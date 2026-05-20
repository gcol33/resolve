"""I/O retry helper for transient filesystem stalls.

Long-running training jobs that save checkpoints to flaky drives (NTFS over a
slow disk, network share, drive briefly oversubscribed during heavy load) can
hit transient OS-level I/O faults: ``WinError 121`` ("The semaphore timeout
period has expired"), ``OSError(22, 'Invalid argument')`` on a half-open
handle, or generic file-system errors. These almost always resolve themselves
within seconds, but an unhandled exception in a checkpoint write kills hours
of training.

``retry_io`` wraps a callable and retries it on ``OSError`` with exponential
backoff. It is exported from ``resolve_core`` for downstream use, and is also
applied as the default wrapper around ``Trainer.save``, ``Trainer.load``, and
``Predictor.load`` (see ``__init__.py``).
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_io(
    fn: Callable[[], T],
    *,
    what: str = "I/O",
    max_tries: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
) -> T:
    """Run ``fn()`` retrying on ``OSError`` with exponential backoff.

    Parameters
    ----------
    fn
        Zero-argument callable to run. Wrap the real call in ``lambda:``.
    what
        Short label used in retry log lines (e.g. ``"Trainer.save(path)"``).
    max_tries
        Total attempts including the first. Default 6.
    base_delay
        Seconds to sleep after the first failure. Default 1.0.
    max_delay
        Cap on the per-attempt sleep. Default 32.0.

    Returns
    -------
    Whatever ``fn()`` returns on its first successful attempt.

    Raises
    ------
    OSError
        The last ``OSError`` if every attempt fails.
    """
    delay = base_delay
    last_exc: OSError | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except OSError as e:
            last_exc = e
            try:
                print(
                    f"[retry_io] {what}: OSError attempt {attempt}/{max_tries}: {e!r}",
                    flush=True,
                )
            except Exception:
                pass
            if attempt == max_tries:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, max_delay)
    assert last_exc is not None
    raise last_exc
