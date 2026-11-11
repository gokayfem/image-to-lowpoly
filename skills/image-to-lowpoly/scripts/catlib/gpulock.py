"""Exclusive GPU lock shared by every rendering Blender process (CONTRACT section 1).

Stdlib only, so it imports inside Blender's bundled Python 3.13 as well as the analysis venv.

Usage (inside the rendering Blender script, first thing, held until process exit):

    from catlib.gpulock import acquire_gpu_lock
    acquire_gpu_lock()              # blocks; released automatically when the process exits

or as a context manager:

    with gpu_lock():
        ...render...

The lock is an exclusive ``fcntl.flock`` on ``<project>/.cache/gpu.lock``. The kernel drops
a flock when its file descriptor is closed, which includes any kind of process exit, so a
crashed renderer can never leave a stale lock behind. Callers of the rendering script
(e.g. scripts/evaluate.py) must NOT take the lock themselves.
"""
from __future__ import annotations

import atexit
import contextlib
import fcntl
import os
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent  # scripts/ (holds catlib/)


def _case_root() -> Path:
    """The case directory (set by catlib.paths in the calling process)."""
    return Path(os.environ.get("LOWPOLY_CASE") or os.getcwd())

LOCK_PATH = _case_root() / ".cache" / "gpu.lock"

_held_fd: int | None = None


def _log(msg: str) -> None:
    t = time.time()
    print(f"[gpulock {time.strftime('%H:%M:%S', time.localtime(t))} t={t:.3f} pid={os.getpid()}] {msg}",
          file=sys.stderr, flush=True)


def acquire_gpu_lock(poll_s: float = 0.5, report_every_s: float = 15.0, timeout_s: float | None = None,
                     label: str = "") -> float:
    """Block until this process holds the exclusive GPU lock. Returns seconds spent waiting.

    Idempotent within one process. The lock is held until release_gpu_lock() or process exit.
    """
    global _held_fd
    if _held_fd is not None:
        return 0.0
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.time()
    last = -report_every_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = time.time() - t0
            if waited - last >= report_every_s:
                holder = ""
                try:
                    holder = Path(LOCK_PATH).read_text().strip()
                except OSError:
                    pass
                _log(f"waiting for GPU lock {LOCK_PATH} ({waited:.0f}s){' held by: ' + holder if holder else ''}")
                last = waited
            if timeout_s is not None and waited > timeout_s:
                os.close(fd)
                raise TimeoutError(f"GPU lock not acquired within {timeout_s}s")
            time.sleep(poll_s)
    waited = time.time() - t0
    # record the holder for humans (informational only; the flock is what matters)
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"pid={os.getpid()} since={time.strftime('%Y-%m-%dT%H:%M:%S')} {label}\n".encode())
    except OSError:
        pass
    _held_fd = fd
    atexit.register(release_gpu_lock)
    _log(f"acquired GPU lock after waiting {waited:.2f}s {label}".rstrip())
    return waited


def release_gpu_lock() -> None:
    global _held_fd
    if _held_fd is None:
        return
    fd, _held_fd = _held_fd, None
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    _log("released GPU lock")


@contextlib.contextmanager
def gpu_lock(**kwargs):
    """Context manager form of acquire_gpu_lock()/release_gpu_lock()."""
    already = _held_fd is not None
    acquire_gpu_lock(**kwargs)
    try:
        yield LOCK_PATH
    finally:
        if not already:
            release_gpu_lock()
