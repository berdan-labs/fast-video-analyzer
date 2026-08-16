"""Small dependency-free process and generated-output resource snapshots."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any


def directory_usage(
    root: str | Path,
    *,
    include_reclaimable: bool = False,
) -> dict[str, int]:
    """Count regular files/bytes, optionally classifying disposable checkpoints.

    ``Path.rglob`` combined with ``is_symlink``/``is_file``/``stat`` performs
    several metadata probes for every generated entry.  Resource telemetry is
    published repeatedly while a project is finalized, so recurse with
    ``os.scandir`` and use each ``DirEntry``'s one ``stat(follow_symlinks=False)``
    result for both classification and sizing.  This keeps symlinked
    directories and files out of the inventory while avoiding the temporary
    ``Path`` objects and second path lookup that ``os.walk`` requires; the
    operation remains accounting-only and never follows source/model links.
    """

    target = Path(root)
    file_count = 0
    total_bytes = 0
    reclaimable_file_count = 0
    reclaimable_bytes = 0
    # Treat a symlink supplied as the root argument as outside the generated
    # output inventory for the same safety boundary as linked descendants.
    if target.is_symlink() or not target.exists():
        usage = {"file_count": 0, "bytes": 0}
        if include_reclaimable:
            usage.update({"reclaimable_file_count": 0, "reclaimable_bytes": 0})
        return usage
    def visit(directory: str, relative_parts: tuple[str, ...]) -> None:
        nonlocal file_count, total_bytes, reclaimable_file_count, reclaimable_bytes
        try:
            entries = os.scandir(directory)
        except OSError:
            # Match os.walk's default onerror behavior: a transiently
            # inaccessible subtree contributes no files rather than failing
            # an accounting-only telemetry refresh.
            return
        with entries:
            for entry in entries:
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                mode = stat.st_mode
                if S_ISDIR(mode):
                    visit(entry.path, (*relative_parts, entry.name))
                    continue
                if not S_ISREG(mode):
                    continue
                size = int(stat.st_size)
                total_bytes += size
                file_count += 1
                child_parts = (*relative_parts, entry.name)
                reclaimable = child_parts[:2] == (".state", "cache") or child_parts[:3] in {
                    (".state", "checkpoints", "visual-frames"),
                    (".state", "checkpoints", "ocr"),
                }
                if reclaimable:
                    reclaimable_file_count += 1
                    reclaimable_bytes += size

    visit(os.fspath(target), ())
    usage = {"file_count": file_count, "bytes": total_bytes}
    if include_reclaimable:
        usage.update(
            {
                "reclaimable_file_count": reclaimable_file_count,
                "reclaimable_bytes": reclaimable_bytes,
            }
        )
    return usage


def process_memory_usage() -> dict[str, int | None]:
    """Return current/peak resident memory where the host exposes it."""

    current: int | None = None
    peak: int | None = None
    if os.name == "nt":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        try:
            win_dll = getattr(ctypes, "WinDLL", None)
            if win_dll is None:
                ok = 0
            else:
                kernel32 = win_dll("kernel32", use_last_error=True)
                psapi = win_dll("psapi", use_last_error=True)
                get_current_process = kernel32.GetCurrentProcess
                get_current_process.restype = ctypes.c_void_p
                get_process_memory_info = psapi.GetProcessMemoryInfo
                get_process_memory_info.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_Counters),
                    ctypes.c_ulong,
                ]
                get_process_memory_info.restype = ctypes.c_int
                process = get_current_process()
                ok = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
        except (AttributeError, OSError):
            ok = 0
        if ok:
            current = int(counters.WorkingSetSize)
            peak = int(counters.PeakWorkingSetSize)
    else:
        try:
            import resource

            getrusage = getattr(resource, "getrusage", None)
            usage_self = getattr(resource, "RUSAGE_SELF", None)
            if not callable(getrusage) or usage_self is None:
                raise OSError("resource.getrusage is unavailable")
            value = int(getrusage(usage_self).ru_maxrss)
            peak = value * (1024 if sys.platform != "darwin" else 1)
            current = peak
        except (ImportError, OSError, ValueError):
            pass
    return {"current_rss_bytes": current, "peak_rss_bytes": peak}


def resource_snapshot(root: str | Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"memory": process_memory_usage()}
    if root is not None:
        snapshot["output"] = directory_usage(root, include_reclaimable=True)
    return snapshot


__all__ = ["directory_usage", "process_memory_usage", "resource_snapshot"]
