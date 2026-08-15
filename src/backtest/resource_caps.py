"""Apply quiet-mode CPU/RAM/thread caps so the search doesn't take over the box.

Targets:
  - CPU: BelowNormal process priority + affinity limited to ceil(cpu_count*0.20)
  - RAM: best-effort Job Object cap on Windows; advisory soft GC elsewhere
  - Threads: pin BLAS/LGBM/OMP/MKL to 1 thread per process

Call `apply_quiet_mode()` once at process start, before importing heavy libs
that read these env vars (numpy, lightgbm, etc).
"""
from __future__ import annotations

import math
import os
import sys


def set_thread_caps() -> None:
    """Pin underlying math libs to single-threaded so we don't burn cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "LIGHTGBM_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def set_cpu_affinity_pct(pct: float = 0.20) -> int:
    """Restrict the process to a fraction of available CPU cores. Returns N cores."""
    try:
        import psutil
        n_total = psutil.cpu_count(logical=True) or 1
        n_use = max(1, math.ceil(n_total * pct))
        p = psutil.Process()
        try:
            cores = list(range(n_use))
            p.cpu_affinity(cores)
        except (AttributeError, OSError):
            pass
        return n_use
    except Exception:
        return -1


def set_low_priority() -> None:
    try:
        import psutil
        p = psutil.Process()
        if sys.platform.startswith("win"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        pass


def set_memory_cap_mb(cap_mb: int = 4096) -> bool:
    """Best-effort hard cap on process RSS via Windows Job Object. Returns True if set."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Constants
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = ctypes.c_size_t(cap_mb * 1024 * 1024)

        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            return False

        handle = kernel32.GetCurrentProcess()
        if not kernel32.AssignProcessToJobObject(job, handle):
            return False
        return True
    except Exception:
        return False


def apply_quiet_mode(cpu_pct: float = 0.20, mem_cap_mb: int = 4096,
                     verbose: bool = True) -> dict:
    """One-shot: apply all caps. Call at process start."""
    set_thread_caps()
    n_cores = set_cpu_affinity_pct(cpu_pct)
    set_low_priority()
    mem_set = set_memory_cap_mb(mem_cap_mb)
    info = {
        "cpu_cores_used": n_cores,
        "cpu_pct_target": cpu_pct,
        "mem_cap_mb": mem_cap_mb,
        "mem_cap_enforced": mem_set,
        "priority": "BelowNormal" if sys.platform.startswith("win") else "nice+10",
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "LIGHTGBM_NUM_THREADS")},
    }
    if verbose:
        print(f"[quiet-mode] {info}", file=sys.stderr)
    return info
