"""Uso de memoria del proceso y utilidades para devolverla al sistema (sin interfaz, sin dependencias nuevas).

`process_memory_mb()` mide la RAM que usa la app ahora mismo: en Windows con la API del sistema (ctypes) y en
Linux con /proc; en macOS no hay una lectura sencilla sin psutil y devuelve None. `gpu_memory_mb()` mide la
memoria de vídeo que PyTorch tiene reservada (None si no hay CUDA o torch todavía no se cargó).
"""

from __future__ import annotations

import gc
import sys


def process_memory_mb() -> float | None:
    """Memoria privada (RAM) que usa este proceso, en MB (None si no se puede medir en este sistema)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.PagefileUsage / 1e6      # memoria privada: lo que de verdad se libera
            return None
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="ascii") as f:
                pages = int(f.read().split()[1])
            import os

            return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, AttributeError, ImportError):
        pass
    return None


def gpu_memory_mb() -> float | None:
    """Memoria de vídeo reservada por PyTorch, en MB (None sin CUDA o si torch aún no se ha cargado)."""
    torch = sys.modules.get("torch")      # no se importa aquí: cargar torch tarda varios segundos
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e6
    except Exception:                      # un driver raro no debe romper la interfaz
        pass
    return None


def release_python_memory() -> None:
    """Recoge la basura de Python y devuelve a la GPU la memoria que PyTorch tiene en caché."""
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Memoria libre del equipo (para el semáforo del pie de la ventana)
# ---------------------------------------------------------------------------
def gpu_free_total_mb() -> tuple[float, float] | None:
    """(libre, total) de la memoria de vídeo del equipo, en MB. Cuenta también lo que usan otros programas.
    None sin CUDA o si torch aún no se ha cargado."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1e6, total / 1e6
    except Exception:
        pass
    return None


def ram_free_total_mb() -> tuple[float, float] | None:
    """(libre, total) de la RAM del equipo, en MB (None si no se puede medir en este sistema)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys / 1e6, status.ullTotalPhys / 1e6
            return None
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    info[key] = float(rest.split()[0]) / 1000      # kB -> MB
            return info["MemAvailable"], info["MemTotal"]
    except (OSError, ValueError, KeyError, AttributeError, ImportError):
        pass
    return None


def level(free_mb: float, total_mb: float) -> str:
    """Semáforo de una memoria: "bad" (casi llena), "warn" (justa) u "ok". Mira el porcentaje libre y un mínimo
    absoluto (8 GB con 700 MB libres está peor que 64 GB con el mismo 10 %)."""
    if total_mb <= 0:
        return "ok"
    ratio = free_mb / total_mb
    if ratio < 0.10 or free_mb < 700:
        return "bad"
    if ratio < 0.25 or free_mb < 1800:
        return "warn"
    return "ok"


def worst(*levels: str | None) -> str:
    """El peor de varios niveles (los None se ignoran)."""
    order = ("ok", "warn", "bad")
    return max((lv for lv in levels if lv), key=order.index, default="ok")
