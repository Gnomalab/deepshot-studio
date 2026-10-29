"""Elección del dispositivo de inferencia: CUDA (NVIDIA), MPS (Apple Silicon) o CPU."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# En Mac, las operaciones que MPS aún no soporta caen a CPU en lugar de fallar.
# Debe fijarse antes de importar torch.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class DeviceInfo:
    kind: str           # "cuda", "mps" o "cpu"
    name: str           # nombre legible para la barra de estado
    vram_gb: float | None = None


def detect_device(preferred: str = "auto") -> DeviceInfo:
    """Devuelve el mejor dispositivo disponible.

    `preferred` permite forzar "cuda", "mps" o "cpu"; con "auto" se elige
    CUDA > MPS > CPU. Si el preferido no está disponible, se usa CPU.
    """
    import torch  # importación diferida: tarda un par de segundos

    want_cuda = preferred in ("auto", "cuda")
    want_mps = preferred in ("auto", "mps")

    if want_cuda and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo("cuda", props.name, round(props.total_memory / 1024**3, 1))
    if want_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo("mps", "Apple Silicon (MPS)")
    return DeviceInfo("cpu", "CPU")
