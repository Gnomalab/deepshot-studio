"""Cálculos de profundidad de una foto, para compararlos y elegir con cuál seguir (sin interfaz).

Cada vez que se calcula (o se recupera de la caché) el mapa de una foto con unos ajustes concretos (modelo, calidad,
refinado) se guarda una entrada con esos ajustes, lo que tardó y el mapa. La lista de una foto deja ver los tiempos
de cada modelo y calidad, y elegir cuál se usa en el resto del proceso (rampa, áreas, desenfoque, exportación…).

Los mapas se guardan perezosamente: una entrada que viene de la caché en disco solo lee su PNG cuando se necesita,
así que recuperar de golpe todos los cálculos de una foto no cuesta memoria ni tiempo.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepshot.depth import (
    QUALITY_PRESETS,
    DepthRun,
    DepthSettings,
    cache_path,
    file_hash,
    load_depth_png,
    read_timing,
    settings_for,
)

MAX_RUNS = 8                 # cálculos que se recuerdan por foto (con mapas de decenas de megapíxeles pesan)


def quality_label(settings: DepthSettings) -> str:
    """Nombre de la calidad de unos ajustes («Máxima 1280 px ×3»), o solo la resolución si no es un preset."""
    for preset in QUALITY_PRESETS.values():
        if (preset.processing_resolution, preset.ensemble_size) == (settings.processing_resolution,
                                                                     settings.ensemble_size):
            suffix = f" ×{preset.ensemble_size}" if preset.ensemble_size > 1 else ""
            return f"{QCoreApplication.translate('Depth', preset.label)} {preset.processing_resolution} px{suffix}"
    extra = f" ×{settings.ensemble_size}" if settings.ensemble_size > 1 else ""
    return f"{settings.processing_resolution or 'auto'} px{extra}"


def quality_id(settings: DepthSettings) -> str | None:
    """Id del preset de calidad que corresponde a estos ajustes, o None."""
    for preset in QUALITY_PRESETS.values():
        if (preset.processing_resolution, preset.ensemble_size) == (settings.processing_resolution,
                                                                     settings.ensemble_size):
            return preset.id
    return None


@dataclass
class RunEntry:
    """Un cálculo de profundidad de una foto."""

    settings: DepthSettings
    run: DepthRun | None = None             # tiempos; None si se desconocen
    cache_file: Path | None = None          # PNG en la caché desde el que se puede leer el mapa
    depth: np.ndarray | None = None         # el mapa, si ya está en memoria

    @property
    def key(self) -> str:
        return self.settings.key()

    @property
    def seconds(self) -> float | None:
        return self.run.seconds if self.run is not None else None

    def get_depth(self) -> np.ndarray | None:
        """El mapa (lo lee de la caché la primera vez). None si ya no se puede leer."""
        if self.depth is None and self.cache_file is not None and self.cache_file.is_file():
            self.depth = load_depth_png(self.cache_file)
        return self.depth


class RunList:
    """Los cálculos recordados de una foto, del más antiguo al más reciente."""

    def __init__(self, max_runs: int = MAX_RUNS) -> None:
        self.max_runs = max_runs
        self.entries: list[RunEntry] = []

    def get(self, key: str) -> RunEntry | None:
        return next((e for e in self.entries if e.key == key), None)

    def add(self, entry: RunEntry, keep_key: str = "") -> None:
        """Añade un cálculo; si ya había uno con los mismos ajustes lo sustituye. Si se pasa del máximo se olvidan los
        más antiguos (nunca el de `keep_key`, el que está en uso)."""
        self.entries = [e for e in self.entries if e.key != entry.key]
        self.entries.append(entry)
        while len(self.entries) > self.max_runs:
            victim = next((e for e in self.entries if e.key != keep_key and e is not entry), None)
            if victim is None:
                break
            self.entries.remove(victim)

    def remove(self, key: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.key != key]
        return len(self.entries) != before

    def keep_only(self, key: str) -> None:
        self.entries = [e for e in self.entries if e.key == key]


def discover(photo: Path, model_ids: list[str]) -> list[RunEntry]:
    """Cálculos de esta foto que ya están en la caché de disco, con sus tiempos, para todos los modelos, calidades y
    con o sin refinado. Solo mira si existen los archivos; no lee los mapas."""
    digest = file_hash(photo)
    found = []
    for model in model_ids:
        for quality in QUALITY_PRESETS:
            for refine in (0.0, 1.0):
                settings = settings_for(model=model, quality=quality, refine=refine)
                png = cache_path(photo, settings, photo_hash=digest)
                if png.is_file():
                    t = read_timing(png)
                    run = DepthRun(model, settings.processing_resolution, settings.ensemble_size, refine,
                                   infer_seconds=t.get("infer_seconds"),
                                   refine_seconds=float(t.get("refine_seconds", 0.0)), from_cache=True)
                    found.append(RunEntry(settings, run, png))
    return found


def format_seconds(seconds: float | None) -> str:
    """12,3 s · «—» si no se conoce."""
    if seconds is None:
        return "—"
    return f"{seconds:.1f} s".replace(".", ",") if seconds < 100 else f"{int(round(seconds // 60))} min {int(round(seconds % 60)):02d} s"
