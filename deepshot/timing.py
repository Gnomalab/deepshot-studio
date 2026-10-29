"""Tiempo transcurrido y estimado de las tareas largas (sin interfaz).

`ProgressClock` cuenta el tiempo de una tarea y estima lo que falta de dos maneras:
  - con la fracción de avance, si la tarea la da (exportar, descargar): total ≈ transcurrido / fracción;
  - con lo que tardó otras veces la misma tarea (`Durations`), si no da fracción (calcular la profundidad).
`Durations` guarda en los ajustes (QSettings o cualquier objeto con value / setValue) los últimos tiempos
de cada tipo de tarea y devuelve la mediana, que resiste a un cálculo raro (p. ej. el que incluye cargar el modelo).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

MIN_ELAPSED_FOR_ESTIMATE = 1.5     # antes de este tiempo la fracción da estimaciones demasiado bailongas
MIN_FRACTION_FOR_ESTIMATE = 0.02
KEEP_DURATIONS = 5                 # tiempos recordados por tipo de tarea


@dataclass(frozen=True)
class Snapshot:
    elapsed: float                  # segundos desde que empezó
    remaining: float | None         # segundos que faltan, o None si aún no se puede estimar
    overrun: bool = False           # tarda más de lo habitual y no hay otra estimación


class ProgressClock:
    """Cronómetro de una tarea con estimación del tiempo que falta."""

    def __init__(self, expected: float | None = None, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self.expected = expected if expected and expected > 0 else None   # lo que tardó otras veces
        self._start = now()
        self._fraction: float | None = None

    def update(self, fraction: float | None) -> None:
        """Avance 0..1 (None o negativo = no se puede medir)."""
        self._fraction = None if fraction is None or fraction < 0 else min(float(fraction), 1.0)

    def elapsed(self) -> float:
        return max(self._now() - self._start, 0.0)

    def snapshot(self) -> Snapshot:
        elapsed = self.elapsed()
        fraction = self._fraction
        if fraction is not None and fraction >= MIN_FRACTION_FOR_ESTIMATE and elapsed >= MIN_ELAPSED_FOR_ESTIMATE:
            return Snapshot(elapsed, max(elapsed / fraction - elapsed, 0.0))
        if self.expected is not None and fraction is None:
            if elapsed <= self.expected:
                return Snapshot(elapsed, self.expected - elapsed)
            return Snapshot(elapsed, None, overrun=True)
        return Snapshot(elapsed, None)


def format_clock(seconds: float) -> str:
    """12 → «0:12» · 65 → «1:05» · 3700 → «1:01:40»."""
    total = max(int(round(seconds)), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Durations:
    """Tiempos recientes de cada tipo de tarea, guardados en los ajustes de la app."""

    def __init__(self, settings) -> None:
        self._settings = settings

    @staticmethod
    def _key(kind: str) -> str:
        return f"timing/{kind}"

    def recent(self, kind: str) -> list[float]:
        value = self._settings.value(self._key(kind), [])
        if isinstance(value, (str, int, float)):           # QSettings devuelve un solo valor como texto
            value = [value]
        out = []
        for item in value or []:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                pass
        return out

    def expected(self, kind: str) -> float | None:
        """Mediana de los últimos tiempos de esta tarea, o None si nunca se hizo."""
        recent = self.recent(kind)
        return statistics.median(recent) if recent else None

    def record(self, kind: str, seconds: float) -> None:
        if seconds <= 0.2:                                  # tareas instantáneas (caché): no aportan nada
            return
        self._settings.setValue(self._key(kind), [*self.recent(kind), round(seconds, 2)][-KEEP_DURATIONS:])
