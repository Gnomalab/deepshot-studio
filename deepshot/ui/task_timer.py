"""Cronómetro de las tareas largas para la interfaz: emite cada segundo el texto de tiempo y estimación.

Uso: `start()` al empezar, `update(fracción)` con cada avance, `text()` para añadirlo al mensaje y `stop()`
al acabar (devuelve los segundos que tardó, para recordarlos en `timing.Durations`).
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

from deepshot.timing import ProgressClock, format_clock


class TaskTimer(QObject):
    changed = Signal(str)            # texto nuevo de tiempo (cada segundo y con cada avance)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._clock: ProgressClock | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(lambda: self.changed.emit(self.text()))

    def start(self, expected: float | None = None) -> None:
        """Empieza a contar. `expected` = lo que tardó otras veces (para tareas sin fracción de avance)."""
        self._clock = ProgressClock(expected)
        self._timer.start()

    def update(self, fraction: float | None) -> None:
        if self._clock is not None:
            self._clock.update(fraction)

    def is_running(self) -> bool:
        return self._clock is not None

    def stop(self) -> float:
        """Para el cronómetro y devuelve los segundos transcurridos (0 si no estaba en marcha)."""
        self._timer.stop()
        elapsed = self._clock.elapsed() if self._clock is not None else 0.0
        self._clock = None
        return elapsed

    def text(self) -> str:
        """«0:12 · quedan ~0:35», «0:12 · más de lo habitual» o solo «0:12» si aún no se puede estimar."""
        if self._clock is None:
            return ""
        snap = self._clock.snapshot()
        text = format_clock(snap.elapsed)
        if snap.remaining is not None:
            text += " · " + QCoreApplication.translate("TaskTimer", "quedan ~{0}").format(format_clock(snap.remaining))
        elif snap.overrun:
            text += " · " + QCoreApplication.translate("TaskTimer", "tarda más de lo habitual")
        return text


def with_time(message: str, timer: TaskTimer) -> str:
    """Mensaje de progreso con el tiempo al final (sin tocar el mensaje si el cronómetro no corre)."""
    text = timer.text()
    return f"{message}  ·  {text}" if text else message
