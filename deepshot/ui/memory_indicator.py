"""Indicador de memoria libre (VRAM de la GPU y RAM) con semáforo: botón plano que se repinta solo cada pocos segundos.

Lo usan el pie de la ventana principal y la ventana del Mapa rápido (y la aplicación básica por separado). No depende
del editor. Al pulsarlo emite `clicked`: quien lo pone decide qué liberar (ver `MainWindow.free_memory`)."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QPushButton, QWidget

from deepshot import memory


class MemoryIndicator(QPushButton):
    def __init__(self, parent: QWidget | None = None, interval_ms: int = 3000) -> None:
        super().__init__(parent)
        self.setObjectName("MemoryIndicator")        # el color lo da el QSS según la propiedad «tone»
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(interval_ms)
        self.refresh()

    @staticmethod
    def levels() -> tuple[str, str]:
        """(nivel de la VRAM, nivel de la RAM) del equipo: "ok" / "warn" / "bad" ("ok" si no se puede medir)."""
        gpu, ram = memory.gpu_free_total_mb(), memory.ram_free_total_mb()
        return (memory.level(*gpu) if gpu else "ok", memory.level(*ram) if ram else "ok")

    def refresh(self) -> None:
        """Pone la memoria en uso (VRAM y RAM) con el color del semáforo."""
        gpu, ram = memory.gpu_free_total_mb(), memory.ram_free_total_mb()
        parts = []
        if gpu:
            parts.append(self.tr("VRAM {0:.1f}/{1:.1f} GB").format((gpu[1] - gpu[0]) / 1000, gpu[1] / 1000))
        if ram:
            parts.append(self.tr("RAM {0:.1f}/{1:.1f} GB").format((ram[1] - ram[0]) / 1000, ram[1] / 1000))
        self.setVisible(bool(parts))
        self.setText("● " + " · ".join(parts))
        tone = memory.worst(memory.level(*gpu) if gpu else None, memory.level(*ram) if ram else None)
        self.setToolTip(self.tr(
            "Memoria en uso del equipo (VRAM de la GPU y RAM). Verde: de sobra · amarillo: justa · rojo: casi llena. "
            "Pulsa para liberar memoria (descarga el modelo y suelta cachés; el proyecto no cambia)."))
        if self.property("tone") != tone:
            self.setProperty("tone", tone)
            self.style().unpolish(self)
            self.style().polish(self)
