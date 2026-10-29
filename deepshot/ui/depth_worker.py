"""Ejecuta la estimación de profundidad en segundo plano sin congelar la ventana."""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from deepshot.depth import DepthEstimator, DepthSettings
from deepshot.ui.depth_errors import friendly_error

log = logging.getLogger(__name__)


class DepthWorker(QObject):
    """Lanza `DepthEstimator.estimate` en un hilo y avisa con señales (hilo principal).

    progress(mensaje, fracción)  fracción = -1 si no se puede medir (barra indeterminada)
    finished(ruta_foto, mapa)    mapa: np.ndarray float32, 0 = cerca, 1 = lejos
    failed(ruta_foto, mensaje, detalle)   detalle = traza técnica para el diálogo de error
    """

    progress = Signal(str, float)
    finished = Signal(str, object)
    failed = Signal(str, str, str)

    def __init__(self, estimator: DepthEstimator, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.estimator = estimator
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    def start(self, image_path: Path, settings: DepthSettings = DepthSettings()) -> bool:
        """Empieza un cálculo. Devuelve False si ya hay uno en marcha."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        # Hilo "daemon": si se cierra la app a mitad de cálculo, no impide salir.
        threading.Thread(target=self._run, args=(Path(image_path), settings), daemon=True).start()
        return True

    def _run(self, image_path: Path, settings: DepthSettings) -> None:
        def report(message: str, fraction: float | None) -> None:
            self.progress.emit(message, -1.0 if fraction is None else float(fraction))

        try:
            depth = self.estimator.estimate(image_path, settings, report)
            self.finished.emit(str(image_path), depth)
        except Exception as exc:
            log.exception("Fallo al calcular la profundidad de %s", image_path)
            self.failed.emit(str(image_path), _friendly_error(exc), traceback.format_exc())
        finally:
            self._busy = False


# Se conserva el nombre antiguo: el código y los tests que lo importan de aquí siguen valiendo.
_friendly_error = friendly_error
