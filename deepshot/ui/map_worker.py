"""Calcula un mapa auxiliar (deepshot/maps.py) en segundo plano sin congelar la ventana."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from deepshot import maps

log = logging.getLogger(__name__)


class MapWorker(QObject):
    """Lanza `maps.compute` en un hilo y avisa con señales (llegan al hilo principal).

    progress(ruta_foto, id_mapa, mensaje, fracción)
    finished(ruta_foto, id_mapa, mapa)        mapa: float32 0..1
    failed(ruta_foto, id_mapa, mensaje, sin_modelo)   sin_modelo = True si lo que falta es el modelo
    Solo hay un cálculo a la vez (el modelo ocupa memoria): `start` devuelve False si ya hay uno en marcha.
    """

    progress = Signal(str, str, str, float)
    finished = Signal(str, str, object)
    failed = Signal(str, str, str, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._busy: tuple[str, str] | None = None

    def busy(self) -> tuple[str, str] | None:
        """(foto, mapa) del cálculo en marcha, o None."""
        return self._busy

    def start(self, photo: Path, map_id: str, depth: np.ndarray, crop: tuple | None = None, force: bool = False) -> bool:
        with self._lock:
            if self._busy is not None:
                return False
            self._busy = (str(photo), map_id)
        # Hilo «daemon»: si se cierra la app a mitad de cálculo, no impide salir.
        threading.Thread(target=self._run, args=(Path(photo), map_id, depth, crop, force), daemon=True).start()
        return True

    def _run(self, photo: Path, map_id: str, depth: np.ndarray, crop: tuple | None, force: bool) -> None:
        def report(message: str, fraction: float) -> None:
            self._emit(self.progress, str(photo), map_id, message, float(fraction))

        try:
            if force:                                      # «Recalcular»: se tira la copia guardada
                try:
                    maps.cache_path(map_id, photo, depth, crop).unlink(missing_ok=True)
                except OSError:
                    pass
            mask = maps.compute(map_id, photo, depth, crop, report)
            self._busy = None
            self._emit(self.finished, str(photo), map_id, mask)
        except maps.MapError as exc:
            self._busy = None
            self._emit(self.failed, str(photo), map_id, str(exc), True)
        except Exception as exc:
            log.exception("Fallo al calcular el mapa «%s» de %s", map_id, photo)
            self._busy = None
            self._emit(self.failed, str(photo), map_id, str(exc), False)

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:                               # la ventana ya se cerró
            pass
