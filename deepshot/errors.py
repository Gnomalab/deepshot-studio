"""Errores inesperados: nunca se pierden en silencio.

Una app de escritorio no tiene consola visible, así que una excepción dentro de un botón o de un
hilo desaparecía sin dejar rastro. Aquí se enganchan los ganchos de Python para que:
  - la traza completa quede en el archivo de registro;
  - la ventana principal reciba un aviso (señal) y se lo enseñe al usuario;
  - un fallo nativo (cierre brusco) deje su traza en `crash.log` de la carpeta de registros.
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import time
import traceback

from PySide6.QtCore import QObject, Signal

from deepshot.paths import data_subdir

log = logging.getLogger(__name__)


class ErrorReporter(QObject):
    """Recibe las excepciones no capturadas y las reenvía a la interfaz.

    reported(resumen, traza): se emite en el hilo de la interfaz (conexión en cola entre hilos).
    Un mismo error repetido en menos de `QUIET_SECONDS` se registra pero no se vuelve a avisar,
    para no llenar la pantalla de diálogos si falla algo en cada movimiento del ratón.
    """

    reported = Signal(str, str)
    QUIET_SECONDS = 20.0

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._last: dict[str, float] = {}

    def handle(self, exc_type, exc, tb) -> None:
        trace = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("Error inesperado:\n%s", trace)
        key = f"{exc_type.__name__}:{exc}"
        now = time.monotonic()
        if now - self._last.get(key, -1e9) < self.QUIET_SECONDS:
            return
        self._last[key] = now
        try:
            self.reported.emit(f"{exc_type.__name__}: {exc}", trace)
        except RuntimeError:
            pass        # la ventana ya no existe


_original_excepthook = sys.excepthook
_crash_file = None


def install(reporter: ErrorReporter) -> None:
    """Activa los ganchos de errores (hilo principal, hilos de trabajo y fallos nativos)."""
    global _crash_file

    def excepthook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _original_excepthook(exc_type, exc, tb)
            return
        reporter.handle(exc_type, exc, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        reporter.handle(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    try:
        _crash_file = open(data_subdir("logs") / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(_crash_file)      # cierre brusco (fallo nativo): deja la traza de Python
    except (OSError, RuntimeError):
        pass


def uninstall() -> None:
    """Restaura los ganchos originales (los tests lo usan para no dejar nada instalado)."""
    global _crash_file
    sys.excepthook = _original_excepthook
    threading.excepthook = threading.__excepthook__
    if _crash_file is not None:
        try:
            faulthandler.disable()
            _crash_file.close()
        except (OSError, ValueError):
            pass
        _crash_file = None
