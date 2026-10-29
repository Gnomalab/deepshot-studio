"""Panel de registro: mensajes de `logging` dentro de la app, con limpiar, copiar y recargar."""

from __future__ import annotations

import logging
from collections import deque

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from deepshot.paths import log_file


class _Emitter(QObject):
    """Puente entre logging (cualquier hilo) y la interfaz (hilo principal)."""

    message = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.message.emit(self.format(record))
        except RuntimeError:
            pass  # el panel ya se destruyó (la app se está cerrando)


def setup_file_logging() -> None:
    """Guarda el registro también en un archivo rotatorio (1 MB × 3) de la carpeta de datos.

    Así, si algo falla, queda constancia aunque el panel de registro estuviera cerrado.
    Se llama una sola vez al arrancar; no hace nada si ya está activo.
    """
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    if any(getattr(h, "_deepshot_file", False) for h in root.handlers):
        return
    try:
        handler = RotatingFileHandler(log_file(), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        return  # carpeta sin permiso de escritura: se sigue sin archivo
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s"))
    handler._deepshot_file = True
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


class LogPanel(QWidget):
    """Registro de actividad con barra de herramientas."""

    MAX_LINES = 2000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(self.MAX_LINES)

        clear = QPushButton(self.tr("Limpiar"))
        clear.setToolTip(self.tr("Borra los mensajes de este panel (el archivo de registro no se toca)"))
        clear.clicked.connect(self.clear)
        copy = QPushButton(self.tr("Copiar"))
        copy.setToolTip(self.tr("Copia todo el registro al portapapeles, para enviarlo si hay un problema"))
        copy.clicked.connect(self.copy_all)
        reload = QPushButton(self.tr("Recargar"))
        reload.setToolTip(self.tr("Vuelve a cargar los últimos mensajes guardados en el archivo de registro "
                                  "(incluye los de sesiones anteriores)"))
        reload.clicked.connect(lambda _checked=False: self.reload_from_file())   # `checked` no son las líneas
        folder = QPushButton(self.tr("Abrir carpeta"))
        folder.setToolTip(self.tr("Abre la carpeta donde se guarda el archivo de registro"))
        folder.clicked.connect(self.open_folder)
        row = QHBoxLayout()
        row.setContentsMargins(4, 4, 4, 0)
        row.setSpacing(4)
        for button in (clear, copy, reload, folder):
            row.addWidget(button)
        row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.text)

        self.handler = QtLogHandler()
        self.handler.emitter.message.connect(self.text.appendPlainText)
        logging.getLogger().addHandler(self.handler)

    # --- Acciones -------------------------------------------------------------
    def clear(self) -> None:
        self.text.clear()

    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.text.toPlainText())

    def reload_from_file(self, lines: int = 500) -> None:
        """Sustituye el contenido por las últimas `lines` líneas del archivo de registro."""
        try:
            with open(log_file(), "r", encoding="utf-8", errors="replace") as f:
                tail = deque(f, maxlen=lines)
        except OSError:
            tail = deque()
        self.text.setPlainText("".join(tail).rstrip("\n"))
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_file().parent)))

    def detach(self) -> None:
        """Quita el handler (al cerrar la ventana) para no escribir en un widget destruido."""
        logging.getLogger().removeHandler(self.handler)
