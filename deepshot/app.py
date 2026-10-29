"""Punto de entrada de la aplicación."""

from __future__ import annotations

import logging
import os
import sys

from PySide6.QtCore import QTimer

from deepshot import errors
from deepshot.bootstrap import _migrate_settings, create_app  # noqa: F401  (create_app se importa desde aquí)
from deepshot.errors import ErrorReporter
from deepshot.ui.log_panel import setup_file_logging
from deepshot.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_file_logging()   # registro también en archivo (carpeta de datos)
    app = create_app()
    window = MainWindow()
    # Los errores inesperados se registran y se muestran en vez de perderse (ver errors.py).
    reporter = ErrorReporter(window)
    reporter.reported.connect(window.show_unexpected_error)
    errors.install(reporter)
    window.show()
    # Pantalla de inicio (proyecto nuevo / abrir / recientes). Se puede desactivar en su casilla o con
    # DEEPSHOT_NO_START=1 (las pruebas y los scripts no deben bloquearse en un diálogo).
    # Tras ella, si no hay ningún modelo instalado, el aviso del primer arranque (lleva al gestor de modelos).
    if os.environ.get("DEEPSHOT_NO_START") != "1":
        QTimer.singleShot(0, window.startup_dialogs)
    code = app.exec()
    errors.uninstall()
    return code


if __name__ == "__main__":
    sys.exit(main())
