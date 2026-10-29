"""Arranque común de las aplicaciones (la completa y la básica): QApplication con nombre, idioma, icono y tema.

Está aparte de `app.py` para que la aplicación básica (`quick_app.py`) no cargue la ventana principal ni el editor."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from deepshot import APP_NAME, LEGACY_APP_NAME, ORG_NAME, __version__, i18n
from deepshot.paths import resource
from deepshot.ui import wheel_guard
from deepshot.ui.theme import apply_theme


def _migrate_settings() -> None:
    """Copia una vez los ajustes guardados con el nombre anterior de la app.

    QSettings guarda por organización + nombre de app; al cambiar el nombre se perderían
    la geometría de la ventana, el modelo activo y la última carpeta usada.
    """
    new = QSettings()
    if new.allKeys():
        return                      # ya hay ajustes con el nombre nuevo
    old = QSettings(ORG_NAME, LEGACY_APP_NAME)
    for key in old.allKeys():
        new.setValue(key, old.value(key))
    new.sync()


def create_app(argv: list[str] | None = None) -> QApplication:
    """Crea (o reutiliza) la QApplication con nombre, icono y tema."""
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)   # QSettings guarda en una ruta por usuario
    app.setApplicationVersion(__version__)
    _migrate_settings()
    i18n.install(app)          # idioma de la interfaz (ajustes o variable DEEPSHOT_LANG)
    app.setWindowIcon(QIcon(str(resource("icons", "app.svg"))))
    apply_theme(app)
    if not getattr(app, "_wheel_guard", None):          # la rueda no cambia desplegables ni casillas sin Ctrl
        app._wheel_guard = wheel_guard.install(app)
    return app
