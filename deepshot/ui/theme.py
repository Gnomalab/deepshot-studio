"""Aplica el tema elegido: estilo Fusion + paleta Qt + hoja QSS generada (ver `deepshot/themes.py`).

El tema y el aspecto (redondeo, densidad) se guardan en QSettings (`appearance/theme`, `appearance/radius`,
`appearance/density`) y se pueden cambiar en caliente con `apply_theme(app, ...)`: los widgets pintados a mano leen los
colores con `color("nombre")` cada vez que se dibujan, así que se actualizan solos con un `update()`.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from deepshot import themes

log = logging.getLogger(__name__)

# Fuentes del sistema por orden de preferencia: Windows, macOS, genérica.
FONT_FAMILIES = ["Segoe UI", ".AppleSystemUIFont", "Helvetica Neue", "Arial"]

_state = {"theme": themes.FACTORY_THEMES[themes.DEFAULT_THEME], "appearance": themes.Appearance()}


def current_theme() -> themes.Theme:
    return _state["theme"]


def current_appearance() -> themes.Appearance:
    return _state["appearance"]


def color(name: str) -> QColor:
    """Un color de la paleta del tema actual («text», «accent», «line»…)."""
    return QColor(_state["theme"].color(name))


def _palette(theme: themes.Theme) -> QPalette:
    """Paleta para los elementos que QSS no cubre (diálogos nativos de Qt, tooltips, enlaces…)."""
    c = {k: QColor(theme.color(k)) for k in themes.COLOR_KEYS}
    p = QPalette()
    p.setColor(QPalette.Window, c["window"])
    p.setColor(QPalette.WindowText, c["text"])
    p.setColor(QPalette.Base, c["base"])
    p.setColor(QPalette.AlternateBase, c["panel"])
    p.setColor(QPalette.Text, c["text"])
    p.setColor(QPalette.Button, c["button"])
    p.setColor(QPalette.ButtonText, c["text"])
    p.setColor(QPalette.ToolTipBase, c["popup"])
    p.setColor(QPalette.ToolTipText, c["text"])
    p.setColor(QPalette.Highlight, c["accent"])
    p.setColor(QPalette.HighlightedText, c["on_accent"])
    p.setColor(QPalette.PlaceholderText, c["text_dim"])
    p.setColor(QPalette.Link, c["link"])
    p.setColor(QPalette.LinkVisited, c["link"])
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c["text_disabled"])
    return p


def saved_choice() -> tuple[str, themes.Appearance]:
    """Tema y aspecto guardados en los ajustes (los de fábrica si no hay)."""
    s = QSettings()
    try:
        appearance = themes.Appearance(radius=int(s.value("appearance/radius", 3)),
                                       density=str(s.value("appearance/density", "standard")),
                                       icons=str(s.value("appearance/icons", "color"))).validated()
    except (TypeError, ValueError):
        appearance = themes.Appearance()
    return str(s.value("appearance/theme", themes.DEFAULT_THEME)), appearance


def save_choice(theme_key: str, appearance: themes.Appearance) -> None:
    s = QSettings()
    s.setValue("appearance/theme", theme_key)
    s.setValue("appearance/radius", appearance.radius)
    s.setValue("appearance/density", appearance.density)
    s.setValue("appearance/icons", appearance.icons)


def apply_theme(app: QApplication, theme_key: str | None = None, appearance: themes.Appearance | None = None) -> None:
    """Aplica estilo, paleta, fuente y QSS a toda la aplicación. Sin argumentos usa lo guardado en los ajustes."""
    if theme_key is None or appearance is None:
        saved_key, saved_appearance = saved_choice()
        theme_key = theme_key if theme_key is not None else saved_key
        appearance = appearance if appearance is not None else saved_appearance
    theme = themes.get_theme(theme_key)
    appearance = appearance.validated()
    _state.update(theme=theme, appearance=appearance)

    app.setStyle("Fusion")  # estilo idéntico en Windows y macOS
    app.setPalette(_palette(theme))
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSizeF(9.5)
    app.setFont(font)
    try:
        app.setStyleSheet(themes.build_qss(theme, appearance))
    except (OSError, KeyError, ValueError) as exc:
        log.warning("No se pudo aplicar la hoja de estilo del tema %s: %s", theme.key, exc)
