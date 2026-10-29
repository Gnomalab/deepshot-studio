"""Disposición de los paneles de la ventana: plantillas de fábrica y las que guarda el usuario.

Los paneles (biblioteca, ajustes y registro) son acoplables. Una plantilla de fábrica los coloca y dimensiona por
código; una del usuario es el estado completo de la ventana (`QMainWindow.saveState`, que incluye paneles flotantes y
tamaños) guardado como JSON en `<datos>/layouts/<nombre>.json`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QT_TRANSLATE_NOOP, Qt

from deepshot.paths import data_subdir

log = logging.getLogger(__name__)

DEFAULT_LAYOUT = "edit"


def _arrange(window, library: bool, settings: bool, log_panel: bool, library_w: int = 190, settings_w: int = 360,
             log_h: int = 160) -> None:
    """Coloca los tres paneles en su sitio de siempre (sin flotar) y fija cuáles se ven y cuánto miden."""
    for dock, area in ((window.library_dock, Qt.LeftDockWidgetArea), (window.settings_dock, Qt.RightDockWidgetArea),
                       (window.log_dock, Qt.BottomDockWidgetArea)):
        dock.setFloating(False)
        window.addDockWidget(area, dock)
    window.library_dock.setVisible(library)
    window.settings_dock.setVisible(settings)
    window.log_dock.setVisible(log_panel)
    if library:
        window.resizeDocks([window.library_dock], [library_w], Qt.Horizontal)
    if settings:
        window.resizeDocks([window.settings_dock], [settings_w], Qt.Horizontal)
    if log_panel:
        window.resizeDocks([window.log_dock], [log_h], Qt.Vertical)


# (clave, nombre visible, cómo se coloca)
FACTORY: list[tuple[str, str, dict]] = [
    ("edit", QT_TRANSLATE_NOOP("Layouts", "Edición (biblioteca y ajustes)"),
     dict(library=True, settings=True, log_panel=False)),
    ("review", QT_TRANSLATE_NOOP("Layouts", "Revisión (visor grande)"),
     dict(library=True, settings=False, log_panel=False, library_w=150)),
    ("small", QT_TRANSLATE_NOOP("Layouts", "Pantalla pequeña"),
     dict(library=False, settings=True, log_panel=False, settings_w=300)),
    ("debug", QT_TRANSLATE_NOOP("Layouts", "Con registro de actividad"),
     dict(library=True, settings=True, log_panel=True)),
]


def factory_names() -> list[tuple[str, str]]:
    return [(key, label) for key, label, _ in FACTORY]


def apply_factory(window, key: str) -> None:
    """Aplica una plantilla de fábrica a la ventana (KeyError si no existe)."""
    for k, _label, options in FACTORY:
        if k == key:
            _arrange(window, **options)
            return
    raise KeyError(f"Plantilla de disposición desconocida: {key}")


def reset(window) -> None:
    """Restablecer disposición: la de edición, como al instalar."""
    apply_factory(window, DEFAULT_LAYOUT)


# --- Las del usuario ----------------------------------------------------------------------------------
def layouts_dir() -> Path:
    return data_subdir("layouts")


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")[:60]


def list_user() -> list[str]:
    names = []
    for path in sorted(layouts_dir().glob("*.json")):
        try:
            names.append(str(json.loads(path.read_text(encoding="utf-8")).get("name") or path.stem))
        except (OSError, ValueError, AttributeError):
            continue
    return names


def _path(name: str) -> Path:
    return layouts_dir() / f"{safe_name(name)}.json"


def exists_user(name: str) -> bool:
    return bool(safe_name(name)) and _path(name).is_file()


def save_user(window, name: str) -> Path:
    """Guarda la disposición actual (estado de los paneles y, aparte, el tamaño y la posición de la ventana)."""
    if not safe_name(name):
        raise ValueError("El nombre de la disposición está vacío.")
    payload = {"name": name.strip(),
               "state": base64.b64encode(bytes(window.saveState())).decode("ascii"),
               "geometry": base64.b64encode(bytes(window.saveGeometry())).decode("ascii")}
    path = _path(name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Disposición guardada: %s", path)
    return path


def apply_user(window, name: str) -> bool:
    """Restaura una disposición del usuario. False si el archivo no existe o no se puede aplicar."""
    try:
        data = json.loads(_path(name).read_text(encoding="utf-8"))
        state = QByteArray(base64.b64decode(data["state"]))
        geometry = QByteArray(base64.b64decode(data["geometry"])) if data.get("geometry") else None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("No se pudo leer la disposición %s: %s", name, exc)
        return False
    if geometry is not None:
        window.restoreGeometry(geometry)
    return bool(window.restoreState(state))


def delete_user(name: str) -> bool:
    path = _path(name)
    if path.is_file():
        path.unlink()
        return True
    return False
