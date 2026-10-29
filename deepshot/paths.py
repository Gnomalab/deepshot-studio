"""Rutas de recursos, válidas en desarrollo y en el ejecutable de PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path


def resources_dir() -> Path:
    """Carpeta `resources/` (tema, iconos).

    En desarrollo está junto al paquete; dentro de un ejecutable de PyInstaller,
    los datos se descomprimen en `sys._MEIPASS`.
    """
    bundle = getattr(sys, "_MEIPASS", None)
    base = Path(bundle) if bundle else Path(__file__).resolve().parent.parent
    return base / "resources"


def resource(*parts: str) -> Path:
    """Ruta a un recurso concreto, p. ej. resource("icons", "open.svg")."""
    return resources_dir().joinpath(*parts)


def app_data_dir() -> Path:
    """Carpeta local de datos de la app (pesos del modelo, caché de profundidad).

    Windows: %LOCALAPPDATA%/Gnomalab Estudio/Deepshot Studio
    macOS:   ~/Library/Application Support/Deepshot Studio
    Otros:   ~/.local/share/deepshot-studio
    Se puede forzar otra ruta con la variable de entorno DEEPSHOT_DATA_DIR.
    Si existe la carpeta del nombre anterior de la app, se renombra (conserva los modelos).
    """
    import os

    from deepshot import APP_NAME, LEGACY_APP_NAME

    override = os.environ.get("DEEPSHOT_DATA_DIR")
    if override:
        path = Path(override)
    else:
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            parent, name, old = Path(base) / "Gnomalab Estudio", APP_NAME, LEGACY_APP_NAME
        elif sys.platform == "darwin":
            parent, name, old = Path.home() / "Library" / "Application Support", APP_NAME, LEGACY_APP_NAME
        else:
            parent = Path.home() / ".local" / "share"
            name, old = APP_NAME.lower().replace(" ", "-"), LEGACY_APP_NAME.lower()
        path = parent / name
        _migrate_legacy(parent / old, path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _migrate_legacy(old: Path, new: Path) -> None:
    """Renombra la carpeta de datos del nombre anterior al nuevo, una sola vez.

    Solo si existe la antigua y aún no la nueva. Los marcadores de descarga de los modelos
    guardan rutas relativas, así que los modelos siguen funcionando tras el cambio.
    """
    if old.is_dir() and not new.exists():
        try:
            old.rename(new)
        except OSError:
            pass   # p. ej. un archivo bloqueado: se usa la carpeta nueva y no se pierde nada


def data_subdir(name: str) -> Path:
    """Subcarpeta de la carpeta de datos (se crea si no existe): logs, masks, presets…"""
    path = app_data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file() -> Path:
    """Archivo donde se guarda el registro de actividad (para diagnosticar fallos)."""
    return data_subdir("logs") / "deepshot.log"
