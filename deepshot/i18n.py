"""Idioma de la interfaz: español (el idioma de los textos del código) e inglés.

Los textos están escritos en español y marcados como traducibles (`self.tr`, `QCoreApplication.translate`,
`QT_TRANSLATE_NOOP`). El inglés es un archivo de traducción de Qt: `resources/translations/deepshot_en.ts` (fuente,
editable con Qt Linguist) compilado a `deepshot_en.qm` (el que carga la aplicación). Para volver a generarlos:
`python tools/update_translations.py`.

La elección se guarda en QSettings (`app/language`: «auto», «es» o «en»); sin nada guardado la aplicación
arranca en inglés (el español es opcional, decisión de 2026-10-20). Cambiarla en Preferencias se aplica al
reiniciar la aplicación: los textos ya creados no se retraducen en caliente.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QSettings, QTranslator

from deepshot.paths import resource

log = logging.getLogger(__name__)

LANGUAGES = ("es", "en")
AUTO = "auto"                    # el idioma del sistema
DEFAULT_CHOICE = "en"            # decisión de Juanjo (2026-10-20): la aplicación arranca en inglés; el español es opcional
SOURCE_LANGUAGE = "es"           # el idioma en el que están escritos los textos del código

_translators: list[QTranslator] = []        # se guardan: un QTranslator recogido por el recolector deja de traducir
_active = SOURCE_LANGUAGE


def translations_dir() -> Path:
    return resource("translations")


def saved_choice() -> str:
    """«auto», «es» o «en»: la variable de entorno DEEPSHOT_LANG (pruebas y soporte) manda sobre los ajustes."""
    forced = os.environ.get("DEEPSHOT_LANG", "").strip().lower()
    if forced in LANGUAGES:
        return forced
    choice = str(QSettings().value("app/language", DEFAULT_CHOICE))
    return choice if choice in (*LANGUAGES, AUTO) else DEFAULT_CHOICE


def save_choice(choice: str) -> None:
    QSettings().setValue("app/language", choice if choice in (*LANGUAGES, AUTO) else DEFAULT_CHOICE)


def resolve(choice: str, system_locale: str | None = None) -> str:
    """El idioma que se usará: el elegido o, en «auto», el del sistema (español si lo es; si no, inglés)."""
    if choice in LANGUAGES:
        return choice
    name = (system_locale or QLocale.system().name()).lower()
    return "es" if name.startswith("es") else "en"


def active() -> str:
    """El idioma con el que se instaló la interfaz en esta ejecución."""
    return _active


def install(app: QCoreApplication, choice: str | None = None) -> str:
    """Instala las traducciones del idioma elegido (por defecto, el de los ajustes) y devuelve el idioma usado.

    El español no necesita traductor propio (son los textos del código), pero sí las traducciones de Qt para sus
    diálogos estándar (botones de abrir y guardar, etc.) si están disponibles."""
    global _active
    for translator in _translators:
        app.removeTranslator(translator)
    _translators.clear()
    language = resolve(choice if choice is not None else saved_choice())
    _active = language
    if language == "en":
        _load(app, translations_dir(), "deepshot_en")
    else:
        _load(app, Path(QLibraryInfo.path(QLibraryInfo.TranslationsPath)), "qtbase_es")
        _load(app, Path(QLibraryInfo.path(QLibraryInfo.TranslationsPath)), "qt_es")
        pyside = _pyside_translations()
        if pyside is not None:
            _load(app, pyside, "qtbase_es")
    QLocale.setDefault(QLocale("en" if language == "en" else "es"))
    return language


def _pyside_translations() -> Path | None:
    try:
        import PySide6

        path = Path(PySide6.__file__).parent / "translations"
        return path if path.is_dir() else None
    except (ImportError, OSError):
        return None


def _load(app: QCoreApplication, folder: Path, name: str) -> bool:
    translator = QTranslator(app)
    if translator.load(name, str(folder)):
        app.installTranslator(translator)
        _translators.append(translator)
        return True
    log.info("Traducción no encontrada: %s en %s", name, folder)
    return False
