"""Idioma de la interfaz: el inglés está completo y coherente con el español, y se elige y se instala bien."""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402

from deepshot import i18n  # noqa: E402

TS = Path(__file__).resolve().parents[1] / "resources" / "translations" / "deepshot_en.ts"
SPANISH_ONLY = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]")
PLACEHOLDER = re.compile(r"\{[^{}]*\}|%[sdf%]|%\d+")
TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*")


def ts_messages():
    root = ET.parse(TS).getroot()
    for ctx in root.findall("context"):
        for m in ctx.findall("message"):
            yield ctx.findtext("name"), m.findtext("source") or "", m.find("translation")


def test_every_message_is_translated_and_keeps_placeholders_and_html():
    messages = list(ts_messages())
    assert len(messages) > 700                                             # la extracción no se quedó corta
    for ctx, source, translation in messages:
        text = translation.text or ""
        where = f"{ctx}: {source[:60]!r}"
        assert translation.get("type") != "unfinished" and text.strip(), where        # nada sin traducir
        assert sorted(PLACEHOLDER.findall(source)) == sorted(PLACEHOLDER.findall(text)), where   # {0}, {app}, %s…
        assert sorted(TAG.findall(source)) == sorted(TAG.findall(text)), where         # etiquetas HTML iguales
        assert not SPANISH_ONLY.search(text), f"queda español en la traducción: {where}"
        assert source.count("&") == text.count("&") or "&" not in source, where       # atajos de menú (&File)


def test_compiled_translation_loads_and_translates_known_texts():
    from PySide6.QtCore import QTranslator

    translator = QTranslator()
    assert translator.load("deepshot_en", str(i18n.translations_dir()))
    assert translator.translate("MainWindow", "&Archivo") == "&File"
    assert translator.translate("Help", "Lienzo") == "Canvas"
    assert translator.translate("Models", "No se pudo copiar el modelo: {0}") == "The model could not be copied: {0}"
    assert translator.translate("MainWindow", "texto que no existe") == ""


def test_language_choice_resolution_and_environment_override(monkeypatch, tmp_path):
    from deepshot.app import create_app

    QSettings.setDefaultFormat(QSettings.IniFormat)                     # los ajustes de la prueba, no los del usuario
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    create_app([])
    assert i18n.resolve("es") == "es" and i18n.resolve("en") == "en"
    assert i18n.resolve("auto", "es_ES") == "es" and i18n.resolve("auto", "es_MX") == "es"
    assert i18n.resolve("auto", "en_US") == "en" and i18n.resolve("auto", "de_DE") == "en"     # otro idioma: inglés
    monkeypatch.setenv("DEEPSHOT_LANG", "en")
    assert i18n.saved_choice() == "en"
    monkeypatch.delenv("DEEPSHOT_LANG")
    assert i18n.saved_choice() == "en"                                  # sin nada guardado: inglés (decisión del autor)
    monkeypatch.setenv("DEEPSHOT_LANG", "klingon")
    QSettings().setValue("app/language", "es")
    assert i18n.saved_choice() == "es"                                                          # valor raro del entorno: se ignora
    QSettings().setValue("app/language", "raro")
    assert i18n.saved_choice() == "en"
    QSettings().setValue("app/language", "auto")
    assert i18n.saved_choice() == "auto"                                # «Automático» sigue siendo una opción
    QSettings().remove("app/language")


def test_install_switches_texts_and_back():
    from deepshot.app import create_app

    app = create_app([])
    try:
        assert i18n.install(app, "en") == "en" and i18n.active() == "en"
        assert QCoreApplication.translate("MainWindow", "&Archivo") == "&File"
        assert QCoreApplication.translate("Depth", "Rápida") == "Fast"
        assert QCoreApplication.translate("MainWindow", "texto sin traducción") == "texto sin traducción"   # no inventa
        assert i18n.install(app, "es") == "es"
        assert QCoreApplication.translate("MainWindow", "&Archivo") == "&Archivo"
    finally:
        i18n.install(app, "es")


# --- Ninguna cadena de la interfaz queda en español al usar el inglés ---------------------------------------------
LEFTOVER = re.compile(r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(el|la|los|las|para|con|sin|una|que|por|hay|foto|mapa|profundidad|cálculo|"
                      r"carpeta|archivo|guardar|cargar|calcular|ajustes)\b", re.I)
ALLOWED = {"Español"}                                   # el nombre de cada idioma va en ese idioma


def collect_texts(root):
    from PySide6.QtWidgets import QComboBox, QListWidget, QMenu, QWidget

    texts = []
    for widget in [root, *root.findChildren(QWidget)]:
        for attr in ("text", "toolTip", "windowTitle", "placeholderText", "statusTip"):
            fn = getattr(widget, attr, None)
            if callable(fn):
                try:
                    value = fn()
                except TypeError:
                    continue
                if isinstance(value, str):
                    texts.append(value)
        if isinstance(widget, QComboBox):
            texts += [widget.itemText(i) for i in range(widget.count())]
        if isinstance(widget, QListWidget):
            texts += [widget.item(i).text() for i in range(widget.count())]
        if isinstance(widget, QMenu):
            def walk(menu):
                for action in menu.actions():
                    texts.extend([action.text(), action.toolTip()])
                    if action.menu():
                        action.menu().aboutToShow.emit()
                        walk(action.menu())
            walk(widget)
    return texts


def test_english_ui_has_no_spanish_text_left(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QTextBrowser

    from deepshot.app import create_app
    from deepshot.ui.about import AboutDialog
    from deepshot.ui.help import GuideDialog, ShortcutsDialog
    from deepshot.ui.main_window import MainWindow
    from deepshot.ui.preferences import PreferencesDialog
    from deepshot.ui.start_dialog import StartDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    app = create_app([])
    i18n.install(app, "en")
    window = None
    try:
        window = MainWindow()
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        texts = []
        for step in range(4):                                           # cada paso enseña sus propios paneles
            window.set_step(step)
            app.processEvents()
            texts += collect_texts(window)
        for action in window.menuBar().actions():
            texts.append(action.text())
        texts += [dock.windowTitle() for dock in (window.library_dock, window.settings_dock, window.log_dock)]
        info = {"app": "Deepshot Studio", "tagline": "tag", "ver": "1", "url": "u", "org": "o", "author": "a", "date": "d"}
        dialogs = [PreferencesDialog(window), StartDialog(["x.deepshot"], window), AboutDialog(info, window),
                   GuideDialog(window), ShortcutsDialog(window)]
        for dialog in dialogs:
            texts += collect_texts(dialog)
            for browser in dialog.findChildren(QTextBrowser):
                texts.append(re.sub(r"<[^>]+>", " ", browser.toPlainText()))
        # Algunas palabras inglesas coinciden con las españolas de la lista («Del», «se»): solo cuentan las que
        # además llevan acentos o letras propias del español, o las frases de la lista sin esas coincidencias.
        left = sorted({t[:100] for t in texts if t and t not in ALLOWED and LEFTOVER.search(re.sub(r"\bDel\b", "", t))
                       and not re.search(r"\b(the|and|of|is|to|in|with)\b", t, re.I)})
        assert not left, left
    finally:
        if window is not None:
            window.ask_before_closing = False
            window._probe.wait(120)
            window.close()
            window.editor.shutdown()
        i18n.install(app, "es")
