"""Controles propios: campo numérico arrastrable, biblioteca, registro y presets."""

import logging
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from deepshot import presets  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.project import Project  # noqa: E402
from deepshot.ui.log_panel import LogPanel  # noqa: E402
from deepshot.ui.value_slider import ValueSlider  # noqa: E402
from deepshot.ui.widgets import LibraryList  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return create_app([])


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))


def _pair(app):
    box = QWidget()
    layout = QVBoxLayout(box)
    a, b = ValueSlider("A", 0.5), ValueSlider("B", 0.5)
    layout.addWidget(a)
    layout.addWidget(b)
    box.resize(300, 100)
    box.show()
    app.processEvents()
    return box, a, b


# --- ValueSlider: el campo de texto nunca debe quedarse tapando el control ---------------------
def test_click_opens_editor_and_enter_applies(app):
    box, a, _b = _pair(app)
    changes = []
    a.valueChanged.connect(changes.append)
    QTest.mouseClick(a, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    assert a.is_editing()
    a._editor.setText("0,75")                       # coma decimal admitida
    QTest.keyClick(a._editor, Qt.Key_Return)
    assert not a.is_editing() and a.value() == 0.75 and changes == [0.75]
    box.deleteLater()


def test_escape_cancels_editor(app):
    box, a, _b = _pair(app)
    QTest.mouseClick(a, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    a._editor.setText("0.9")
    QTest.keyClick(a._editor, Qt.Key_Escape)
    assert not a.is_editing() and a.value() == 0.5
    box.deleteLater()


def test_editor_closes_when_another_slider_is_clicked(app):
    """Error visto en uso real: varios deslizadores se quedaban con el campo de texto encima."""
    box, a, b = _pair(app)
    QTest.mouseClick(a, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    assert a.is_editing()
    QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    assert not a.is_editing() and b.is_editing()
    box.deleteLater()


def test_editor_closes_when_hidden_or_disabled(app):
    box, a, b = _pair(app)
    QTest.mouseClick(a, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    a.setEnabled(False)
    assert not a.is_editing()
    QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, QPoint(20, 10))
    box.hide()
    app.processEvents()
    assert not b.is_editing()
    box.deleteLater()


def test_drag_changes_value_without_opening_editor(app):
    box, a, _b = _pair(app)
    press = QPoint(10, 10)
    QTest.mousePress(a, Qt.LeftButton, Qt.NoModifier, press)
    QTest.mouseMove(a, QPoint(100, 10))
    QTest.mouseRelease(a, Qt.LeftButton, Qt.NoModifier, QPoint(100, 10))
    assert a.value() > 0.5 and not a.is_editing()
    box.deleteLater()


# --- Biblioteca -------------------------------------------------------------------------------
def test_library_double_click_on_empty_space_requests_import(app):
    lib = LibraryList()
    lib.resize(200, 200)
    lib.show()
    got = []
    lib.importRequested.connect(lambda: got.append(1))
    QTest.mouseDClick(lib.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    assert got == [1]
    lib.deleteLater()


def test_library_delete_key_requests_removal(app):
    from PySide6.QtWidgets import QListWidgetItem

    lib = LibraryList()
    for name in ("a.png", "b.png"):
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        lib.addItem(item)
    lib.show()
    lib.item(1).setSelected(True)
    removed = []
    lib.removeRequested.connect(removed.append)
    QTest.keyClick(lib, Qt.Key_Delete)
    assert removed == [["b.png"]]
    lib.deleteLater()


# --- Registro --------------------------------------------------------------------------------------
def test_log_panel_clear_copy_and_reload(app, tmp_path):
    from deepshot.ui.log_panel import setup_file_logging

    setup_file_logging()
    panel = LogPanel()
    log = logging.getLogger("deepshot.test_widgets")
    log.setLevel(logging.INFO)
    log.info("mensaje de prueba 123")
    app.processEvents()
    assert "mensaje de prueba 123" in panel.text.toPlainText()
    panel.copy_all()
    assert "mensaje de prueba 123" in QApplication.clipboard().text()
    panel.clear()
    assert panel.text.toPlainText() == ""
    for h in logging.getLogger().handlers:
        h.flush()
    panel.reload_from_file()
    assert "mensaje de prueba 123" in panel.text.toPlainText()    # recuperado del archivo
    panel.detach()
    panel.deleteLater()


# --- Presets ---------------------------------------------------------------------------------------
def test_every_builtin_preset_is_valid():
    for kind in presets.KINDS:
        for key, _name in presets.builtin_names(kind):
            out = presets.apply_to_project_dict(kind, Project().to_dict(), presets.builtin(kind, key))
            Project.from_dict(out).validate()


def test_user_presets_roundtrip_and_delete():
    p = Project()
    p.ramp.stops[0].color = (0.2, 0.4, 0.6)
    presets.save_user("ramp", "Mi rampa: prueba/1", presets.from_project("ramp", p))
    assert "Mi rampa_ prueba_1" in presets.list_user("ramp")      # nombre saneado para el archivo
    loaded = presets.load_user("ramp", "Mi rampa: prueba/1")
    assert loaded["ramp"]["stops"][0]["color"] == [0.2, 0.4, 0.6]
    assert presets.list_user("style") == []                        # cada tipo tiene su carpeta
    assert presets.delete_user("ramp", "Mi rampa: prueba/1") and presets.list_user("ramp") == []


def test_invalid_user_preset_is_reported():
    (presets.user_path("style", "roto")).parent.mkdir(parents=True, exist_ok=True)
    presets.user_path("style", "roto").write_text("{no es json", encoding="utf-8")
    with pytest.raises(ValueError):
        presets.load_user("style", "roto")
    with pytest.raises(ValueError):
        presets.save_user("style", "   ", {})


def test_style_keeps_user_areas_and_painted_mask_but_resets_look():
    p = Project()
    p.areas.append(__import__("deepshot.areas", fromlist=["Area"]).Area(name="mía"))
    p.composite.painted_mask = "/x/mascara.png"
    p.composite.mode = "multiply"
    p.bands = []
    base = p.to_dict()
    out = presets.apply_to_project_dict("style", base, presets.builtin("style", "haze"))
    assert [a["name"] for a in out["areas"]] == ["mía"]            # sus áreas se conservan
    assert out["composite"]["painted_mask"] == "/x/mascara.png"    # y su máscara pintada
    assert out["composite"]["mode"] == "normal"                    # el resto de la mezcla vuelve a su valor
    assert base["composite"]["mode"] == "multiply"                 # sin tocar el original
