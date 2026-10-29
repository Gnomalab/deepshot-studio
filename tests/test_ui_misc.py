"""Rampa de color (doble clic y color en vivo), selector de presets y errores inesperados."""

import os
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from deepshot import errors  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.tonemap import ColorRamp, RampStop  # noqa: E402
from deepshot.ui import ramp_editor as re_mod  # noqa: E402
from deepshot.ui.color_pick import pick_color  # noqa: E402
from deepshot.ui.ramp_editor import HANDLE, RampBar, RampEditor  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return create_app([])


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))


def bar_with_ramp(app):
    bar = RampBar()
    bar.resize(300, 60)
    bar.set_ramp(ColorRamp([RampStop(0.0, (1, 1, 1)), RampStop(0.5, (0.5, 0.5, 0.5)), RampStop(1.0, (0, 0, 0))]))
    bar.show()
    app.processEvents()
    return bar


def handle_point(bar, i):
    r = bar._bar_rect()
    return QPoint(int(bar._x_of(bar.ramp.stops[i].position)), int(r.bottom() + HANDLE))


def test_double_click_on_a_handle_asks_for_its_color(app):
    bar = bar_with_ramp(app)
    asked = []
    bar.colorRequested.connect(asked.append)
    QTest.mouseDClick(bar, Qt.LeftButton, Qt.NoModifier, handle_point(bar, 1))
    assert asked == [1] and len(bar.ramp.stops) == 3                # no añade ninguna parada
    bar.deleteLater()


def test_double_click_on_the_gradient_adds_a_stop(app):
    bar = bar_with_ramp(app)
    asked = []
    bar.colorRequested.connect(asked.append)
    QTest.mouseDClick(bar, Qt.LeftButton, Qt.NoModifier, QPoint(int(bar._x_of(0.25)), 10))
    assert asked == [] and len(bar.ramp.stops) == 4
    assert bar.ramp.stops[-1].position == pytest.approx(0.25, abs=0.01)
    bar.deleteLater()


def test_color_pick_live_preview_then_accept_or_cancel(app, monkeypatch):
    ed = RampEditor()
    ramp = ColorRamp([RampStop(0.0, (1, 1, 1)), RampStop(1.0, (0, 0, 0))])
    ed.set_ramp(ramp)
    changes, commits = [], []
    ed.changed.connect(lambda: changes.append(tuple(ramp.stops[1].color)))
    ed.committed.connect(lambda: commits.append(1))

    def accepted(parent, initial, title, on_live):
        on_live((0.9, 0.1, 0.1))
        on_live((0.2, 0.8, 0.3))
        on_live((0.1, 0.2, 0.9))
        return (0.1, 0.2, 0.9)

    monkeypatch.setattr(re_mod, "pick_color", accepted)
    ed._pick_color(1)
    assert changes == [(0.9, 0.1, 0.1), (0.2, 0.8, 0.3), (0.1, 0.2, 0.9)]    # la imagen cambia mientras se elige
    assert ramp.stops[1].color == (0.1, 0.2, 0.9) and commits == [1]          # y se apila un solo deshacer

    def cancelled(parent, initial, title, on_live):
        on_live((0.5, 0.5, 0.5))
        on_live(initial)                                                      # al cancelar se restaura
        return None

    monkeypatch.setattr(re_mod, "pick_color", cancelled)
    commits.clear()
    ed._pick_color(1)
    assert ramp.stops[1].color == (0.1, 0.2, 0.9) and commits == []
    ed.deleteLater()


def test_pick_color_dialog_reports_live_changes_and_restores_on_cancel(app, monkeypatch):
    """El diálogo real: cada color elegido se comunica y cancelar devuelve el inicial."""
    from PySide6.QtWidgets import QColorDialog

    seen = []

    def fake_exec(self):
        self.setCurrentColor(QColor.fromRgbF(0.2, 0.4, 0.6))
        self.setCurrentColor(QColor.fromRgbF(0.8, 0.1, 0.1))
        return QColorDialog.Rejected

    monkeypatch.setattr(QColorDialog, "exec", fake_exec)
    assert pick_color(None, (0.5, 0.5, 0.5), "t", seen.append) is None
    assert len(seen) >= 3 and seen[-1] == (0.5, 0.5, 0.5)                    # último aviso: el color original
    assert seen[0][0] == pytest.approx(0.2, abs=0.01)

    seen.clear()
    monkeypatch.setattr(QColorDialog, "exec", lambda self: (self.setCurrentColor(QColor.fromRgbF(0.3, 0.6, 0.9)),
                                                              QColorDialog.Accepted)[1])
    got = pick_color(None, (0.5, 0.5, 0.5), "t", seen.append)
    assert got == pytest.approx((0.3, 0.6, 0.9), abs=0.01) and seen[-1] == got


def test_ramp_preset_bar_applies_and_user_presets_are_listed(app, monkeypatch):
    ed = RampEditor()
    ramp = ColorRamp()
    ed.set_ramp(ramp)
    done = []
    ed.committed.connect(lambda: done.append(1))
    bar = ed.preset
    idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("builtin", "contrast"))
    bar.combo.activated.emit(idx)
    assert len(ramp.stops) == 4 and ramp.interpolation == "linear" and done == [1]
    assert bar.combo.currentIndex() == idx                                    # se queda mostrando el preset elegido
    assert bar.save_current("Mi rampa")
    assert any(bar.combo.itemData(i) == ("user", "Mi rampa") for i in range(bar.combo.count()))
    ramp.stops[:] = [RampStop(0.0, (0, 0, 0))]
    idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("user", "Mi rampa"))
    bar.combo.activated.emit(idx)
    assert len(ramp.stops) == 4
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    bar.delete("Mi rampa")
    assert not any(bar.combo.itemData(i) == ("user", "Mi rampa") for i in range(bar.combo.count()))
    ed.deleteLater()


def test_preset_bar_overwrite_asks_and_empty_name_is_refused(app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    ed = RampEditor()
    ed.set_ramp(ColorRamp())
    bar = ed.preset
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    assert not bar.save_current("   ")
    assert bar.save_current("X")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    assert not bar.save_current("X")                                          # no sustituye si se responde que no
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    assert bar.save_current("X")
    ed.deleteLater()


# --- Errores inesperados -------------------------------------------------------------------------------
def test_error_reporter_logs_emits_and_deduplicates(app, caplog):
    rep = errors.ErrorReporter()
    got = []
    rep.reported.connect(lambda summary, trace: got.append((summary, trace)))
    try:
        raise ValueError("boom")
    except ValueError:
        et, ev, tb = sys.exc_info()
    with caplog.at_level("ERROR"):
        rep.handle(et, ev, tb)
        rep.handle(et, ev, tb)                       # el mismo error seguido: se registra pero no se repite el aviso
    assert len(got) == 1 and "ValueError: boom" in got[0][0] and "Traceback" in got[0][1]
    assert caplog.text.count("Error inesperado") == 2
    try:
        raise KeyError("otro")
    except KeyError:
        rep.handle(*sys.exc_info())
    assert len(got) == 2


def test_hooks_capture_main_and_thread_exceptions(app):
    rep = errors.ErrorReporter()
    got = []
    rep.reported.connect(lambda s, t: got.append(s))
    errors.install(rep)
    try:
        try:
            raise RuntimeError("en el principal")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())

        def work():
            raise OSError("en un hilo")

        t = threading.Thread(target=work)
        t.start()
        t.join()
        app.processEvents()
        assert any("en el principal" in s for s in got) and any("en un hilo" in s for s in got)
    finally:
        errors.uninstall()
    assert sys.excepthook is errors._original_excepthook


def test_preset_menu_save_action_asks_for_the_name(app, monkeypatch):
    """Dispara la acción real del menú: Qt le pasa `checked` (False) y eso no debe tomarse como el nombre."""
    from PySide6.QtWidgets import QInputDialog, QMenu

    ed = RampEditor()
    ed.set_ramp(ColorRamp())
    bar = ed.preset
    asked = []
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: (asked.append(1), ("Desde el menu", True))[1])

    class AutoMenu(QMenu):
        def exec(self, *args, **kwargs):                  # en vez de abrirse, pulsa «Guardar…»
            self.actions()[0].trigger()

    import deepshot.ui.preset_bar as preset_bar
    monkeypatch.setattr(preset_bar, "QMenu", AutoMenu)
    bar._show_menu()
    assert asked == [1]
    assert any(bar.combo.itemData(i) == ("user", "Desde el menu") for i in range(bar.combo.count()))
    from deepshot import presets
    presets.delete_user(bar.kind, "Desde el menu")
    ed.deleteLater()


# --- Preset elegido y vista previa al pasar el ratón ----------------------------------------------------------
def test_preset_bar_keeps_showing_the_chosen_preset_until_settings_change(app):
    from deepshot.ui.preset_bar import PresetBar

    state = {"v": 0}
    applied = []
    bar = PresetBar("ramp", "Cargar…", lambda: {"v": state["v"]})
    bar.chosen.connect(lambda preset: (applied.append(preset), state.update(v=1)))     # «aplicar» cambia los ajustes
    idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("builtin", "contrast"))
    bar.combo.activated.emit(idx)
    assert applied and bar.combo.currentIndex() == idx                                  # se queda mostrando su nombre
    assert bar.combo.currentText() == "Contraste fuerte"
    state["v"] = 2                                                                      # el usuario toca algo
    bar.sync()
    assert bar.combo.currentIndex() == 0                                                # vuelve al texto guía
    state["v"] = 1                                                                      # y lo deshace
    bar.sync()
    assert bar.combo.currentIndex() == idx                                              # el nombre vuelve
    bar.refresh()                                                                       # rehacer la lista no lo pierde
    assert bar.combo.currentIndex() == idx
    bar.combo.activated.emit(0)                                                         # elegir el texto guía no hace nada
    assert len(applied) == 1


def test_preset_bar_previews_on_hover_and_only_ends_the_preview_without_a_choice(app):
    from PySide6.QtTest import QTest

    from deepshot.ui.preset_bar import PresetBar

    bar = PresetBar("ramp", "Cargar…", lambda: {"v": 0})
    events = []
    bar.previewed.connect(lambda preset: events.append(("preview", sorted(preset))))
    bar.previewEnded.connect(lambda: events.append(("end",)))
    bar.chosen.connect(lambda preset: events.append(("chosen",)))
    idx = next(i for i in range(bar.combo.count()) if bar.combo.itemData(i) == ("builtin", "haze"))
    bar.combo.highlighted.emit(idx)                                                     # el ratón pasa por encima
    assert events == [("preview", ["ramp"])]
    bar.combo.highlighted.emit(0)                                                       # y sobre el texto guía: se acaba
    assert events[-1] == ("end",)
    events.clear()
    bar.combo.highlighted.emit(idx)
    QTest.qWait(300)                                                                    # la lista no está abierta: se acabó
    assert events == [("preview", ["ramp"]), ("end",)]
    events.clear()
    bar.combo.highlighted.emit(idx)
    bar.combo.activated.emit(idx)                                                       # elige: no hay «fin», se aplica
    QTest.qWait(300)
    assert events == [("preview", ["ramp"]), ("chosen",)]


def test_wheel_over_combos_and_spin_boxes_scrolls_the_panel_instead_unless_ctrl_is_held(app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QScrollArea, QVBoxLayout, QWidget

    from deepshot.ui import wheel_guard

    guard = wheel_guard.install(QApplication.instance())
    try:
        page = QWidget()
        layout = QVBoxLayout(page)
        combo = QComboBox()
        combo.addItems(["a", "b", "c"])
        spin = QDoubleSpinBox()
        layout.addWidget(combo)
        layout.addWidget(spin)
        for _ in range(40):
            layout.addWidget(QComboBox())                                   # relleno para que haya scroll
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        area.resize(300, 200)
        area.show()
        QApplication.processEvents()

        def wheel(widget, mods=Qt.NoModifier, delta=-120):
            ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, delta), Qt.NoButton, mods,
                             Qt.NoScrollPhase, False)
            QApplication.sendEvent(widget, ev)

        wheel(combo)
        wheel(spin)
        assert combo.currentIndex() == 0 and spin.value() == 0.0                # sin Ctrl: no cambian
        assert area.verticalScrollBar().value() > 0                              # y el panel se desplazó
        wheel(combo, Qt.ControlModifier)
        wheel(spin, Qt.ControlModifier, 120)
        assert combo.currentIndex() == 1 and spin.value() != 0.0                 # con Ctrl sí cambian
        area.close()
    finally:
        QApplication.instance().removeEventFilter(guard)
