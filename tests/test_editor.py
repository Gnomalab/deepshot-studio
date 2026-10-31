"""Fase 3: sesión de edición (proyecto por foto, deshacer, colocar áreas, vista previa)."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtGui import QUndoStack  # noqa: E402

from deepshot.app import create_app  # noqa: E402
from deepshot.ui.editor import EditSession  # noqa: E402


import pytest  # noqa: E402

_sessions = []


@pytest.fixture(autouse=True)
def _stop_threads():
    """Cada test detiene los hilos de render de las sesiones que creó."""
    yield
    from PySide6.QtCore import QCoreApplication, QEvent

    while _sessions:
        session = _sessions.pop()
        session.shutdown()
        # Borrado ordenado por Qt (como hace la ventana al cerrarse), no por el recolector de Python.
        for panel in session.panels():
            panel.deleteLater()
        session.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _setup(tmp_path):
    app = create_app([])
    photo = tmp_path / "foto prueba.png"
    Image.new("RGB", (120, 80), (90, 120, 160)).save(photo)
    depth = np.tile(np.linspace(0.1, 0.9, 120, dtype=np.float32), (80, 1))
    stack = QUndoStack()
    session = EditSession(stack)
    session.set_photo(photo, depth)
    _sessions.append(session)
    return app, session, stack, photo, depth


def _wait_result(app, session, timeout=5.0):
    session.result = None
    t = time.time()
    while session.result is None and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return session.result


def test_new_project_fits_ramp_and_renders(tmp_path):
    app, session, *_ = _setup(tmp_path)
    p = session.project
    assert p is not None and 0.1 <= p.ramp.range_min < p.ramp.range_max <= 0.9
    session._on_changed()
    img = _wait_result(app, session)
    assert img is not None and img.width() == 120 and img.height() == 80


def test_edit_commit_undo_redo(tmp_path):
    app, session, stack, *_ = _setup(tmp_path)
    session.mix_panel.opacity.setValue(0.5, emit=True)
    session.mix_panel.opacity.editingFinished.emit()
    assert stack.count() == 1 and session.project.composite.opacity == 0.5
    stack.undo()
    assert session.project.composite.opacity == 0.0
    assert session.mix_panel.opacity.value() == 0.0          # el panel se actualiza
    stack.redo()
    assert session.project.composite.opacity == 0.5


def test_no_undo_step_without_real_change(tmp_path):
    _app, session, stack, *_ = _setup(tmp_path)
    session._on_committed()
    assert stack.count() == 0


def test_place_area_uses_depth_under_click(tmp_path):
    _app, session, *_ = _setup(tmp_path)
    assert session.place_area_at(0.25, 0.5)                   # crea un área si no hay
    area = session.project.areas[0]
    expected_z = session.render.preview_depth()[40, 30] * session.project.scene.relief
    assert abs(area.center[2] - expected_z) < 1e-5
    ov = session.overlays()[0]
    assert abs(ov["cx"] - 0.25 * 120) < 1.0 and abs(ov["cy"] - 40) < 1.0   # contorno donde se hizo clic


def test_each_photo_keeps_its_own_project(tmp_path):
    _app, session, _stack, photo, depth = _setup(tmp_path)
    session.project.base = "black"
    other = tmp_path / "otra.png"
    Image.new("RGB", (120, 80), (10, 10, 10)).save(other)
    session.set_photo(other, depth)
    assert session.project.base == "ramp"
    session.set_photo(photo, depth)
    assert session.project.base == "black"


def test_settings_json_roundtrip(tmp_path):
    _app, session, stack, *_ = _setup(tmp_path)
    session.project.bands.append(session.bands_panel.new_item())
    path = tmp_path / "ajustes.json"
    session.save_settings(path)
    session.project.bands.clear()
    session.load_settings(path)
    assert len(session.project.bands) == 1 and stack.count() == 1


# --- Look: pila de capas de ajuste -------------------------------------------
def test_look_panel_estilo_photoshop_orden_arrastrar_renombrar_apagar_y_solo(tmp_path):
    from PySide6.QtCore import QModelIndex, Qt

    from deepshot.look import LookLayer

    app, session, stack, *_ = _setup(tmp_path)
    panel, p = session.look_panel, session.project
    assert p.look == [] and panel.isEnabled()
    panel.add_item(LookLayer.new("levels"))
    panel.add_item(LookLayer.new("saturation"))
    assert [l.kind for l in p.look] == ["saturation", "levels"] and panel.list.currentRow() == 0   # lo nuevo entra arriba
    panel.list.setCurrentRow(1)
    panel._move(-1)                                              # sube los niveles: se aplican después de la saturación
    assert [l.kind for l in p.look] == ["levels", "saturation"] and panel.list.currentRow() == 0
    assert not panel.up_button.isEnabled() and panel.down_button.isEnabled()
    panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), 2)    # arrastrar la fila de arriba al final
    app.processEvents()
    assert [l.kind for l in p.look] == ["saturation", "levels"] and panel.list.currentRow() == 1
    assert [panel.list.item(i).text() for i in range(2)] == ["Saturación", "Niveles"]      # la lista sigue al proyecto

    entry = panel.list.item(0)                                   # renombrar (doble clic o F2 en la lista)
    entry.setText("Color de la piel")
    assert p.look[0].name == "Color de la piel" and panel.list.item(0).text() == "Color de la piel"
    panel.list.item(0).setText("")                               # un nombre vacío vuelve al del efecto
    assert p.look[0].name == "" and panel.list.item(0).text() == "Saturación"
    panel.toggle_row_enabled(0)                                  # el ojo de la fila: bypass (también se exporta así)
    assert p.look[0].enabled is False
    panel.toggle_row_enabled(0)
    assert p.look[0].enabled is True

    panel.list.setCurrentRow(0)
    panel.param_sliders["amount"].setValue(-1.0, emit=True)      # el deslizador escribe en los ajustes del efecto
    assert p.look[0].params["amount"] == -1.0
    panel.limit.set_current_index(1)
    panel._on_limit(1)
    assert p.look[0].limit == "depth" and panel.range_box.isVisibleTo(panel)
    panel._duplicate()                                           # la copia queda justo encima, con su nombre
    assert [l.kind for l in p.look] == ["saturation", "saturation", "levels"] and p.look[0].params == p.look[1].params
    panel._remove()
    assert [l.kind for l in p.look] == ["saturation", "levels"]


def test_look_cambia_la_imagen_final(tmp_path):
    from deepshot.look import LookLayer

    app, session, *_ = _setup(tmp_path)
    before = _wait_result(app, session).pixelColor(60, 40).value()
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.6                                  # sube los negros: la imagen se oscurece
    session.project.look.append(layer)
    session.look_panel.refresh_list(select=0)
    session.look_panel.emit_both()
    after = _wait_result(app, session).pixelColor(60, 40).value()
    assert after < before


def test_a_look_preset_replaces_the_stack_in_one_undo_step_and_previews_on_hover(tmp_path):
    from deepshot import presets
    from deepshot.look import LookLayer

    app, session, stack, *_ = _setup(tmp_path)
    p = session.project
    session.look_panel.add_item(LookLayer.new("levels"))
    steps = stack.index()
    sent = []
    session.render.request = lambda project, mask=None: sent.append(project)
    preset = presets.builtin("look", "old_film")
    session.preview_preset("look", preset)                                   # al pasar el ratón: se ve sin aplicar
    assert [l.kind for l in sent[-1].look] == [l["kind"] for l in preset["look"]]
    assert [l.kind for l in session.project.look] == ["levels"]
    session.look_panel.preset_bar.chosen.emit(preset)                        # al elegirlo: sustituye la pila
    assert [l.kind for l in session.project.look] == [l["kind"] for l in preset["look"]]
    assert stack.index() == steps + 1
    stack.undo()
    assert [l.kind for l in session.project.look] == ["levels"]
