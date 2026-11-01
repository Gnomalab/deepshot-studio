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


def test_pre_and_post_fx_are_two_independent_stacks_with_undo_solo_and_presets(tmp_path):
    from deepshot import presets
    from deepshot.look import LookLayer

    app, session, stack, *_ = _setup(tmp_path)
    p = session.project
    assert p.pre == [] and p.look == [] and session.look_stack.current_panel() is session.look_panel
    session.look_stack.show_stack("pre")
    assert session.look_stack.current_panel() is session.pre_panel
    session.pre_panel.add_item(LookLayer.new("saturation"))                  # entra en Pre FX, no en Post FX
    assert [l.kind for l in p.pre] == ["saturation"] and p.look == []
    session.look_panel.add_item(LookLayer.new("levels"))
    assert [l.kind for l in p.look] == ["levels"] and len(p.pre) == 1
    # Solo: cada pila tiene el suyo y la vista previa lo respeta.
    p.pre[0].enabled = False
    session.pre_panel.toggle_row_solo(0)
    assert session._preview_project().pre[0].enabled is True
    session.pre_panel.toggle_row_solo(0)
    # Un preset de look se aplica a la pila de su pestaña, con un solo paso de deshacer.
    preset = presets.builtin("look", "old_film")
    steps = stack.index()
    session.pre_panel.preset_bar.chosen.emit(preset)
    p = session.project                                                      # el preset rehace el proyecto
    assert [l.kind for l in p.pre] == [l["kind"] for l in preset["look"]] and [l.kind for l in p.look] == ["levels"]
    assert stack.index() == steps + 1
    stack.undo()
    assert [l.kind for l in session.project.pre] == ["saturation"]
    # Vista previa al pasar el ratón por un preset de Pre FX: solo cambia Pre.
    sent = []
    session.render.request = lambda project, mask=None: sent.append(project)
    session.preview_preset("look", preset, "pre")
    assert [l.kind for l in sent[-1].pre] == [l["kind"] for l in preset["look"]] and [l.kind for l in sent[-1].look] == ["levels"]
    # Se guardan y se recuperan en el JSON del proyecto (y un proyecto antiguo, sin «pre», sigue abriendo).
    p = session.project
    again = type(p).from_dict(p.to_dict())
    assert [l.kind for l in again.pre] == ["saturation"]
    old = p.to_dict()
    old.pop("pre")
    assert type(p).from_dict(old).pre == []


def test_limit_to_a_map_in_the_panel_computes_on_demand_and_applies_in_preview_and_export(tmp_path, monkeypatch):
    from deepshot import export as ex
    from deepshot import maps
    from deepshot.look import LookLayer
    from deepshot.project import Project

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))       # caché de mapas vacía
    app, session, stack, photo, depth = _setup(tmp_path)
    subject = np.zeros((80, 120), np.float32)
    subject[:, 60:] = 1.0                                        # el sujeto es la mitad derecha
    monkeypatch.setattr(maps, "sam_installed", lambda: True)
    monkeypatch.setattr(maps, "segment_at", lambda image, point: np.stack([subject]))
    panel = session.look_panel
    layer = LookLayer.new("levels")
    layer.params["black"] = 0.9
    panel.add_item(layer)
    panel._on_limit(2)                                           # «Mapa»
    assert layer.limit == "mask" and layer.mask_id == "subject"
    assert panel.map_state["subject"] == "missing" and "subject" not in session.current_maps()
    dark_before = _wait_result(app, session)
    assert dark_before.pixelColor(100, 40).value() > 30          # sin mapa calculado, el efecto no actúa
    panel.map_button.click()                                     # «Calcular sujeto»: en segundo plano
    t = time.time()
    while "subject" not in session.current_maps() and time.time() - t < 10:
        app.processEvents()
        time.sleep(0.01)
    assert panel.map_state["subject"] == "ready" and session.pre_panel.map_state["subject"] == "ready"
    session.result = None
    session._request_render()
    img = _wait_result(app, session)
    assert img.pixelColor(100, 40).value() < 15 and img.pixelColor(20, 40).value() > 30     # solo la mitad derecha
    layer.mask_invert = True
    panel.emit_both()
    t = time.time()                                              # puede llegar antes un render anterior: se espera al nuevo
    while time.time() - t < 5:
        img = _wait_result(app, session)
        if img is not None and img.pixelColor(100, 40).value() > 30:
            break
    assert img.pixelColor(100, 40).value() > 30 and img.pixelColor(20, 40).value() < 15     # todo menos el sujeto
    # Exportación: el mapa viaja a la imagen final a resolución completa.
    opts = ex.ExportOptions("final", "png8", tmp_path / "out", "x")
    out = ex.export_effect(Project.from_dict(session.project.to_dict()), depth, photo, opts, masks=session.current_maps())[0]
    with Image.open(out) as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr[40, 100].max() > 30 and arr[40, 20].max() < 15


def test_without_the_model_the_map_button_reports_it_and_nothing_breaks(tmp_path, monkeypatch):
    from deepshot import maps
    from deepshot.look import LookLayer

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "datos"))
    app, session, *_ = _setup(tmp_path)
    monkeypatch.setattr(maps, "sam_installed", lambda: False)
    panel = session.look_panel
    panel.add_item(LookLayer.new("levels"))
    panel._on_limit(2)
    panel.map_button.click()
    t = time.time()
    while panel.map_state.get("subject") == "busy" or (panel.map_state.get("subject") == "missing"
                                                       and time.time() - t < 5 and not panel.map_text):
        app.processEvents()
        time.sleep(0.01)
    assert panel.map_state["subject"] == "nomodel" and "SAM" in panel.map_text
    assert session.current_maps() == {}


def test_pre_fx_preview_shows_the_photo_with_its_effects_and_the_eye_toggles_on_press(tmp_path):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QStyleOptionViewItem

    from deepshot.look import LookLayer

    app, session, stack, *_ = _setup(tmp_path)
    p = session.project
    assert p.composite.opacity == 0.0 and session.fx_context() == ""
    session.pre_panel.add_item(LookLayer.new("saturation"))
    session.look_panel.add_item(LookLayer.new("levels"))
    assert session._preview_project() is p                          # fuera del paso Look no cambia nada
    session.set_look_active(True)
    assert session.fx_context() == "post" and session._preview_project() is p
    session.look_stack.show_stack("pre")
    assert session.fx_context() == "pre"
    view = session._preview_project()                               # Pre FX: foto con sus efectos, sin Post FX
    assert view.composite.opacity == 1.0 and view.composite.mask == "none" and view.look == [] and len(view.pre) == 1
    assert p.composite.opacity == 0.0 and len(p.look) == 1          # el proyecto real no se toca
    # El ojo actúa al pulsar y su zona es toda la fila: un doble clic seguido no se pierde.
    panel = session.pre_panel
    panel.resize(400, 500)
    panel.show()
    app.processEvents()
    delegate = panel.list.itemDelegate()
    rect = panel.list.visualItemRect(panel.list.item(0))
    eye = delegate._eye_rect(rect).center()
    assert delegate._hit(rect, panel.list.model().index(0, 0), rect.topLeft() + type(eye)(3, 2)) == "eye"
    QTest.mouseClick(panel.list.viewport(), Qt.LeftButton, Qt.NoModifier, eye)
    assert p.pre[0].enabled is False
    # El segundo clic de un doble clic llega como otro evento (y Qt no entrega su «soltar» al delegado): también cuenta.
    index = panel.list.model().index(0, 0)
    option = QStyleOptionViewItem()
    option.rect = rect
    dbl = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(eye), QPointF(eye), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert delegate.editorEvent(dbl, panel.list.model(), option, index) is True
    assert p.pre[0].enabled is True
    session.look_stack.show_stack("look")
    assert session.fx_context() == "post"


def test_focus_picker_announces_press_and_release_so_the_zone_shows_while_pressed(tmp_path):
    app, session, *_ = _setup(tmp_path)
    events = []
    session.focusPicking.connect(events.append)
    session.pick_focus_gesture({"phase": "press", "kind": "pick_focus", "x": 100.0, "y": 40.0})
    session.pick_focus_gesture({"phase": "move", "kind": "pick_focus", "x": 20.0, "y": 40.0})
    assert events == [True] and session.project.lens.enabled
    session.pick_focus_gesture({"phase": "release", "kind": "pick_focus", "x": 20.0, "y": 40.0})
    assert events == [True, False]


def test_export_effect_preview_leaves_out_post_fx_like_the_file(tmp_path):
    from deepshot.look import LookLayer

    app, session, *_ = _setup(tmp_path)
    p = session.project
    session.look_panel.add_item(LookLayer.new("levels"))
    assert session._preview_project() is p
    session.set_export_content("final")
    assert session._preview_project() is p                                   # la imagen final lleva el paso Look
    session.set_export_content("effect")
    assert session._preview_project().look == [] and len(p.look) == 1       # el efecto, sin él; el proyecto intacto
    session.set_export_content("")
    assert session._preview_project() is p
