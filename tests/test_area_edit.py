"""Edición de áreas desde la imagen: matemática (area_edit), gestos de la sesión y ratón en el lienzo."""

import math
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QUndoStack  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from deepshot import area_edit as ae  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.areas import Area, Scene  # noqa: E402
from deepshot.ui.canvas import CanvasView  # noqa: E402
from deepshot.ui.editor import EditSession  # noqa: E402

W, H = 600, 400


def geo(fov=0.0, depth=None):
    return ae.ImageGeometry(W, H, Scene(relief=1.0, fov_deg=fov), depth)


# --- Matemática -----------------------------------------------------------------------------------
@pytest.mark.parametrize("fov", [0.0, 50.0])
def test_pixels_scene_roundtrip(fov):
    g = geo(fov)
    for px, py, z in [(0, 0, 0.0), (300, 200, 0.5), (599, 399, 1.0), (123, 321, 0.3)]:
        x, y = g.to_scene_xy(px, py, z)
        bx, by = g.to_pixels(x, y, z)
        assert abs(bx - px) < 1e-6 and abs(by - py) < 1e-6
        assert abs(g.pixels_to_units(g.units_to_pixels(0.37, z), z) - 0.37) < 1e-9


def test_geometry_matches_scene_point_cloud():
    """El centro de un área colocada con point_at debe dibujarse en el píxel original."""
    depth = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    g = geo(50.0, depth)
    area = ae.place(Area(), (150.0, 100.0), g)
    ov = ae.outline(area, g)
    assert abs(ov["cx"] - 150.0) < 1.0 and abs(ov["cy"] - 100.0) < 1.0


def test_move_follows_pointer_in_pixels():
    g = geo(50.0)
    a0 = Area(center=(0.1, -0.2, 0.4))
    cx0, cy0 = g.to_pixels(*a0.center)
    moved = ae.move(a0, (cx0, cy0), (cx0 + 30, cy0 - 20), g)
    ov = ae.outline(moved, g)
    assert abs(ov["cx"] - (cx0 + 30)) < 1e-6 and abs(ov["cy"] - (cy0 - 20)) < 1e-6
    assert moved.center[2] == a0.center[2]                       # no cambia la profundidad
    assert ae.move(a0, (0, 0), (30, 20), g, axis="x").center[1] == pytest.approx(
        ae.move(a0, (0, 0), (30, 0), g).center[1])               # eje X: no se mueve en vertical
    assert ae.move(a0, (0, 0), (50, 0), g, axis="z").center[2] > a0.center[2]


def test_move_snap_to_surface_uses_depth():
    depth = np.full((H, W), 0.7, dtype=np.float32)
    g = geo(0.0, depth)
    moved = ae.move(Area(center=(0, 0, 0.1)), (0, 0), (100.0, 50.0), g, snap_to_surface=True)
    assert moved.center[2] == pytest.approx(0.7)


def test_scale_handle_makes_semiaxis_equal_to_distance():
    g = geo(0.0)
    a0 = Area(center=(0.0, 0.0, 0.2), scale=(0.2, 0.2, 0.2), rotation=(0, 0, 30.0))
    cx, cy = g.to_pixels(*a0.center)
    ux, _ = ae.axes(30.0)
    pointer = (cx + ux[0] * 90, cy + ux[1] * 90)                 # 90 px sobre el eje X girado
    new = ae.scale_handle(a0, "x", pointer, g)
    assert ae.outline(new, g)["rx"] == pytest.approx(90.0, rel=1e-6)
    assert new.scale[1] == a0.scale[1]                           # el otro eje no cambia
    uni = ae.scale_handle(a0, "x", pointer, g, uniform=True)
    assert uni.scale[0] / a0.scale[0] == pytest.approx(uni.scale[1] / a0.scale[1])
    tiny = ae.scale_handle(a0, "x", (cx, cy), g)
    assert tiny.scale[0] >= ae.MIN_SCALE                         # nunca desaparece


def test_scale_all_ratio_and_axis():
    g = geo(0.0)
    a0 = Area(center=(0, 0, 0.2), scale=(0.2, 0.3, 0.4))
    cx, cy = g.to_pixels(*a0.center)
    new = ae.scale_all(a0, (cx + 50, cy), (cx + 100, cy), g)
    assert new.scale == pytest.approx((0.4, 0.6, 0.8))
    only_x = ae.scale_all(a0, (cx + 50, cy), (cx + 100, cy), g, axis="x")
    assert only_x.scale == pytest.approx((0.4, 0.3, 0.4))


@pytest.mark.parametrize("angle", [0.0, 30.0, -75.0, 140.0])
def test_rotate_handle_roundtrip_with_canvas_handle(angle):
    """El tirador de giro está sobre el eje Y superior: arrastrarlo a su sitio no debe cambiar el giro."""
    g = geo(0.0)
    a0 = Area(center=(0.1, 0.1, 0.2), scale=(0.3, 0.2, 0.2), rotation=(5.0, 6.0, angle))
    ov = ae.outline(a0, g)
    _, uy = ae.axes(angle)
    top = (ov["cx"] - uy[0] * (ov["ry"] + 30), ov["cy"] - uy[1] * (ov["ry"] + 30))
    new = ae.rotate_handle(Area(center=a0.center, scale=a0.scale, rotation=(5.0, 6.0, 0.0)), top, g)
    assert new.rotation[2] == pytest.approx(ae._wrap_angle(angle), abs=1e-6)
    assert new.rotation[:2] == (5.0, 6.0)                        # X e Y no se tocan


def test_rotate_free_counterclockwise_and_snap():
    g = geo(0.0)
    a0 = Area(center=(0, 0, 0.2), rotation=(0, 0, 10.0))
    cx, cy = g.to_pixels(*a0.center)
    new = ae.rotate_free(a0, (cx + 100, cy), (cx, cy - 100), g)   # de la derecha hacia arriba: +90°
    assert new.rotation[2] == pytest.approx(100.0)
    snapped = ae.rotate_free(a0, (cx + 100, cy), (cx + 100, cy - 28), g, snap=True)
    assert snapped.rotation[2] == pytest.approx(25.0)                # el giro (15,6°) se redondea a 15°
    ay = ae.rotate_free(a0, (0, 0), (40, 0), g, axis="y")
    assert ay.rotation == (0, 20.0, 10.0)


def test_draw_radius_and_free_box():
    depth = np.full((H, W), 0.5, dtype=np.float32)
    g = geo(0.0, depth)
    area = ae.draw(Area(), (300, 200), (340, 230), g)             # radio 50 px
    ov = ae.outline(area, g)
    assert ov["rx"] == pytest.approx(50.0) and ov["ry"] == pytest.approx(50.0)
    assert abs(ov["cx"] - 300) < 1.0 and area.center[2] == pytest.approx(0.5)
    free = ae.outline(ae.draw(Area(), (300, 200), (340, 230), g, free=True), g)
    assert free["rx"] == pytest.approx(40.0) and free["ry"] == pytest.approx(30.0)


# --- Sesión de edición: gestos ------------------------------------------------------------------------
_sessions = []


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    from PySide6.QtCore import QCoreApplication, QEvent

    while _sessions:
        s = _sessions.pop()
        s.shutdown()
        for panel in s.panels():
            panel.deleteLater()
        s.style_bar.deleteLater()
        s.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def make_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    create_app([])
    photo = tmp_path / "foto.png"
    Image.new("RGB", (W, H), (90, 120, 160)).save(photo)
    depth = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32), (H, 1))
    stack = QUndoStack()
    session = EditSession(stack)
    session.set_photo(photo, depth)
    _sessions.append(session)
    return session, stack


def g(kind, phase, x=0.0, y=0.0, index=-1, mods=0, axis=""):
    return {"kind": kind, "phase": phase, "index": index, "x": x, "y": y, "mods": mods, "axis": axis}


def test_draw_gesture_creates_one_area_and_one_undo_step(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.gesture(g("draw", "press", 300, 200))
    s.gesture(g("draw", "move", 360, 200))
    s.gesture(g("draw", "move", 380, 200))
    s.gesture(g("draw", "release", 380, 200))
    assert len(s.project.areas) == 1 and stack.count() == 1
    ov = s.overlays()[0]
    assert ov["rx"] == pytest.approx(80.0, abs=0.5) and ov["selected"]
    stack.undo()
    assert s.project.areas == [] and s.areas_panel.list.count() == 0
    stack.redo()
    assert len(s.project.areas) == 1


def test_draw_click_without_drag_uses_default_size(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.gesture(g("draw", "press", 300, 200))
    s.gesture(g("draw", "release", 300, 200))
    assert s.project.areas[0].scale == s.DEFAULT_DRAW_SCALE


def test_cancel_removes_created_area_and_restores_moved(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.gesture(g("draw", "press", 300, 200))
    s.gesture(g("draw", "move", 360, 200))
    s.gesture(g("draw", "cancel"))
    assert s.project.areas == [] and stack.count() == 0
    s.areas_panel.add_item(Area(center=(0, 0, 0.2)))
    before = s.project.areas[0].center
    stack_count = stack.count()
    s.gesture(g("move", "press", 300, 200, index=0))
    s.gesture(g("move", "move", 380, 150, index=0))
    assert s.project.areas[0].center != before
    s.gesture(g("move", "cancel"))
    assert s.project.areas[0].center == before and stack.count() == stack_count


def test_move_gesture_moves_and_undoes_as_one_step(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.areas_panel.add_item(Area(center=(0, 0, 0.2)))
    n = stack.count()
    cx, cy = s.overlays()[0]["cx"], s.overlays()[0]["cy"]
    s.gesture(g("move", "press", cx, cy, index=0))
    for i in range(1, 6):
        s.gesture(g("move", "move", cx + 10 * i, cy + 4 * i, index=0))
    s.gesture(g("move", "release", cx + 50, cy + 20, index=0))
    ov = s.overlays()[0]
    assert ov["cx"] == pytest.approx(cx + 50, abs=1e-6) and ov["cy"] == pytest.approx(cy + 20, abs=1e-6)
    assert stack.count() == n + 1                      # un solo paso para todo el arrastre
    assert s.areas_panel.center[0].value() == pytest.approx(s.project.areas[0].center[0], abs=1e-3)  # panel al día
    stack.undo()
    assert s.overlays()[0]["cx"] == pytest.approx(cx, abs=1e-6)


def test_click_on_other_area_selects_it_without_undo_step(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.areas_panel.add_item(Area(center=(-0.5, 0, 0.2)))
    s.areas_panel.add_item(Area(center=(0.5, 0, 0.2)))
    n = stack.count()
    assert s.selected_area == 1
    s.gesture(g("move", "press", 10, 10, index=0))
    s.gesture(g("move", "release", 10, 10, index=0))
    assert s.selected_area == 0 and stack.count() == n


def test_place_gesture_follows_pointer_on_surface(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.gesture(g("place", "press", 150, 100))             # sin áreas: crea una
    s.gesture(g("place", "move", 450, 300))
    s.gesture(g("place", "release", 450, 300))
    area = s.project.areas[0]
    expected_z = s.render.preview_depth()[300 * s.render.preview_depth().shape[0] // H,
                                          450 * s.render.preview_depth().shape[1] // W] * s.project.scene.relief
    assert area.center[2] == pytest.approx(expected_z, abs=0.01)
    assert s.overlays()[0]["cx"] == pytest.approx(450, abs=2.0)


def test_wheel_adjustments_merge_into_one_undo_step(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    app = create_app([])
    s.areas_panel.add_item(Area(center=(0, 0, 0.2), scale=(0.3, 0.3, 0.3)))
    n = stack.count()
    for _ in range(4):
        s.adjust_selected("scale", 1)
    s.adjust_selected("softness", 2)
    s.adjust_selected("depth", -3)
    a = s.project.areas[0]
    assert a.scale[0] == pytest.approx(0.3 * 1.06 ** 4) and a.edge_softness == pytest.approx(0.6)
    assert a.center[2] == pytest.approx(0.2 - 0.06)
    deadline = time.time() + 3
    while stack.count() == n and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert stack.count() == n + 1


def test_rename_and_color_survive_json_and_undo(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.areas_panel.add_item(Area(name="A"))
    s.areas_panel.add_item(Area(name="B"))
    s.areas_panel.name.setText("Cara")
    s.areas_panel.name.editingFinished.emit()
    assert s.project.areas[1].name == "Cara"
    assert s.areas_panel.list.item(1).text().startswith("Cara")
    s.project.areas[1].ui_color = (0.1, 0.2, 0.3)
    s.save_settings(tmp_path / "x.json")
    s.project.areas.clear()
    s.load_settings(tmp_path / "x.json")
    assert s.project.areas[1].name == "Cara" and s.project.areas[1].ui_color == (0.1, 0.2, 0.3)
    assert s.project.areas[0].ui_color is None
    # Colores distintos por posición cuando no se eligió ninguno
    assert s.overlays()[0]["color"] != s.overlays()[1]["color"] or s.project.areas[1].ui_color


def test_style_applies_and_undoes(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    from deepshot import presets

    before = s.project.to_dict()
    s.apply_style(presets.builtin("style", "photo_bw"))
    assert s.project.composite.photo_grayscale and s.project.composite.mode == "multiply"
    assert s.ramp_editor.ramp is s.project.ramp                  # el editor de rampa sigue enlazado
    stack.undo()
    assert s.project.to_dict() == before


# --- Lienzo: ratón y teclado -------------------------------------------------------------------------------
def canvas_with_area(angle=0.0):
    create_app([])
    c = CanvasView()
    c.resize(800, 600)
    img = QImage(W, H, QImage.Format_RGB32)
    img.fill(QColor("gray"))
    c.set_image(img)
    c.show()
    c.set_overlays([{"cx": 300.0, "cy": 200.0, "rx": 100.0, "ry": 60.0, "angle": angle, "box": False,
                     "selected": True, "strength": 1.0, "soft": 0.5, "color": (1, 0.8, 0.3), "name": "A",
                     "index": 0}])
    events = []
    c.gesture.connect(events.append)
    return c, events


def to_view(c, x, y):
    return c.mapFromScene(QPointF(x, y))


def test_canvas_drag_inside_area_emits_move_gesture():
    c, events = canvas_with_area()
    start = to_view(c, 300, 200)
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(c.viewport(), start + QPoint(25, 10))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, start + QPoint(25, 10))
    phases = [e["phase"] for e in events]
    assert phases[0] == "press" and phases[-1] == "release" and "move" in phases
    assert all(e["kind"] == "move" and e["index"] == 0 for e in events)
    assert abs(events[0]["x"] - 300) < 2 and abs(events[0]["y"] - 200) < 2
    c.deleteLater()


def test_canvas_hits_handles_before_body():
    c, events = canvas_with_area(angle=0.0)
    h = c._handles(c._overlays[0])
    for name in ("scale_x", "scale_y", "rotate"):
        events.clear()
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, h[name].toPoint())
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, h[name].toPoint())
        assert events and events[0]["kind"] == name, name
    c.deleteLater()


def test_canvas_outside_area_pans_instead_of_gesture():
    c, events = canvas_with_area()
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, to_view(c, 30, 30))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, to_view(c, 30, 30))
    assert events == []
    c.deleteLater()


def test_canvas_rotated_area_hit_test():
    c, events = canvas_with_area(angle=90.0)       # girada 90°: el eje X largo queda vertical
    # (300, 200 + 90) está dentro de la elipse girada (semieje 100 hacia abajo), no dentro de la sin girar
    assert c._hit_test(QPointF(to_view(c, 300, 290))) == (0, "move")
    assert c._hit_test(QPointF(to_view(c, 390, 200))) is None
    c.deleteLater()


def test_canvas_keyboard_modal_g_x_confirm_and_cancel():
    c, events = canvas_with_area()
    hints = []
    c.hint.connect(hints.append)
    start = to_view(c, 300, 200)
    c._cursor_pos = start
    QTest.keyClick(c, Qt.Key_G)
    assert events and events[0]["kind"] == "move" and events[0]["phase"] == "press"
    QTest.keyClick(c, Qt.Key_X)
    assert events[-1]["phase"] == "axis" and events[-1]["axis"] == "x"
    QTest.mouseMove(c.viewport(), start + QPoint(20, 5))
    assert events[-1]["phase"] == "move"
    QTest.keyClick(c, Qt.Key_Return)
    assert events[-1]["phase"] == "release"
    events.clear()
    QTest.keyClick(c, Qt.Key_S)
    QTest.keyClick(c, Qt.Key_Escape)
    assert [e["phase"] for e in events] == ["press", "cancel"] and events[0]["kind"] == "scale_all"
    assert any("G" in h for h in hints)
    c.deleteLater()


def test_canvas_modal_ignored_without_selected_area_or_with_tool():
    c, events = canvas_with_area()
    c.set_overlays([])
    c._cursor_pos = to_view(c, 300, 200)
    QTest.keyClick(c, Qt.Key_G)
    assert events == []
    c.deleteLater()


def test_canvas_tool_draw_emits_gesture_with_press_point():
    c, events = canvas_with_area()
    c.set_tool("draw")
    start = to_view(c, 100, 100)
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(c.viewport(), start + QPoint(30, 0))
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, start + QPoint(30, 0))
    assert events[0]["kind"] == "draw" and abs(events[0]["x"] - 100) < 2
    c.deleteLater()


def test_canvas_wheel_modifiers_emit_area_wheel():
    from PySide6.QtCore import QPointF as P
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtCore import QPoint as Pt

    c, _events = canvas_with_area()
    got = []
    c.areaWheel.connect(lambda kind, steps: got.append((kind, steps)))
    for mod, name in ((Qt.ControlModifier, "scale"), (Qt.ShiftModifier, "depth"), (Qt.AltModifier, "softness")):
        ev = QWheelEvent(P(100, 100), P(100, 100), Pt(0, 0), Pt(0, 120), Qt.NoButton, mod, Qt.NoScrollPhase, False)
        c.wheelEvent(ev)
    assert got == [("scale", 1.0), ("depth", 1.0), ("softness", 1.0)]
    zoom_before = c.transform().m11()
    ev = QWheelEvent(P(100, 100), P(100, 100), Pt(0, 0), Pt(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    c.wheelEvent(ev)
    assert c.transform().m11() > zoom_before and len(got) == 3     # sin modificador: zoom
    c.deleteLater()


def test_area_outlines_and_handles_are_hidden_while_painting_the_mask():
    """Con el pincel o el borrador activos no se dibujan contornos ni tiradores de áreas (fallo visto por Juanjo)."""
    c, _events = canvas_with_area()
    drawn = []
    original = c._draw_area
    c._draw_area = lambda painter, ov, to_view_: (drawn.append(1), original(painter, ov, to_view_))[1]
    c.grab()
    assert drawn                                           # en modo normal sí se dibujan
    for tool in ("paint", "erase"):
        drawn.clear()
        c.set_tool(tool)
        c.grab()
        assert not drawn, tool
    for tool in ("draw", "place", ""):                    # las herramientas de áreas sí necesitan verlas
        drawn.clear()
        c.set_tool(tool)
        c.grab()
        assert drawn, tool
    c.deleteLater()


def test_pan_tilt_rotate_around_vertical_and_horizontal_axes_and_snap():
    a0 = Area(center=(0, 0, 0.2), rotation=(10.0, 20.0, 30.0))
    pan = ae.pan_tilt(a0, "pan", (0, 0), (40, 99))               # solo cuenta el movimiento horizontal: 40 px = 20°
    assert pan.rotation == (10.0, 40.0, 30.0)
    tilt = ae.pan_tilt(a0, "tilt", (0, 0), (99, -20))            # solo el vertical: hacia arriba, -10°
    assert tilt.rotation == (0.0, 20.0, 30.0)
    assert ae.pan_tilt(a0, "pan", (0, 0), (62, 0), snap=True).rotation[1] == pytest.approx(50.0)   # +31° → 30° → 20+30


def test_canvas_pan_and_tilt_handles_start_their_gestures():
    c, events = canvas_with_area()
    handles = c._handles(c._overlays[0])
    assert c._hit_test(handles["pan"]) == (0, "pan") and c._hit_test(handles["tilt"]) == (0, "tilt")
    assert handles["pan"].x() < c._handles(c._overlays[0])["scale_x"].x()      # el paneo está a la izquierda
    c._begin_gesture("pan", 0, handles["pan"])
    assert events[-1]["kind"] == "pan" and events[-1]["phase"] == "press"
    c.deleteLater()


def test_pan_and_tilt_gestures_change_rotation_in_one_undo_step_each(tmp_path, monkeypatch):
    s, stack = make_session(tmp_path, monkeypatch)
    s.areas_panel.add_item(Area(center=(0, 0, 0.2)))
    base = stack.count()
    s.gesture(g("pan", "press", 100, 100, index=0))
    s.gesture(g("pan", "move", 140, 100, index=0))
    s.gesture(g("pan", "release", 140, 100, index=0))
    assert s.project.areas[0].rotation[1] == pytest.approx(20.0) and stack.count() == base + 1
    s.gesture(g("tilt", "press", 100, 100, index=0))
    s.gesture(g("tilt", "move", 100, 120, index=0))
    s.gesture(g("tilt", "release", 100, 120, index=0))
    assert s.project.areas[0].rotation[0] == pytest.approx(10.0) and stack.count() == base + 2
    stack.undo()
    assert s.project.areas[0].rotation[0] == 0.0
