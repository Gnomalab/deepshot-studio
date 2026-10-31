"""Vista 3D del relieve: cámara orbital, dibujo de la nube, sombreado y el visor con ratón."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from deepshot import view3d as v3  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.areas import Scene  # noqa: E402
from deepshot.ui.view3d_widget import View3D  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return create_app([])


# --- Cámara -------------------------------------------------------------------------------------------
def test_target_projects_to_the_screen_center_from_any_angle():
    cam = v3.OrbitCamera(yaw=40, pitch=25, distance=5)
    for yaw, pitch in ((0, 0), (90, 10), (-120, 60), (180, -45)):
        cam.set_view(yaw, pitch)
        x, y, z = cam.project(cam.target[None, :], 800, 600)
        assert x[0] == pytest.approx(400, abs=1e-6) and y[0] == pytest.approx(300, abs=1e-6)
        assert z[0] == pytest.approx(cam.distance, abs=1e-6)


def test_front_view_matches_the_2d_view():
    """Con yaw = pitch = 0, derecha = +x, arriba = +y y adelante = +z, como la escena 2D."""
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=4, target=np.array([0.0, 0.0, 0.0]))
    right, up, forward = cam.basis()
    assert np.allclose(right, [1, 0, 0]) and np.allclose(up, [0, 1, 0]) and np.allclose(forward, [0, 0, 1])
    sx, sy, _ = cam.project(np.array([[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]]), 800, 600)
    assert sx[0] > 400 > sx[1] and sy[0] < 300 < sy[1]            # arriba-derecha en pantalla = +x, +y


def test_nearer_points_look_bigger_and_points_behind_the_camera_are_dropped():
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=4, target=np.array([0.0, 0.0, 0.0]))
    sx, _, _ = cam.project(np.array([[1.0, 0, -1.0], [1.0, 0, 1.0]]), 800, 600)   # el primero está más cerca
    assert abs(sx[0] - 400) > abs(sx[1] - 400)
    sx, sy, _ = cam.project(np.array([[0.0, 0, -9.0]]), 800, 600)                  # detrás de la cámara
    assert np.isnan(sx[0]) and np.isnan(sy[0])


def test_orbit_clamps_pitch_wraps_yaw_and_zoom_is_limited():
    cam = v3.OrbitCamera()
    cam.orbit(0, 500)
    assert cam.pitch == v3.MAX_PITCH
    cam.orbit(0, -1000)
    assert cam.pitch == v3.MIN_PITCH
    cam.set_view(170, 0)
    cam.orbit(30, 0)
    assert -180 <= cam.yaw < 180 and cam.yaw == pytest.approx(-160)
    for _ in range(100):
        cam.zoom(2)
    assert cam.distance >= 0.4
    for _ in range(100):
        cam.zoom(0.5)
    assert cam.distance <= 40.0


def test_pan_moves_the_target_in_the_screen_plane_and_reset_restores():
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=4, target=np.array([0.0, 0.0, 0.0]))
    cam.pan(100, 0, 600)                         # arrastrar a la derecha: la escena sigue al ratón
    assert cam.target[0] < 0 and cam.target[1] == pytest.approx(0) and cam.target[2] == pytest.approx(0)
    cam.reset(0.7)
    assert cam.target.tolist() == [0.0, 0.0, 0.7] and cam.yaw == 28.0


# --- Rejilla, sombreado y dibujo ------------------------------------------------------------------------
def test_grid_size_keeps_aspect_and_caps_points():
    cols, rows = v3.grid_size(6000, 4000)
    assert cols * rows <= v3.MAX_POINTS * 1.05 and cols / rows == pytest.approx(1.5, abs=0.02)
    assert v3.grid_size(40, 30) == (40, 30)                       # una foto pequeña no se reescala
    assert min(v3.grid_size(4000, 10)) >= 8


def test_shade_leaves_flat_surfaces_alone_and_darkens_or_lightens_slopes():
    colors = np.full((20, 30, 3), 200, dtype=np.uint8)
    flat = np.zeros((20, 30))
    assert (v3.shade(colors, flat, 1.5, 0.4) == 200).all()
    ramp = np.tile(np.linspace(0, 1, 30), (20, 1))                # inclinada en x
    shaded = v3.shade(colors, ramp, 1.5, 0.4)
    assert shaded.min() != shaded.max() or shaded[0, 0, 0] != 200
    assert (v3.shade(colors, ramp, 1.5, 0.0) == 200).all()         # fuerza 0: sin sombreado


def grid(depth_fn, cols=40, rows=30, relief=1.0):
    d = depth_fn(*np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows)))
    return Scene(relief=relief).points(d.astype(np.float32)), d


def test_render_flat_plane_fills_the_center_and_leaves_background_at_the_edges():
    pts, _ = grid(lambda x, y: np.zeros_like(x))
    colors = np.full((30, 40, 3), (10, 200, 30), dtype=np.uint8)
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=3.6, target=np.array([0.0, 0.0, 0.0]))
    img = v3.render_cloud(pts, colors, cam, 400, 300)
    assert img.shape == (300, 400, 4)
    assert tuple(img[150, 200]) == (10, 200, 30, 255)             # el centro está cubierto, sin agujeros
    assert tuple(img[2, 2]) == v3.BACKGROUND
    covered = (img[..., :3] == (10, 200, 30)).all(axis=-1)
    assert covered[100:200, 130:270].all()                         # sin huecos dentro de la superficie


def test_nearer_surface_hides_the_farther_one():
    """Dos puntos en la misma línea de visión: se ve el de delante."""
    pts = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])          # (1 fila, 2 columnas): cerca y lejos
    colors = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=4, target=np.array([0.0, 0.0, 0.0]))
    img = v3.render_cloud(pts, colors, cam, 200, 200)
    assert tuple(img[100, 100, :3]) == (255, 0, 0)
    # Mismo resultado si se invierte el orden de los puntos en la rejilla.
    img2 = v3.render_cloud(pts[:, ::-1], colors[:, ::-1], cam, 200, 200)
    assert tuple(img2[100, 100, :3]) == (255, 0, 0)


def test_orbiting_changes_the_picture_and_relief_is_visible_from_the_side():
    pts, _ = grid(lambda x, y: np.where((abs(x - .5) < .2) & (abs(y - .5) < .2), 0.0, 1.0), relief=1.0)
    colors = np.full((30, 40, 3), 150, dtype=np.uint8)
    front = v3.render_cloud(pts, colors, v3.OrbitCamera(0, 0, 3.6, target=np.array([0.0, 0, .5])), 400, 300)
    side = v3.render_cloud(pts, colors, v3.OrbitCamera(80, 0, 3.6, target=np.array([0.0, 0, .5])), 400, 300)
    assert not np.array_equal(front, side)
    # De frente todo el relieve cae dentro del rectángulo de la foto; de lado se ve estrecho (profundidad).
    cols_front = np.flatnonzero((front[..., :3] != v3.BACKGROUND[:3]).any(axis=-1).any(axis=0))
    cols_side = np.flatnonzero((side[..., :3] != v3.BACKGROUND[:3]).any(axis=-1).any(axis=0))
    assert len(cols_side) < len(cols_front)


def test_render_with_everything_behind_the_camera_returns_background():
    pts, _ = grid(lambda x, y: np.zeros_like(x))
    colors = np.full((30, 40, 3), 99, dtype=np.uint8)
    cam = v3.OrbitCamera(yaw=0, pitch=0, distance=0.5, target=np.array([0.0, 0.0, 30.0]))
    img = v3.render_cloud(pts, colors, cam, 100, 80)
    assert (img == np.array(v3.BACKGROUND, dtype=np.uint8)).all()


# --- Visor ----------------------------------------------------------------------------------------------------
def make_widget(app):
    w = View3D()
    w.resize(500, 380)
    w.show()
    depth = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (80, 1))
    tex = QImage(120, 80, QImage.Format_RGB888)
    tex.fill(0x3080c0)
    w.set_data(depth, Scene(relief=1.0), tex)
    app.processEvents()
    return w


def test_widget_renders_and_orbits_with_the_mouse(app):
    w = make_widget(app)
    before = w.camera.copy()
    first = w.grab().toImage()
    QTest.mousePress(w, Qt.LeftButton, Qt.NoModifier, QPoint(250, 190))
    QTest.mouseMove(w, QPoint(300, 170))
    QTest.mouseRelease(w, Qt.LeftButton, Qt.NoModifier, QPoint(300, 170))
    assert w.camera.yaw < before.yaw and w.camera.pitch < before.pitch    # derecha y arriba
    assert w.grab().toImage() != first                                    # la imagen cambia al girar
    w.deleteLater()


def test_widget_pan_zoom_reset_and_view_keys(app):
    w = make_widget(app)
    start_target = w.camera.target.copy()
    QTest.mousePress(w, Qt.LeftButton, Qt.ShiftModifier, QPoint(200, 200))
    QTest.mouseMove(w, QPoint(260, 230))
    QTest.mouseRelease(w, Qt.LeftButton, Qt.ShiftModifier, QPoint(260, 230))
    assert not np.allclose(w.camera.target, start_target)                 # Mayús+arrastrar: mueve
    d = w.camera.distance
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QWheelEvent

    ev = QWheelEvent(QPointF(250, 190), QPointF(250, 190), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    w.wheelEvent(ev)
    assert w.camera.distance < d                                          # rueda hacia delante: acerca
    for key, (yaw, pitch) in ((Qt.Key_1, (0, 0)), (Qt.Key_2, (90, 0)), (Qt.Key_7, (0, 89))):
        QTest.keyClick(w, key)
        assert (w.camera.yaw, w.camera.pitch) == (yaw, pitch)
    QTest.mouseDClick(w, Qt.LeftButton, Qt.NoModifier, QPoint(250, 190))  # doble clic: restablecer
    assert w.camera.yaw == 28.0 and w.camera.distance == 3.6
    w.deleteLater()


def test_widget_without_data_shows_message_and_follows_relief_changes(app):
    w = View3D()
    w.resize(400, 300)
    w.show()
    app.processEvents()
    w.grab()                                                              # sin datos: no falla
    assert not w.has_data()
    depth = np.tile(np.linspace(0, 1, 60, dtype=np.float32), (40, 1))
    w.set_data(depth, Scene(relief=0.5), None)
    first = w._render_frame().copy()
    w.set_data(depth, Scene(relief=2.0), None)                            # más relieve: otra superficie
    assert w._render_frame() != first
    assert w.camera.target[2] == pytest.approx(0.25)                      # el objetivo se centró con el primer relieve
    w.deleteLater()


# --- Acantilados del relieve --------------------------------------------------------------------------------
def test_cliff_points_fill_walls_with_the_nearer_color_and_none_on_flat_relief():
    flat, _ = grid(lambda x, y: np.zeros_like(x))
    pts, cols = v3.cliff_points(flat, np.full((30, 40, 3), 7, dtype=np.uint8))
    assert len(pts) == 0 and len(cols) == 0
    step_up, _ = grid(lambda x, y: np.where(x < 0.5, 0.0, 1.0), relief=1.0)     # una pared en la mitad
    colors = np.zeros((30, 40, 3), dtype=np.uint8)
    colors[:, :20] = (255, 0, 0)                                               # lado cercano: rojo
    colors[:, 20:] = (0, 0, 255)                                               # lado lejano: azul
    pts, cols = v3.cliff_points(step_up, colors)
    assert len(pts) > 0 and (cols == (255, 0, 0)).all()                        # la pared toma el color del primer plano
    assert pts[:, 2].min() >= 0 and pts[:, 2].max() <= 1                       # puntos entre los dos niveles


def test_cliff_points_are_capped_and_the_wall_is_closed_in_the_render():
    rng = np.random.default_rng(1)
    noisy = rng.random((120, 160)).astype(np.float32)                           # relieve muy ruidoso: muchos saltos
    pts = Scene(relief=1.0).points(noisy)
    colors = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    extra = v3.cliff_points(pts, colors, max_total=20_000)
    assert 0 < len(extra[0]) <= 20_000 * 1.1
    # Una pared vertical vista de lado no deja agujeros: con puntos extra cubre más píxeles que sin ellos.
    wall, _ = grid(lambda x, y: np.where(x < 0.5, 0.0, 1.0), cols=60, rows=40)
    col = np.full((40, 60, 3), 200, dtype=np.uint8)
    cam = v3.OrbitCamera(yaw=60, pitch=10, distance=3.6, target=np.array([0.0, 0.0, 0.5]))
    plain = v3.render_cloud(wall, col, cam, 400, 300)
    filled = v3.render_cloud(wall, col, cam, 400, 300, extra=v3.cliff_points(wall, col))
    bg = np.array(v3.BACKGROUND[:3], dtype=np.uint8)
    assert (filled[..., :3] != bg).any(axis=-1).sum() > (plain[..., :3] != bg).any(axis=-1).sum()


def test_widget_draws_fast_while_moving_and_refines_afterwards(app):
    import time

    w = make_widget(app)
    QTest.keyClick(w, Qt.Key_7)                       # cualquier movimiento de la cámara
    assert w._fast                                    # durante el movimiento: media resolución
    fast = w._render_frame().copy()
    assert fast.width() == w.width() // 2
    t = time.time()
    while w._fast and time.time() - t < 2:
        app.processEvents()
    assert not w._fast                                # al dejar de moverse: calidad completa
    assert w._render_frame().width() == w.width()
    w.deleteLater()


# --- Calidad 3D --------------------------------------------------------------------------------------
def test_quality_levels_grow_and_the_grid_follows_them():
    levels = [v3.QUALITY_POINTS[k] for k in ("low", "medium", "high", "ultra", "max")]
    assert levels == sorted(levels) and v3.QUALITY_POINTS["medium"] == v3.MAX_POINTS
    cols_low, rows_low = v3.grid_size(4000, 3000, v3.QUALITY_POINTS["low"])
    cols_hi, rows_hi = v3.grid_size(4000, 3000, v3.QUALITY_POINTS["ultra"])
    assert cols_low * rows_low < cols_hi * rows_hi <= v3.QUALITY_POINTS["ultra"] * 1.05
    assert v3.grid_size(300, 200, v3.QUALITY_POINTS["max"]) == (300, 200)       # nunca más que la foto


def test_auto_quality_picks_the_highest_level_that_fits_the_time_budget():
    big = 24_000_000
    fast, slow = 0.5e-6, 5e-6                                  # segundos por punto: equipo rápido / lento
    assert v3.auto_points(fast, big) == v3.QUALITY_POINTS["ultra"]
    assert v3.auto_points(slow, big) == v3.QUALITY_POINTS["medium"]            # nunca baja de «Media»
    assert v3.auto_points(fast, 100_000) == v3.QUALITY_POINTS["medium"]        # foto pequeña: no hay más puntos
    mid = v3.auto_points(1.5e-6, big)
    assert mid == v3.QUALITY_POINTS["high"]


def big_widget(app):
    """Visor con una foto de 1200 × 800 (más puntos que el nivel «Media»)."""
    w = View3D()
    w.resize(400, 300)
    w.show()
    depth = np.tile(np.linspace(0, 1, 1200, dtype=np.float32), (800, 1))
    tex = QImage(1200, 800, QImage.Format_RGB888)
    tex.fill(0x3080c0)
    w.set_data(depth, Scene(relief=1.0), tex)
    return w


def test_widget_grid_follows_the_chosen_quality(app):
    w = big_widget(app)
    w.set_quality("low")
    w._render_frame()
    low = w._grid.shape[0] * w._grid.shape[1]
    w.set_quality("high")
    w._render_frame()
    high = w._grid.shape[0] * w._grid.shape[1]
    assert low <= v3.QUALITY_POINTS["low"] * 1.05 and low < high <= v3.QUALITY_POINTS["high"] * 1.05
    w.set_quality("nonsense")                         # un valor desconocido vuelve a «automática»
    assert w.quality() == v3.AUTO
    w.deleteLater()


def test_widget_auto_quality_measures_once_and_settles(app):
    w = big_widget(app)
    assert w.quality() == v3.AUTO and w.max_points() == v3.QUALITY_POINTS["medium"]   # arranca en «Media»
    w._render_frame()                                 # el primer dibujo completo mide el coste
    assert w._auto_points is not None
    chosen = w._auto_points
    w._render_frame()
    assert w._auto_points == chosen                   # no se vuelve a medir (sin saltos de calidad)
    w.set_quality("auto")                             # mismo valor: no cambia nada
    assert w._auto_points == chosen
    w.set_quality("max")
    assert w.max_points() == v3.QUALITY_POINTS["max"]
    w.deleteLater()


def test_fast_mode_draws_a_thinned_grid_when_quality_is_high(app):
    w = big_widget(app)
    w.set_quality("max")
    w._prepare()
    w._fast = True
    frame = w._render_frame()                         # no debe tardar con un millón de puntos: se aclara la rejilla
    assert frame is not None and frame.width() == w.width() // 2
    w.deleteLater()


def _move_event(pos):
    """Movimiento del ratón sin botones pulsados, entregado directamente al visor."""
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(QMouseEvent.MouseMove, pos, pos, Qt.NoButton, Qt.NoButton, Qt.NoModifier)


def test_orientation_gizmo_aligns_the_view_with_each_axis_and_highlights_on_hover(app):
    w = View3D()
    w.resize(500, 400)
    w.set_data(np.full((40, 60), 0.5, np.float32), Scene(), None)
    w.show()
    w.camera.set_view(28.0, 16.0)
    balls = {name: pt for name, pt, _toward in w._gizmo_balls()}
    assert set(balls) == {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    # Cada bola se encuentra donde se dibuja, y pulsarla pone la cámara mirando desde ese lado.
    for name, (yaw, pitch) in (("+X", (90.0, 0.0)), ("-X", (-90.0, 0.0)), ("+Z", (180.0, 0.0)), ("-Z", (0.0, 0.0))):
        w.camera.set_view(28.0, 16.0)
        balls = {n: pt for n, pt, _toward in w._gizmo_balls()}              # la cámara volvió a su sitio
        w.mouseMoveEvent(_move_event(balls[name]))
        assert w._axis_hover == name
        QTest.mouseClick(w, Qt.LeftButton, Qt.NoModifier, balls[name].toPoint())
        assert (w.camera.yaw, w.camera.pitch) == (yaw, pitch), name
    w.camera.set_view(40.0, 0.0)
    top = {n: pt for n, pt, _ in w._gizmo_balls()}["+Y"]
    QTest.mouseClick(w, Qt.LeftButton, Qt.NoModifier, top.toPoint())
    assert w.camera.yaw == 40.0 and w.camera.pitch == 89.0                 # arriba: conserva el giro
    assert w._drag == ""                                                   # pulsar el gizmo no empieza a orbitar
    w.leaveEvent(None)
    assert w._axis_hover == ""
    assert w.gizmo_hit(QPointF(5, 395)) == ""                              # lejos del gizmo no hay nada


# --- G / S / R con el teclado en 3D ------------------------------------------------------------------------
def _press_event(pos):
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(QMouseEvent.MouseButtonPress, pos, pos, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def _modal_widget(app):
    from deepshot.areas import Area

    w = make_widget(app)
    area = Area(name="A", center=(0.0, 0.0, 0.3), scale=(0.4, 0.3, 0.2))
    w.set_areas([area], 0)
    events = []
    w.areaGesture.connect(events.append)
    return w, events


def test_modal_g_moves_the_selected_area_with_the_mouse_and_x_locks_the_axis(app):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QCursor
    from PySide6.QtTest import QTest

    w, events = _modal_widget(app)
    QCursor.setPos(w.mapToGlobal(QPoint(200, 150)))
    QTest.keyClick(w, Qt.Key_G)
    assert w._modal is not None and events[0]["phase"] == "press"
    QTest.keyClick(w, Qt.Key_X)                                          # solo en X
    w.mouseMoveEvent(_move_event(QPointF(QPoint(260, 190))))
    app.processEvents()
    moves = [e for e in events if e["phase"] == "move"]
    assert moves and moves[-1]["center"][0] != 0.0
    assert moves[-1]["center"][1] == pytest.approx(0.0) and moves[-1]["center"][2] == pytest.approx(0.3)
    QTest.keyClick(w, Qt.Key_Return)                                      # Intro acepta
    assert events[-1]["phase"] == "release" and w._modal is None
    w.close()                                                             # que no recoja el ratón del siguiente
    w.deleteLater()


def test_modal_s_scales_by_the_distance_to_the_center_and_esc_cancels(app):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QCursor
    from PySide6.QtTest import QTest

    w, events = _modal_widget(app)
    cx, cy = w._screen(np.array([[0.0, 0.0, 0.3]]))
    QCursor.setPos(w.mapToGlobal(QPoint(int(cx[0]) + 20, int(cy[0]))))
    QTest.keyClick(w, Qt.Key_S)
    w.mouseMoveEvent(_move_event(QPointF(QPoint(int(cx[0]) + 40, int(cy[0])))))               # el doble de lejos del centro
    app.processEvents()
    last = [e for e in events if e["phase"] == "move"][-1]
    assert last["scale"][0] == pytest.approx(0.8, rel=0.05) and last["scale"][2] == pytest.approx(0.4, rel=0.05)
    QTest.keyClick(w, Qt.Key_Escape)
    assert events[-1]["phase"] == "cancel" and w._modal is None
    w.close()                                                             # que no recoja el ratón del siguiente
    w.deleteLater()


def test_modal_r_rotates_and_a_click_accepts_and_g_does_nothing_without_a_selected_area(app):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QCursor
    from PySide6.QtTest import QTest

    w, events = _modal_widget(app)
    cx, cy = w._screen(np.array([[0.0, 0.0, 0.3]]))
    QCursor.setPos(w.mapToGlobal(QPoint(int(cx[0]) + 40, int(cy[0]))))
    QTest.keyClick(w, Qt.Key_R)
    QTest.keyClick(w, Qt.Key_Z)                                           # giro sobre el eje Z (la vista frontal lo ve)
    w.mouseMoveEvent(_move_event(QPointF(QPoint(int(cx[0]), int(cy[0]) - 40))))
    app.processEvents()
    last = [e for e in events if e["phase"] == "move"][-1]
    assert abs(last["rotation"][2]) > 10.0
    w.mousePressEvent(_press_event(QPointF(int(cx[0]), int(cy[0]) - 40)))
    assert events[-1]["phase"] == "release"
    w.set_areas([], -1)
    events.clear()
    QTest.keyClick(w, Qt.Key_G)
    assert w._modal is None and events == []
    w.close()                                                             # que no recoja el ratón del siguiente
    w.deleteLater()


def test_modal_keys_are_not_stolen_by_global_shortcuts_like_z(app):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QCursor, QKeyEvent
    from PySide6.QtTest import QTest

    w, _events = _modal_widget(app)
    ev = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Z, Qt.NoModifier, "z")
    ev.ignore()
    w.event(ev)
    assert not ev.isAccepted()                                            # fuera de G / S / R, el atajo global sí vale
    QCursor.setPos(w.mapToGlobal(QPoint(200, 150)))
    QTest.keyClick(w, Qt.Key_G)
    ev.ignore()
    w.event(ev)
    assert ev.isAccepted()                                                # en G / S / R, la Z es del eje
    QTest.keyClick(w, Qt.Key_Escape)
    w.close()
    w.deleteLater()


# --- Planos de banda y capa de foco en 3D ----------------------------------------------------------------------
def test_band_plane_is_flat_without_pan_and_tilt_and_leans_like_the_effect_with_them():
    from deepshot import tonemap
    from deepshot.view3d import band_plane

    flat = band_plane(0.5, 0.0, 0.0, 0.0, 1.0, 2.0, 1.5)
    assert flat.shape == (4, 3) and np.allclose(flat[:, 2], 1.0)                     # z0 = 0,5 · relieve 2
    leaning = band_plane(0.5, 30.0, 0.0, 0.0, 1.0, 2.0, 1.5)
    assert leaning[1, 2] > 1.0 > leaning[0, 2]                                        # pan > 0: se aleja hacia la derecha
    assert leaning[0, 2] + leaning[1, 2] == pytest.approx(2.0)                        # pasa por la altura del centro
    # Coincide con el efecto: la profundidad inclinada vale la de la banda justo en el plano dibujado.
    depth = np.full((20, 30), 0.5, dtype=np.float32)
    tilted = tonemap.tilted_depth(depth, 30.0, 0.0, 2.0)
    x_right = (29 + 0.5) / 30 * 2.0 - 1.0
    assert tilted[10, 29] == pytest.approx(0.5 - x_right * 1.5 * np.tan(np.radians(30.0)) / 2.0, abs=1e-4)
    assert band_plane(0.5, 0.0, 45.0, 0.0, 1.0, 1.0, 1.0)[0, 2] > band_plane(0.5, 0.0, 45.0, 0.0, 1.0, 1.0, 1.0)[2, 2]


def test_widget_draws_band_planes_and_tints_the_focus_zone_green(app):
    from deepshot.view3d import band_plane

    w = make_widget(app)
    base = w._render_frame().copy()
    mask = np.full((20, 30), 255, dtype=np.uint8)
    w.set_focus_overlay(mask)
    tinted = w._render_frame()
    assert tinted != base                                                             # el relieve se tiñe de verde
    pix = tinted.pixelColor(tinted.width() // 2, tinted.height() // 2)
    assert pix.green() > pix.red() and pix.green() > pix.blue()
    w.set_focus_overlay(None)
    assert w._render_frame() == base                                                  # al quitarla vuelve el color de antes
    from deepshot.area3d import BandFrame

    frame = BandFrame(0.5, 0.1, 20.0, 0.0, 0.0, 1.0, 1.0, 1.5)
    w.set_bands([(band_plane(0.5, 20.0, 0.0, 0.0, 1.0, 1.0, 1.5), (1.0, 1.0, 1.0), frame, 0)], selected=0)
    w.grab()                                                                          # dibujar los planos no falla
    w.close()
    w.deleteLater()
