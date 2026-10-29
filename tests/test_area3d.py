"""Edición de áreas en 3D: geometría, rayos, gestos y su uso desde el visor y la ventana."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from deepshot import area3d as a3  # noqa: E402
from deepshot.app import create_app  # noqa: E402
from deepshot.areas import Area, Scene, rotation_matrix  # noqa: E402
from deepshot.ui.view3d_widget import View3D  # noqa: E402
from deepshot.view3d import OrbitCamera  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return create_app([])


def cam(yaw=0.0, pitch=0.0, distance=4.0, target=(0, 0, 0)):
    return OrbitCamera(yaw=yaw, pitch=pitch, distance=distance, target=np.array(target, dtype=np.float64))


def ray_at(camera, point, w=800, h=600):
    """Rayo de la cámara que pasa justo por un punto del mundo."""
    sx, sy, _ = camera.project(np.asarray(point, dtype=np.float64)[None, :], w, h)
    return a3.ray_through(camera, float(sx[0]), float(sy[0]), w, h)


# --- Rotaciones -----------------------------------------------------------------------------------
@pytest.mark.parametrize("angles", [(0, 0, 0), (30, 20, 10), (-45, 60, 170), (10, -80, -120), (90, 30, 0)])
def test_euler_roundtrip_matches_rotation_matrix(angles):
    r = rotation_matrix(*angles).astype(np.float64)
    back = a3.euler_from_matrix(r)
    assert np.allclose(rotation_matrix(*back), r, atol=1e-5)


def test_euler_handles_gimbal_lock():
    r = rotation_matrix(25, 90, 40).astype(np.float64)
    assert np.allclose(rotation_matrix(*a3.euler_from_matrix(r)), r, atol=1e-5)


# --- Contornos -------------------------------------------------------------------------------------
def test_wireframe_points_lie_on_the_ellipsoid_and_box_surface():
    area = Area(shape="ellipsoid", center=(0.3, -0.2, 0.6), scale=(0.5, 0.3, 0.2), rotation=(20, 35, 50))
    lines = a3.wireframe(area)
    assert len(lines) == 3
    for line in lines:
        assert np.allclose(area.distance(line.astype(np.float32)), 1.0, atol=1e-4)
    box = Area(shape="box", center=(0, 0, 0.5), scale=(0.4, 0.2, 0.1), rotation=(10, 20, 30))
    edges = a3.wireframe(box)
    assert len(edges) == 12
    for e in edges:
        assert np.allclose(box.distance(e.astype(np.float32)), 1.0, atol=1e-4)     # las aristas están en la superficie


def test_scale_handles_sit_at_the_positive_ends_of_each_local_axis():
    area = Area(center=(0.1, 0.2, 0.3), scale=(0.5, 0.3, 0.2), rotation=(0, 0, 90))
    h = a3.scale_handle_points(area)
    assert np.allclose(h["x"], [0.1, 0.2 + 0.5, 0.3], atol=1e-6)        # X local girado 90° apunta a +y del mundo
    assert np.allclose(h["z"], [0.1, 0.2, 0.3 + 0.2], atol=1e-6)


# --- Rayos y proyecciones -------------------------------------------------------------------------
def test_center_ray_points_at_the_target_and_param_on_axis_recovers_distance():
    c = cam(yaw=30, pitch=20)
    ray = a3.ray_through(c, 400, 300, 800, 600)
    assert np.allclose(ray.direction, (c.target - c.position()) / np.linalg.norm(c.target - c.position()), atol=1e-9)
    origin = np.array([0.0, 0.0, 0.0])
    point = origin + a3.AXES["x"] * 0.37
    s = a3.param_on_axis(ray_at(c, point), origin, a3.AXES["x"])
    assert s == pytest.approx(0.37, abs=1e-6)
    assert a3.param_on_axis(a3.Ray(np.array([0.0, 0, -4]), np.array([1.0, 0, 0])), origin, a3.AXES["x"]) is None


def test_hit_plane_rejects_parallel_and_behind():
    ray = a3.Ray(np.array([0.0, 0, -4]), np.array([0.0, 0, 1.0]))
    assert np.allclose(a3.hit_plane(ray, np.zeros(3), np.array([0, 0, 1.0])), [0, 0, 0])
    assert a3.hit_plane(ray, np.zeros(3), np.array([1.0, 0, 0])) is None                  # paralelo
    assert a3.hit_plane(ray, np.array([0, 0, -9.0]), np.array([0, 0, 1.0])) is None       # detrás


# --- Gestos ----------------------------------------------------------------------------------------
def test_move_along_axis_changes_only_that_axis():
    c = cam(yaw=25, pitch=15)
    area = Area(center=(0.1, 0.2, 0.3))
    start = ray_at(c, area.center)
    now = ray_at(c, np.array(area.center) + a3.AXES["z"] * 0.25)
    moved = a3.move_along_axis(area, "z", start, now)
    assert np.allclose(moved.center, (0.1, 0.2, 0.55), atol=1e-6)
    assert a3.move_along_axis(area, "x", start, now).center[2] == pytest.approx(0.3)    # el eje x no mueve z


def test_move_in_view_plane_follows_the_mouse_in_the_screen_plane():
    c = cam(yaw=0, pitch=0, target=(0, 0, 0.3))
    area = Area(center=(0.0, 0.0, 0.3))
    moved = a3.move_in_view_plane(area, c, ray_at(c, (0, 0, 0.3)), ray_at(c, (0.4, -0.2, 0.3)))
    assert np.allclose(moved.center, (0.4, -0.2, 0.3), atol=1e-6)
    c2 = cam(yaw=50, pitch=30, target=(0, 0, 0.3))                                       # desde otro ángulo: no cambia la distancia
    m2 = a3.move_in_view_plane(area, c2, ray_at(c2, (0, 0, 0.3)), ray_at(c2, (0.1, 0.1, 0.3)))
    d0 = np.dot(np.array(m2.center) - np.array(area.center), c2.basis()[2])
    assert d0 == pytest.approx(0.0, abs=1e-6)


def test_scale_along_axis_sets_semiaxis_to_distance_and_uniform_scales_all():
    c = cam(yaw=20, pitch=10)
    area = Area(center=(0, 0, 0.5), scale=(0.4, 0.2, 0.1), rotation=(0, 0, 0))
    target = np.array(area.center) + a3.AXES["y"] * 0.6
    new = a3.scale_along_axis(area, "y", ray_at(c, target))
    assert np.allclose(new.scale, (0.4, 0.6, 0.1), atol=1e-6)
    uni = a3.scale_along_axis(area, "y", ray_at(c, target), uniform=True)
    assert np.allclose(uni.scale, (0.4 * 3, 0.6, 0.1 * 3), atol=1e-6)                    # el semieje y pasa de 0.2 a 0.6: ×3
    tiny = a3.scale_along_axis(area, "x", ray_at(c, np.array(area.center)))              # sobre el centro: nunca 0
    assert tiny.scale[0] >= a3.MIN_SCALE


def test_rotate_about_world_axis_composes_with_the_existing_rotation_and_snaps():
    c = cam(yaw=0, pitch=0, target=(0, 0, 0))
    area = Area(center=(0, 0, 0), rotation=(0, 0, 0))
    r0 = ray_at(c, (0.5, 0, 0))
    r1 = ray_at(c, (0, 0.5, 0))                                                          # 90° en sentido antihorario
    turned = a3.rotate_about_axis(area, "z", r0, r1)
    assert np.allclose(rotation_matrix(*turned.rotation), rotation_matrix(0, 0, 90), atol=1e-4)
    tilted = Area(center=(0, 0, 0), rotation=(30, 0, 0))
    again = a3.rotate_about_axis(tilted, "z", r0, r1)
    expected = a3.axis_rotation("z", np.pi / 2) @ rotation_matrix(30, 0, 0).astype(np.float64)
    assert np.allclose(rotation_matrix(*again.rotation), expected, atol=1e-4)
    r2 = ray_at(c, (0.5, 0.11, 0))                                                       # ≈ 12.4°: con Ctrl, 15°
    snapped = a3.rotate_about_axis(area, "z", r0, r2, snap=True)
    assert snapped.rotation[2] == pytest.approx(15.0, abs=1e-3)


# --- Visor -----------------------------------------------------------------------------------------
def make_view(app, areas, selected=0):
    w = View3D()
    w.resize(800, 600)
    w.show()
    depth = np.tile(np.linspace(0, 1, 120, dtype=np.float32), (80, 1))
    w.set_data(depth, Scene(relief=1.0), None)
    w.camera.set_view(0, 0)
    w.set_areas(areas, selected)
    app.processEvents()
    return w


def to_qpoint(w, world):
    sx, sy = w._screen(np.asarray(world, dtype=np.float64)[None, :])
    return QPoint(int(round(sx[0])), int(round(sy[0])))


def test_hit_test_finds_each_handle_and_other_areas(app):
    a = Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3))
    b = Area(name="B", center=(1.0, 0.0, 0.5), scale=(0.2, 0.2, 0.2))
    w = make_view(app, [a, b], 0)
    c = np.array(a.center)
    length = a3.gizmo_length(w.camera)
    assert w.hit_test(QPointF(to_qpoint(w, c)))[0] == "move_free"
    assert w.hit_test(QPointF(to_qpoint(w, a3.scale_handle_points(a)["x"])))[0] == "scale_x"
    assert w.hit_test(QPointF(to_qpoint(w, c + a3.AXES["y"] * length * 0.9)))[0] == "move_y"
    assert w.hit_test(QPointF(to_qpoint(w, a3.ring_points(c, "z", length * 0.8)[10])))[0] == "rotate_z"
    kind, index = w.hit_test(QPointF(to_qpoint(w, a3.wireframe(b)[0][5])))              # contorno de otra área
    assert (kind, index) == ("select", 1)
    assert w.hit_test(QPointF(5, 5)) == ("", -1)
    w.deleteLater()


def test_dragging_an_arrow_emits_press_moves_release_and_a_click_only_selects(app):
    a = Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3))
    w = make_view(app, [a], 0)
    events = []
    w.areaGesture.connect(events.append)
    c = np.array(a.center)
    length = a3.gizmo_length(w.camera)
    start = to_qpoint(w, c + a3.AXES["x"] * length * 0.9)
    QTest.mousePress(w, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(w, start + QPoint(40, 0))
    QTest.mouseMove(w, start + QPoint(80, 0))
    QTest.mouseRelease(w, Qt.LeftButton, Qt.NoModifier, start + QPoint(80, 0))
    phases = [e["phase"] for e in events]
    assert phases[0] == "press" and phases[-1] == "release" and phases.count("move") >= 1
    assert all(e["kind"] == "move_x" for e in events)
    last = [e for e in events if e["phase"] == "move"][-1]
    assert last["center"][0] > 0.05 and last["center"][1] == pytest.approx(0.0, abs=1e-6)       # solo se movió en x
    events.clear()
    QTest.mouseClick(w, Qt.LeftButton, Qt.NoModifier, start)                                      # clic sin mover
    assert [e["phase"] for e in events] == ["press", "release"]                                   # no hay «move»
    w.deleteLater()


def test_escape_cancels_the_gesture_and_empty_space_still_orbits(app):
    a = Area(name="A", center=(0.0, 0.0, 0.5), scale=(0.3, 0.3, 0.3))
    w = make_view(app, [a], 0)
    events = []
    w.areaGesture.connect(events.append)
    start = to_qpoint(w, a3.scale_handle_points(a)["y"])
    QTest.mousePress(w, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(w, start + QPoint(0, -30))
    QTest.keyClick(w, Qt.Key_Escape)
    assert events[-1]["phase"] == "cancel"
    QTest.mouseRelease(w, Qt.LeftButton, Qt.NoModifier, start)
    assert [e["phase"] for e in events].count("release") == 0
    yaw = w.camera.yaw
    QTest.mousePress(w, Qt.LeftButton, Qt.NoModifier, QPoint(700, 80))                           # espacio vacío: órbita
    QTest.mouseMove(w, QPoint(740, 80))
    QTest.mouseRelease(w, Qt.LeftButton, Qt.NoModifier, QPoint(740, 80))
    assert w.camera.yaw != yaw
    w.deleteLater()


def test_scale_by_factor_scales_all_semiaxes_or_only_the_chosen_one_with_a_floor():
    from deepshot import area3d
    from deepshot.areas import Area

    a = Area(scale=(0.4, 0.3, 0.2))
    assert area3d.scale_by_factor(a, 2.0).scale == pytest.approx((0.8, 0.6, 0.4))
    assert area3d.scale_by_factor(a, 2.0, "y").scale == pytest.approx((0.4, 0.6, 0.2))
    assert min(area3d.scale_by_factor(a, 0.0).scale) >= area3d.MIN_SCALE


def test_tilted_area_gets_its_real_silhouette_in_2d_and_a_flat_one_keeps_the_ellipse():
    from deepshot import area_edit
    from deepshot.areas import Area, Scene

    geo = area_edit.ImageGeometry(600, 400, Scene(), None)
    flat = Area(center=(0.0, 0.0, 0.3), scale=(0.5, 0.3, 0.3))
    assert area_edit.silhouette(flat, geo) is None and "poly" not in area_edit.outline(flat, geo)
    tilted = Area(center=(0.0, 0.0, 0.3), scale=(0.5, 0.3, 0.3), rotation=(0.0, 60.0, 0.0))
    out = area_edit.outline(tilted, geo)
    xs = [p[0] for p in out["poly"]]
    # Girada 60° sobre el eje vertical, el área se ve más estrecha que su elipse plana.
    assert len(out["poly"]) >= 8 and (max(xs) - min(xs)) / 2.0 < out["rx"] * 0.75
    box = Area(shape="box", scale=(0.4, 0.4, 0.4), rotation=(30.0, 30.0, 0.0))
    assert len(area_edit.silhouette(box, geo)) >= 4
