"""Máscara pintada con pincel: capa (paint.py), integración con el render y con deshacer."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtGui import QUndoStack  # noqa: E402

from deepshot.app import create_app  # noqa: E402
from deepshot.paint import MAX_SIDE, MaskLayer, layer_size  # noqa: E402
from deepshot.project import Project, render  # noqa: E402
from deepshot.ui.editor import EditSession  # noqa: E402

W, H = 400, 300


def test_layer_size_limits_long_side():
    assert layer_size((400, 300)) == (400, 300)
    w, h = layer_size((6000, 4000))
    assert w == MAX_SIDE and h == round(4000 * MAX_SIDE / 6000)


def test_hard_brush_paints_disc_and_erase_removes_it():
    m = MaskLayer((W, H))
    m.dab(200, 150, radius=30, hardness=1.0, opacity=1.0)
    assert m.data[150, 200] == 255 and m.data[150, 225] == 255       # dentro del disco
    assert m.data[150, 250] == 0 and m.data[10, 10] == 0             # fuera
    m.dab(200, 150, radius=30, hardness=1.0, opacity=1.0, erase=True)
    assert m.is_empty()


def test_soft_brush_has_gradient_and_opacity_accumulates():
    m = MaskLayer((W, H))
    m.dab(200, 150, radius=40, hardness=0.0, opacity=0.5)
    row = m.data[150, 200:240].astype(int)
    assert row[0] > row[20] > row[39] >= 0                            # baja del centro al borde
    centre_once = int(m.data[150, 200])
    assert 120 <= centre_once <= 135                                  # ~50 % con una pasada
    m.dab(200, 150, radius=40, hardness=0.0, opacity=0.5)
    assert m.data[150, 200] > centre_once + 50                        # la segunda pasada acumula
    assert m.data[150, 200] <= 255


def test_stroke_is_continuous_between_distant_points():
    m = MaskLayer((W, H))
    m.begin_stroke()
    m.stroke_to(20, 150, 6, 1.0, 1.0)
    m.stroke_to(380, 150, 6, 1.0, 1.0)                                 # salto grande entre eventos del ratón
    assert (m.data[150, 20:381] == 255).all()                          # sin huecos
    m.begin_stroke()
    m.stroke_to(20, 20, 3, 1.0, 1.0)                                    # un trazo nuevo no se une al anterior
    assert m.data[100, 20] == 0


def test_brush_near_border_does_not_fail_and_outside_is_ignored():
    m = MaskLayer((W, H))
    assert m.dab(-5, -5, 20, 0.5, 1.0) is not None and m.data[0, 0] > 0
    assert m.dab(5000, 5000, 20, 0.5, 1.0) is None


def test_layer_is_downscaled_for_big_photos_but_brush_uses_photo_pixels():
    m = MaskLayer((6000, 4000))
    assert m.width == MAX_SIDE
    m.dab(3000, 2000, radius=300, hardness=1.0, opacity=1.0)           # 300 px de la foto
    painted = int((m.data > 0).sum())
    expected = np.pi * (300 * m.scale) ** 2
    assert painted == pytest.approx(expected, rel=0.05)
    assert m.to_float((600, 400)).shape == (400, 600)


def test_snapshot_restore_fill_invert_and_files(tmp_path):
    m = MaskLayer((W, H))
    blob = m.snapshot()
    m.fill(1.0)
    assert (m.data == 255).all()
    m.invert()
    assert m.is_empty()
    m.dab(100, 100, 20, 1.0, 1.0)
    m.save(tmp_path / "sub" / "m.png")
    other = MaskLayer.load(tmp_path / "sub" / "m.png", (W, H))
    assert np.array_equal(other.data, m.data)
    m.restore(blob)
    assert m.is_empty()
    # PNG de otro tamaño y de 16 bits: se adapta
    Image.fromarray((np.linspace(0, 65535, 50 * 40).reshape(40, 50)).astype(np.uint16)).save(tmp_path / "g16.png")
    big = MaskLayer.load(tmp_path / "g16.png", (W, H))
    assert big.data.shape == (H, W) and big.data[0, 0] < 5 and big.data[-1, -1] > 250


# --- Integración con el render ----------------------------------------------------------------------------
def test_render_uses_painted_mask_array():
    depth = np.tile(np.linspace(0, 1, 40, dtype=np.float32), (30, 1))
    photo = np.zeros((30, 40, 3), dtype=np.float32)
    photo[..., 0] = 1.0                                                  # foto roja
    p = Project()
    p.composite.opacity, p.composite.mask = 1.0, "painted"
    mask = np.zeros((30, 40), dtype=np.float32)
    mask[:, :20] = 1.0                                                   # solo la mitad izquierda
    out = render(p, depth, photo, painted_mask=mask)
    assert out[10, 5, 0] == pytest.approx(1.0) and out[10, 5, 1] == pytest.approx(0.0)    # foto roja
    assert out[10, 35, 0] == pytest.approx(out[10, 35, 1])                                 # rampa gris
    p.composite.invert_mask = True
    inv = render(p, depth, photo, painted_mask=mask)
    assert inv[10, 35, 1] == pytest.approx(0.0) and inv[10, 5, 0] == pytest.approx(inv[10, 5, 1])


def test_render_with_missing_mask_file_does_not_crash(tmp_path):
    depth = np.full((10, 10), 0.5, dtype=np.float32)
    p = Project()
    p.composite.opacity, p.composite.mask, p.composite.painted_mask = 1.0, "painted", str(tmp_path / "no.png")
    render(p, depth, np.zeros((10, 10, 3), dtype=np.float32))


# --- Sesión de edición --------------------------------------------------------------------------------------
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
    app = create_app([])
    photo = tmp_path / "foto.png"
    Image.new("RGB", (W, H), (255, 0, 0)).save(photo)
    depth = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32), (H, 1))
    stack = QUndoStack()
    session = EditSession(stack)
    session.set_photo(photo, depth)
    _sessions.append(session)
    return app, session, stack


def wait_result(app, session, timeout=5.0):
    session.result = None
    t = time.time()
    while session.result is None and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return session.result


def stroke(session, points, kind="paint"):
    first = points[0]
    session.gesture({"kind": kind, "phase": "press", "index": -1, "x": first[0], "y": first[1], "mods": 0, "axis": ""})
    for x, y in points[1:]:
        session.gesture({"kind": kind, "phase": "move", "index": -1, "x": x, "y": y, "mods": 0, "axis": ""})
    last = points[-1]
    session.gesture({"kind": kind, "phase": "release", "index": -1, "x": last[0], "y": last[1], "mods": 0, "axis": ""})


def test_painting_requires_painted_mask_selected(tmp_path, monkeypatch):
    _app, s, stack = make_session(tmp_path, monkeypatch)
    stroke(s, [(100, 100), (150, 100)])
    assert s.mask_layer(create=False) is None and stack.count() == 0


def test_paint_stroke_undo_redo_and_persistence(tmp_path, monkeypatch):
    app, s, stack = make_session(tmp_path, monkeypatch)
    panel = s.mix_panel
    panel.mask.setCurrentIndex(panel.mask.findData("painted"))
    panel._set("mask", "painted")                                        # elegir «pintada» en el desplegable
    assert s.project.composite.painted_mask.endswith(".png")             # se asignó su archivo
    n = stack.count()
    panel.brush_size.setValue(10.0)
    panel.brush_hardness.setValue(1.0)
    stroke(s, [(100, 100), (200, 100)])
    layer = s.mask_layer()
    assert layer.data[100, 150] == 255 and stack.count() == n + 1         # un trazo = un paso
    assert os.path.isfile(s.project.composite.painted_mask)              # guardado en disco
    stack.undo()
    assert layer.is_empty()
    stack.redo()
    assert layer.data[100, 150] == 255
    # El archivo guardado se recupera al reabrir la sesión
    s.masks.clear()
    again = s.mask_layer()
    assert again.data[100, 150] == 255


def test_painted_mask_changes_the_rendered_image(tmp_path, monkeypatch):
    app, s, stack = make_session(tmp_path, monkeypatch)
    panel = s.mix_panel
    panel.opacity.setValue(1.0, emit=True)
    panel.mask.setCurrentIndex(panel.mask.findData("painted"))
    panel._set("mask", "painted")
    panel.brush_size.setValue(20.0)
    panel.brush_hardness.setValue(1.0)
    stroke(s, [(100, 150), (130, 150)])
    img = wait_result(app, s)
    deadline = time.time() + 5
    while time.time() < deadline:                                         # la última vista previa
        img = wait_result(app, s, 1.0) or img
        c = img.pixelColor(int(115 / W * img.width()), int(150 / H * img.height()))
        if c.red() > 200 and c.green() < 60:
            break
    c = img.pixelColor(int(115 / W * img.width()), int(150 / H * img.height()))
    assert c.red() > 200 and c.green() < 60                               # donde se pintó: la foto (roja)
    far = img.pixelColor(int(350 / W * img.width()), int(50 / H * img.height()))
    assert abs(far.red() - far.green()) < 5                               # fuera: la rampa gris


def test_erase_fill_invert_clear_and_load_are_undoable(tmp_path, monkeypatch):
    _app, s, stack = make_session(tmp_path, monkeypatch)
    panel = s.mix_panel
    panel.mask.setCurrentIndex(panel.mask.findData("painted"))
    panel._set("mask", "painted")
    layer = s.mask_layer()
    s.edit_mask("fill")
    assert (layer.data == 255).all()
    s.edit_mask("invert")
    assert layer.is_empty()
    s.edit_mask("fill")
    panel.brush_hardness.setValue(1.0)
    panel.brush_size.setValue(10.0)
    stroke(s, [(100, 100), (160, 100)], kind="erase")
    assert layer.data[100, 130] == 0 and layer.data[200, 300] == 255
    stack.undo()
    assert layer.data[100, 130] == 255
    png = tmp_path / "m.png"
    Image.fromarray(np.zeros((30, 40), dtype=np.uint8) + 128).save(png)
    s.edit_mask("load", png)
    assert abs(int(layer.data[10, 10]) - 128) <= 1
    stack.undo()
    assert layer.data[10, 10] == 255
    s.edit_mask("clear")
    assert layer.is_empty()


def test_cancel_stroke_restores_mask(tmp_path, monkeypatch):
    _app, s, stack = make_session(tmp_path, monkeypatch)
    panel = s.mix_panel
    panel.mask.setCurrentIndex(panel.mask.findData("painted"))
    panel._set("mask", "painted")
    n = stack.count()
    s.gesture({"kind": "paint", "phase": "press", "index": -1, "x": 100, "y": 100, "mods": 0, "axis": ""})
    s.gesture({"kind": "paint", "phase": "move", "index": -1, "x": 200, "y": 100, "mods": 0, "axis": ""})
    assert not s.mask_layer().is_empty()
    s.gesture({"kind": "paint", "phase": "cancel", "index": -1, "x": 200, "y": 100, "mods": 0, "axis": ""})
    assert s.mask_layer().is_empty() and stack.count() == n


def test_activate_paint_shortcut_selects_painted_mask(tmp_path, monkeypatch):
    _app, s, stack = make_session(tmp_path, monkeypatch)
    assert s.mix_panel.activate_paint("paint")
    assert s.project.composite.mask == "painted" and s.mix_panel.paint_tool() == "paint"
    s.mix_panel.activate_paint("paint")                                   # pulsar de nuevo la desactiva
    assert s.mix_panel.paint_tool() == ""
    s.mix_panel.activate_paint("erase")
    s.mix_panel.mask.setCurrentIndex(s.mix_panel.mask.findData("none"))
    s.mix_panel._set("mask", "none")
    assert s.mix_panel.paint_tool() == ""                                 # sin máscara pintada no hay pincel


def test_brush_ghost_follows_size_and_hardness(tmp_path, monkeypatch):
    _app, s, _stack = make_session(tmp_path, monkeypatch)
    s.mix_panel.brush_size.setValue(10.0)
    s.mix_panel.brush_hardness.setValue(0.25)
    ghost = s.brush("paint")
    assert ghost["rx"] == pytest.approx(0.10 * W / 2)                     # 10 % del lado largo = diámetro
    assert ghost["soft"] == pytest.approx(0.75)
