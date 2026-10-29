"""Modo sencillo: normalizar, invertir, mapa de normales y exportación; y su ventana."""

from pathlib import Path
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from deepshot import simple_depth as sd


def ramp(h=40, w=60):
    return np.tile(np.linspace(0.2, 0.6, w, dtype=np.float32), (h, 1))      # 0 = cerca: la izquierda es lo cercano


def test_normalize_stretches_to_full_range_and_flat_maps_give_zeros():
    out = sd.normalize_range(ramp())
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)
    assert sd.normalize_range(np.full((4, 4), 0.3, np.float32)).max() == 0.0
    spiky = ramp()
    spiky[0, 0] = 5.0                                                          # un punto suelto aplasta el rango
    assert sd.normalize_range(spiky, 1.0, 99.0)[20, 30] > sd.normalize_range(spiky)[20, 30]


def test_finish_depth_near_bright_inverts_and_normalize_is_optional():
    d = ramp()
    bright = sd.finish_depth(d, normalize=True, near_bright=True)
    dark = sd.finish_depth(d, normalize=True, near_bright=False)
    assert bright[0, 0] == pytest.approx(1.0) and dark[0, 0] == pytest.approx(0.0)       # lo cercano (izquierda)
    assert np.allclose(bright, 1.0 - dark)
    raw = sd.finish_depth(d, normalize=False, near_bright=False)
    assert raw.min() == pytest.approx(0.2) and raw.max() == pytest.approx(0.6)


def test_normal_map_points_away_from_a_slope_and_follows_the_convention():
    d = ramp()                                       # cerca a la izquierda = alto a la izquierda: la superficie baja hacia la derecha
    opengl = sd.normal_map(d, strength=2.0, convention="opengl")
    assert opengl.shape == d.shape + (3,) and opengl.dtype == np.float32 and 0.0 <= opengl.min() <= opengl.max() <= 1.0
    centre = opengl[20, 30]
    assert centre[0] > 0.5 + 0.01                    # la normal mira hacia la derecha (+X): la pendiente baja a la derecha
    assert centre[1] == pytest.approx(0.5, abs=1e-3) and centre[2] > 0.5       # sin pendiente vertical; Z hacia fuera
    # Una pendiente vertical (cerca arriba): OpenGL y DirectX solo difieren en el verde, invertido.
    v = np.tile(np.linspace(0.2, 0.6, 40, dtype=np.float32)[:, None], (1, 60))
    gl, dx = sd.normal_map(v, 2.0, "opengl"), sd.normal_map(v, 2.0, "directx")
    assert gl[20, 30, 1] == pytest.approx(1.0 - dx[20, 30, 1], abs=1e-5) and gl[20, 30, 1] != pytest.approx(0.5, abs=0.01)
    assert np.allclose(gl[..., 0], dx[..., 0]) and np.allclose(gl[..., 2], dx[..., 2])
    # Con más intensidad las normales se inclinan más.
    assert abs(sd.normal_map(d, 5.0)[20, 30, 0] - 0.5) > abs(sd.normal_map(d, 0.5)[20, 30, 0] - 0.5)
    # Vectores unitarios.
    n = opengl * 2 - 1
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-4)
    with pytest.raises(ValueError):
        sd.normal_map(d, 1.0, "metal")


def test_export_simple_writes_depth_and_normal_without_overwriting(tmp_path):
    opts = sd.SimpleOptions(depth_map=True, normal=True, format="png16", folder=tmp_path / "out")
    first = sd.export_simple(ramp(), "mi foto.jpg", opts)
    assert [p.name for p in first] == ["mi foto_depth.png", "mi foto_normal.png"]
    second = sd.export_simple(ramp(), "mi foto.jpg", opts)
    assert [p.name for p in second] == ["mi foto_depth_2.png", "mi foto_normal_2.png"]          # no pisa
    depth_png, normal_png = Image.open(first[0]), Image.open(first[1])
    assert depth_png.size == (60, 40) and depth_png.mode.startswith("I")                          # gris de 16 bits
    assert normal_png.size == (60, 40) and len(normal_png.getbands()) == 3
    exr = sd.export_simple(ramp(), "x.png", sd.SimpleOptions(format="exr32", folder=tmp_path))
    assert exr[0].suffix == ".exr" and exr[0].is_file()
    with pytest.raises(ValueError):
        sd.export_simple(ramp(), "x.png", sd.SimpleOptions(depth_map=False, normal=False, folder=tmp_path))


def test_file_tag_and_json_next_to_each_image(tmp_path):
    assert sd.file_tag("marigold-v1-0", "high") == "marigold-v1-0_high"
    assert sd.file_tag("a/b", "fast", 512) == "a-b_fast_512px"
    opts = sd.SimpleOptions(folder=tmp_path, tag="m_fast", save_json=True, info={"model": "m", "quality": "fast"})
    written = sd.export_simple(ramp(), "f.jpg", opts)
    assert [p.name for p in written] == ["f_depth_m_fast.png", "f_depth_m_fast.png.json"]
    import json
    data = json.loads(written[1].read_text(encoding="utf-8"))
    assert data["model"] == "m" and data["file"] == "f_depth_m_fast.png" and data["near_bright"] is True


# --- Ventana ---------------------------------------------------------------------------------------------
def test_dialog_creates_the_requested_maps_for_every_photo_and_reports(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from deepshot.app import create_app
    from deepshot.ui.main_window import MainWindow
    from deepshot.ui.simple_depth_dialog import SimpleDepthDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ini"))
    create_app([])
    win = MainWindow()
    win.ask_before_closing = False
    photos = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.new("RGB", (60, 40), (90, 120, 160)).save(p)
        photos.append(p)

    calls = []

    class FakeEstimator:
        loaded_model = None

        def estimate(self, path, settings, progress=None, use_cache=True):
            calls.append((path.name, settings.processing_resolution))
            return ramp()

    win.estimator = FakeEstimator()
    dialog = SimpleDepthDialog(win)
    dialog._worker.estimator = win.estimator
    assert not dialog.run_button.isEnabled()                              # sin fotos no se puede crear
    dialog.add_photos(photos + photos)                                    # los repetidos no se añaden
    assert dialog.photos.count() == 2 and dialog.run_button.isEnabled()
    dialog.folder_edit.setText(str(tmp_path / "salida"))
    dialog.normal_check.setChecked(True)
    dialog.resolution_combo.setCurrentIndex(dialog.resolution_combo.findData(512))
    dialog.run()
    import time

    deadline = time.time() + 20
    while dialog._worker.busy and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        QApplication.processEvents()
    assert [c[0] for c in calls] == ["a.png", "b.png"] and all(c[1] == 512 for c in calls)    # la resolución elegida
    names = sorted(p.name for p in (tmp_path / "salida").iterdir())
    tag = "marigold-v1-0_high_512px"                                    # modelo, calidad y resolución elegida
    assert names == [f"a_depth_{tag}.png", f"a_normal_{tag}.png", f"b_depth_{tag}.png", f"b_normal_{tag}.png"]
    assert "2 de 2" in dialog.status.text() and dialog.run_button.isEnabled()
    assert dialog.results.count() == 2 and "a.png" in dialog.results.item(0).text() and "Listo en" in dialog.results.item(0).text()
    assert "512 px" in dialog.result_label.text() and "b.png" in dialog.result_label.text()   # lo último, en verde
    assert dialog.memory_button.objectName() == "MemoryIndicator"
    # Las opciones se recuerdan.
    assert QSettings().value("simple/normal", False, type=bool) is True
    # Sin ningún resultado marcado, no hay nada que crear.
    dialog.depth_check.setChecked(False)
    dialog.normal_check.setChecked(False)
    assert not dialog.run_button.isEnabled()
    dialog.deleteLater()
    win._probe.wait()
    win.close()


def test_start_screen_has_a_quick_depth_button_that_opens_the_simple_mode(tmp_path, monkeypatch):
    from deepshot.app import create_app
    from deepshot.ui.start_dialog import StartDialog

    create_app([])
    dialog = StartDialog([], None, False)
    assert dialog.quick_button.isEnabled() and dialog.quick_button.toolTip()
    dialog.quick_button.click()
    assert dialog.action == "quick"


# --- Visor de revisión rápida ---------------------------------------------------------------------------
def test_quick_viewer_reads_png8_png16_exr_and_the_json_next_to_them(tmp_path):
    from deepshot.app import create_app
    from deepshot.ui.quick_viewer import QuickViewer, load_for_review

    create_app([])
    opts = sd.SimpleOptions(folder=tmp_path, tag="m_fast", save_json=True, info={"model": "m", "quality": "fast"})
    files = []
    for fmt in ("png8", "png16", "exr32"):
        opts.format = fmt
        files += [p for p in sd.export_simple(ramp(), "f.jpg", opts) if p.suffix != ".json"]
    infos = {p.suffix + str(i): load_for_review(p)[1] for i, p in enumerate(files)}
    formats = [i.format for i in infos.values()]
    assert formats == ["PNG 8 bits", "PNG 16 bits", "EXR 32 bits"]
    first = next(iter(infos.values()))
    assert (first.width, first.height) == (60, 40) and first.model == "m" and first.quality == "fast"
    assert first.size_text.endswith("B")
    viewer = QuickViewer(files + [tmp_path / "f_depth_m_fast.png.json"])         # el .json no cuenta como imagen
    assert len(viewer.files) == 3 and "60 × 40" in viewer.details.text() and "Modelo: m" in viewer.details.text()
    viewer.step(1)
    assert viewer.index == 1 and "(2/3)" in viewer.title.text()
    viewer.step(5)                                                                # da la vuelta
    assert viewer.index == 0
    viewer.deleteLater()
    assert QuickViewer([]).title.text()                                           # sin imágenes no falla


# --- Aplicación básica (sin editor) ---------------------------------------------------------------------
def test_quick_app_does_not_load_the_editor_and_builds_its_own_host(tmp_path):
    import subprocess
    import sys

    code = ("import sys, deepshot.quick_app as q\n"
            "bad = [m for m in sys.modules if m in ('deepshot.ui.main_window', 'deepshot.ui.editor', 'deepshot.ui.canvas',"
            " 'deepshot.ui.editor_panels', 'deepshot.app')]\n"
            "print(bad)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert out.stdout.strip() == "[]", out.stdout + out.stderr


def test_quick_dialog_works_with_a_standalone_host(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from deepshot import quick_app
    from deepshot.app import create_app
    from deepshot.ui.simple_depth_dialog import SimpleDepthDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ini"))
    create_app([])
    host = quick_app.build_host()
    dialog = SimpleDepthDialog(host)
    assert dialog.model_combo.count() >= 1 and dialog.quality_combo.count() == 3 and dialog.resolution_combo.count() > 1
    dialog.add_library_photos()                                   # sin biblioteca: no añade nada ni falla
    assert dialog.photos.count() == 0 and not dialog.run_button.isEnabled()
    dialog.deleteLater()


def test_memory_indicator_widget_shows_levels_and_the_dialog_frees_memory(tmp_path, monkeypatch):
    from deepshot import memory
    from deepshot.app import create_app
    from deepshot.ui.memory_indicator import MemoryIndicator
    from deepshot.ui.simple_depth_dialog import SimpleDepthDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    create_app([])
    monkeypatch.setattr(memory, "gpu_free_total_mb", lambda: (200.0, 8000.0))
    monkeypatch.setattr(memory, "ram_free_total_mb", lambda: (9000.0, 16000.0))
    w = MemoryIndicator()
    assert w.property("tone") == "bad" and "VRAM 7.8/8.0 GB" in w.text() and MemoryIndicator.levels() == ("bad", "ok")
    from deepshot import quick_app

    host = quick_app.build_host()
    unloaded = []
    host.estimator.unload = lambda: unloaded.append(1)
    dialog = SimpleDepthDialog(host)
    dialog.memory_button.click()
    assert unloaded == [1] and "liberada" in dialog.status.text()
    dialog.deleteLater()
    w.deleteLater()


def test_hover_help_does_not_resize_the_quick_dialog(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from deepshot import quick_app
    from deepshot.app import create_app
    from deepshot.ui.simple_depth_dialog import SimpleDepthDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ini"))
    app = create_app([])
    dialog = SimpleDepthDialog(quick_app.build_host())
    dialog.show()
    app.processEvents()
    before = (dialog.size(), dialog.layout().sizeHint(), dialog.hint.height())
    for text in ("", "Corto", "Un texto de ayuda bastante largo " * 12):          # lo que saldría al pasar por varios controles
        dialog.hint.setText(text)
        app.processEvents()
        assert (dialog.size(), dialog.layout().sizeHint(), dialog.hint.height()) == before
    dialog.close()
    dialog.deleteLater()


def test_quick_dialog_opens_the_model_manager_and_refreshes_the_model_list(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from deepshot import quick_app
    from deepshot.app import create_app
    from deepshot.ui.simple_depth_dialog import SimpleDepthDialog

    monkeypatch.setenv("DEEPSHOT_DATA_DIR", str(tmp_path / "data"))
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "ini"))
    create_app([])
    host = quick_app.build_host()
    opened = []

    def fake_manager():
        opened.append(1)
        host.model_combo.addItem("Modelo nuevo", "nuevo")                       # lo que cambiaría el gestor
        host.model_combo.setCurrentIndex(host.model_combo.findData("nuevo"))

    host.open_model_manager = fake_manager
    dialog = SimpleDepthDialog(host)
    assert dialog.models_button.isVisibleTo(dialog) and dialog.models_button.toolTip()
    before = dialog.model_combo.count()
    dialog.models_button.click()
    assert opened == [1] and dialog.model_combo.count() == before + 1 and dialog.model_combo.currentData() == "nuevo"
    host2 = quick_app.build_host()
    host2.open_model_manager = None                                             # un anfitrión sin gestor: no hay botón
    assert not SimpleDepthDialog(host2).models_button.isVisibleTo(dialog)
    dialog.deleteLater()


def test_quick_mode_is_one_click_away_from_the_header_and_the_start_screen(tmp_path, monkeypatch):
    from deepshot.app import create_app
    from deepshot.ui.start_dialog import StartDialog

    create_app([])
    start = StartDialog([], None)
    assert start.quick_button.objectName() == "Primary" and start.new_button.objectName() == "Primary"
    start._quick()
    assert start.action == "quick"
